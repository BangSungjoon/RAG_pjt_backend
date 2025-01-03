import os
from typing import List, Optional

import json

# 라이브러리 임포트
from kiwipiepy import Kiwi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수를 로드하는 도구
from dotenv import load_dotenv

# FastAPI는 Python 기반 웹 프레임워크
from fastapi import FastAPI

# CORS (Cross-Origin Resource Sharing) 설정을 위한 미들웨어
from fastapi.middleware.cors import CORSMiddleware

# LangChain 라이브러리에서 RetrievalQA 체인을 가져옴
from langchain.chains import RetrievalQA

# Pinecone 벡터 저장소와 관련된 도구
from langchain_pinecone import PineconeVectorStore

# Upstage 모델: ChatUpstage(LLM) 및 UpstageEmbeddings(임베딩 생성기)
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings

# Pinecone 라이브러리: 벡터 검색 관리
from pinecone import Pinecone, ServerlessSpec

# 데이터 유효성 검사를 위한 Pydantic의 BaseModel
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# upstage model 초기화
# ChatUpstage: 대화형 AI 모델
# UpstageEmbeddings: 텍스트를 벡터로 변환하는 모델
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="solar-embedding-1-large")

# Pinecone 초기화
# API 키를 환경 변수에서 가져옴
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "kb-docs"
index = pc.Index(index_name)

# Pinecone에 사용할 벡터 인덱스 이름
# index_name = "finance-pdt"

# OpenAI를 사용하여 쿼리 텍스트 벡터화
gpt_api_key = os.getenv("OPENAI_API_KEY")

# LangChain의 OpenAI LLM 설정
chat = ChatOpenAI(
    model="gpt-4o",  # 또는 "gpt-4o"
    openai_api_key=gpt_api_key,
    temperature=0.7,
    max_tokens=1000
)

# Pinecone 인덱스 생성
# 인덱스가 존재하지 않으면 새 인덱스를 생성
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,          # 인덱스 이름
        dimension=4096,           # 벡터 차원 (Upstage 임베딩 출력 크기)
        metric="cosine",          # 유사도 계산 방식 (코사인 유사도)
        spec=ServerlessSpec(
            cloud="aws",          # 클라우드 제공자
            region="us-east-1"    # 리전
        ),
    )

# Pinecone VectorStore 초기화
# - 벡터 인덱스와 임베딩 생성기를 사용하여 벡터 저장소 생성
pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

# Pinecone Retriever 설정
# - 검색 알고리즘으로 'mmr' 사용
# - 가장 관련성 높은 5개의 문서를 반환하도록 설정
pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # 'similarity' 또는 'mmr'
    search_kwargs={"k": 5, "diversity": 0.7}  # 최대 5개의 관련 문서를 반환
)

# FastAPI 애플리케이션 초기화
app = FastAPI()

# CORS 미들웨어 추가
# - 모든 출처에서 요청을 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 모든 도메인 허용
    allow_credentials=True,     # 쿠키 허용
    allow_methods=["*"],        # 모든 HTTP 메서드 허용
    allow_headers=["*"],        # 모든 헤더 허용
)

# 데이터 모델 정의

# chatMessage: 대화 메시지 구조
class ChatMessage(BaseModel):
    role: str                    # 메시지의 역할 (user/assistant)
    content: str                 # 메시지 내용

# - `AssistantRequest`: 대화 요청 데이터 구조
class AssistantRequest(BaseModel):
    message: str                 # 사용자 입력 메시지
    thread_id: Optional[str] = None  # 대화 스레드 ID (선택)

# - `ChatRequest`: 대화 요청 메시지의 목록 구조
class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # 대화 메시지 리스트

# - `MessageRequest`: 단일 사용자 입력 메시지 구조
class MessageRequest(BaseModel):
    message: str                 # 사용자 입력 메시지

# 채팅 엔드포인트
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):

    # 형태소 분석 및 명사 추출
    kiwi = Kiwi()
    keywords = [token.form for token in kiwi.tokenize(req.messages[-1].content) if token.tag in ['NNG', 'NNP']]  # 일반 명사와 고유 명사 추출
    print(keywords)
    # 기본 file_detail 값 생성
    file_detail_values = []
    # 조건에 따라 file_detail 값 추가
    if "대출" in keywords or "융자" in keywords or "자금" in keywords:
        file_detail_values.extend(["담보대출", "자동차대출", "신용대출"])
    if "예금" in keywords or "저축" in keywords or "예치" in keywords:
        file_detail_values.append("예금")
    if "적금" in keywords or "정기적금" in keywords or "월저축" in keywords:
        file_detail_values.append("적금")
    if "통장" in keywords or "계좌" in keywords or "입출금" in keywords:
        file_detail_values.append("입출금자유")
    if "주택" in keywords or "집" in keywords or "부동산" in keywords:
        file_detail_values.append("담보대출")
    if "차" in keywords or "자동차" in keywords or "차량" in keywords:
        file_detail_values.append("자동차대출")

    # OR 조건으로 필터 생성
    metadata_filter = {
        "file_detail": {"$in": file_detail_values}  # file_detail이 리스트 중 하나에 포함
    }
    print(metadata_filter)
    # Solar-Embedding으로 쿼리 텍스트를 벡터화
    query_vector = embedding_upstage.embed_query(req.messages[-1].content)

    # Pinecone에서 쿼리 실행
    namespace = "kb_namespace"  # 검색할 namespace 설정
    results = index.query(
        vector=query_vector,     # 쿼리 벡터
        namespace=namespace,     # 검색할 namespace
        top_k=8,                 # 검색 결과 개수
        include_values=False,    # 벡터 값 포함 여부
        include_metadata=True,    # 메타데이터 포함 여부
        filter=metadata_filter  # 메타메이터 필터 on
    )
    print('성공')
    # 데이터 가공 및 병합
    grouped_data = {}

    for match in results["matches"]:
        file_name = match["metadata"].get("file_name", "")  # 파일 이름
        file_detail = match["metadata"].get("file_detail", "")  # 파일 상세
        file_text = match["metadata"].get("text", "")  # 파일 내용

        # 같은 file_name 그룹화
        if file_name not in grouped_data:
            grouped_data[file_name] = {"file_detail": file_detail, "texts": []}
        grouped_data[file_name]["texts"].append(file_text)

    # 데이터 확인
    for file_name, data in grouped_data.items():
        print(f"File Name: {file_name}")
        print(f"Category: {data['file_detail']}")
        print("Contents:")
        for text in data["texts"]:
            print(f"- {text[:100]}...")  # 첫 100자만 출력
        print("\n")

    # 검색된 결과를 CONTEXT로 변환
    result_docs = "\n".join([match["metadata"].get("text", "") for match in results["matches"]])
    # 전체 대화 히스토리를 템플릿에 전달
    conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])

    # LLM 구성
    # llm = ChatUpstage()

    # 메시지 정의
    system_message = SystemMessage(content=f"""
    You are an assistant for question-answering tasks specialized in financial products.
    You will be given JSON DATA and user conversation history.

    Structure of JSON DATA will be

    Product Name:
    Category:
    Description:

    JSON DATA: {grouped_data}
    user conversation history : {conversation_history}


    You must follow the rules below.

    1. Each file name represents the product name.
    2. The category indicates the type of product.
    3. Only use JSON DATA to find relevant product information.
    4. Do not mix content between different products.
    5. If the product cannot be identified, respond with: "죄송합니다. 해당 상품 정보를 찾을 수 없습니다."
    6. If the input is unrelated to financial product consultations, respond with: "저는 KB 국민은행의 금융 상품과 정보만 제공해드릴 수 있습니다. 다시 입력해주세요."
    7. If the input concerns financial institutions other than KB 국민은행, respond with: "저는 KB 국민은행의 금융 상품과 정보만 제공해드릴 수 있습니다. 다시 입력해주세요."
    8. Answer naturally without including "Product Name", "Category", or "Description" in the response.
    9. Base your responses only on the product information and user conversation history.

    """)


    # 사용자 질문
    human_message = HumanMessage(content=req.messages[-1].content)

    # LangChain LLM 호출
    response = chat([system_message, human_message]).content

    return {
        "reply": response,
    }

# 건강 체크 엔드포인트
@app.get("/health")
@app.get("/")
async def health_check():
    """
    - 애플리케이션의 상태를 확인하기 위한 기본 엔드포인트
    """
    return {"status": "ok"}

# 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn

    # FastAPI 애플리케이션을 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
