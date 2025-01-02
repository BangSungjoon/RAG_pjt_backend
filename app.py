import os
from typing import List, Optional

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
    # keywords = kiwi.tokenize(req.messages[-1].content)
    # tokens = kiwi.tokenize(req.messages[-1].content)
    # keywords = [token.form for token in tokens if token.tag.startswith("NN")]  # 명사 태그 추출
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
        top_k=6,                 # 검색 결과 개수
        include_values=False,    # 벡터 값 포함 여부
        include_metadata=True,    # 메타데이터 포함 여부
        filter=metadata_filter  # 메타메이터 필터 on
    )
    print('성공')
    for match in results["matches"]:
        file_name = match["metadata"].get("file_name", "Unknown")  # file_name 추출, 없으면 "Unknown"
        print(f"File Name: {file_name}")
    # 검색된 결과를 CONTEXT로 변환
    result_docs = "\n".join([match["metadata"].get("text", "") for match in results["matches"]])
    print('성공1')
    # 전체 대화 히스토리를 템플릿에 전달
    # conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
    print('성공2')
    # LLM 구성
    # llm = ChatUpstage()
    # 메시지 정의
    system_message = SystemMessage(content=f"""
    You are an assistant for question-answering tasks specialized in financial products. 
    Use the following context to answer the question while considering the user's circumstances:
    Context: {result_docs}
    """)


    # 사용자 질문
    human_message = HumanMessage(content=req.messages[-1].content)

    # LangChain LLM 호출
    response = chat([system_message, human_message]).content
    print('llm 성공')
    # 프롬프트 정의
    # prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """
    #             You are an assistant for question-answering tasks specialized in financial products. Follow these rules carefully:
    #             Rule 1. Use the following retrieved context to answer the question while considering the user's circumstances.
    #             Rule 2. If the user's circumstances are clear, recommend one specific financial product that is most appropriate.
    #             Rule 3. If the user's circumstances are unclear, ask clarifying questions to better understand their situation before recommending a product.
    #             Rule 4. If no appropriate recommendation can be made, say: "질문을 이해하지 못하였습니다. 상세하게 질문해주세요."
    #             Rule 5. Only include financial products or information directly relevant to the user's input. Do not provide unrelated product details or context.
    #             Rule 6. Classify financial products into the following categories:
    #                 - 예금 (Deposits): For users looking for fixed-term savings with interest.
    #                 - 적금 (Savings Plans): For users wanting to save a fixed amount monthly for a specific term.
    #                 - 입출금자유 (Free Savings Accounts): For users needing flexibility in deposits and withdrawals.
    #                 - 담보대출 (Secured Loans): For users providing collateral to borrow money (e.g., housing).
    #                 - 신용대출 (Unsecured Loans): For users borrowing based on creditworthiness without collateral.
    #                 - 자동차대출 (Auto Loans): For users financing the purchase of a vehicle.
    #             Rule 7. Ensure that every sentence ends with a space to improve readability.
    #             ---
    #             사용자와의 지난 대화: {history}
    #             ---
    #             CONTEXT:
    #             {context}
    #             """,
    #         ),
    #         ("user", "{input}"),
    #     ]
    # )
    # print(prompt_template)
    # print('성공3')
    # # LLM Chain 정의
    # chain = prompt_template | llm | StrOutputParser()
    # print('성공4')
    # # LLM Chain 호출
    # try:
    #     response = chain.invoke({"context": result_docs, "input": req.messages[-1].content, "history": conversation_history})
    # except Exception as e:
    #     print(f"Error during chain invocation: {e}")

    # # 결과 출력
    # print(f"LLM Response: {response}")

    return {
        "reply": response,
    }
    # prompt_template = f"""
    # 너는 전문 금융 컨설턴트 역할을 맡고 있다.
    # 다음은 사용자와의 대화 기록이다:
    # {conversation_history}
    
    # 사용자의 최신 질문에 대해 연결된 데이터베이스에서 가장 관련성 높은 정보를 찾아 답하라.
    # 질문: "{req.messages[-1].content}"
    # """
    
    # QA 실행
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat_upstage,
    #     chain_type="stuff",
    #     retriever=pinecone_retriever,
    #     return_source_documents=True
    # )
    
    # result = qa(prompt_template)
    # return {
    #     "reply": result['result'],
    #     "sources": result['source_documents']
    # }

# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")
#
#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id
#
#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )
#
#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)
#
#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value
#
#     return {"reply": assistant_reply, "thread_id": thread_id}


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
