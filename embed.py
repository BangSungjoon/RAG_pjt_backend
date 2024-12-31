import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from multiprocessing import Pool

load_dotenv()

# upstage models
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "finance-pdt"
pdf_folder_path = "pdf_files"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start")

# 벡터 스토어 생성
pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

# 문서 처리 및 벡터화 함수
def process_pdf(pdf_file):
    try:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        print(f"Processing {pdf_file}...")

        # PDF 로드
        document_parse_loader = UpstageDocumentParseLoader(
            pdf_path,
            output_format='html',  # 결과물 형태 : HTML
            coordinates=False  # 이미지 OCR 좌표계 가지고 오지 않기
        )
        docs = document_parse_loader.load()
        print(f"{pdf_file} parsed into documents.")

        # 문서 분할 및 벡터화
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        PineconeVectorStore.from_documents(splits, embedding_upstage, index_name=index_name)
        print(f"{pdf_file} successfully stored in Pinecone.")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

# 멀티프로세싱 처리 함수, 로컬 컴퓨터 cpu 코어 수에 따라 빨라진다길래 적용해 봄
def main():
    # PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

    # 멀티프로세싱 처리
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_pdf, pdf_files)

if __name__ == "__main__":
    main()
