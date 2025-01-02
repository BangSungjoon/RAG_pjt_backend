## app.py
### 주요 역할
1. Pinecone와 Upstage 모델 초기화
    - Pinecone을 활용해 텍스트 임베딩을 저장하고 검색합니다.
    - Upstage 모델을 사용해 사용자의 입력 데이터를 벡터화하고, LLM(ChatUpstage)을 통해 답변을 생성합니다.
2. FastAPI 구성
    - FastAPI로 RESTful API를 구축하며, CORS 정책을 설정해 외부 도메인에서의 요청을 허용합니다.
3. 엔드포인트
    - /chat: 사용자 메시지를 처리하고 Pinecone 기반 문서 검색과 LLM을 활용해 답변을 반환합니다.
    - /health: 애플리케이션 상태를 확인합니다.
4. 디버깅 및 확장 가능성
    - Pinecone에서 검색된 문서를 확인할 수 있도록 로그를 추가했습니다.
    - 필요한 경우 추가적인 엔드포인트를 쉽게 정의할 수 있도록 구조화되었습니다.