import os # 파이썬 코드와 기본 운영 체제 사이의 다리 역할하는 라이브러리
from PyPDF2 import PdfReader # PDF 파일에서 텍스트를 추출하는 라이브러리(v3.0.1)
import streamlit as st # streamlit(v1.29.0)
# langchain 대규모 언어 모델(LLM)을 효율적으로 다루기 위한 프레임워크(v0.3.27)
from langchain.text_splitter import CharacterTextSplitter # 긴 텍스트를 잘게 나누는 도구
from langchain_openai import OpenAIEmbeddings # 텍스트의 의미를 추출하여 수치화(Vector)하는 도구 (AI가 내용을 비교/검색할 때 사용)
from langchain_openai import ChatOpenAI # OpenAI의 GPT 모델을 연결하여 대화나 요약 같은 실제 답변을 생성하는 엔진
from langchain_community.vectorstores import FAISS # 텍스트를 벡터로 변환하여 저장하고 검색하는 벡터 데이터베이스
from langchain.chains.question_answering import load_qa_chain # 여러 개의 문서 조각(chunks) 중에서 질문과 관련된 내용을 참조하여 답변을 생성하도록 하는 '연결 고리(Chain)'
from langchain_community.callbacks import get_openai_callback # OpenAI API를 사용할 때 발생하는 토큰 사용량과 비용을 실시간으로 추적하는 도구
import openai  # 키 유효성 검사를 위해 추가, OpenAI 공식 파이썬 라이브러리(v2.15.0)

# API 키 유효성 검사 함수
def check_api_key(api_key):
    try:
        # 아주 작은 요청을 보내서 키가 작동하는지 확인
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False
# 텍스트 전처리 함수
def process_text(text, api_key): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # 텍스트 단위를 1,000자 단위로 자름
        chunk_overlap=200, # 문맥이 끊기지 않게 200자 겹치게 자름
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents
# 웹 UI 구현
def main(): 
    # 웹 브라우저 탭의 이름과 메인 화면의 제목 설정
    st.set_page_config(page_title="박수연의 논문 초록 요약기", page_icon="📄")
    st.title("📄 박수연의 논문 초록 요약 서비스")
    st.divider()

    # 사이드바 설정: API 입력용
    with st.sidebar:
        st.title("설정")
        # API type: API 키가 노출되지 않도록 **** 형태로 입력
        user_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
        
        # 키 입력 여부에 따른 상태 메시지 표시
        if user_api_key:
            if check_api_key(user_api_key):
                st.success("✅ 연결되었습니다!")
            else:
                st.error("❌ 유효하지 않은 키입니다. 다시 확인해 주세요.")
        else:
            st.warning("🔑 API Key를 입력해 주세요.")
            
        st.markdown("[API Key 발급받기](https://platform.openai.com/api-keys)")
    # pdf 파일 업로드 버튼
    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        # 키 검증이 실패하면 진행하지 않음
        if not user_api_key or not check_api_key(user_api_key):
            st.info("먼저 유효한 OpenAI API Key를 입력해 주세요.")
            st.stop() # 유효한 API 키가 없다면 코드 실행 즉시 중단

        # 업로드된 PDF의 모든 페이지를 돌며 글자를 뽑아내 하나의 긴 문장(text)로 합침
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 긴 텍스트를 잘게 나누고 벡터화(숫자 변환)
        documents = process_text(text, user_api_key)
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."

        if query:
            docs = documents.similarity_search(query) # 질문과 가장 관련된 텍스트 조각을 골라냄
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=user_api_key, temperature=0.1) # gpt-3.5-turbo-16k 모델 사용, API 키는 사용자 입력, 사실에 기반한 답변 설정(0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with st.spinner('PDF 내용을 분석하여 요약 중입니다...'): # 요약하는 중에 나오는 로딩 애니메이션
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)

            st.subheader('-- 요약 결과 --')
            st.write(response) # AI가 생성한 최종 요약 결과를 화면에 뿌려줌
            st.caption(f"발생 비용: ${cost.total_cost:.4f}") # 이 요약 한 번을 위해 OpenAI에 지불해야 할 실제 금액 출력

if __name__ == '__main__':
    main() # 웹 서버를 시작하는 main() 함수를 안전하게 호출하기 위해 사용