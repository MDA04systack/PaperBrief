import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import openai  # 키 유효성 검사를 위해 추가

# API 키 유효성 검사 함수
def check_api_key(api_key):
    try:
        # 아주 작은 요청을 보내서 키가 작동하는지 확인합니다.
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def process_text(text, api_key): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main(): 
    st.set_page_config(page_title="논문 초록 요약기", page_icon="📄")
    st.title("📄 박수연의 논문 초록 요약 서비스")
    st.divider()

    # 사이드바 설정
    with st.sidebar:
        st.title("설정")
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

    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        # 키 검증이 실패하면 진행하지 않음
        if not user_api_key or not check_api_key(user_api_key):
            st.info("먼저 유효한 OpenAI API Key를 입력해 주세요.")
            st.stop()

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text, user_api_key)
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=user_api_key, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with st.spinner('PDF 내용을 분석하여 요약 중입니다...'):
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)

            st.subheader('-- 요약 결과 --')
            st.write(response)
            st.caption(f"발생 비용: ${cost.total_cost:.4f}")

if __name__ == '__main__':
    main()