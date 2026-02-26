# 📄 박수연의 논문 초록 요약 서비스 (PaperBrief)

![Streamlit]([https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white](https://paperbrief-mda04systack.streamlit.app/))
## 1. 프로젝트의 주요 기능과 목적, 접속 링크

**목적:**  
복잡하고 긴 논문이나 문서(PDF)를 빠르고 핵심적으로 요약하여 사용자가 내용을 쉽게 파악할 수 있도록 돕습니다. Langchain 기반의 RAG (Retrieval-Augmented Generation) 시스템을 통해 문서 내용 전반에 대해 정확도 높은 요약을 제공합니다.

**주요 기능:**
- OpenAI API 키 연동 및 유효성 검사
- PDF 파일 업로드 및 텍스트 추출 (`PyPDF2` 활용)
- 문서 분할, 임베딩을 거친 벡터 데이터베이스(`FAISS`) 구성
- 사용자의 목적에 맞춘 논문 텍스트 맞춤형 요약 (3~5문장 길이 제한)
- 한 번의 요약 시 사용되는 OpenAI API 비용 안내

**접속 링크:**
- **웹 서비스 (Streamlit Cloud):** [https://paperbrief-mda04systack.streamlit.app/](https://paperbrief-mda04systack.streamlit.app/)

---

## 2. 설치 방법

로컬 환경에서 프로젝트를 실행하려면 다음 과정을 따라주세요:

```bash
# 1. 저장소 클론
git clone https://github.com/MDA04systack/PaperBrief.git

# 2. 프로젝트 디렉토리로 이동
cd PaperBrief/pdfsumbot

# 3. 가상환경 생성 및 활성화 (선택 사항)
python -m venv venv
# Windows (cmd):
venv\Scripts\activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# 4. 필수 라이브러리 설치
pip install -r requirements.txt

# 5. 스트림릿 앱 실행
streamlit run make_pdf_summary_site.py
```

---

## 3. 문제 해결 방법

앱 사용 중 문제가 발생할 경우 아래 항목들을 참고하세요.

- **`유효하지 않은 키입니다.` 에러가 날 경우:**  
  [OpenAI API 설정 페이지](https://platform.openai.com/api-keys)에서 유효한 비밀 키(Secret Key)를 발급받았는지, 오타나 공백이 포함되어 있지 않은지 확인하세요.

- **`결제 수단을 등록하세요` 라는 식의 오류가 발생할 경우 (OpenAI API 과금 정책):**  
  OpenAI API의 무료 크레딧이 소진되었거나 결제 수단이 등록되지 않았을 가능성이 높습니다. 사용량 페이지를 확인하여 한도를 늘려보세요.

- **업로드된 PDF 파일 인식 오류:**  
  비밀번호가 걸려있는 PDF이거나 이미지 기반(스캔된 형태)의 PDF일 경우 텍스트를 추출하지 못할 수 있습니다. 문자열 인식이 가능한 PDF를 업로드하세요.

---

## 4. 지원 창구

앱 이용 및 기능 개선에 대해 문의사항이나 버그 리포트가 필요하신 경우 아래 창구를 이용해주세요.

- **이슈 등록:** [GitHub Issue 페이지](https://github.com/MDA04systack/PaperBrief/issues)에 글을 남겨주세요.
- **이메일(선택):** 개발자의 이메일을 통해 다이렉트로 연락 가능합니다. (예: `MDA04systack@example.com` - 본인의 이메일로 변경 요망)

---

## 5. 라이선스 정보

이 프로젝트는 **MIT 라이선스**를 따릅니다. 누구나 자유롭게 사용, 복제, 수정 및 배포할 수 있습니다. 자세한 내용은 `LICENSE` 파일을 확인하시기 바랍니다.

---

## 6. 변경 로그 (Changelog)

- **v1.0.0 (2026-02-26)**
  - 초기 버전 배포 (`streamlit` 서비스)
  - `PyPDF2`를 활용한 텍스트 추출 모듈 구현
  - `gpt-3.5-turbo-16k` 및 `Langchain`을 활용한 내용 요약 시스템 구성
  - 비용 계산 및 UI 인터페이스 추가
  - Streamlit Cloud 환경 구성

