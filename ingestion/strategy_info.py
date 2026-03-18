"""
Strategy metadata registry: descriptions, pros, cons, recommendations,
install requirements for all loader and splitter strategies.
Used by the Streamlit UI to render info popovers.
"""
from __future__ import annotations
from typing import TypedDict


class StrategyInfo(TypedDict):
    name: str
    icon: str
    description: str
    pros: list[str]
    cons: list[str]
    recommended: str
    best_for: list[str]
    requires: list[str]       # pip package names
    import_check: str | None  # python module name to check availability


# ---------------------------------------------------------------------------
# Loader strategies
# ---------------------------------------------------------------------------

LOADER_STRATEGY_INFO: dict[str, dict[str, StrategyInfo]] = {
    "pdf": {
        "page": StrategyInfo(
            name="Page-based (pypdf)",
            icon="📄",
            description=(
                "pypdf로 각 페이지를 독립 섹션으로 추출합니다. "
                "텍스트 기반 PDF에 최적화된 기본 방식입니다."
            ),
            pros=[
                "추가 설치 불필요 (기본 제공)",
                "빠른 처리 속도",
                "페이지 번호 메타데이터 자동 보존",
            ],
            cons=[
                "표/복잡 레이아웃 인식 없음",
                "스캔(이미지) PDF 처리 불가",
                "2단 컬럼 레이아웃 왜곡 가능",
            ],
            recommended="일반 텍스트 PDF, 논문, 보고서",
            best_for=["학술논문", "일반보고서", "단순구조 PDF"],
            requires=["pypdf"],
            import_check="pypdf",
        ),
        "pdfplumber": StrategyInfo(
            name="Layout-aware (pdfplumber)",
            icon="📐",
            description=(
                "pdfplumber로 레이아웃을 분석하며 텍스트를 추출합니다. "
                "표와 위치 정보를 정밀하게 보존합니다."
            ),
            pros=[
                "표 추출 정확도 높음",
                "페이지 레이아웃 인식",
                "좌표 기반 텍스트 정렬",
            ],
            cons=[
                "pypdf보다 처리 속도 느림",
                "pdfplumber 별도 설치 필요",
                "이미지 기반 PDF 처리 불가",
            ],
            recommended="표/데이터가 많은 재무제표, 계약서, 규정 문서",
            best_for=["재무제표", "계약서", "데이터시트"],
            requires=["pdfplumber"],
            import_check="pdfplumber",
        ),
        "pymupdf": StrategyInfo(
            name="High-speed (PyMuPDF / fitz)",
            icon="⚡",
            description=(
                "PyMuPDF(fitz)로 초고속 텍스트 추출합니다. "
                "폰트·위치 정보, 이미지 감지까지 지원합니다."
            ),
            pros=[
                "매우 빠른 처리 속도",
                "풍부한 메타데이터 (폰트·위치)",
                "대용량 PDF 배치 처리에 최적",
            ],
            cons=[
                "패키지 설치 크기 큼",
                "AGPL 라이선스 주의",
                "pymupdf 별도 설치 필요",
            ],
            recommended="대용량 PDF 배치 처리, 속도가 최우선인 환경",
            best_for=["대용량PDF", "배치처리", "고속처리"],
            requires=["pymupdf"],
            import_check="fitz",
        ),
        "fulltext": StrategyInfo(
            name="Full-text (pypdf)",
            icon="📋",
            description=(
                "전체 문서를 하나의 텍스트 블록으로 병합합니다. "
                "페이지 경계를 무시하고 연속 텍스트로 처리합니다."
            ),
            pros=[
                "페이지 단절 없이 문장 연속성 유지",
                "추가 설치 불필요",
                "단순한 처리 구조",
            ],
            cons=[
                "섹션 메타데이터 없음",
                "매우 긴 청크가 발생할 수 있음",
                "chunker 의존도 높아짐",
            ],
            recommended="짧은 단일 주제 문서, 페이지 구분이 의미 없는 경우",
            best_for=["짧은문서", "단일주제", "연속텍스트"],
            requires=["pypdf"],
            import_check="pypdf",
        ),
        "ocr": StrategyInfo(
            name="OCR (pytesseract + pdf2image)",
            icon="🔍",
            description=(
                "스캔/이미지 PDF를 OCR로 텍스트 추출합니다. "
                "Tesseract OCR 엔진과 poppler가 사전 설치되어야 합니다."
            ),
            pros=[
                "스캔 PDF 처리 가능",
                "이미지 기반 문서 텍스트 복원",
            ],
            cons=[
                "처리 속도 매우 느림",
                "Tesseract + poppler 별도 설치 필요",
                "인식 정확도 제한적",
            ],
            recommended="스캔된 문서, 이미지 기반 PDF, 구형 문서",
            best_for=["스캔PDF", "이미지PDF", "구형문서"],
            requires=["pdf2image", "pytesseract"],
            import_check="pytesseract",
        ),
    },
    "docx": {
        "heading": StrategyInfo(
            name="Heading-based (현재 방식)",
            icon="📑",
            description=(
                "Heading 1~9, Title, Subtitle 스타일을 섹션 경계로 인식합니다. "
                "구조화된 Word 문서에 최적화된 방식입니다."
            ),
            pros=[
                "문서의 논리 구조 그대로 보존",
                "섹션 제목이 메타데이터로 저장",
                "표 내용 자동 포함",
            ],
            cons=[
                "Heading 스타일 없는 문서에 취약",
                "하위 헤딩 계층 구조 미분리",
            ],
            recommended="공식 보고서, 매뉴얼, 구조화된 제안서",
            best_for=["공식문서", "매뉴얼", "제안서"],
            requires=["python-docx"],
            import_check="docx",
        ),
        "paragraph": StrategyInfo(
            name="Paragraph-based",
            icon="¶",
            description=(
                "비어있지 않은 각 단락을 독립 섹션으로 분리합니다. "
                "제목 스타일이 없는 문서에도 동작합니다."
            ),
            pros=[
                "세밀한 단위로 분리 가능",
                "제목 스타일 없어도 동작",
                "단락 경계 정확히 보존",
            ],
            cons=[
                "섹션이 너무 짧아질 수 있음",
                "의미적으로 연관된 단락 분리 가능성",
                "섹션 수 과다 발생 가능",
            ],
            recommended="뉴스 기사, 에세이, 제목 없는 비정형 문서",
            best_for=["뉴스기사", "에세이", "비정형문서"],
            requires=["python-docx"],
            import_check="docx",
        ),
        "fulltext": StrategyInfo(
            name="Full-text",
            icon="📋",
            description="전체 문서를 하나의 텍스트 블록으로 병합합니다.",
            pros=["문맥 연속성 최대화", "구현 단순"],
            cons=["섹션 메타데이터 없음", "매우 긴 텍스트 발생 가능"],
            recommended="짧은 메모, 단일 주제 문서",
            best_for=["짧은메모", "단일주제"],
            requires=["python-docx"],
            import_check="docx",
        ),
    },
    "txt": {
        "fulltext": StrategyInfo(
            name="Full-text",
            icon="📄",
            description="전체 텍스트 파일을 하나의 섹션으로 로드합니다.",
            pros=["단순하고 빠름", "추가 설치 불필요"],
            cons=["섹션 메타데이터 없음"],
            recommended="짧은 텍스트 파일, 로그, 메모",
            best_for=["메모", "로그파일", "짧은텍스트"],
            requires=[],
            import_check=None,
        ),
        "paragraph": StrategyInfo(
            name="Paragraph-based",
            icon="¶",
            description="빈 줄로 구분된 단락들을 각각 섹션으로 분리합니다.",
            pros=["단락 경계 보존", "구조 있는 텍스트에 유용"],
            cons=["빈줄 없으면 단일 섹션으로 처리"],
            recommended="구조화된 텍스트, 항목별 정리 문서",
            best_for=["구조화텍스트", "항목문서"],
            requires=[],
            import_check=None,
        ),
    },
    "md": {
        "heading": StrategyInfo(
            name="Heading-based (# ## ###)",
            icon="📑",
            description=(
                "Markdown 헤딩(#, ##, ###)을 섹션 경계로 인식합니다. "
                "추가 라이브러리 없이 정규식으로 처리합니다."
            ),
            pros=["문서 구조 그대로 보존", "헤딩 텍스트가 메타데이터로 저장", "추가 설치 불필요"],
            cons=["헤딩 없는 파일에는 단일 섹션으로 처리"],
            recommended="README, 기술 문서, Wiki 페이지",
            best_for=["README", "기술문서", "Wiki"],
            requires=[],
            import_check=None,
        ),
        "fulltext": StrategyInfo(
            name="Full-text",
            icon="📋",
            description="전체 Markdown 파일을 하나의 텍스트로 로드합니다.",
            pros=["단순", "빠름"],
            cons=["구조 정보 손실"],
            recommended="짧은 MD 파일",
            best_for=["짧은MD파일"],
            requires=[],
            import_check=None,
        ),
    },
    "html": {
        "tag": StrategyInfo(
            name="Tag-based (BeautifulSoup)",
            icon="🏷️",
            description=(
                "h1~h6 태그를 섹션 경계로 인식하고 본문 텍스트를 추출합니다. "
                "script/style 태그는 자동 제거됩니다."
            ),
            pros=[
                "웹 문서 구조 보존",
                "태그 자동 제거",
                "스크립트·스타일 노이즈 제거",
            ],
            cons=[
                "beautifulsoup4 설치 필요",
                "복잡한 JS 렌더링 DOM 처리 불가",
            ],
            recommended="웹 페이지, HTML 문서",
            best_for=["웹페이지", "HTML문서"],
            requires=["beautifulsoup4"],
            import_check="bs4",
        ),
        "fulltext": StrategyInfo(
            name="Full-text (BeautifulSoup)",
            icon="📋",
            description="HTML 태그를 모두 제거한 순수 텍스트로 로드합니다.",
            pros=["단순", "빠름", "태그 노이즈 없음"],
            cons=["구조 정보 손실", "beautifulsoup4 필요"],
            recommended="짧은 HTML, 섹션 구분이 불필요한 경우",
            best_for=["단순HTML"],
            requires=["beautifulsoup4"],
            import_check="bs4",
        ),
    },
    "csv": {
        "row_batch": StrategyInfo(
            name="Row-batch (pandas)",
            icon="📊",
            description=(
                "CSV를 pandas로 읽어 N행씩 하나의 섹션으로 묶습니다. "
                "각 행을 'column: value' 형식으로 변환합니다."
            ),
            pros=[
                "구조화 데이터 처리에 최적",
                "컬럼명 자동 메타데이터",
                "배치 크기 조절 가능",
            ],
            cons=[
                "pandas 설치 필요",
                "자유 텍스트 컬럼에 비효율",
            ],
            recommended="테이블형 데이터, FAQ, 상품 목록",
            best_for=["FAQ", "상품목록", "테이블데이터"],
            requires=["pandas"],
            import_check="pandas",
        ),
    },
}

# ---------------------------------------------------------------------------
# Splitter strategies
# ---------------------------------------------------------------------------

SPLITTER_STRATEGY_INFO: dict[str, StrategyInfo] = {
    "sliding_window": StrategyInfo(
        name="Sliding Window (현재 방식)",
        icon="🪟",
        description=(
            "고정 크기 윈도우를 슬라이드하며 문장 경계에서 끊습니다. "
            "한국어·영어 혼합 텍스트에 맞게 최적화되어 있습니다."
        ),
        pros=[
            "일관된 청크 크기 유지",
            "구현 안정적·검증됨",
            "한국어 문장 경계(。!?) 지원",
            "추가 의존성 없음",
        ],
        cons=[
            "의미 단위 무시 가능성",
            "중요한 문장 중간에서 분할될 수 있음",
        ],
        recommended="범용, 한영 혼합 문서, 빠른 처리가 필요한 환경",
        best_for=["범용", "한영혼합", "빠른처리"],
        requires=[],
        import_check=None,
    ),
    "recursive": StrategyInfo(
        name="Recursive Character Split",
        icon="🔄",
        description=(
            "구분자 우선순위(\\n\\n → \\n → . → 공백) 순으로 재귀 분할합니다. "
            "LangChain의 RecursiveCharacterTextSplitter와 동일한 방식입니다."
        ),
        pros=[
            "자연스러운 단락·문장 경계 분할",
            "LangChain 호환 방식",
            "추가 의존성 없음",
        ],
        cons=[
            "청크 크기 불균일 발생 가능",
            "짧은 문장이 많으면 과다 분할",
        ],
        recommended="일반 텍스트 문서, 단락 구조가 있는 문서, 표준 RAG 파이프라인",
        best_for=["일반문서", "단락구조", "표준RAG"],
        requires=[],
        import_check=None,
    ),
    "sentence": StrategyInfo(
        name="Sentence-based",
        icon="📝",
        description=(
            "문장 단위로 분리 후 N문장씩 묶어 청크를 구성합니다. "
            "한국어(。!?) 및 영어(.!?) 문장 경계를 모두 지원합니다."
        ),
        pros=[
            "의미 단위 정확히 보존",
            "문장 경계에서만 분할",
            "문맥 손실 최소화",
        ],
        cons=[
            "청크 크기 불균일",
            "매우 긴 단일 문장에 취약",
            "sentence_count 파라미터 튜닝 필요",
        ],
        recommended="논문, 법률 문서, 계약서, 의미 단위가 중요한 경우",
        best_for=["논문", "법률문서", "정밀RAG"],
        requires=[],
        import_check=None,
    ),
    "semantic": StrategyInfo(
        name="Semantic Split (임베딩 기반)",
        icon="🧠",
        description=(
            "연속 문장들의 임베딩 코사인 유사도가 threshold 이하로 떨어지는 지점을 "
            "자동으로 분할 경계로 설정합니다. 주제 전환을 자동 감지합니다."
        ),
        pros=[
            "의미적 경계 자동 최적화",
            "주제 전환 지점 자동 감지",
            "최고 품질 RAG 결과",
        ],
        cons=[
            "처리 속도 느림 (임베딩 API 추가 호출)",
            "임베딩 비용 추가 발생",
            "semantic_threshold 튜닝 필요",
            "짧은 문서에는 오히려 비효율",
        ],
        recommended="고품질 RAG, 주제가 다양한 긴 문서, 정확도 최우선 환경",
        best_for=["고품질RAG", "긴문서", "정확도최우선"],
        requires=[],
        import_check=None,
    ),
}

# ---------------------------------------------------------------------------
# Default loader strategy per file type
# ---------------------------------------------------------------------------

DEFAULT_LOADER_STRATEGY: dict[str, str] = {
    "pdf": "page",
    "docx": "heading",
    "txt": "fulltext",
    "md": "heading",
    "html": "tag",
    "csv": "row_batch",
}
