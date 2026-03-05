"""
Query definitions for the vector-DB benchmark CLI.

The Streamlit UI (app.py) accepts live user input — no hardcoded queries needed there.
This module is used only by main.py (benchmark CLI) for automated repeat runs.
"""

# Default queries used by `make run` (the automated benchmark).
# Edit this list to match the content of your PDF documents.
CLI_BENCHMARK_QUERIES: list[str] = [
    "벡터 데이터베이스에서 고차원 벡터를 유사도 기반으로 검색하는 방법",
    "HNSW 알고리즘을 이용한 근사 최근접 이웃 검색의 원리와 특징",
    "트랜스포머 모델의 셀프 어텐션 메커니즘 작동 방식 설명",
    "클라우드 환경에서 마이크로서비스와 컨테이너를 이용한 배포 전략",
    "자연어 처리를 위한 텍스트 임베딩 기술의 역할과 활용",
    "MLOps 파이프라인에서 모델 배포 자동화와 모니터링 구성 방법",
]
