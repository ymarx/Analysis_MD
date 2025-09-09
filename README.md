# 🌊 사이드스캔 소나 기물 탐지 분석 시스템

해저에 설치된 기뢰 형상 물체(기물)를 사이드스캔 소나 데이터에서 자동으로 탐지하는 **Multi-Environment AI 시스템**입니다.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 프로젝트 개요

**🆕 Multi-Environment Support (2025-09-09 업데이트)**

본 시스템은 **로컬 CPU**, **로컬 GPU**, **클라우드 환경**에서 모두 동작하는 적응형 AI 플랫폼입니다:

### 핵심 기능
- **🔄 자동 환경 감지**: CPU → GPU → 클라우드 자동 최적화
- **📁 XTF 파일 파싱**: 사이드스캔 소나 raw 데이터 읽기 및 처리
- **🗺️ 좌표 매핑**: 기물 위치와 소나 데이터 간의 정확한 매핑
- **🎨 전처리 파이프라인**: 워터컬럼 제거, 정규화, 노이즈 제거, 대비 향상
- **🏔️ 지형 분류**: 모래, 뻘, 암반 등 지형별 적응형 처리
- **🤖 CNN 기물 탐지**: ResNet + CBAM 어텐션 메커니즘
- **📊 종합 평가**: 다중 지표 성능 분석
- **☁️ 클라우드 배포**: Runpod GPU 클라우드 자동 배포

### 지원 환경
| 환경 | 성능 | 설정 시간 | 비용 |
|------|------|----------|------|
| **로컬 CPU** | 기준 (1x) | 0분 | 무료 |
| **로컬 GPU** | 5-15배 | 30분 | GPU 구매비 |
| **Runpod RTX 4090** | 15-25배 | 5분 | $0.35-0.69/시간 |
| **Runpod A100-80GB** | 20-50배 | 5분 | $1.5-3/시간 |

## 📁 프로젝트 구조

```
sidescan_sonar_detection/
├── config/                    # 설정 파일
│   ├── __init__.py
│   ├── settings.py           # 전역 설정 관리
│   └── paths.py              # 파일 경로 관리
├── src/                      # 소스 코드
│   ├── data_processing/      # 데이터 처리 모듈
│   │   ├── xtf_reader.py     # XTF 파일 파싱
│   │   ├── coordinate_mapper.py  # 좌표 매핑
│   │   └── preprocessor.py   # 전처리 파이프라인
│   ├── feature_extraction/   # 특징 추출 (예정)
│   ├── models/              # 탐지 모델 (예정)
│   ├── evaluation/          # 성능 평가 (예정)
│   ├── utils/               # 유틸리티 (예정)
│   └── interactive/         # 인터랙티브 도구 (예정)
├── notebooks/               # Jupyter 노트북
│   └── 01_exploratory_analysis.ipynb
├── [샘플]데이터/            # 샘플 데이터
├── datasets/                # 연구용 데이터셋
├── data/                    # 처리된 데이터 (자동 생성)
│   ├── processed/
│   ├── augmented/
│   └── annotations/
├── outputs/                 # 출력 결과 (자동 생성)
│   ├── models/
│   ├── figures/
│   └── results/
├── main.py                  # 메인 실행 파일
├── requirements.txt         # 패키지 의존성
└── README.md               # 이 파일
```

## 🚀 빠른 시작

### ⚡ 자동 설치 및 실행 (권장)

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/mine-detection.git
cd mine-detection

# 2. 자동 환경 설치
chmod +x scripts/install.sh
./scripts/install.sh --auto

# 3. 환경 활성화
source mine_detection_env/bin/activate

# 4. 자동 환경 감지 실행
python main.py --device auto
```

### 🖥️ 환경별 실행

#### 로컬 CPU (기존 방식, 변경 없음)
```bash
python main.py
```

#### 로컬 GPU (자동 감지)
```bash
python main.py --device auto  # CUDA/MPS 자동 감지
python main.py --device cuda  # NVIDIA GPU 직접 지정
python main.py --device mps   # Apple Silicon 직접 지정
```

#### Runpod 클라우드 배포
```bash
# API 키 설정
export RUNPOD_API_KEY="your-api-key"

# 자동 배포 (RTX 4090)
python scripts/deploy_runpod.py --action deploy --gpu-type "RTX 4090"

# 고성능 배포 (A100-80GB)
python scripts/deploy_runpod.py --action deploy --gpu-type "A100-80GB"
```

### 🔍 성능 벤치마크

```bash
# 현재 환경 성능 측정
python scripts/benchmark_performance.py --save

# 모든 환경 비교 (GPU 있는 경우)
python scripts/benchmark_performance.py --device auto --full
```

## 📊 주요 기능

### XTF 파일 처리
```python
from src.data_processing.xtf_reader import XTFReader

# XTF 파일 로드
reader = XTFReader('path/to/file.xtf')
reader.load_file()
ping_data = reader.parse_pings()

# Intensity 매트릭스 추출
intensity_matrix = reader.extract_intensity_matrix(channel=0)
```

### 좌표 매핑
```python
from src.data_processing.coordinate_mapper import CoordinateMapper, CoordinateTransformer

# 좌표 변환기 초기화
transformer = CoordinateTransformer(utm_zone=52)
mapper = CoordinateMapper(transformer)

# 픽셀 <-> 지리좌표 변환
pixel_coords = mapper.geo_to_pixel(longitude, latitude)
geo_coords = mapper.pixel_to_geo(ping_idx, sample_idx)
```

### 전처리 파이프라인
```python
from src.data_processing.preprocessor import Preprocessor, PreprocessingConfig

# 전처리 설정
config = PreprocessingConfig(
    remove_water_column=True,
    normalize_intensity=True,
    apply_denoising=True,
    enhance_contrast=True
)

# 전처리 실행
preprocessor = Preprocessor(config)
result = preprocessor.process(intensity_data)
```

## 🗂️ 데이터 형식

### 입력 데이터
- **XTF 파일**: 사이드스캔 소나 raw 데이터
- **BMP 이미지**: 변환된 이미지 데이터 (참조용)
- **Excel 파일**: 기물 위치 좌표 정보

### 출력 데이터
- **처리된 intensity 데이터**: NumPy 배열 형태
- **기물 마스크**: 바이너리 마스크
- **바운딩 박스**: JSON/CSV 형태의 좌표 정보
- **메타데이터**: 처리 결과 및 품질 메트릭

## ⚙️ 설정 옵션

### XTF 처리 설정
```python
XTF_CONFIG = {
    'max_pings_per_load': 1000,  # 메모리 효율성을 위한 배치 크기
    'channels': {'port': 0, 'starboard': 1}
}
```

### 전처리 설정
```python
preprocess_config = PreprocessingConfig(
    remove_water_column=True,     # 워터컬럼 제거
    water_column_width=50,        # 워터컬럼 폭
    normalize_intensity=True,     # 강도 정규화
    normalization_method='minmax', # 정규화 방법
    apply_denoising=True,         # 노이즈 제거
    denoising_method='gaussian',  # 노이즈 제거 방법
    enhance_contrast=True,        # 대비 향상
    contrast_method='clahe',      # 대비 향상 방법
    terrain_adaptive=True         # 지형별 적응형 처리
)
```

## 📈 성능 최적화

### 메모리 관리
- `max_pings_per_load` 설정으로 메모리 사용량 제한
- 배치 처리로 대용량 파일 처리
- 처리된 데이터 캐싱으로 반복 작업 최소화

### 처리 속도
- NumPy 벡터화 연산 활용
- OpenCV 최적화된 이미지 처리
- 병렬 처리 지원 (향후 구현 예정)

## 🔧 문제 해결

### 일반적인 문제들

1. **XTF 파일 로드 실패**
   - `pyxtf` 라이브러리 설치 확인
   - 파일 경로 및 권한 확인

2. **좌표 매핑 오류**
   - UTM 존 설정 확인 (한국: 52존)
   - 위경도 데이터 형식 확인

3. **메모리 부족**
   - `max_pings_per_load` 값 감소
   - 배치 크기 조정

### 로그 확인
```bash
# 디버그 모드로 실행
python main.py --mode sample --log-level DEBUG
```

## 🧪 테스트

```bash
# 테스트 실행 (향후 구현 예정)
pytest tests/
```

## 📖 추가 문서

- [API 문서](docs/api/) (향후 제공)
- [알고리즘 가이드](docs/algorithms/) (향후 제공)
- [성능 벤치마크](docs/benchmarks/) (향후 제공)

## 🤝 기여하기

1. 이슈 등록으로 문제점 보고
2. 개선 사항 제안
3. 코드 리뷰 및 피드백

## 📋 개발 현황

### Phase 1: 데이터 처리 (완료 ✅)
- [x] XTF 파일 리더 및 파서
- [x] 좌표 매핑 시스템  
- [x] 기본 전처리 파이프라인
- [x] 탐색적 분석 노트북

### Phase 2: AI 모델 (완료 ✅)
- [x] 특징 추출 모듈 (HOG, LBP, Gabor, SfS)
- [x] 데이터 증강 시스템
- [x] CNN 기반 탐지 모델 (ResNet + CBAM)
- [x] 종합 평가 시스템

### Phase 3: Multi-Environment (완료 ✅)
- [x] 자동 디바이스 감지 및 관리
- [x] GPU 최적화 (CUDA, MPS)
- [x] Runpod 클라우드 자동 배포
- [x] 성능 벤치마킹 시스템
- [x] Docker 컨테이너화

### Phase 4: 향상된 기능 (예정 🔄)
- [ ] 지형 적응형 처리 고도화
- [ ] 앙상블 모델 (CNN + 전통적 특징)
- [ ] 실시간 처리 파이프라인
- [ ] 웹 기반 인터페이스
- [ ] REST API 서버

## 🛠️ 기술 스택

### 🧠 AI/ML
- **PyTorch 2.1+**: 딥러닝 프레임워크
- **OpenCV 4.8+**: 컴퓨터 비전
- **scikit-learn**: 전통적 ML 알고리즘
- **NumPy 1.26**: 수치 계산 (호환성 최적화)

### 🖥️ 컴퓨팅 환경
- **CUDA 11.8**: NVIDIA GPU 가속
- **Apple MPS**: Apple Silicon 최적화  
- **Docker**: 컨테이너화 배포
- **Runpod API**: 클라우드 GPU 관리

### 📊 데이터 처리
- **pyxtf**: 사이드스캔 소나 데이터 파싱
- **UTM**: 좌표계 변환
- **pandas/matplotlib**: 데이터 분석 및 시각화

## 📚 문서

- **[설치 가이드](docs/installation_guide.md)**: 환경별 상세 설치 방법
- **[사용법 가이드](docs/usage_guide.md)**: 모듈별 사용법 및 고급 기능  
- **[배포 계획서](docs/gpu_cloud_deployment_plan.md)**: GPU/클라우드 배포 전략
- **[API 문서](docs/api/)**: 코드 레퍼런스 (향후 제공)

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해 주세요.

---

**© 2024 사이드스캔 소나 기물 탐지 연구팀**