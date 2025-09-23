# 기뢰 탐지 분석 프로젝트 (Mine Detection Analysis)

해저 사이드스캔 소나 데이터를 활용한 기뢰 및 해저 객체 탐지 시스템

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 프로젝트 개요

이 프로젝트는 사이드스캔 소나 데이터(XTF 파일)를 분석하여 해저에 있는 기뢰나 기타 객체를 자동으로 탐지하는 머신러닝 파이프라인입니다.

### 🆕 주요 업데이트 (2024-09-16)
- **🔧 통합 파이프라인**: 모듈화된 단계별 실행
- **📍 좌표 매핑 검증**: 98.4% 정확도 달성
- **🎯 앙상블 최적화**: Optuna 기반 하이퍼파라미터 튜닝
- **🗺️ 지형 분석**: 해저 지형 특성 고려 분류

### 핵심 기능
- **📁 XTF 파일 읽기**: 사이드스캔 소나 데이터 파싱
- **📊 강도 데이터 추출**: 포트/스타보드 채널 처리
- **🗺️ 좌표 매핑**: 픽셀 ↔ GPS 좌표 정확한 변환
- **🏷️ 레이블 생성**: 어노테이션 기반 자동 샘플링
- **🔍 특징 추출**: 통계/텍스처/형태/주파수 특징
- **🤖 앙상블 분류**: 다중 모델 최적화 및 스태킹
- **🏔️ 지형 분석**: 해저 지형적 맥락 고려

## 📁 프로젝트 구조

```
Analysis_MD/
├── pipeline/                      # 📦 통합 파이프라인 (신규)
│   ├── unified_pipeline.py       # 메인 실행 파일
│   └── modules/                   # 모듈별 구성요소
│       ├── xtf_reader.py         # XTF 파일 읽기
│       ├── xtf_extractor.py      # 강도 데이터 추출
│       ├── coordinate_mapper.py   # 좌표 매핑
│       ├── label_generator.py     # 레이블 생성
│       ├── feature_extractor.py   # 특징 추출
│       ├── ensemble_optimizer.py  # 앙상블 최적화
│       ├── mine_classifier.py     # 분류기
│       └── terrain_analyzer.py    # 지형 분석
├── data/
│   ├── processed/                 # 처리된 데이터
│   │   ├── coordinate_mappings/   # 좌표 매핑 결과
│   │   ├── features/             # 추출된 특징
│   │   └── xtf_extracted/        # XTF 추출 데이터
│   └── raw/                      # 원본 데이터
├── datasets/                     # 데이터셋
│   ├── Location_MDGPS.xlsx       # GPS 좌표 데이터
│   ├── PH_annotation.png         # 어노테이션 이미지
│   └── Pohang_*/                 # XTF 원본 데이터
├── src/                          # 기존 소스 코드
├── config/                       # 설정 파일
├── deprecated/                   # 더 이상 사용하지 않는 파일들
├── real_data_pipeline.py         # ✅ 활성 - 실제 데이터 파이프라인
├── process_edgetech_complete.py  # ✅ 활성 - 에지텍 처리
└── README.md                     # 이 파일
```

## 🔧 설치 및 설정

### 필수 패키지 설치

```bash
pip install -r requirements.txt
```

주요 의존성:
- numpy, pandas, scikit-learn
- opencv-python, pillow
- matplotlib, seaborn
- optuna (앙상블 최적화용)
- pyxtf (XTF 파일 처리용, 선택적)

## 🚀 사용법

### 1. 통합 파이프라인 (추천)

**전체 파이프라인 실행:**
```bash
python pipeline/unified_pipeline.py \
    --xtf datasets/Pohang_*/original/*.XTF \
    --gps datasets/Location_MDGPS.xlsx \
    --annotation datasets/PH_annotation.png \
    --output results/
```

**모듈별 실행:**
```bash
# 특정 단계만 실행
python pipeline/unified_pipeline.py \
    --mode modular \
    --steps read extract map feature classify \
    --xtf datasets/Pohang_*/original/*.XTF
```

### 2. 개별 파이프라인

**실제 데이터 처리:**
```bash
python real_data_pipeline.py
```

**에지텍 데이터 처리:**
```bash
python process_edgetech_complete.py
```

## 📊 파이프라인 단계별 설명

### 1. XTF 읽기 (XTF Reader)
- 사이드스캔 소나 XTF 파일 읽기
- 메타데이터 및 네비게이션 정보 추출

### 2. 데이터 추출 (XTF Extractor)
- 포트/스타보드 채널 강도 데이터 추출
- 데이터 전처리 및 정규화

### 3. 좌표 매핑 (Coordinate Mapper)
- 픽셀 좌표 ↔ GPS 좌표 변환
- 어노테이션 이미지와 실제 좌표 매핑
- **중요**: 180도 회전 + 좌우 반전 변환 적용

### 4. 레이블 생성 (Label Generator)
- 어노테이션 기반 positive/negative 샘플 생성
- 배경 영역 자동 샘플링

### 5. 특징 추출 (Feature Extractor)
- **통계적 특징**: 평균, 표준편차, 왜도, 첨도
- **텍스처 특징**: LBP, GLCM, Gabor 필터
- **형태학적 특징**: 침식, 팽창, 영역 특성
- **주파수 특징**: FFT, 스펙트럼 분석

### 6. 앙상블 최적화 (Ensemble Optimizer)
- 개별 모델 하이퍼파라미터 최적화 (Optuna)
- 보팅 앙상블 조합 최적화
- 스태킹 앙상블 구성

### 7. 분류 (Mine Classifier)
- 최적화된 앙상블 모델 사용
- 확률 예측 및 성능 평가

### 8. 지형 분석 (Terrain Analyzer)
- 해저 지형 특성 분석
- 지형적 맥락을 고려한 분류 개선

## 📈 성능 지표

현재 달성된 성능:
- **좌표 매핑 정확도**: 98.4% (상관계수 0.9839)
- **특징 추출**: 13개 방법으로 100+ 특징
- **앙상블 최적화**: 5개 기본 모델 + 스태킹

## 🔬 데이터 분석

### 좌표 변환 분석
- **원본 → 어노테이션**: 180도 회전 + 좌우 반전
- **GPS 진행방향**: 서→동 (PH_01 → PH_25)
- **매핑 신뢰도**: 평균 0.83, 선형 관계 R² = 0.968

### 데이터 통계
- **GPS 포인트**: 25개 (PH_01 ~ PH_25)
- **이미지 크기**: 1024 × 3862 픽셀
- **좌표 범위**: 위도 36.593°N, 경도 129.509°~129.514°E

## 🛠 개발자 가이드

### 새로운 특징 추가
```python
# pipeline/modules/feature_extractor.py
def _extract_custom_features(self, patches):
    # 새로운 특징 추출 로직
    return features, feature_names
```

### 새로운 분류기 추가
```python
# pipeline/modules/ensemble_optimizer.py
def _optimize_custom_model(self, features, labels):
    # 새로운 모델 최적화 로직
    return {'score': score, 'params': params, 'model': model}
```

## 📋 주의사항

1. **XTF 파일 크기**: 대용량 파일의 경우 메모리 사용량 주의
2. **좌표 변환**: 어노테이션 이미지는 원본 대비 변환된 상태
3. **GPU 사용**: scikit-learn 기반이므로 CPU 최적화됨
4. **데이터 경로**: 절대 경로 사용 권장

## 🔄 업데이트 내역

- **v1.0.0** (2024-09-16): 통합 파이프라인 및 모듈화 완료
- **v0.9.0**: 좌표 매핑 시스템 검증 완료
- **v0.8.0**: 기본 특징 추출 및 분류 파이프라인

## 📧 문의

프로젝트 관련 문의: YMARX

---

**🚀 다음 단계**: `datasets`의 simulation 데이터를 활용한 특징 추출 및 탐지 성능 평가