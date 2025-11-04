# Mine Labeling Project

사이드 스캔 소나 XTF 데이터에서 기뢰를 라벨링하는 독립 프로젝트

## 프로젝트 개요

Klein 3900 사이드 스캔 소나 데이터(XTF)에서 추출한 강도 데이터(.npy)에 기뢰 위치를 정확하게 라벨링하고, 머신러닝 학습용 데이터셋을 생성합니다.

### 주요 기능

- ✅ XTF 파일 전체 추출 (Full extraction, 샘플링 없음)
- ✅ 방향 보존된 이미지 생성
- ✅ 바운딩 박스 좌표 매핑 (해상도 차이 처리)
- ✅ NPY 데이터 라벨링 (픽셀 마스크 + 좌표 정보)
- ✅ 샘플링 (기뢰:배경 = 1:5 비율)
- ✅ 데이터 증강 (9가지 기법)
- ✅ 검증 및 시각화

## 빠른 시작

### 설치

```bash
# 1. 저장소 클론 또는 다운로드
cd mine_labeling_project

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 설치 확인
python -c "import numpy, PIL, matplotlib; print('설치 완료!')"
```

상세 설치 방법은 [INSTALL.md](INSTALL.md)를 참조하세요.

### 사용법

```bash
# 기존 검증된 데이터 사용 예시
python -c "
import numpy as np
import json

# 검증된 flipped 데이터 로드
data = np.load('verified_data/flipped_20251104/flipped_labeled_intensity_data.npz', allow_pickle=True)

print(f'강도 데이터: {data[\"intensity\"].shape}')
print(f'라벨 마스크: {data[\"labels\"].shape}')
print(f'기뢰 개수: {len(json.loads(str(data[\"metadata\"])))}')
"
```

상세 사용법은 [USAGE.md](USAGE.md)를 참조하세요.

## 프로젝트 구조

```
mine_labeling_project/
├── README.md                    # 이 파일
├── INSTALL.md                   # 설치 가이드
├── USAGE.md                     # 사용 가이드
├── VERIFIED_MODULES.md          # 검증된 모듈 목록
├── requirements.txt             # Python 의존성
│
├── config/                      # 설정 파일
├── data/                        # 데이터 디렉토리
│   ├── raw/                    # 원본 XTF
│   ├── extracted/              # 추출된 NPY
│   ├── images/                 # 생성된 이미지
│   ├── annotations/            # 어노테이션
│   ├── labels/                 # 라벨 데이터
│   └── samples/                # 샘플 및 증강
│
├── verified_data/              # 검증된 기존 데이터
│   └── flipped_20251104/       # 2025-11-04 검증 완료
│
├── outputs/                    # 타임스탬프별 출력
│
├── src/mine_labeling/          # 소스 코드
│   ├── extractors/            # XTF 추출기
│   ├── converters/            # 이미지 변환기
│   ├── annotation/            # 어노테이션 도구
│   ├── labeling/              # 라벨링 처리기
│   ├── sampling/              # 샘플링
│   ├── augmentation/          # 데이터 증강
│   ├── validation/            # 검증 도구
│   └── utils/                 # 유틸리티
│
├── scripts/                    # 실행 스크립트
└── docs/                       # 문서
```

## 주요 데이터 형식

### NPZ 라벨 데이터

```python
{
    'intensity': (5137, 6400) float32,  # 원본 강도 데이터
    'labels': (5137, 6400) uint8,       # 픽셀별 마스크 (0/1)
    'metadata': JSON string              # 바운딩 박스 좌표 + 메타정보
}
```

- **좌표 정보**: metadata에 25개 기뢰의 픽셀 좌표
- **마스크**: labels 배열에 픽셀별 클래스 (0=배경, 1=기뢰)

자세한 내용은 [docs/HOW_TO_USE_NPZ_LABELS.md](docs/HOW_TO_USE_NPZ_LABELS.md)를 참조하세요.

## 검증된 데이터

프로젝트에는 2025-11-04에 검증 완료된 데이터가 포함되어 있습니다:

```
verified_data/flipped_20251104/
├── flipped_mine_label_mask.npy                    # 이진 마스크
├── flipped_labeled_intensity_data.npz             # 통합 데이터
├── flipped_mapped_annotations.json                # 좌표 정보
└── flipped_coordinate_mapping_visualization.png   # 시각화
```

### 검증 내역

- ✅ 차원 검증: PASS
- ✅ 스케일 팩터 검증: PASS (X=6.25, Y=1.0 flip)
- ✅ 라벨 마스크 검증: PASS (25개 모든 기뢰 확인)
- ✅ 시각적 검증: PASS (BMP 이미지와 정확히 일치)

## 핵심 원칙

### Y축 방향 처리

**중요**: 기존 제공된 BMP 이미지가 Y축 flip된 상태였습니다.

- **새로운 이미지 생성**: Flip 하지 않음 (방향 보존)
- **기존 데이터 사용**: Flipped 좌표 그대로 사용
- **설정 옵션**: `apply_flip` 옵션으로 조건부 적용

### 좌표 변환 공식

```python
# BMP (1024 × 5137) → NPY (6400 × 5137)

X_npy = X_bmp × 6.25     # X축 스케일
Y_npy = (5137 - 1) - Y_bmp  # Y축 flip (기존 데이터용)
```

## 문서

- [INSTALL.md](INSTALL.md) - 설치 가이드
- [USAGE.md](USAGE.md) - 사용 방법
- [VERIFIED_MODULES.md](VERIFIED_MODULES.md) - 검증된 모듈 목록
- [docs/HOW_TO_USE_NPZ_LABELS.md](docs/HOW_TO_USE_NPZ_LABELS.md) - NPZ 라벨 사용법
- [docs/LABEL_STRUCTURE_COMPARISON.md](docs/LABEL_STRUCTURE_COMPARISON.md) - 라벨 구조 비교
- [docs/REVISED_PROJECT_PLAN.md](docs/REVISED_PROJECT_PLAN.md) - 프로젝트 계획

## 라이선스

이 프로젝트는 ECMiner 프로젝트의 일부입니다.

## 지원

문의사항은 프로젝트 관리자에게 연락하세요.

---

**버전**: 1.0.0
**최종 업데이트**: 2025-11-04
**상태**: Production Ready (검증 완료)
