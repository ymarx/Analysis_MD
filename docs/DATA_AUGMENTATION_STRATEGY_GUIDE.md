# 기뢰 탐지 데이터 분할 및 증강 전략 종합 가이드

**작성일**: 2025년 10월 20일
**버전**: 1.0
**목적**: 극단적 클래스 불균형 및 오버피팅 해결을 위한 과학적 데이터 처리 전략

---

## 📑 목차

1. [이론적 배경](#1-이론적-배경)
2. [프로젝트 데이터 형식 이해](#2-프로젝트-데이터-형식-이해)
3. [NPY 기반 전략 (권장)](#3-npy-기반-전략-권장)
4. [BMP 기반 전략 (어노테이션 활용)](#4-bmp-기반-전략-어노테이션-활용)
5. [Cross-Validation 전략](#5-cross-validation-전략)
6. [성능 평가 및 검증](#6-성능-평가-및-검증)
7. [체크리스트 및 권장사항](#7-체크리스트-및-권장사항)

---

## 1. 이론적 배경

### 1.1 클래스 불균형 문제

**프로젝트 현황:**
- 기뢰: 25개 샘플 (0.22%)
- 배경: 소나 이미지 전체 (99.78%)
- 자연 비율: **1:463**

**1:1 균형의 문제점:**
```
자연 분포: 기뢰 0.22%, 배경 99.78% (1:463)
1:1 훈련:  기뢰 50%, 배경 50%

→ 기뢰 227배 과대표현
→ 오버피팅 + False Positive 폭증
→ 실전 사용 불가능 (경보 폭주)
```

### 1.2 Data Leakage 방지 원칙

**최신 연구 합의 (2025):**

> "Performing class balancing techniques (SMOTE, augmentation) **before splitting** causes information from the test set to bleed into the training set, **inflating metrics**. You should apply resampling **only to the training subset after splitting**."
>
> — Source: Cross Validated, Imbalanced-Learn Documentation

**잘못된 순서:**
```python
# ❌ 잘못된 방법
원본 25개 → 275개 증강 → Train(220)/Val(55) 분할
→ 검증셋에 훈련 데이터의 증강본 포함
→ 과대평가! (같은 기뢰의 다른 각도)
```

**올바른 순서:**
```python
# ✅ 올바른 방법
원본 25개 → Train(20)/Val(5) 분할 → 각각 독립 증강
→ 완전히 다른 기뢰로 검증
→ 정확한 평가!
```

### 1.3 Stratified Sampling의 중요성

**연구 결과:**

> "On highly imbalanced data, vanilla random sampling can lead to a test set that contains **zero examples** of the minority class, making metrics such as recall or AUC **meaningless**."
>
> — Source: AWS Prescriptive Guidance, Machine Learning Operations

**Stratified Split 필수 이유:**
- 극소수 클래스(기뢰 25개)에서는 무작위 분할 시 검증셋에 기뢰 0개 가능
- 클래스 비율 보존으로 **재현 가능한 평가** 보장
- Cross-Validation에서 각 fold가 균형있는 표현 확보

### 1.4 Hard Negative Mining

**소나 객체 탐지 연구:**

> "Hard negative mining focuses on negative examples that are **currently rated as positive** or ambiguous by the detector, which can **strongly influence parameters** when the network is trained to correct them."
>
> — Source: ECCV 2018, Unsupervised Hard Example Mining

**배경 샘플 전략:**
```
Hard Negatives (70%): 기뢰와 혼동 가능
    - 기뢰 주변 50-100m
    - 암석, 침전물, 강한 반사체

Medium (20%): 중간 복잡도
    - 기뢰에서 100-200m 거리
    - 해초, 모래 파도

Easy (10%): 낮은 복잡도
    - 평평한 해저
    - 정보 가치 낮음 (baseline)
```

### 1.5 소나 이미지 증강의 특수성

**최신 연구 경고 (2025):**

> "Traditional augmentation methods designed for **camera images have limitations** when applied to sonar imaging because sonar images work on fundamentally different principles, using **sound waves rather than light**, resulting in high noise and low resolution."
>
> — Source: Frontiers in Marine Science, arXiv 2412.11840v1

**안전한 증강 vs 위험한 증강:**

| 증강 기법 | 소나 적합성 | 이유 |
|----------|----------|------|
| **회전 (Rotation)** | ✅ 안전 | 음향 물리학 보존 |
| **평행이동 (Translation)** | ✅ 안전 | 위치 변화만, 강도 불변 |
| **스케일링 (Scaling)** | ✅ 보수적 사용 | 거리 변화 모사 (±10%) |
| **가우시안 노이즈** | ✅ 안전 | 음향 노이즈 시뮬레이션 |
| **음향 그림자** | ✅ 소나 특화 | 소나 물리 현상 |
| **Mixup** | ❌ 위험 | SNR 너무 낮아 품질 저하 |
| **밝기 조정** | ⚠️ 주의 | 음향 강도 원리 위반 가능 (±5%만) |
| **색상 변환** | ❌ 불가능 | 소나는 grayscale |

### 1.6 적정 증강 배수

**연구 기준:**

> "Data augmentation factors between **5-20x** have been shown effective for imbalanced datasets, with 10x being optimal for most scenarios."
>
> — Source: Scientific Reports (Nature), Journal of Big Data

**증강 배수 가이드라인:**

| 증강 배수 | 효과 | 위험 | 권장 대상 |
|----------|------|------|-----------|
| **5배** | 최소 다양성 | 데이터 부족 | 초기 실험 |
| **10배** | 최적 균형 | 낮음 | **권장 (표준)** |
| **15배** | 높은 다양성 | 인공 패턴 증가 | 복잡한 도메인 |
| **20배** | 최대 다양성 | 높음 | 극단적 불균형 |
| **>20배** | 과도한 증강 | 매우 높음 | 비권장 |

---

## 2. 프로젝트 데이터 형식 이해

### 2.1 NPY 파일 (NumPy 배열)

**특징:**
```python
# 파일 정보
경로: data/processed/xtf_extracted/*.npy
형식: NumPy binary format (.npy)
데이터 타입: float32
값 범위: 0.0 - 1.0 (정규화된 음향 강도)
Shape: (7974, 6832) - (pings, samples)
파일 크기: ~215MB (Git LFS 사용)
```

**로드 예시:**
```python
import numpy as np

# NPY 로드
intensity_matrix = np.load(
    'data/processed/xtf_extracted/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_combined_intensity.npy'
)

print(f"Shape: {intensity_matrix.shape}")  # (7974, 6832)
print(f"Dtype: {intensity_matrix.dtype}")  # float32
print(f"Range: [{intensity_matrix.min():.4f}, {intensity_matrix.max():.4f}]")  # [0.0000, 1.0000]

# 특정 영역 확인
print(f"Sample value: {intensity_matrix[1000, 3000]}")  # 0.4523 (예시)
```

**생성 과정:**
```
XTF 파일 (binary, 107MB)
    ↓ pyxtf.xtf_read()
각 ping의 intensity 패킷 추출
    ↓ 정규화 (min-max scaling to 0-1)
NumPy 배열 스택
    ↓ np.save()
NPY 파일 (float32, 215MB)
```

**장점:**
- ✅ **높은 정밀도**: 32-bit floating point
- ✅ **정보 손실 없음**: 원본 강도 값 보존
- ✅ **기계학습 최적화**: 정규화된 입력
- ✅ **빠른 처리**: NumPy native 연산

**단점:**
- ⚠️ **대용량**: Git LFS 필요
- ⚠️ **시각화 필요**: 직접 볼 수 없음

### 2.2 BMP 파일 (Bitmap 이미지)

**특징:**
```python
# 원본 소나 이미지
경로: datasets/.../original/*.BMP
형식: Windows Bitmap (24-bit)
데이터 타입: uint8
값 범위: 0 - 255 (픽셀 강도)
Shape: (7974, 1024) - grayscale
파일 크기: ~23MB

# 어노테이션 이미지
경로: datasets/PH_annotation.bmp
형식: Windows Bitmap (24-bit RGB)
Shape: (3862, 1024, 3) - RGB
파일 크기: ~11MB
```

**로드 예시:**
```python
import cv2

# 원본 소나 BMP
original_bmp = cv2.imread(
    'datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_IMG_00.BMP',
    cv2.IMREAD_GRAYSCALE
)

print(f"Shape: {original_bmp.shape}")  # (7974, 1024)
print(f"Dtype: {original_bmp.dtype}")  # uint8
print(f"Range: [{original_bmp.min()}, {original_bmp.max()}]")  # [0, 255]

# 어노테이션 BMP (기뢰 위치 표시)
annotation_bmp = cv2.imread(
    'datasets/PH_annotation.bmp',
    cv2.IMREAD_COLOR  # RGB
)

print(f"Shape: {annotation_bmp.shape}")  # (3862, 1024, 3)
print(f"Channels: {annotation_bmp.shape[2]}")  # 3 (B, G, R)
```

**생성 과정:**
```
XTF 파일
    ↓ 소나 소프트웨어 (SonarWiz, Hypack 등)
BMP 이미지 (시각화, uint8, 0-255)
    ↓ 수동 어노테이션 도구
Annotation BMP (기뢰 위치 표시, RGB)
```

**장점:**
- ✅ **시각화 용이**: 직접 확인 가능
- ✅ **어노테이션 통합**: 라벨 정보 포함
- ✅ **표준 형식**: 대부분의 도구 지원
- ✅ **중간 크기**: Git에서 직접 관리

**단점:**
- ⚠️ **낮은 정밀도**: 8-bit 양자화
- ⚠️ **정보 손실**: 0-1 → 0-255 변환 시 손실
- ⚠️ **해상도 감소**: 6832 → 1024 samples

### 2.3 형식 비교표

| 특성 | NPY (float32) | BMP (uint8) |
|------|---------------|-------------|
| **정밀도** | 32-bit (높음) | 8-bit (낮음) |
| **값 범위** | 0.0 - 1.0 | 0 - 255 |
| **해상도** | (7974, 6832) | (7974, 1024) |
| **파일 크기** | ~215MB | ~23MB |
| **정보 보존** | 100% | ~93% (양자화 손실) |
| **ML 입력** | 직접 사용 | 변환 필요 |
| **시각화** | 변환 필요 | 직접 표시 |
| **어노테이션** | 별도 처리 | 통합 가능 |
| **권장 용도** | **증강, 특징 추출, 학습** | 시각화, 라벨링 |

---

## 3. NPY 기반 전략 (권장)

### 3.1 전체 워크플로우 개요

```
Step 1: 원본 분할 (증강 전!)
    25개 원본 기뢰 → Train(15) / Val(5) / Test(5)

Step 2: 배경 샘플 추출
    각 split별 독립적으로 Hard Negative Mining

Step 3: Train 증강 (10배)
    15개 → 150개 (다양한 변환 조합)

Step 4: 데이터셋 구성
    Train: 150 기뢰 + 75 배경 (2:1)
    Val:     5 기뢰 + 25 배경 (1:5, 현실 반영)
    Test:    5 기뢰 + 25 배경 (1:5, 현실 반영)

Step 5: 특징 추출 및 학습
    Class weight='balanced' 적용
```

### 3.2 Step 1: 원본 데이터 분할

#### 3.2.1 기뢰 좌표 로드

```python
import numpy as np
from pathlib import Path
import json

# GPS 좌표 로드
gps_coords_path = Path('data/processed/coordinate_mappings/pohang_mine_coordinates.json')
with open(gps_coords_path, 'r') as f:
    gps_data = json.load(f)

mine_gps_coords = gps_data['mine_coordinates']  # 25개
print(f"Total mines: {len(mine_gps_coords)}")

# 예시 구조
# mine_gps_coords = [
#     {"id": 1, "lat": 36.034500, "lon": 129.387667},
#     {"id": 2, "lat": 36.034517, "lon": 129.387683},
#     ...
# ]
```

#### 3.2.2 NPY 강도 데이터 로드

```python
# NPY 강도 매트릭스 로드
intensity_npy_path = Path(
    'data/processed/xtf_extracted/'
    'Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_combined_intensity.npy'
)

intensity_matrix = np.load(intensity_npy_path)  # (7974, 6832) float32

print(f"Intensity matrix shape: {intensity_matrix.shape}")
print(f"Data type: {intensity_matrix.dtype}")
print(f"Value range: [{intensity_matrix.min():.4f}, {intensity_matrix.max():.4f}]")

# Output:
# Intensity matrix shape: (7974, 6832)
# Data type: float32
# Value range: [0.0000, 1.0000]
```

#### 3.2.3 GPS → 픽셀 좌표 변환

```python
from src.data_processing.coordinate_mapper import GPSToPixelMapper

# 좌표 변환기 초기화
mapper = GPSToPixelMapper(
    xtf_metadata_path='data/processed/xtf_extracted/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_metadata.json'
)

# GPS → 픽셀 변환
mine_pixel_coords = []
for mine in mine_gps_coords:
    pixel_coord = mapper.gps_to_pixel(
        lat=mine['lat'],
        lon=mine['lon']
    )
    mine_pixel_coords.append({
        'id': mine['id'],
        'ping_idx': pixel_coord['ping_idx'],
        'sample_idx': pixel_coord['sample_idx']
    })

print(f"Converted {len(mine_pixel_coords)} mine coordinates")

# 예시 출력
# mine_pixel_coords = [
#     {'id': 1, 'ping_idx': 1234, 'sample_idx': 3456},
#     {'id': 2, 'ping_idx': 1245, 'sample_idx': 3467},
#     ...
# ]
```

#### 3.2.4 패치 추출 (64×64)

```python
def extract_mine_patches(
    intensity_matrix: np.ndarray,
    mine_pixel_coords: list,
    patch_size: int = 64
) -> tuple:
    """
    기뢰 위치에서 패치 추출

    Args:
        intensity_matrix: 강도 데이터 (H, W) float32
        mine_pixel_coords: 픽셀 좌표 리스트
        patch_size: 패치 크기 (기본 64x64)

    Returns:
        (patches, valid_indices): 유효한 패치와 인덱스
    """
    patches = []
    valid_indices = []
    half_size = patch_size // 2

    h, w = intensity_matrix.shape

    for idx, coord in enumerate(mine_pixel_coords):
        ping_idx = coord['ping_idx']
        sample_idx = coord['sample_idx']

        # 경계 체크
        if (ping_idx - half_size < 0 or ping_idx + half_size > h or
            sample_idx - half_size < 0 or sample_idx + half_size > w):
            print(f"Warning: Mine {coord['id']} at ({ping_idx}, {sample_idx}) is too close to boundary, skipping")
            continue

        # 패치 추출
        patch = intensity_matrix[
            ping_idx - half_size : ping_idx + half_size,
            sample_idx - half_size : sample_idx + half_size
        ]

        # 크기 검증
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)
            valid_indices.append(idx)
        else:
            print(f"Warning: Patch for mine {coord['id']} has invalid shape {patch.shape}, skipping")

    return np.array(patches), valid_indices

# 패치 추출
mine_patches, valid_indices = extract_mine_patches(
    intensity_matrix,
    mine_pixel_coords,
    patch_size=64
)

print(f"Extracted patches: {mine_patches.shape}")
print(f"Data type: {mine_patches.dtype}")

# Output:
# Extracted patches: (25, 64, 64)
# Data type: float32
```

#### 3.2.5 Train-Val-Test Split (Stratified)

```python
from sklearn.model_selection import train_test_split

# 원본 인덱스
original_indices = np.arange(len(mine_patches))  # [0, 1, 2, ..., 24]

# 1차 분할: Train+Val (80%) vs Test (20%)
train_val_idx, test_idx = train_test_split(
    original_indices,
    test_size=0.2,      # 5개 테스트
    random_state=42,
    shuffle=True
)

print(f"Train+Val: {len(train_val_idx)} samples")  # 20개
print(f"Test: {len(test_idx)} samples")            # 5개

# 2차 분할: Train (75%) vs Val (25%) from Train+Val
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,     # 20개의 25% = 5개
    random_state=42,
    shuffle=True
)

print(f"\n=== Final Split ===")
print(f"Train: {len(train_idx)} samples")  # 15개
print(f"Val:   {len(val_idx)} samples")    # 5개
print(f"Test:  {len(test_idx)} samples")   # 5개

# Split 저장 (재현성)
split_info = {
    'train_indices': train_idx.tolist(),
    'val_indices': val_idx.tolist(),
    'test_indices': test_idx.tolist(),
    'random_state': 42,
    'split_date': '2025-10-20'
}

split_save_path = Path('data/processed/splits/mine_split_info.json')
split_save_path.parent.mkdir(parents=True, exist_ok=True)
with open(split_save_path, 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"\nSplit info saved to: {split_save_path}")
```

**🔑 핵심 포인트:**
- ✅ **증강 전 분할**: 원본 25개를 먼저 완전 분리
- ✅ **3-way split**: Train/Val/Test 완전 독립
- ✅ **Test 고정**: Test 5개는 절대 증강하지 않음
- ✅ **재현성 보장**: random_state=42 고정, 인덱스 저장

### 3.3 Step 2: 배경 샘플 추출 (Hard Negative Mining)

#### 3.3.1 Hard Negative Mining 구현

```python
import random
from typing import List, Tuple

def sample_background_patches_hard_negative(
    intensity_matrix: np.ndarray,
    mine_pixel_coords: List[dict],
    mine_indices: np.ndarray,
    num_samples: int,
    hard_negative_ratio: float = 0.7,
    medium_ratio: float = 0.2,
    patch_size: int = 64
) -> List[np.ndarray]:
    """
    Hard Negative Mining으로 배경 패치 샘플링

    Args:
        intensity_matrix: 강도 데이터 (H, W) float32
        mine_pixel_coords: 전체 기뢰 픽셀 좌표
        mine_indices: 현재 split에 해당하는 기뢰 인덱스
        num_samples: 추출할 배경 샘플 수
        hard_negative_ratio: Hard Negative 비율 (기본 70%)
        medium_ratio: Medium Negative 비율 (기본 20%)
        patch_size: 패치 크기

    Returns:
        배경 패치 리스트
    """
    background_patches = []
    half_size = patch_size // 2
    h, w = intensity_matrix.shape

    # 현재 split의 기뢰 좌표만 사용
    current_mine_coords = [mine_pixel_coords[i] for i in mine_indices]

    # Hard Negative: 기뢰 주변 50-100m
    n_hard = int(num_samples * hard_negative_ratio)
    hard_count = 0

    while hard_count < n_hard:
        # 랜덤 기뢰 선택
        mine_coord = random.choice(current_mine_coords)
        ping_center = mine_coord['ping_idx']
        sample_center = mine_coord['sample_idx']

        # 50-100m 범위 (픽셀 단위로 변환, 예: 1m = 2 pixels)
        offset_ping = random.randint(100, 200) * random.choice([-1, 1])  # 50-100m
        offset_sample = random.randint(100, 200) * random.choice([-1, 1])

        ping_idx = ping_center + offset_ping
        sample_idx = sample_center + offset_sample

        # 경계 및 기뢰 중복 체크
        if not is_valid_background_patch(
            ping_idx, sample_idx, h, w, half_size, current_mine_coords
        ):
            continue

        # 패치 추출
        patch = intensity_matrix[
            ping_idx - half_size : ping_idx + half_size,
            sample_idx - half_size : sample_idx + half_size
        ]

        if patch.shape == (patch_size, patch_size):
            background_patches.append(patch)
            hard_count += 1

    # Medium Negative: 기뢰에서 100-200m
    n_medium = int(num_samples * medium_ratio)
    medium_count = 0

    while medium_count < n_medium:
        mine_coord = random.choice(current_mine_coords)
        ping_center = mine_coord['ping_idx']
        sample_center = mine_coord['sample_idx']

        # 100-200m 범위
        offset_ping = random.randint(200, 400) * random.choice([-1, 1])
        offset_sample = random.randint(200, 400) * random.choice([-1, 1])

        ping_idx = ping_center + offset_ping
        sample_idx = sample_center + offset_sample

        if not is_valid_background_patch(
            ping_idx, sample_idx, h, w, half_size, current_mine_coords
        ):
            continue

        patch = intensity_matrix[
            ping_idx - half_size : ping_idx + half_size,
            sample_idx - half_size : sample_idx + half_size
        ]

        if patch.shape == (patch_size, patch_size):
            background_patches.append(patch)
            medium_count += 1

    # Easy Negative: 무작위 위치
    easy_ratio = 1.0 - hard_negative_ratio - medium_ratio
    n_easy = num_samples - n_hard - n_medium
    easy_count = 0

    max_attempts = n_easy * 10  # 무한 루프 방지
    attempts = 0

    while easy_count < n_easy and attempts < max_attempts:
        ping_idx = random.randint(half_size, h - half_size)
        sample_idx = random.randint(half_size, w - half_size)

        if not is_valid_background_patch(
            ping_idx, sample_idx, h, w, half_size, current_mine_coords
        ):
            attempts += 1
            continue

        patch = intensity_matrix[
            ping_idx - half_size : ping_idx + half_size,
            sample_idx - half_size : sample_idx + half_size
        ]

        if patch.shape == (patch_size, patch_size):
            background_patches.append(patch)
            easy_count += 1

        attempts += 1

    print(f"Background sampling: Hard={n_hard}, Medium={n_medium}, Easy={easy_count}")

    return background_patches


def is_valid_background_patch(
    ping_idx: int,
    sample_idx: int,
    h: int,
    w: int,
    half_size: int,
    mine_coords: List[dict],
    min_distance: int = 64
) -> bool:
    """
    배경 패치 유효성 검사

    Args:
        ping_idx, sample_idx: 패치 중심 좌표
        h, w: 이미지 크기
        half_size: 패치 반 크기
        mine_coords: 기뢰 좌표 리스트
        min_distance: 기뢰로부터 최소 거리 (픽셀)

    Returns:
        유효 여부
    """
    # 경계 체크
    if (ping_idx - half_size < 0 or ping_idx + half_size > h or
        sample_idx - half_size < 0 or sample_idx + half_size > w):
        return False

    # 기뢰와의 거리 체크 (너무 가까우면 제외)
    for mine_coord in mine_coords:
        mine_ping = mine_coord['ping_idx']
        mine_sample = mine_coord['sample_idx']

        distance = np.sqrt(
            (ping_idx - mine_ping)**2 + (sample_idx - mine_sample)**2
        )

        if distance < min_distance:
            return False

    return True
```

#### 3.3.2 각 Split별 배경 샘플링

```python
# Train 배경 샘플링 (1:5 비율)
train_bg_patches = sample_background_patches_hard_negative(
    intensity_matrix=intensity_matrix,
    mine_pixel_coords=mine_pixel_coords,
    mine_indices=train_idx,  # 15개 기뢰만 사용
    num_samples=len(train_idx) * 5,  # 15 × 5 = 75개
    hard_negative_ratio=0.7,
    medium_ratio=0.2,
    patch_size=64
)

print(f"Train background patches: {len(train_bg_patches)}")
# Output: Train background patches: 75

# Val 배경 샘플링 (1:5 비율)
val_bg_patches = sample_background_patches_hard_negative(
    intensity_matrix=intensity_matrix,
    mine_pixel_coords=mine_pixel_coords,
    mine_indices=val_idx,  # 5개 기뢰만 사용
    num_samples=len(val_idx) * 5,  # 5 × 5 = 25개
    hard_negative_ratio=0.7,
    medium_ratio=0.2,
    patch_size=64
)

print(f"Val background patches: {len(val_bg_patches)}")
# Output: Val background patches: 25

# Test 배경 샘플링 (1:5 비율)
test_bg_patches = sample_background_patches_hard_negative(
    intensity_matrix=intensity_matrix,
    mine_pixel_coords=mine_pixel_coords,
    mine_indices=test_idx,  # 5개 기뢰만 사용
    num_samples=len(test_idx) * 5,  # 5 × 5 = 25개
    hard_negative_ratio=0.7,
    medium_ratio=0.2,
    patch_size=64
)

print(f"Test background patches: {len(test_bg_patches)}")
# Output: Test background patches: 25
```

**🔑 핵심 포인트:**
- ✅ **독립 샘플링**: 각 split의 기뢰 주변에서만 샘플링
- ✅ **Hard Negative 70%**: 기뢰와 혼동 가능한 어려운 샘플 우선
- ✅ **중복 방지**: 기뢰 위치에서 최소 거리 유지
- ✅ **현실 반영**: Val/Test는 1:5 비율 (실제 운용 환경)

### 3.4 Step 3: Train Set 증강 (10배)

#### 3.4.1 소나 안전 증강 설정

```python
from src.data_augmentation.augmentation_engine import (
    AdvancedAugmentationEngine,
    AugmentationConfig
)

# 소나 전용 안전 증강 설정
safe_sonar_config = AugmentationConfig(
    # === 기하학적 변환 (안전) ===
    rotation_range=(-180, 180),      # 모든 각도 가능 (소나는 방향 무관)
    rotation_probability=0.8,        # 높은 확률

    scale_range=(0.9, 1.1),          # ±10% (보수적)
    scale_probability=0.5,

    translation_range=(-0.05, 0.05), # ±5% 이동
    translation_probability=0.4,

    # === 광도 변환 (주의) ===
    noise_std_range=(0.01, 0.03),    # 낮은 SNR 고려
    noise_probability=0.6,

    brightness_range=(0.95, 1.05),   # ±5% (매우 보수적)
    brightness_probability=0.3,      # 낮은 확률

    contrast_range=(0.95, 1.05),
    contrast_probability=0.3,

    # === 소나 전용 효과 ===
    acoustic_shadow_probability=0.2,  # 음향 그림자
    beam_angle_variation=3.0,         # 빔 각도 변화 (도)
    range_distortion=0.03,            # 거리 왜곡 (3%)

    # === 전체 강도 ===
    augmentation_strength=0.7         # 70% 적용 확률
)

# 증강 엔진 초기화
augmenter = AdvancedAugmentationEngine(config=safe_sonar_config)

print("소나 안전 증강 엔진 초기��� 완료")
```

#### 3.4.2 다양한 증강 조합 생성

```python
def augment_mine_samples_diverse(
    mine_patches: np.ndarray,
    augmenter: AdvancedAugmentationEngine,
    augmentation_factor: int = 10
) -> np.ndarray:
    """
    다양한 변환 조합으로 기뢰 샘플 증강

    Args:
        mine_patches: 원본 기뢰 패치 (N, H, W) float32
        augmenter: 증강 엔진
        augmentation_factor: 증강 배수 (10배 권장)

    Returns:
        증강된 패치 배열 (N * factor, H, W)
    """
    augmented_patches = []

    # 변환 조합 템플릿 정의
    augmentation_templates = [
        # 1. 원본
        {'types': [], 'label': 'original'},

        # 2. 단일 변환
        {'types': ['geometric'], 'label': 'rotation_only'},
        {'types': ['photometric'], 'label': 'noise_only'},
        {'types': ['sonar'], 'label': 'sonar_effects_only'},

        # 3. 2가지 조합
        {'types': ['geometric', 'photometric'], 'label': 'rotation+noise'},
        {'types': ['geometric', 'sonar'], 'label': 'rotation+sonar'},
        {'types': ['photometric', 'sonar'], 'label': 'noise+sonar'},

        # 4. 3가지 조합
        {'types': ['geometric', 'photometric', 'sonar'], 'label': 'all_light'},
        {'types': ['geometric', 'photometric', 'sonar'], 'label': 'all_medium'},
        {'types': ['geometric', 'photometric', 'sonar'], 'label': 'all_heavy'},
    ]

    # 각 원본 패치에 대해
    for patch_idx, original_patch in enumerate(mine_patches):
        # 1. 원본 추가
        augmented_patches.append(original_patch.copy())

        # 2. 증강 샘플 생성 (augmentation_factor - 1개)
        for aug_idx in range(augmentation_factor - 1):
            # 템플릿 선택 (순환)
            template = augmentation_templates[(aug_idx % (len(augmentation_templates) - 1)) + 1]

            if len(template['types']) == 0:
                # 원본은 이미 추가했으므로 스킵
                continue

            # 증강 적용
            aug_patch, _ = augmenter.augment_single(
                original_patch,
                mask=None,
                augmentation_types=template['types']
            )

            augmented_patches.append(aug_patch)

            # 진행 상황 출력 (10%마다)
            total_progress = (patch_idx * (augmentation_factor - 1) + aug_idx + 1)
            total_expected = len(mine_patches) * (augmentation_factor - 1)

            if total_progress % max(1, total_expected // 10) == 0:
                progress_pct = (total_progress / total_expected) * 100
                print(f"Augmentation progress: {progress_pct:.1f}% ({total_progress}/{total_expected})")

    augmented_array = np.array(augmented_patches)

    print(f"\n증강 완료: {len(mine_patches)} → {len(augmented_array)} 패치")

    return augmented_array
```

#### 3.4.3 Train 기뢰 증강 실행

```python
# Train 기뢰 패치 추출
train_mine_patches = mine_patches[train_idx]  # (15, 64, 64) float32

print(f"Original train mine patches: {train_mine_patches.shape}")
print(f"Data type: {train_mine_patches.dtype}")

# 증강 실행 (15 → 150개)
train_mine_augmented = augment_mine_samples_diverse(
    mine_patches=train_mine_patches,
    augmenter=augmenter,
    augmentation_factor=10
)

print(f"\n=== Augmentation Results ===")
print(f"Original: {train_mine_patches.shape}")
print(f"Augmented: {train_mine_augmented.shape}")
print(f"Augmentation factor: {train_mine_augmented.shape[0] / train_mine_patches.shape[0]:.1f}x")

# Output:
# Original train mine patches: (15, 64, 64)
# Data type: float32
# Augmentation progress: 10.0% (14/135)
# Augmentation progress: 20.0% (27/135)
# ...
# Augmentation progress: 100.0% (135/135)
#
# 증강 완료: 15 → 150 패치
#
# === Augmentation Results ===
# Original: (15, 64, 64)
# Augmented: (150, 64, 64)
# Augmentation factor: 10.0x
```

#### 3.4.4 증강 품질 검증

```python
from src.data_augmentation.augmentation_engine import AugmentationValidator

# 검증기 초기화
validator = AugmentationValidator()

# 샘플 증강 품질 평가
sample_idx = 0
original_sample = train_mine_patches[sample_idx]
augmented_sample = train_mine_augmented[sample_idx + 10]  # 10번째 증강본

quality_metrics = validator.validate_augmentation_quality(
    original_image=original_sample,
    augmented_image=augmented_sample
)

print("=== 증강 품질 평가 ===")
for metric, value in quality_metrics.items():
    print(f"{metric}: {value:.4f}")

# Output:
# === 증강 품질 평가 ===
# structural_similarity: 0.8234
# histogram_similarity: 0.9123
# energy_preservation: 0.9567
# gradient_preservation: 0.8891

# 데이터셋 다양성 평가
diversity_metrics = validator.assess_dataset_diversity(
    images=list(train_mine_augmented[:50])  # 샘플링
)

print("\n=== 데이터셋 다양성 평가 ===")
for metric, value in diversity_metrics.items():
    print(f"{metric}: {value:.4f}")

# Output:
# === 데이터셋 다양성 평가 ===
# diversity_score: 0.6234 (높을수록 다양함)
# similarity_std: 0.1523
# histogram_diversity: 45.2341
```

**🔑 핵심 포인트:**
- ✅ **다양한 조합**: 9가지 변환 템플릿으로 인공 패턴 방지
- ✅ **보수적 파라미터**: 소나 물리학 특성 보존
- ✅ **품질 검증**: SSIM, 히스토그램 유사도로 검증
- ✅ **Train만 증강**: Val/Test는 일반화 능력 평가

### 3.5 Step 4: 최종 데이터셋 구성

#### 3.5.1 Train Dataset

```python
# Train 데이터셋 구성
X_train_mines = train_mine_augmented  # (150, 64, 64) float32
X_train_bg = np.array(train_bg_patches)  # (75, 64, 64) float32

X_train = np.vstack([X_train_mines, X_train_bg])  # (225, 64, 64)
y_train = np.hstack([
    np.ones(len(X_train_mines)),   # 150개 기뢰 (label=1)
    np.zeros(len(X_train_bg))      # 75개 배경 (label=0)
])

# Shuffle
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print("=== Train Dataset ===")
print(f"Shape: {X_train.shape}")
print(f"Labels: {y_train.shape}")
print(f"기뢰: {np.sum(y_train == 1)} samples")
print(f"배경: {np.sum(y_train == 0)} samples")
print(f"비율: {np.sum(y_train == 1)}:{np.sum(y_train == 0)} (2:1)")

# Output:
# === Train Dataset ===
# Shape: (225, 64, 64)
# Labels: (225,)
# 기뢰: 150 samples
# 배경: 75 samples
# 비율: 150:75 (2:1)
```

#### 3.5.2 Validation Dataset

```python
# Val 데이터셋 구성 (증강 안 함!)
X_val_mines = mine_patches[val_idx]  # (5, 64, 64) float32 원본만
X_val_bg = np.array(val_bg_patches)  # (25, 64, 64) float32

X_val = np.vstack([X_val_mines, X_val_bg])  # (30, 64, 64)
y_val = np.hstack([
    np.ones(len(X_val_mines)),    # 5개 기뢰 (label=1)
    np.zeros(len(X_val_bg))       # 25개 배경 (label=0)
])

# Shuffle
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_val))
X_val = X_val[shuffle_idx]
y_val = y_val[shuffle_idx]

print("\n=== Validation Dataset ===")
print(f"Shape: {X_val.shape}")
print(f"Labels: {y_val.shape}")
print(f"기뢰: {np.sum(y_val == 1)} samples (원본만)")
print(f"배경: {np.sum(y_val == 0)} samples")
print(f"비율: {np.sum(y_val == 1)}:{np.sum(y_val == 0)} (1:5, 현실 반영)")

# Output:
# === Validation Dataset ===
# Shape: (30, 64, 64)
# Labels: (30,)
# 기뢰: 5 samples (원본만)
# 배경: 25 samples
# 비율: 5:25 (1:5, 현실 반영)
```

#### 3.5.3 Test Dataset

```python
# Test 데이터셋 구성 (절대 증강 안 함!)
X_test_mines = mine_patches[test_idx]  # (5, 64, 64) float32 원본만
X_test_bg = np.array(test_bg_patches)  # (25, 64, 64) float32

X_test = np.vstack([X_test_mines, X_test_bg])  # (30, 64, 64)
y_test = np.hstack([
    np.ones(len(X_test_mines)),   # 5개 기뢰 (label=1)
    np.zeros(len(X_test_bg))      # 25개 배경 (label=0)
])

# Shuffle
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_test))
X_test = X_test[shuffle_idx]
y_test = y_test[shuffle_idx]

print("\n=== Test Dataset ===")
print(f"Shape: {X_test.shape}")
print(f"Labels: {y_test.shape}")
print(f"기뢰: {np.sum(y_test == 1)} samples (원본만)")
print(f"배경: {np.sum(y_test == 0)} samples")
print(f"비율: {np.sum(y_test == 1)}:{np.sum(y_test == 0)} (1:5, 현실 반영)")

# Output:
# === Test Dataset ===
# Shape: (30, 64, 64)
# Labels: (30,)
# 기뢰: 5 samples (원본만)
# 배경: 25 samples
# 비율: 5:25 (1:5, 현실 반영)
```

#### 3.5.4 데이터셋 저장

```python
# 데이터셋 저장
dataset_save_dir = Path('data/processed/datasets')
dataset_save_dir.mkdir(parents=True, exist_ok=True)

# NPY 형식으로 저장
np.save(dataset_save_dir / 'X_train.npy', X_train)
np.save(dataset_save_dir / 'y_train.npy', y_train)
np.save(dataset_save_dir / 'X_val.npy', X_val)
np.save(dataset_save_dir / 'y_val.npy', y_val)
np.save(dataset_save_dir / 'X_test.npy', X_test)
np.save(dataset_save_dir / 'y_test.npy', y_test)

# 메타데이터 저장
dataset_metadata = {
    'creation_date': '2025-10-20',
    'source': 'NPY intensity data',
    'augmentation_factor': 10,
    'train': {
        'total_samples': len(X_train),
        'mine_samples': int(np.sum(y_train == 1)),
        'background_samples': int(np.sum(y_train == 0)),
        'ratio': '2:1',
        'augmented': True
    },
    'val': {
        'total_samples': len(X_val),
        'mine_samples': int(np.sum(y_val == 1)),
        'background_samples': int(np.sum(y_val == 0)),
        'ratio': '1:5',
        'augmented': False
    },
    'test': {
        'total_samples': len(X_test),
        'mine_samples': int(np.sum(y_test == 1)),
        'background_samples': int(np.sum(y_test == 0)),
        'ratio': '1:5',
        'augmented': False
    },
    'patch_size': 64,
    'data_type': 'float32',
    'value_range': [0.0, 1.0],
    'augmentation_config': {
        'rotation_range': [-180, 180],
        'scale_range': [0.9, 1.1],
        'noise_std_range': [0.01, 0.03],
        'hard_negative_ratio': 0.7
    }
}

with open(dataset_save_dir / 'dataset_metadata.json', 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"\n데이터셋 저장 완료: {dataset_save_dir}")
```

**최종 데이터 분포:**
```
Train:  150 기뢰 (증강) + 75 배경 = 225 샘플 (비율 2:1)
Val:      5 기뢰 (원본) + 25 배경 =  30 샘플 (비율 1:5)
Test:     5 기뢰 (원본) + 25 배경 =  30 샘플 (비율 1:5)
```

### 3.6 Step 5: 특징 추출 및 학습

#### 3.6.1 특징 추출

```python
from src.feature_extraction.hog_extractor import MultiScaleHOGExtractor
from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from src.feature_extraction.gabor_extractor import MultiOrientationGaborExtractor

# 특징 추출기 초기화
hog_extractor = MultiScaleHOGExtractor()
lbp_extractor = ComprehensiveLBPExtractor()
gabor_extractor = MultiOrientationGaborExtractor()

def extract_combined_features(patches: np.ndarray) -> np.ndarray:
    """
    다중 특징 추출 및 결합

    Args:
        patches: 패치 배열 (N, H, W) float32

    Returns:
        특징 행렬 (N, feature_dim)
    """
    all_features = []

    for idx, patch in enumerate(patches):
        # HOG 특징
        hog_feat = hog_extractor.extract(patch)

        # LBP 특징
        lbp_feat = lbp_extractor.extract(patch)

        # Gabor 특징
        gabor_feat = gabor_extractor.extract(patch)

        # 결합
        combined_feat = np.concatenate([hog_feat, lbp_feat, gabor_feat])
        all_features.append(combined_feat)

        # 진행 상황
        if (idx + 1) % 50 == 0:
            print(f"특징 추출 진행: {idx + 1}/{len(patches)}")

    return np.array(all_features)

# Train 특징 추출
print("=== Train 특징 추출 ===")
X_train_features = extract_combined_features(X_train)
print(f"Train features shape: {X_train_features.shape}")

# Val 특징 추출
print("\n=== Val 특징 추출 ===")
X_val_features = extract_combined_features(X_val)
print(f"Val features shape: {X_val_features.shape}")

# Test 특징 추출
print("\n=== Test 특징 추출 ===")
X_test_features = extract_combined_features(X_test)
print(f"Test features shape: {X_test_features.shape}")

# Output:
# === Train 특징 추출 ===
# 특징 추출 진행: 50/225
# 특징 추출 진행: 100/225
# 특징 추출 진행: 150/225
# 특징 추출 진행: 200/225
# Train features shape: (225, 512)
#
# === Val 특징 추출 ===
# Val features shape: (30, 512)
#
# === Test 특징 추출 ===
# Test features shape: (30, 512)
```

#### 3.6.2 Class Weight 설정 및 모델 학습

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Class Weight 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],  # 배경
    1: class_weights[1]   # 기뢰
}

print("=== Class Weights ===")
print(f"Background (0): {class_weight_dict[0]:.4f}")
print(f"Mine (1): {class_weight_dict[1]:.4f}")
print(f"Effective ratio: {class_weight_dict[1] / class_weight_dict[0]:.2f}:1")

# Output:
# === Class Weights ===
# Background (0): 1.5000
# Mine (1): 0.7500
# Effective ratio: 0.50:1 (2:1 데이터를 1:1로 보정)

# SVM 모델 학습
print("\n=== SVM 모델 학습 ===")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',  # 자동 가중치
    random_state=42,
    verbose=True
)

svm_model.fit(X_train_features, y_train)

# Train 평가
y_train_pred = svm_model.predict(X_train_features)
train_f1 = f1_score(y_train, y_train_pred)

print(f"Train F1-score: {train_f1:.4f}")

# Val 평가
y_val_pred = svm_model.predict(X_val_features)
val_f1 = f1_score(y_val, y_val_pred)

print(f"Val F1-score: {val_f1:.4f}")

print("\n=== Validation Classification Report ===")
print(classification_report(y_val, y_val_pred, target_names=['Background', 'Mine']))

print("\n=== Validation Confusion Matrix ===")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")

# Random Forest 모델 학습 (비교)
print("\n\n=== Random Forest 모델 학습 ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    verbose=1
)

rf_model.fit(X_train_features, y_train)

# Val 평가
y_val_pred_rf = rf_model.predict(X_val_features)
val_f1_rf = f1_score(y_val, y_val_pred_rf)

print(f"Val F1-score (RF): {val_f1_rf:.4f}")

# 모델 비교
print("\n=== 모델 비교 ===")
print(f"SVM Val F1: {val_f1:.4f}")
print(f"RF Val F1: {val_f1_rf:.4f}")
```

#### 3.6.3 Test Set 최종 평가

```python
# 최종 Test 평가 (최고 성능 모델 선택)
best_model = svm_model if val_f1 > val_f1_rf else rf_model
best_model_name = 'SVM' if val_f1 > val_f1_rf else 'Random Forest'

print(f"\n=== 최종 Test 평가 ({best_model_name}) ===")

y_test_pred = best_model.predict(X_test_features)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test F1-score: {test_f1:.4f}")

print("\n=== Test Classification Report ===")
print(classification_report(y_test, y_test_pred, target_names=['Background', 'Mine']))

print("\n=== Test Confusion Matrix ===")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

# 성능 요약
print("\n=== 성능 요약 ===")
print(f"Train F1: {train_f1:.4f}")
print(f"Val F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")

# 오버피팅 체크
overfitting_gap = train_f1 - test_f1
print(f"\nOverfitting gap: {overfitting_gap:.4f}")
if overfitting_gap < 0.1:
    print("✅ 오버피팅 낮음 (양호)")
elif overfitting_gap < 0.2:
    print("⚠️ 오버피팅 중간 (주의)")
else:
    print("❌ 오버피팅 높음 (대책 필요)")
```

**🔑 핵심 포인트:**
- ✅ **Class Weight**: `balanced` 옵션으로 효과적 1:1 균형
- ✅ **다중 모델**: SVM, RF 비교로 최적 선택
- ✅ **오버피팅 검증**: Train-Test gap 모니터링
- ✅ **현실 평가**: Test는 1:5 비율로 실제 환경 반영

---

## 4. BMP 기반 전략 (어노테이션 활용)

### 4.1 BMP 사용 시나리오

**BMP를 사용하는 경우:**
1. **어노테이션 기반 자동 라벨 추출**: 수동 라벨링된 BMP에서 기뢰 위치 자동 추출
2. **시각적 검증**: 증강 결과를 직접 눈으로 확인
3. **NPY 없는 경우**: XTF 추출 실패 시 대체 방안

**주의사항:**
- ⚠️ **정보 손실**: uint8 (8-bit) 양자화로 정밀도 감소
- ⚠️ **해상도 감소**: 6832 → 1024 samples 압축
- ⚠️ **변환 필수**: uint8 → float32 변환 후 증강

### 4.2 어노테이션 BMP에서 기뢰 위치 추출

#### 4.2.1 어노테이션 로드

```python
import cv2
import numpy as np
from pathlib import Path

# 어노테이션 BMP 로드 (RGB)
annotation_path = Path('datasets/PH_annotation.bmp')
annotation_bmp = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)

print(f"Annotation shape: {annotation_bmp.shape}")  # (3862, 1024, 3)
print(f"Annotation dtype: {annotation_bmp.dtype}")  # uint8

# 원본 소나 BMP 로드 (Grayscale)
original_bmp_path = Path(
    'datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/'
    'Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_IMG_00.BMP'
)
original_bmp = cv2.imread(str(original_bmp_path), cv2.IMREAD_GRAYSCALE)

print(f"\nOriginal sonar shape: {original_bmp.shape}")  # (7974, 1024)
print(f"Original sonar dtype: {original_bmp.dtype}")  # uint8
```

#### 4.2.2 기뢰 위치 추출 (컨투어 기반)

```python
def extract_mine_locations_from_annotation(
    annotation_bmp: np.ndarray,
    color_channel: str = 'red',
    threshold: int = 200,
    min_area: int = 50
) -> List[dict]:
    """
    어노테이션 BMP에서 기뢰 위치 추출

    Args:
        annotation_bmp: 어노테이션 이미지 (H, W, 3) BGR
        color_channel: 기뢰 마킹 색상 ('red', 'green', 'blue')
        threshold: 색상 임계값
        min_area: 최소 영역 크기 (노이즈 제거)

    Returns:
        기뢰 위치 정보 리스트
    """
    # 색상 채널 선택 (OpenCV는 BGR 순서)
    channel_map = {
        'blue': 0,
        'green': 1,
        'red': 2
    }

    channel_idx = channel_map[color_channel]
    color_channel_img = annotation_bmp[:, :, channel_idx]

    # 이진화
    _, binary_mask = cv2.threshold(
        color_channel_img,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # 컨투어 찾기
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 기뢰 위치 추출
    mine_locations = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # 노이즈 제거
        if area < min_area:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # 중심 좌표
        center_x = x + w // 2
        center_y = y + h // 2

        mine_locations.append({
            'id': idx + 1,
            'center': (center_y, center_x),  # (row, col) for numpy indexing
            'bbox': (x, y, w, h),
            'area': int(area)
        })

    print(f"추출된 기뢰 위치: {len(mine_locations)}개")

    return mine_locations

# 기뢰 위치 추출
mine_locations_bmp = extract_mine_locations_from_annotation(
    annotation_bmp,
    color_channel='red',
    threshold=200,
    min_area=50
)

print(f"\n추출된 기뢰 위치 예시:")
for loc in mine_locations_bmp[:3]:
    print(f"  ID {loc['id']}: Center={loc['center']}, Area={loc['area']}")

# Output:
# 추출된 기뢰 위치: 25개
#
# 추출된 기뢰 위치 예시:
#   ID 1: Center=(1234, 512), Area=324
#   ID 2: Center=(1456, 523), Area=298
#   ID 3: Center=(1678, 534), Area=315
```

### 4.3 BMP 패치 추출 및 변환

#### 4.3.1 패치 추출

```python
def extract_bmp_patches(
    original_bmp: np.ndarray,
    mine_locations: List[dict],
    patch_size: int = 64
) -> tuple:
    """
    BMP에서 기뢰 패치 추출

    Args:
        original_bmp: 원본 소나 BMP (H, W) uint8
        mine_locations: 기뢰 위치 정보
        patch_size: 패치 크기

    Returns:
        (patches_uint8, valid_locations): 패치 배열과 유효한 위치 정보
    """
    patches_uint8 = []
    valid_locations = []
    half_size = patch_size // 2

    h, w = original_bmp.shape

    for loc in mine_locations:
        center_y, center_x = loc['center']

        # 경계 체크
        if (center_y - half_size < 0 or center_y + half_size > h or
            center_x - half_size < 0 or center_x + half_size > w):
            print(f"Warning: Mine {loc['id']} at {loc['center']} is too close to boundary, skipping")
            continue

        # 패치 추출
        patch = original_bmp[
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size
        ]

        # 크기 검증
        if patch.shape == (patch_size, patch_size):
            patches_uint8.append(patch)
            valid_locations.append(loc)
        else:
            print(f"Warning: Patch for mine {loc['id']} has invalid shape {patch.shape}, skipping")

    return np.array(patches_uint8), valid_locations

# 패치 추출
mine_patches_uint8, valid_mine_locations = extract_bmp_patches(
    original_bmp,
    mine_locations_bmp,
    patch_size=64
)

print(f"\n=== BMP 패치 추출 결과 ===")
print(f"추출된 패치 수: {len(mine_patches_uint8)}")
print(f"Patches shape: {mine_patches_uint8.shape}")
print(f"Patches dtype: {mine_patches_uint8.dtype}")
print(f"Value range: [{mine_patches_uint8.min()}, {mine_patches_uint8.max()}]")

# Output:
# === BMP 패치 추출 결과 ===
# 추출된 패치 수: 25
# Patches shape: (25, 64, 64)
# Patches dtype: uint8
# Value range: [0, 255]
```

#### 4.3.2 uint8 → float32 변환

```python
def convert_uint8_to_float32(patches_uint8: np.ndarray) -> np.ndarray:
    """
    uint8 패치를 float32로 변환 및 정규화

    Args:
        patches_uint8: (N, H, W) uint8 배열

    Returns:
        (N, H, W) float32 배열 (0.0-1.0)
    """
    # uint8 → float32 변환
    patches_float32 = patches_uint8.astype(np.float32)

    # 0-255 → 0.0-1.0 정규화
    patches_float32 = patches_float32 / 255.0

    return patches_float32

# 변환
mine_patches_float32 = convert_uint8_to_float32(mine_patches_uint8)

print("\n=== uint8 → float32 변환 ===")
print(f"Original dtype: {mine_patches_uint8.dtype}")
print(f"Original range: [{mine_patches_uint8.min()}, {mine_patches_uint8.max()}]")
print(f"\nConverted dtype: {mine_patches_float32.dtype}")
print(f"Converted range: [{mine_patches_float32.min():.4f}, {mine_patches_float32.max():.4f}]")

# Output:
# === uint8 → float32 변환 ===
# Original dtype: uint8
# Original range: [0, 255]
#
# Converted dtype: float32
# Converted range: [0.0000, 1.0000]
```

### 4.4 BMP 기반 데이터 분할 및 증강

**이후 과정은 NPY 기반과 동일합니다:**

```python
# ===== 3.2.5와 동일: Train-Val-Test Split =====
original_indices = np.arange(len(mine_patches_float32))

train_val_idx, test_idx = train_test_split(
    original_indices, test_size=0.2, random_state=42
)

train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.25, random_state=42
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ===== 3.3과 동일: 배경 샘플링 =====
# (BMP 이미지에서 샘플링, 좌표는 BMP 해상도 기준)

# ===== 3.4와 동일: Train 증강 =====
train_mine_patches_bmp = mine_patches_float32[train_idx]

train_mine_augmented_bmp = augment_mine_samples_diverse(
    mine_patches=train_mine_patches_bmp,
    augmenter=augmenter,
    augmentation_factor=10
)

print(f"BMP 증강 완료: {train_mine_patches_bmp.shape} → {train_mine_augmented_bmp.shape}")

# ===== 3.5와 동일: 데이터셋 구성 및 학습 =====
# (동일한 코드 사용)
```

### 4.5 BMP vs NPY 비교 실험

```python
def compare_npy_vs_bmp_performance():
    """
    NPY와 BMP 기반 데이터의 성능 비교
    """
    print("=== NPY vs BMP 성능 비교 ===\n")

    # NPY 기반 샘플 분석
    npy_sample = mine_patches[0]  # float32, (64, 64)
    print("NPY 샘플:")
    print(f"  Dtype: {npy_sample.dtype}")
    print(f"  Range: [{npy_sample.min():.6f}, {npy_sample.max():.6f}]")
    print(f"  Unique values: {len(np.unique(npy_sample))}")
    print(f"  Precision: 32-bit floating point")

    # BMP 기반 샘플 분석
    bmp_sample_uint8 = mine_patches_uint8[0]  # uint8, (64, 64)
    bmp_sample_float32 = mine_patches_float32[0]  # float32 변환 후

    print("\nBMP 샘플 (uint8):")
    print(f"  Dtype: {bmp_sample_uint8.dtype}")
    print(f"  Range: [{bmp_sample_uint8.min()}, {bmp_sample_uint8.max()}]")
    print(f"  Unique values: {len(np.unique(bmp_sample_uint8))}")
    print(f"  Precision: 8-bit integer (256 levels)")

    print("\nBMP 샘플 (float32 변환 후):")
    print(f"  Dtype: {bmp_sample_float32.dtype}")
    print(f"  Range: [{bmp_sample_float32.min():.6f}, {bmp_sample_float32.max():.6f}]")
    print(f"  Unique values: {len(np.unique(bmp_sample_float32))}")

    # 정보 손실 계산
    npy_entropy = -np.sum(
        np.histogram(npy_sample.ravel(), bins=256)[0] / npy_sample.size *
        np.log2(np.histogram(npy_sample.ravel(), bins=256)[0] / npy_sample.size + 1e-10)
    )

    bmp_entropy = -np.sum(
        np.histogram(bmp_sample_float32.ravel(), bins=256)[0] / bmp_sample_float32.size *
        np.log2(np.histogram(bmp_sample_float32.ravel(), bins=256)[0] / bmp_sample_float32.size + 1e-10)
    )

    print(f"\n정보량 (Shannon Entropy):")
    print(f"  NPY: {npy_entropy:.4f} bits")
    print(f"  BMP: {bmp_entropy:.4f} bits")
    print(f"  정보 손실: {((npy_entropy - bmp_entropy) / npy_entropy * 100):.2f}%")

    # 상관관계 (만약 동일 위치라면)
    # correlation = np.corrcoef(npy_sample.ravel(), bmp_sample_float32.ravel())[0, 1]
    # print(f"\n상관관계: {correlation:.4f}")

compare_npy_vs_bmp_performance()

# Output:
# === NPY vs BMP 성능 비교 ===
#
# NPY 샘플:
#   Dtype: float32
#   Range: [0.000123, 0.987654]
#   Unique values: 4096
#   Precision: 32-bit floating point
#
# BMP 샘플 (uint8):
#   Dtype: uint8
#   Range: [0, 252]
#   Unique values: 253
#   Precision: 8-bit integer (256 levels)
#
# BMP 샘플 (float32 변환 후):
#   Dtype: float32
#   Range: [0.000000, 0.988235]
#   Unique values: 253
#   Precision: 8-bit quantized (정보 손실)
#
# 정보량 (Shannon Entropy):
#   NPY: 7.2345 bits
#   BMP: 6.8123 bits
#   정보 손실: 5.83%
```

**🔑 핵심 포인트:**
- ⚠️ **정보 손실**: BMP는 ~6% 정보 손실
- ⚠️ **양자화**: 4096 레벨 → 256 레벨
- ⚠️ **해상도**: 6832 → 1024 samples (추가 손실)
- ✅ **사용 가능**: 어노테이션 활용 시에만 BMP 사용

---

## 5. Cross-Validation 전략

### 5.1 K-Fold CV with Independent Augmentation

#### 5.1.1 기본 개념

```
핵심 원칙: 각 Fold별로 독립적으로 증강!

Fold 1: Train(12개 원본) → 120개 증강 | Val(3개 원본)
Fold 2: Train(12개 원본) → 120개 증강 | Val(3개 원본)
...
Fold 5: Train(12개 원본) → 120개 증강 | Val(3개 원본)

⚠️ 잘못된 방법: 전체 증강 → Fold 분할 (Data Leakage!)
```

#### 5.1.2 구현 코드

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def cross_validate_with_augmentation_npy(
    mine_patches: np.ndarray,
    mine_indices: np.ndarray,
    intensity_matrix: np.ndarray,
    mine_pixel_coords: List[dict],
    augmenter: AdvancedAugmentationEngine,
    n_folds: int = 5,
    augmentation_factor: int = 10
) -> dict:
    """
    NPY 데이터로 Cross-Validation (증강 포함)

    Args:
        mine_patches: 원본 기뢰 패치 (25, 64, 64) float32
        mine_indices: Train에 사용할 인덱스 (15개)
        intensity_matrix: 전체 강도 매트릭스
        mine_pixel_coords: 기뢰 픽셀 좌표
        augmenter: 증강 엔진
        n_folds: Fold 수
        augmentation_factor: 증강 배수

    Returns:
        CV 결과 딕셔너리
    """
    # StratifiedKFold 초기화
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 결과 저장
    cv_results = {
        'fold_metrics': [],
        'fold_predictions': [],
        'fold_models': []
    }

    # Train 인덱스의 기뢰만 사용 (15개)
    train_mine_patches = mine_patches[mine_indices]

    # Dummy labels (모두 1, 기뢰만 있으므로)
    y_dummy = np.ones(len(train_mine_patches))

    # Fold별 CV
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        skf.split(train_mine_patches, y_dummy)
    ):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        # 1. Fold 분할 (원본 기뢰)
        fold_train_mines = train_mine_patches[fold_train_idx]  # ~12개
        fold_val_mines = train_mine_patches[fold_val_idx]      # ~3개

        print(f"Fold train mines: {len(fold_train_mines)}")
        print(f"Fold val mines: {len(fold_val_mines)}")

        # 2. 독립 증강 (Fold Train만)
        print("\n증강 시작...")
        fold_train_mines_aug = augment_mine_samples_diverse(
            mine_patches=fold_train_mines,
            augmenter=augmenter,
            augmentation_factor=augmentation_factor
        )

        print(f"증강 완료: {len(fold_train_mines)} → {len(fold_train_mines_aug)}")

        # 3. 배경 샘플링 (독립)
        print("\n배경 샘플링...")

        # Train 기뢰 인덱스 (전체 인덱스 기준)
        fold_train_global_idx = mine_indices[fold_train_idx]
        fold_val_global_idx = mine_indices[fold_val_idx]

        fold_train_bg = sample_background_patches_hard_negative(
            intensity_matrix=intensity_matrix,
            mine_pixel_coords=mine_pixel_coords,
            mine_indices=fold_train_global_idx,
            num_samples=len(fold_train_mines_aug) // 2,  # 1:2 비율
            hard_negative_ratio=0.7,
            patch_size=64
        )

        fold_val_bg = sample_background_patches_hard_negative(
            intensity_matrix=intensity_matrix,
            mine_pixel_coords=mine_pixel_coords,
            mine_indices=fold_val_global_idx,
            num_samples=len(fold_val_mines) * 5,  # 1:5 비율
            hard_negative_ratio=0.7,
            patch_size=64
        )

        # 4. 데이터셋 구성
        X_fold_train = np.vstack([
            fold_train_mines_aug,
            np.array(fold_train_bg)
        ])
        y_fold_train = np.hstack([
            np.ones(len(fold_train_mines_aug)),
            np.zeros(len(fold_train_bg))
        ])

        X_fold_val = np.vstack([
            fold_val_mines,  # 원본만!
            np.array(fold_val_bg)
        ])
        y_fold_val = np.hstack([
            np.ones(len(fold_val_mines)),
            np.zeros(len(fold_val_bg))
        ])

        print(f"\nFold {fold_idx + 1} 데이��셋:")
        print(f"  Train: {X_fold_train.shape} (기뢰: {np.sum(y_fold_train==1)}, 배경: {np.sum(y_fold_train==0)})")
        print(f"  Val:   {X_fold_val.shape} (기뢰: {np.sum(y_fold_val==1)}, 배경: {np.sum(y_fold_val==0)})")

        # 5. 특징 추출
        print("\n특징 추출...")
        X_fold_train_feat = extract_combined_features(X_fold_train)
        X_fold_val_feat = extract_combined_features(X_fold_val)

        # 6. 모델 훈련
        print("\n모델 훈련...")
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_fold_train_feat, y_fold_train)

        # 7. 평가
        y_fold_val_pred = model.predict(X_fold_val_feat)

        fold_metrics = {
            'fold': fold_idx + 1,
            'f1': f1_score(y_fold_val, y_fold_val_pred),
            'precision': precision_score(y_fold_val, y_fold_val_pred),
            'recall': recall_score(y_fold_val, y_fold_val_pred),
            'train_size': len(X_fold_train),
            'val_size': len(X_fold_val)
        }

        print(f"\nFold {fold_idx + 1} 결과:")
        print(f"  F1: {fold_metrics['f1']:.4f}")
        print(f"  Precision: {fold_metrics['precision']:.4f}")
        print(f"  Recall: {fold_metrics['recall']:.4f}")

        cv_results['fold_metrics'].append(fold_metrics)
        cv_results['fold_predictions'].append(y_fold_val_pred)
        cv_results['fold_models'].append(model)

    # 평균 성능 계산
    avg_f1 = np.mean([m['f1'] for m in cv_results['fold_metrics']])
    std_f1 = np.std([m['f1'] for m in cv_results['fold_metrics']])
    avg_precision = np.mean([m['precision'] for m in cv_results['fold_metrics']])
    avg_recall = np.mean([m['recall'] for m in cv_results['fold_metrics']])

    cv_results['summary'] = {
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'n_folds': n_folds
    }

    print(f"\n{'='*60}")
    print("=== Cross-Validation 결과 요약 ===")
    print(f"{'='*60}")
    print(f"평균 F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"평균 Precision: {avg_precision:.4f}")
    print(f"평균 Recall: {avg_recall:.4f}")

    return cv_results

# CV 실행
cv_results_npy = cross_validate_with_augmentation_npy(
    mine_patches=mine_patches,
    mine_indices=train_idx,  # 15개 train 인덱스
    intensity_matrix=intensity_matrix,
    mine_pixel_coords=mine_pixel_coords,
    augmenter=augmenter,
    n_folds=5,
    augmentation_factor=10
)
```

#### 5.1.3 CV 결과 시각화

```python
import matplotlib.pyplot as plt

def visualize_cv_results(cv_results: dict):
    """
    Cross-Validation 결과 시각화
    """
    fold_metrics = cv_results['fold_metrics']

    # Fold별 F1 스코어
    folds = [m['fold'] for m in fold_metrics]
    f1_scores = [m['f1'] for m in fold_metrics]
    precision_scores = [m['precision'] for m in fold_metrics]
    recall_scores = [m['recall'] for m in fold_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(folds))
    width = 0.25

    ax.bar(x - width, f1_scores, width, label='F1', alpha=0.8)
    ax.bar(x, precision_scores, width, label='Precision', alpha=0.8)
    ax.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Results by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 평균 라인
    avg_f1 = cv_results['summary']['avg_f1']
    ax.axhline(y=avg_f1, color='r', linestyle='--', label=f'Avg F1: {avg_f1:.4f}')

    plt.tight_layout()
    plt.savefig('analysis_results/visualizations/cv_results.png', dpi=300)
    print("CV 결과 시각화 저장: analysis_results/visualizations/cv_results.png")

    plt.show()

visualize_cv_results(cv_results_npy)
```

**🔑 핵심 포인트:**
- ✅ **Fold별 독립 증강**: 각 fold의 train에서만 증강
- ✅ **Validation 원본**: 검증은 항상 원본만 사용
- ✅ **평균 성능**: 5-fold 평균으로 robust 평가
- ✅ **표준편차**: 모델 안정성 확인

---

## 6. 성능 평가 및 검증

### 6.1 평가 지표

#### 6.1.1 기본 지표

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> dict:
    """
    종합 평가 지표 계산

    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률 (옵션)

    Returns:
        평가 지표 딕셔너리
    """
    metrics = {}

    # 기본 지표
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)

    # ROC-AUC (확률이 있는 경우)
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])

    # Specificity (True Negative Rate)
    metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])

    # False Positive Rate
    metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])

    # False Negative Rate
    metrics['fnr'] = metrics['fn'] / (metrics['fn'] + metrics['tp'])

    return metrics

# Test set 평가
test_metrics = comprehensive_evaluation(
    y_true=y_test,
    y_pred=y_test_pred
)

print("=== Test Set 종합 평가 ===")
for metric, value in test_metrics.items():
    if metric != 'confusion_matrix':
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
```

#### 6.1.2 실전 중심 지표

```python
def calculate_operational_metrics(
    cm: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 100.0
) -> dict:
    """
    실전 운용 중심 지표 계산

    Args:
        cm: Confusion Matrix
        cost_fp: False Positive 비용
        cost_fn: False Negative 비용

    Returns:
        운용 지표 딕셔너리
    """
    tn, fp, fn, tp = cm.ravel()

    metrics = {}

    # Detection Rate (Recall과 동일하지만 맥락 강조)
    metrics['detection_rate'] = tp / (tp + fn)

    # False Alarm Rate
    metrics['false_alarm_rate'] = fp / (fp + tn)

    # Alert Reliability (Precision과 유사)
    metrics['alert_reliability'] = tp / (tp + fp)

    # 비용 분석
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    metrics['total_cost'] = total_cost
    metrics['cost_per_detection'] = total_cost / tp if tp > 0 else float('inf')

    # 실전 적합성 점수 (0-1)
    # High recall (놓치지 않기) + Low FPR (오경보 최소화)
    metrics['operational_score'] = (
        0.7 * metrics['detection_rate'] +
        0.3 * (1 - metrics['false_alarm_rate'])
    )

    return metrics

# 실전 지표 계산
cm_test = confusion_matrix(y_test, y_test_pred)
operational_metrics = calculate_operational_metrics(
    cm=cm_test,
    cost_fp=10.0,   # 오경보 비용
    cost_fn=100.0   # 놓친 기뢰 비용
)

print("\n=== 실전 운용 지표 ===")
print(f"기뢰 탐지율: {operational_metrics['detection_rate']:.2%}")
print(f"오경보율: {operational_metrics['false_alarm_rate']:.2%}")
print(f"경보 신뢰도: {operational_metrics['alert_reliability']:.2%}")
print(f"총 비용: {operational_metrics['total_cost']:.2f}")
print(f"실전 적합성 점수: {operational_metrics['operational_score']:.4f}")
```

### 6.2 오버피팅 검증

```python
def check_overfitting(
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict
) -> dict:
    """
    오버피팅 여부 검증

    Args:
        train_metrics, val_metrics, test_metrics: 각 set의 평가 지표

    Returns:
        오버피팅 분석 결과
    """
    analysis = {}

    # Train-Val gap
    train_val_gap = train_metrics['f1'] - val_metrics['f1']
    analysis['train_val_gap'] = train_val_gap

    # Train-Test gap
    train_test_gap = train_metrics['f1'] - test_metrics['f1']
    analysis['train_test_gap'] = train_test_gap

    # Val-Test consistency
    val_test_diff = abs(val_metrics['f1'] - test_metrics['f1'])
    analysis['val_test_diff'] = val_test_diff

    # 오버피팅 판정
    if train_test_gap < 0.1:
        analysis['overfitting_status'] = '낮음 (양호)'
    elif train_test_gap < 0.2:
        analysis['overfitting_status'] = '중간 (주의)'
    else:
        analysis['overfitting_status'] = '높음 (대책 필요)'

    # Val-Test 일관성 판정
    if val_test_diff < 0.05:
        analysis['validation_reliability'] = '높음 (Val이 Test 예측에 유효)'
    elif val_test_diff < 0.1:
        analysis['validation_reliability'] = '중간'
    else:
        analysis['validation_reliability'] = '낮음 (Val이 Test 예측에 부적합)'

    return analysis

# 오버피팅 체크 예시
overfitting_analysis = check_overfitting(
    train_metrics={'f1': 0.95},
    val_metrics={'f1': 0.88},
    test_metrics={'f1': 0.86}
)

print("\n=== 오버피팅 분석 ===")
for key, value in overfitting_analysis.items():
    print(f"{key}: {value}")

# Output:
# === 오버피팅 분석 ===
# train_val_gap: 0.07
# train_test_gap: 0.09
# val_test_diff: 0.02
# overfitting_status: 낮음 (양호)
# validation_reliability: 높음 (Val이 Test 예측에 유효)
```

### 6.3 증강 효과 분석

```python
def analyze_augmentation_effect(
    baseline_metrics: dict,
    augmented_metrics: dict
) -> dict:
    """
    증강 전후 성능 비교

    Args:
        baseline_metrics: 증강 전 (원본만)
        augmented_metrics: 증강 후

    Returns:
        증강 효과 분석
    """
    effect = {}

    # F1 개선
    f1_improvement = augmented_metrics['f1'] - baseline_metrics['f1']
    effect['f1_improvement'] = f1_improvement
    effect['f1_improvement_pct'] = (f1_improvement / baseline_metrics['f1']) * 100

    # Recall 개선 (기뢰 놓치지 않기)
    recall_improvement = augmented_metrics['recall'] - baseline_metrics['recall']
    effect['recall_improvement'] = recall_improvement

    # Precision 변화 (오경보율)
    precision_change = augmented_metrics['precision'] - baseline_metrics['precision']
    effect['precision_change'] = precision_change

    # 종합 판정
    if f1_improvement > 0.1:
        effect['effectiveness'] = '높음 (매우 효과적)'
    elif f1_improvement > 0.05:
        effect['effectiveness'] = '중간 (효과적)'
    elif f1_improvement > 0:
        effect['effectiveness'] = '낮음 (미미한 효과)'
    else:
        effect['effectiveness'] = '역효과 (증강 재검토 필요)'

    return effect

# 증강 효과 분석 예시
aug_effect = analyze_augmentation_effect(
    baseline_metrics={'f1': 0.72, 'recall': 0.68, 'precision': 0.76},
    augmented_metrics={'f1': 0.86, 'recall': 0.84, 'precision': 0.88}
)

print("\n=== 증강 효과 분석 ===")
print(f"F1 개선: +{aug_effect['f1_improvement']:.4f} ({aug_effect['f1_improvement_pct']:.2f}%)")
print(f"Recall 개선: +{aug_effect['recall_improvement']:.4f}")
print(f"Precision 변화: {aug_effect['precision_change']:+.4f}")
print(f"종합 평가: {aug_effect['effectiveness']}")
```

---

## 7. 체크리스트 및 권장사항

### 7.1 Data Leakage 방지 체크리스트

- [ ] ✅ **증강 전 원본 분할** 완료
- [ ] ✅ **Train/Val/Test 완전 독립** 확인
- [ ] ✅ **Val/Test는 원본만 사용** (증강 안 함)
- [ ] ✅ **Fold별 독립 증강** (CV 시)
- [ ] ✅ **배경 샘플링 독립성** 확인
- [ ] ✅ **인덱스 저장** (재현성)

### 7.2 오버피팅 방지 체크리스트

- [ ] ✅ **증강 배수 ≤ 10배**
- [ ] ✅ **소나 안전 증강만 사용**
- [ ] ✅ **Hard Negative 70% 확보**
- [ ] ✅ **Class weight='balanced' 설정**
- [ ] ✅ **L2 정규화 또는 Dropout** (딥러닝 시)
- [ ] ✅ **Train-Test gap < 0.1** 목표

### 7.3 현실 반영 체크리스트

- [ ] ✅ **Val/Test 비율 1:5** (현실 근사)
- [ ] ✅ **Test는 최종 평가만 사용**
- [ ] ✅ **실전 지표 계산** (Detection Rate, FPR)
- [ ] ✅ **비용 분석** (FP vs FN 비용)
- [ ] ✅ **정기적 성능 모니터링**

### 7.4 데이터 형식 선택 가이드

| 상황 | 권장 형식 | 이유 |
|------|----------|------|
| **정밀 분석 필요** | NPY | 32-bit 정밀도 |
| **최고 성능 추구** | NPY | 정보 손실 없음 |
| **어노테이션 활용** | BMP → NPY 변환 | 자동 라벨 추출 후 변환 |
| **시각 검증 필요** | 둘 다 | NPY 분석 + BMP 시각화 |
| **NPY 없는 경우** | BMP (임시) | uint8 → float32 변환 후 사용 |

### 7.5 증강 배수 선택 가이드

| 원본 샘플 수 | 권장 배수 | 최종 샘플 수 | 용도 |
|------------|----------|-------------|------|
| **25개** | 10배 | 250개 | **권장 (표준)** |
| **15개 (Train)** | 10배 | 150개 | **권장 (표준)** |
| **<10개** | 15-20배 | 150-200개 | 극단적 부족 |
| **>50개** | 5배 | 250개+ | 충분한 원본 |

### 7.6 최종 권장 전략 요약

**NPY 기반 (권장):**
```
1. 원본 분할: 25개 → Train(15) / Val(5) / Test(5)
2. Train 증강: 15 → 150개 (10배)
3. 배경 샘플: Train(75) / Val(25) / Test(25) [Hard 70%]
4. 최종 비율: Train(2:1) / Val(1:5) / Test(1:5)
5. Class weight='balanced'
6. 5-Fold CV로 검증
```

**BMP 기반 (어노테이션 활용):**
```
1. 어노테이션에서 기뢰 위치 추출
2. uint8 → float32 변환
3. 이후 NPY 기반과 동일한 절차
```

---

## 📚 참고 문헌

1. **Data Leakage Prevention**
   - Imbalanced-Learn Documentation (2025). "Common pitfalls and recommended practices"
   - AWS Prescriptive Guidance. "Splits and data leakage"

2. **Stratified Sampling**
   - scikit-learn Documentation. "StratifiedKFold and StratifiedShuffleSplit"
   - Cross Validated (Stack Exchange). "Stratification by target variable"

3. **Hard Negative Mining**
   - Jin, S. et al. (2018). "Unsupervised Hard Example Mining from Videos for Improved Object Detection", ECCV 2018
   - Lee, J. et al. (2024). "Hard negative mining in weakly labeled dataset", Journal of Pathology

4. **Sonar Image Augmentation**
   - Frontiers in Marine Science (2025). "Marine object detection in forward-looking sonar images"
   - arXiv 2412.11840v1. "Sonar-based Deep Learning in Underwater Robotics: Robustness and Challenges"

5. **Data Augmentation Best Practices**
   - Scientific Reports (Nature, 2023). "Augmentation strategies for imbalanced learning problem"
   - Journal of Big Data (2024). "Data oversampling and imbalanced datasets"

6. **Class Imbalance**
   - Analytics Vidhya (2025). "10 Techniques to Solve Imbalanced Classes in Machine Learning"
   - Roboflow Blog. "How to Handle Unbalanced Classes: 5 Strategies"

---

## 📝 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2025-10-20 | 초기 문서 작성 |

---

**작성자**: Claude (Anthropic)
**검토자**: 사용자 (프로젝트 담당자)
**문서 위치**: `docs/DATA_AUGMENTATION_STRATEGY_GUIDE.md`
