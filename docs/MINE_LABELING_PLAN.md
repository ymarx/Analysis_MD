# 모의기뢰 레이블링 작업 계획서

**작성일**: 2025-10-30
**버전**: 2.0 (수정본)
**목적**: XTF 추출 .npy 데이터에 대한 정확한 기뢰 레이블링 및 검증

---

## 📑 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 현황](#2-데이터-현황)
3. [문제 정의 및 접근 방법](#3-문제-정의-및-접근-방법)
4. [레이블링 전략](#4-레이블링-전략)
5. [구현 세부사항](#5-구현-세부사항)
6. [검증 방법](#6-검증-방법)
7. [예상 출력](#7-예상-출력)
8. [작업 체크리스트](#8-작업-체크리스트)

---

## 1. 프로젝트 개요

### 1.1 목적

포항 해역에 매설된 25개 모의기뢰에 대해:
- XTF 파일에서 추출한 `.npy` 다차원 배열 데이터에 정확한 레이블 생성
- GPS 투하 좌표와 실제 소나 탐지 위치 간의 오프셋 보정
- Annotation 이미지와 일치하는 레이블링 결과 확보
- Feature 추출 및 분류 모델 훈련을 위한 ground truth 제공

### 1.2 작업 범위

- **입력 데이터**:
  - GPS 좌표 (25개 기뢰)
  - XTF 추출 .npy 파일 (intensity 데이터)
  - Annotation 이미지 (검증 기준)

- **출력 데이터**:
  - Binary mask (.npy)
  - Bounding boxes (JSON)
  - Coordinate mapping (CSV)
  - Validation report (이미지 + CSV)

---

## 2. 데이터 현황

### 2.1 GPS 좌표 데이터

**파일**: `모의기물_투하좌표_포항.xlsx`

**형식**: DDMM 분리 형식 (도/분 별도 컬럼)

| 정점 | 위도(도) | 위도(분) | 경도(도) | 경도(분) |
|------|----------|----------|----------|----------|
| PH_01 | 36 | 5.9374 | 129 | 30.5590 |
| PH_02 | 36 | 5.9355 | 129 | 30.5699 |
| ... | ... | ... | ... | ... |
| PH_25 | 36 | 5.9318 | 129 | 30.8461 |

**십진도 변환 공식**:
```
위도 = 36 + 5.9374/60 = 36.098957°N
경도 = 129 + 30.5590/60 = 129.509317°E
```

**좌표 범위**:
- 위도: 36.098863° ~ 36.099003°
- 경도: 129.509317° ~ 129.514102°

### 2.2 XTF 추출 .npy 파일

**파일 목록**:
```
data/processed/xtf_extracted/
├── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_combined_intensity.npy
├── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_port_intensity.npy
├── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_starboard_intensity.npy
├── Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_combined_intensity.npy
├── Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_port_intensity.npy
└── Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_starboard_intensity.npy
```

**데이터 구조**:
- Klein3900: `(200, 6400)` shape
- Edgetech4205: `(200, 6832)` shape
- Dtype: `float32`
- 값 범위: `[0.0, 1.0]` (정규화된 intensity)

### 2.3 Annotation 이미지

**파일**: `datasets/PH_annotation.png`

- 25개 기뢰에 red bounding box 표시
- Ground truth로 활용 (실제 소나 탐지 위치)
- 레이블링 검증 기준

### 2.4 기존 시스템

**검증 완료 모듈**:
- `src/data_processing/coordinate_mapper.py`
  - `CoordinateTransformer`: WGS84 ↔ UTM 변환
  - `CoordinateMapper`: GPS ↔ 픽셀 변환
  - `TargetLocationLoader`: GPS 좌표 로드
  - `create_target_mask()`: Binary mask 생성
  - `get_target_bounding_boxes()`: Bbox 생성

**검증 문서**:
- `POHANG_COORDINATE_MATCHING_REPORT.md`
- `STEP2_GPS_MAPPING_VALIDATION_REPORT.md`

---

## 3. 문제 정의 및 접근 방법

### 3.1 문제 정의

**핵심 문제**: GPS 투하 좌표 ≠ 실제 소나 탐지 위치

**원인 분석**:
1. **물리적 요인**: 기뢰 투하 후 조류, 바람에 의한 위치 이동
2. **측정 오차**: GPS 신호 오차, 선박 위치 오차
3. **시간 지연**: 투하 시점과 침강 완료 시점의 시간차

**관찰된 패턴**:
- POHANG_COORDINATE_MATCHING_REPORT에서 "북동쪽으로 어긋남" 확인
- 평균 거리 130m, 최소 30m 오차

### 3.2 접근 방법

**2단계 보정 전략**:

#### Step 1: GPS 좌표 보정 (위경도 레벨)
```
원본 GPS (투하 위치)
     ↓
오프셋 보정 (+Δlat, +Δlon)
     ↓
보정 GPS (실제 탐지 위치)
```

#### Step 2: XTF 그리드 매핑 (픽셀 레벨)
```
보정 GPS
     ↓
XTF 소나 그리드 (각 픽셀의 실제 GPS)
     ↓
geo_to_pixel() 변환
     ↓
.npy 픽셀 좌표
```

**보정 기준**: Annotation 이미지 (Ground Truth)

**최적화 방법**: 그리드 서치로 최소 오차 오프셋 계산

---

## 4. 레이블링 전략

### 4.1 전체 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: GPS 좌표 로드 및 변환                               │
│ - 모의기물_투하좌표_포항.xlsx 로드                           │
│ - DDMM → 십진도 변환                                         │
│ - WGS84 → UTM 변환                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: XTF 메타데이터 및 소나 그리드 생성                  │
│ - XTF ping 좌표 추출                                         │
│ - CoordinateMapper 초기화                                    │
│ - (200 × 6400) 소나 그리드 생성                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Annotation 기반 오프셋 보정                         │
│ - Annotation 이미지에서 bbox 중심 추출 (25개)                │
│ - 그리드 서치로 최적 오프셋 계산                              │
│ - 보정 전후 오차 비교                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: 레이블 생성                                         │
│ - 보정된 GPS로 Binary mask 생성                              │
│ - Bounding boxes 생성 (30×30 픽셀)                           │
│ - Coordinate mapping 테이블 생성                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: 시각화 및 검증                                      │
│ - Overlay 이미지 생성 (원본 + mask + bbox)                   │
│ - 개별 기뢰 확대 이미지 (25개)                               │
│ - 수치 검증 리포트 (오차 분석)                               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 오프셋 보정 알고리즘

**목표**: Annotation bbox 중심과 GPS→픽셀 변환 결과의 오차 최소화

**방법**: 2D 그리드 서치

```python
# 탐색 범위
offset_lat_range = [0.0, 0.0005]  # 0~50m 북쪽
offset_lon_range = [0.0, 0.0020]  # 0~150m 동쪽
grid_resolution = 20  # 20×20 그리드

# 각 오프셋 조합에 대해
for each (Δlat, Δlon) combination:
    total_error = 0
    for each mine in 25 mines:
        corrected_gps = original_gps + (Δlat, Δlon)
        predicted_pixel = geo_to_pixel(corrected_gps)
        annotation_pixel = ground_truth[mine_id]
        error = euclidean_distance(predicted_pixel, annotation_pixel)
        total_error += error

    if total_error < min_error:
        best_offset = (Δlat, Δlon)
        min_error = total_error

# 결과: 평균 오차가 최소인 오프셋
```

---

## 5. 구현 세부사항

### 5.1 GPS 좌표 로더

**파일**: `scripts/load_mine_coordinates.py`

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_pohang_mine_coordinates(excel_path):
    """
    모의기물_투하좌표_포항.xlsx 로드

    Returns:
        pd.DataFrame: [target_id, latitude, longitude]
    """
    df = pd.read_excel(excel_path, skiprows=2, header=None)
    df.columns = ['col0', 'target_id', 'lat_deg', 'lat_min',
                  'lon_deg', 'lon_min', 'status']

    # PH_로 시작하는 행만 필터링
    df = df[df['target_id'].str.startswith('PH_', na=False)]

    # DDMM → 십진도 변환
    df['latitude'] = df['lat_deg'] + df['lat_min'] / 60.0
    df['longitude'] = df['lon_deg'] + df['lon_min'] / 60.0

    return df[['target_id', 'latitude', 'longitude']]
```

### 5.2 Annotation 파서

**파일**: `scripts/parse_annotation_image.py`

```python
import cv2
import numpy as np
from typing import List, Tuple

def extract_bbox_centers_from_annotation(
    image_path: str
) -> List[Tuple[int, int]]:
    """
    PH_annotation.png에서 red bounding box 중심 좌표 추출

    Returns:
        List[Tuple[int, int]]: [(x1, y1), ..., (x25, y25)]
    """
    img = cv2.imread(image_path)

    # Red color mask (BGR format)
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])
    red_mask = cv2.inRange(img, lower_red, upper_red)

    # Contour detection
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Bounding box 추출 및 중심 계산
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))

    # Y 좌표 기준으로 정렬 (위→아래 순서)
    centers.sort(key=lambda c: c[1])

    return centers
```

### 5.3 오프셋 보정기

**파일**: `scripts/coordinate_corrector.py`

```python
import numpy as np
from typing import List, Tuple
from src.data_processing.coordinate_mapper import CoordinateMapper

class CoordinateCorrector:
    """GPS 좌표와 Annotation 간 오프셋 자동 보정"""

    def __init__(self, annotation_centers: List[Tuple[int, int]]):
        self.annotation_centers = annotation_centers
        self.offset_lat = 0.0
        self.offset_lon = 0.0

    def optimize_offset(
        self,
        gps_coords: List[Tuple[float, float]],
        mapper: CoordinateMapper,
        lat_range: Tuple[float, float] = (0.0, 0.0005),
        lon_range: Tuple[float, float] = (0.0, 0.0020),
        grid_size: int = 20
    ) -> Tuple[float, float, float]:
        """
        그리드 서치로 최적 오프셋 계산

        Returns:
            (offset_lat, offset_lon, avg_error)
        """
        lat_values = np.linspace(lat_range[0], lat_range[1], grid_size)
        lon_values = np.linspace(lon_range[0], lon_range[1], grid_size)

        best_offset_lat = 0.0
        best_offset_lon = 0.0
        min_total_error = float('inf')

        for offset_lat in lat_values:
            for offset_lon in lon_values:
                total_error = 0.0

                for i, (lat, lon) in enumerate(gps_coords):
                    # 보정 적용
                    corrected_lat = lat + offset_lat
                    corrected_lon = lon + offset_lon

                    # GPS → 픽셀 변환
                    ping_idx, sample_idx = mapper.geo_to_pixel(
                        corrected_lon, corrected_lat
                    )

                    # Annotation 좌표
                    anno_x, anno_y = self.annotation_centers[i]

                    # 오차 계산
                    error = np.sqrt(
                        (sample_idx - anno_x)**2 +
                        (ping_idx - anno_y)**2
                    )
                    total_error += error

                # 최소 오차 업데이트
                if total_error < min_total_error:
                    min_total_error = total_error
                    best_offset_lat = offset_lat
                    best_offset_lon = offset_lon

        avg_error = min_total_error / len(gps_coords)

        self.offset_lat = best_offset_lat
        self.offset_lon = best_offset_lon

        return best_offset_lat, best_offset_lon, avg_error

    def apply_correction(
        self,
        latitude: float,
        longitude: float
    ) -> Tuple[float, float]:
        """보정 오프셋 적용"""
        return latitude + self.offset_lat, longitude + self.offset_lon
```

### 5.4 레이블 생성기

**파일**: `scripts/generate_mine_labels.py`

```python
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from src.data_processing.coordinate_mapper import (
    CoordinateMapper, TargetLocation
)

def generate_mine_labels(
    npy_path: str,
    corrected_targets: List[TargetLocation],
    mapper: CoordinateMapper,
    output_dir: str,
    mask_radius: int = 15,
    bbox_size: int = 30
) -> Dict:
    """
    보정된 좌표로 레이블 생성

    Returns:
        Dict: 생성된 파일 경로 및 통계
    """
    # .npy 로드
    intensity = np.load(npy_path)

    # Binary mask 생성
    mask = mapper.create_target_mask(
        target_locations=corrected_targets,
        mask_radius=mask_radius
    )

    # Bounding boxes 생성
    bboxes = mapper.get_target_bounding_boxes(
        target_locations=corrected_targets,
        box_size=bbox_size
    )

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 저장
    base_name = Path(npy_path).stem.replace('_intensity', '')

    # 1. Binary mask
    mask_path = output_path / f'{base_name}_mask.npy'
    np.save(mask_path, mask)

    # 2. Bounding boxes
    bbox_path = output_path / f'{base_name}_bboxes.json'
    with open(bbox_path, 'w') as f:
        json.dump(bboxes, f, indent=2)

    # 3. Coordinate mapping
    mapping_path = output_path / f'{base_name}_coordinate_mapping.csv'
    mapper.export_coordinate_mapping(mapping_path)

    # 통계
    stats = {
        'total_pixels': mask.size,
        'mine_pixels': int(np.sum(mask)),
        'mine_ratio': float(np.sum(mask) / mask.size),
        'num_bboxes': len(bboxes),
        'mask_path': str(mask_path),
        'bbox_path': str(bbox_path),
        'mapping_path': str(mapping_path)
    }

    return stats
```

### 5.5 시각화 도구

**파일**: `scripts/visualize_labeling_results.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict

def visualize_labeling_overlay(
    intensity: np.ndarray,
    mask: np.ndarray,
    bboxes: List[Dict],
    output_path: str
):
    """전체 Overlay 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 1. 원본 intensity
    axes[0].imshow(intensity, cmap='gray', aspect='auto')
    axes[0].set_title('Original Intensity', fontsize=14)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Ping Index')

    # 2. Binary mask 오버레이
    axes[1].imshow(intensity, cmap='gray', aspect='auto')
    axes[1].imshow(mask, cmap='Reds', alpha=0.5, aspect='auto')
    axes[1].set_title('With Binary Mask', fontsize=14)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Ping Index')

    # 3. Bounding boxes
    axes[2].imshow(intensity, cmap='gray', aspect='auto')
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['width'], bbox['height'],
            linewidth=1.5, edgecolor='red', facecolor='none'
        )
        axes[2].add_patch(rect)
        axes[2].text(
            bbox['center_x'], bbox['center_y'],
            bbox['target_id'],
            color='yellow', fontsize=6, ha='center'
        )
    axes[2].set_title('With Bounding Boxes', fontsize=14)
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Ping Index')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_individual_mines(
    intensity: np.ndarray,
    mask: np.ndarray,
    bboxes: List[Dict],
    output_path: str
):
    """개별 기뢰 확대 시각화"""
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for i, bbox in enumerate(bboxes):
        ax = axes[i // 5, i % 5]

        # ROI 추출
        y1, y2 = max(0, bbox['y1']), min(intensity.shape[0], bbox['y2'])
        x1, x2 = max(0, bbox['x1']), min(intensity.shape[1], bbox['x2'])

        roi = intensity[y1:y2, x1:x2]
        mask_roi = mask[y1:y2, x1:x2]

        # 시각화
        ax.imshow(roi, cmap='gray')
        ax.imshow(mask_roi, cmap='Reds', alpha=0.6)

        ax.set_title(
            f"{bbox['target_id']}\n"
            f"Pixel: ({bbox['center_x']}, {bbox['center_y']})\n"
            f"GPS: ({bbox['latitude']:.6f}, {bbox['longitude']:.6f})",
            fontsize=8
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 6. 검증 방법

### 6.1 수치 검증

**검증 메트릭**:
1. **평균 픽셀 오차**: GPS→픽셀 vs Annotation 중심
2. **최대 픽셀 오차**: 최악의 경우 확인
3. **표준편차**: 일관성 평가
4. **Coverage**: 25개 모두 레이블링 확인

**검증 스크립트**: `scripts/validate_labeling_results.py`

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def validate_labeling_results(
    bboxes: List[Dict],
    annotation_centers: List[Tuple[int, int]],
    gps_coords: List[Tuple[float, float]]
) -> pd.DataFrame:
    """레이블링 결과 검증"""

    validation_data = []

    for i, bbox in enumerate(bboxes):
        # GPS 좌표 (보정 후)
        gps_lat, gps_lon = gps_coords[i]

        # 예측 픽셀 좌표
        pred_x = bbox['center_x']
        pred_y = bbox['center_y']

        # Annotation 픽셀 좌표
        anno_x, anno_y = annotation_centers[i]

        # 오차 계산
        error_pixels = np.sqrt((pred_x - anno_x)**2 + (pred_y - anno_y)**2)
        error_meters = error_pixels * 1.0  # 1픽셀 ≈ 1m 가정

        validation_data.append({
            'target_id': bbox['target_id'],
            'gps_lat': gps_lat,
            'gps_lon': gps_lon,
            'predicted_x': pred_x,
            'predicted_y': pred_y,
            'annotation_x': anno_x,
            'annotation_y': anno_y,
            'error_pixels': error_pixels,
            'error_meters': error_meters,
            'bbox_area': bbox['width'] * bbox['height']
        })

    df = pd.DataFrame(validation_data)

    # 통계 출력
    print("\n" + "="*60)
    print("레이블링 검증 결과")
    print("="*60)
    print(f"총 기뢰 수: {len(df)}")
    print(f"평균 오차: {df['error_pixels'].mean():.2f} 픽셀 "
          f"({df['error_meters'].mean():.2f} m)")
    print(f"최소 오차: {df['error_pixels'].min():.2f} 픽셀")
    print(f"최대 오차: {df['error_pixels'].max():.2f} 픽셀")
    print(f"표준편차: {df['error_pixels'].std():.2f} 픽셀")
    print("="*60)

    return df
```

### 6.2 시각적 검증

**검증 이미지**:
1. **Overlay 비교**: 원본 + Mask + Bbox (한 눈에 확인)
2. **개별 확대**: 25개 기뢰 각각의 ROI
3. **Annotation 겹침**: PH_annotation.png와 side-by-side 비교

**검증 기준**:
- ✅ Bbox가 실제 기뢰 위치와 일치
- ✅ Mask 영역이 기뢰 형상을 포함
- ✅ 배경 영역에 False Positive 없음

---

## 7. 예상 출력

### 7.1 디렉토리 구조

```
data/processed/labels/
├── Klein3900_20241011171100/
│   ├── Pohang_Eardo_1_Klein3900_combined_mask.npy
│   ├── Pohang_Eardo_1_Klein3900_combined_bboxes.json
│   ├── Pohang_Eardo_1_Klein3900_combined_coordinate_mapping.csv
│   ├── Pohang_Eardo_1_Klein3900_port_mask.npy
│   ├── Pohang_Eardo_1_Klein3900_port_bboxes.json
│   ├── Pohang_Eardo_1_Klein3900_starboard_mask.npy
│   ├── Pohang_Eardo_1_Klein3900_starboard_bboxes.json
│   ├── offset_correction.json
│   └── validation_report.csv
├── Edgetech4205_20241012110900/
│   └── (동일 구조)
└── visualizations/
    ├── Klein3900_overlay.png
    ├── Klein3900_individual_mines.png
    ├── Edgetech4205_overlay.png
    └── Edgetech4205_individual_mines.png
```

### 7.2 파일 형식 예시

**offset_correction.json**:
```json
{
  "offset_latitude": 0.0002,
  "offset_longitude": 0.0015,
  "offset_lat_meters": 22.2,
  "offset_lon_meters": 132.0,
  "avg_error_pixels": 18.5,
  "max_error_pixels": 42.3,
  "calibration_date": "2025-10-30",
  "calibration_method": "grid_search",
  "grid_size": 20,
  "lat_range": [0.0, 0.0005],
  "lon_range": [0.0, 0.0020]
}
```

**validation_report.csv** (샘플):
```csv
target_id,gps_lat,gps_lon,predicted_x,predicted_y,annotation_x,annotation_y,error_pixels,error_meters,bbox_area
PH_01,36.099157,129.510817,3200,50,3195,48,5.39,5.39,900
PH_02,36.099125,129.510998,3250,52,3248,51,2.24,2.24,900
...
```

---

## 8. 작업 체크리스트

### Phase 1: 데이터 준비
- [x] 올바른 GPS 파일 확인 (`모의기물_투하좌표_포항.xlsx`)
- [x] .npy 파일 구조 확인
- [x] Annotation 이미지 확인
- [x] coordinate_mapper 모듈 검토

### Phase 2: 스크립트 구현
- [ ] `load_mine_coordinates.py` 작성
- [ ] `parse_annotation_image.py` 작성
- [ ] `coordinate_corrector.py` 작성
- [ ] `generate_mine_labels.py` 작성
- [ ] `visualize_labeling_results.py` 작성
- [ ] `validate_labeling_results.py` 작성

### Phase 3: 레이블링 실행
- [ ] GPS 좌표 로드 (25개)
- [ ] Annotation bbox 중심 추출 (25개)
- [ ] 오프셋 보정 실행 (그리드 서치)
- [ ] Klein3900 레이블 생성
- [ ] Edgetech4205 레이블 생성

### Phase 4: 검증 및 시각화
- [ ] 수치 검증 리포트 생성
- [ ] Overlay 이미지 생성
- [ ] 개별 기뢰 확대 이미지 생성
- [ ] Annotation과 육안 비교

### Phase 5: 문서화 및 전달
- [ ] 최종 검증 보고서 작성
- [ ] 레이블 데이터 패키징
- [ ] 다음 단계 (데이터 증강) 연동 확인

---

## 9. 성공 기준

### 9.1 정량적 기준
- ✅ **커버리지**: 25/25 기뢰 레이블링 완료
- ✅ **평균 오차**: < 30 픽셀 (약 30m)
- ✅ **최대 오차**: < 50 픽셀 (약 50m)
- ✅ **일관성**: 표준편차 < 15 픽셀

### 9.2 정성적 기준
- ✅ **시각적 일치**: PH_annotation.png와 육안 확인 시 일치
- ✅ **형상 포함**: Mask가 기뢰 형상을 정확히 포함
- ✅ **False Positive 없음**: 배경 영역에 잘못된 레이블 없음

---

## 10. 다음 단계 연동

레이블링 완료 후:

1. **3단계: 데이터 증강**
   - 레이블된 25개 → 275개 증강
   - Mask/Bbox 동시 증강

2. **4단계: 특징 추출**
   - ROI 기반 패치 추출
   - 14차원 특징 벡터 계산

3. **5단계: 분류 모델 훈련**
   - Labeled feature vectors로 학습
   - 기뢰/배경 분류기 구축

---

**작성자**: 사이드스캔 소나 분석팀
**승인 요청**: 본 계획서를 검토 후 승인 부탁드립니다.
