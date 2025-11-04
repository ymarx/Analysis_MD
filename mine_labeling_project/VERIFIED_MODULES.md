# 검증된 모듈 및 스크립트

## 오늘(2025-11-04) 성공적으로 실행된 스크립트

### 1. 데이터 구조 분석
- **파일**: `scripts/mine_labeling/01_analyze_data_structure.py`
- **상태**: ✅ 성공
- **기능**: NPY, BMP, XML 데이터 차원 분석
- **결과**: `analysis_results/npy_labeling/data_structure_analysis.json`

### 2. 좌표 매핑 (원본)
- **파일**: `scripts/mine_labeling/03_coordinate_mapping.py`
- **상태**: ✅ 성공
- **기능**: BMP 좌표 → NPY 좌표 변환 (6.25배 스케일)
- **결과**: `analysis_results/npy_labeling/mapped_annotations.json`

### 3. 좌표 매핑 검증
- **파일**: `scripts/mine_labeling/04_validate_mapping.py`
- **상태**: ✅ 성공 (모든 검증 PASS)
- **기능**:
  - 차원 검증
  - 스케일 팩터 검증
  - 라벨 마스크 검증
  - 강도 값 검증

### 4. Flipped 좌표 매핑
- **파일**: `scripts/mine_labeling/06_flip_bbox_mapping.py`
- **상태**: ✅ 성공
- **기능**: Y축 flip 적용 좌표 매핑
- **결과**: `analysis_results/npy_labeling/flipped/` (현재 사용 중)

### 5. 원본 vs Flipped 비교
- **파일**: `scripts/mine_labeling/07_compare_original_vs_flipped.py`
- **상태**: ✅ 성공
- **기능**: 두 매핑 방식 시각적 비교
- **결론**: Flipped 방식이 정확함

### 6. 요약 보고서 생성
- **파일**: `scripts/mine_labeling/05_generate_summary_report.py`
- **상태**: ✅ 성공
- **결과**: `LABELING_SUMMARY_REPORT.txt`

---

## 기존 검증된 모듈 (프로젝트에서 사용 중)

### 1. XTF Intensity Extractor ⭐
- **위치**: `src/data_processing/xtf_intensity_extractor.py`
- **클래스**: `XTFIntensityExtractor`
- **상태**: ✅ 프로덕션 검증됨
- **주요 메소드**:
  - `extract_intensity_data(xtf_path, output_dir=None, ping_range=None)`
  - `create_visualization_image(intensity_data, output_path)`
  - `load_intensity_images(directory_path)`

**사용 예시**:
```python
from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor

extractor = XTFIntensityExtractor()
result = extractor.extract_intensity_data(
    'path/to/file.xtf',
    output_dir='data/extracted',
    ping_range=None  # 전체 추출
)

# 반환값
{
    'metadata': IntensityMetadata,
    'ping_data': List[IntensityPing],
    'intensity_images': Dict,
    'navigation_data': Dict
}
```

### 2. XTF Reader
- **위치**: `src/data_processing/xtf_reader.py`
- **상태**: ✅ 검증됨
- **기능**: XTF 파일 파싱

### 3. Coordinate Mapper
- **위치**: `src/data_processing/coordinate_mapper.py`
- **상태**: ✅ 검증됨
- **기능**: 좌표 변환 유틸리티

---

## 재사용 가능한 검증된 코드 패턴

### 1. NPY 데이터 로드 및 확인
```python
import numpy as np

# 데이터 로드
npy_data = np.load('path/to/file.npy')

# 차원 및 타입 확인
print(f'Shape: {npy_data.shape}')
print(f'Dtype: {npy_data.dtype}')
print(f'Range: [{npy_data.min():.4f}, {npy_data.max():.4f}]')
```

### 2. XML 어노테이션 파싱
```python
import xml.etree.ElementTree as ET

tree = ET.parse('path/to/annotation.xml')
root = tree.getroot()

# 이미지 크기
size = root.find('size')
width = int(size.find('width').text)
height = int(size.find('height').text)

# 바운딩 박스
for obj in root.findall('object'):
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
```

### 3. 좌표 변환 (검증됨)
```python
# BMP (1024 × 5137) → NPY (6400 × 5137)
# Y축 flip 적용

def transform_bbox_flipped(bbox_bmp, bmp_height=5137):
    """검증된 좌표 변환 공식"""
    scale_x = 6.25

    # X축: 스케일만
    xmin_npy = int(bbox_bmp['xmin'] * scale_x)
    xmax_npy = int(bbox_bmp['xmax'] * scale_x)

    # Y축: Flip 적용
    ymin_npy = (bmp_height - 1) - bbox_bmp['ymax']
    ymax_npy = (bmp_height - 1) - bbox_bmp['ymin']

    return {
        'xmin': xmin_npy,
        'ymin': ymin_npy,
        'xmax': xmax_npy,
        'ymax': ymax_npy
    }
```

### 4. 라벨 마스크 생성 (검증됨)
```python
def create_label_mask(npy_shape, annotations):
    """검증된 라벨 마스크 생성"""
    label_mask = np.zeros(npy_shape, dtype=np.uint8)

    for ann in annotations:
        bbox = ann['mapped_npy']
        ymin = max(0, min(bbox['ymin'], bbox['ymax']))
        ymax = min(npy_shape[0], max(bbox['ymin'], bbox['ymax']))
        xmin = max(0, bbox['xmin'])
        xmax = min(npy_shape[1], bbox['xmax'])

        label_mask[ymin:ymax, xmin:xmax] = 1

    return label_mask
```

### 5. NPZ 통합 데이터 저장 (검증됨)
```python
import json

def save_labeled_data(intensity, labels, annotations, output_path):
    """검증된 NPZ 저장 형식"""
    np.savez(
        output_path,
        intensity=intensity,
        labels=labels,
        metadata=json.dumps(annotations)
    )
```

---

## 프로젝트에 통합할 모듈

### 우선순위 1: 필수 모듈

1. **XTF Extractor** ✅
   - 소스: `src/data_processing/xtf_intensity_extractor.py`
   - 대상: `mine_labeling_project/src/mine_labeling/extractors/xtf_extractor.py`
   - 수정: 없음 (그대로 사용)

2. **Coordinate Mapper** ✅
   - 소스: `scripts/mine_labeling/06_flip_bbox_mapping.py` (FlippedCoordinateMapper)
   - 대상: `mine_labeling_project/src/mine_labeling/labeling/coordinate_mapper.py`
   - 수정: 클래스 추출 및 리팩토링

3. **NPY Labeler** ✅
   - 소스: `scripts/mine_labeling/06_flip_bbox_mapping.py` (create_label_mask)
   - 대상: `mine_labeling_project/src/mine_labeling/labeling/npy_labeler.py`
   - 수정: 함수 → 클래스 변환

### 우선순위 2: 유틸리티

4. **Data Validator**
   - 소스: `scripts/mine_labeling/04_validate_mapping.py`
   - 대상: `mine_labeling_project/src/mine_labeling/validation/validators.py`

5. **Visualization Tools**
   - 소스: 여러 스크립트의 시각화 함수
   - 대상: `mine_labeling_project/src/mine_labeling/validation/visual_validator.py`

---

## 재사용 계획

### Phase 1: 핵심 모듈 복사
```bash
# XTF Extractor
cp src/data_processing/xtf_intensity_extractor.py \
   mine_labeling_project/src/mine_labeling/extractors/

# Coordinate Mapper (리팩토링 필요)
# 수동 작업
```

### Phase 2: 검증된 패턴 적용
- 좌표 변환 공식
- 라벨 마스크 생성
- NPZ 저장 형식

### Phase 3: 새로운 기능 추가
- 샘플링
- 증강
- Interactive annotation tool

---

## 주의사항

### ⚠️ 방향 보존 이슈
- 기존 `xtf_intensity_extractor.py`의 이미지 생성 함수 확인 필요
- `create_visualization_image()` 메소드가 Y축 flip을 하는지 확인
- 필요시 `origin='upper'` 파라미터 추가

### ⚠️ 좌표 변환
- Flipped 방식(Y축 반전)이 검증된 방식
- 새로운 이미지 생성 시 방향 보존 확인 필수
- 설정 파일에서 flip 여부를 옵션으로 제공

---

## 다음 작업

1. ✅ 기존 `xtf_intensity_extractor.py` 복사
2. ⏳ 이미지 생성 함수의 방향 보존 여부 확인
3. ⏳ Coordinate Mapper 클래스 추출
4. ⏳ NPY Labeler 클래스 작성
5. ⏳ 통합 테스트
