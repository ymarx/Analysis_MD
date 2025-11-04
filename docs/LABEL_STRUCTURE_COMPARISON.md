# 라벨 구조 비교: 이미지 vs NPY 데이터

## 개요

이미지 기반 라벨링과 NPY 다차원 배열 라벨링의 차이점 및 NPY 라벨링의 특수 요구사항 설명

---

## 1. 기존 이미지 라벨링 방식

### 구조
```
이미지 파일 (.jpg, .png, .bmp)
  ↓
어노테이션 파일 (.xml, .json)
  - 바운딩 박스 좌표
  - 클래스 정보
  ↓
머신러닝 학습 시
  - 이미지 로드
  - 어노테이션 파일 로드
  - 매칭하여 사용
```

### 특징
- ✅ **파일 분리**: 이미지와 라벨이 별도 파일
- ✅ **가시성**: 육안으로 이미지 확인 가능
- ✅ **표준 포맷**: PASCAL VOC, COCO 등 표준 형식
- ✅ **도구 풍부**: LabelImg, CVAT 등 어노테이션 도구
- ❌ **픽셀 접근**: 이미지는 2D/3D 배열 (H, W, C)

### 예시
```
dataset/
├── images/
│   └── image_001.jpg        # 이미지 파일
└── annotations/
    └── image_001.xml         # 어노테이션 파일 (별도)
```

---

## 2. NPY 데이터 라벨링의 차이점

### 구조
```
NPY 파일 (.npy)
  - 다차원 배열: (5137, 6400) float32
  - 메타데이터 없음
  - 시각화 불가 (그대로는 볼 수 없음)
  ↓
라벨 데이터를 어떻게 저장/식별?
  ↓
해결책: 구조화된 NPZ + 메타데이터
```

### 핵심 차이점

| 측면 | 이미지 라벨링 | NPY 라벨링 |
|------|--------------|-----------|
| **원본 데이터** | 이미지 파일 (jpg, png) | 다차원 배열 (.npy) |
| **가시성** | 직접 볼 수 있음 | 변환해야 볼 수 있음 |
| **라벨 저장** | 별도 XML/JSON | NPZ 또는 별도 파일 |
| **데이터 접근** | PIL, OpenCV로 로드 | numpy.load() |
| **메타데이터** | 이미지에 포함 (EXIF) | 별도 저장 필요 |
| **라벨 형식** | 바운딩 박스 좌표 | 좌표 + 마스크 배열 |
| **파일 크기** | 작음 (압축됨) | 큼 (비압축 배열) |

---

## 3. NPY 라벨링의 특수 요구사항

### 문제 1: 데이터와 라벨의 연결

**이미지 방식**:
```python
# 파일명으로 매칭
image = load_image("image_001.jpg")
annotation = load_annotation("image_001.xml")
```

**NPY 방식** (문제):
```python
# NPY 파일에는 메타정보가 없음
intensity_data = np.load("data.npy")  # Shape: (5137, 6400)
# 이 데이터가 어떤 파일에서 왔는지 알 수 없음!
# 라벨 파일과 어떻게 매칭?
```

**해결책**:
```python
# NPZ로 통합 저장
data = np.load("labeled_data.npz")
intensity = data['intensity']     # 원본 데이터
labels = data['labels']           # 라벨 마스크
metadata = json.loads(data['metadata'].item())  # 메타정보
```

### 문제 2: 라벨 식별

**이미지 방식**:
```xml
<!-- image_001.xml -->
<object>
  <name>mine</name>
  <bndbox>
    <xmin>100</xmin>
    <ymin>200</ymin>
    <xmax>150</xmax>
    <ymax>250</ymax>
  </bndbox>
</object>
```

**NPY 방식** (문제):
```python
# 이진 마스크만으로는 정보 부족
labels = np.array([
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
    ...
])
# 이 1들이 어떤 객체? 몇 개의 객체? 어떤 클래스?
```

**해결책 1: 인스턴스 세그멘테이션 스타일**
```python
# 각 객체에 고유 ID
instance_mask = np.array([
    [0, 0, 0, 1, 1, 0],  # 0=배경, 1=mine#1
    [0, 0, 2, 2, 2, 0],  # 2=mine#2
    ...
])

# 메타데이터로 ID 설명
metadata = {
    1: {'class': 'mine', 'bbox': [100, 200, 150, 250]},
    2: {'class': 'mine', 'bbox': [300, 400, 350, 450]},
}
```

**해결책 2: 통합 구조 (채택)**
```python
labeled_data = {
    'intensity': np.ndarray,      # 원본 (5137, 6400)
    'labels': np.ndarray,         # 이진 마스크 (5137, 6400)
    'instance_mask': np.ndarray,  # 인스턴스별 ID (5137, 6400)
    'annotations': [              # 각 객체 정보
        {
            'id': 1,
            'class': 'mine',
            'bbox': {'xmin': 4868, 'ymin': 1070, 'xmax': 5187, 'ymax': 1119},
            'area': 15631,
            'pixels': [[1070, 4868], [1070, 4869], ...],  # 실제 픽셀 위치
        },
        ...
    ],
    'metadata': {
        'source_file': 'Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf',
        'extraction_date': '2025-11-04',
        'shape': [5137, 6400],
        'classes': {0: 'background', 1: 'mine'}
    }
}
```

### 문제 3: 학습 시 데이터 로드

**이미지 방식**:
```python
class ImageDataset:
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        annotation = load_xml(self.annotation_paths[idx])
        return image, annotation
```

**NPY 방식** (필요한 것):
```python
class NPYDataset:
    def __getitem__(self, idx):
        # NPZ에서 통합 로드
        data = np.load(self.npz_paths[idx])

        intensity = data['intensity']          # (5137, 6400)
        labels = data['labels']                # (5137, 6400)
        instance_mask = data['instance_mask']  # (5137, 6400)

        # 메타데이터 파싱
        annotations = json.loads(data['metadata'].item())['annotations']

        return {
            'intensity': intensity,
            'labels': labels,
            'instances': instance_mask,
            'annotations': annotations
        }
```

---

## 4. 제안하는 NPY 라벨 구조

### 파일 구조

```
data/labels/
├── labeled_npy/
│   └── sample_001_labeled.npz              # 통합 데이터
│
├── masks/
│   ├── sample_001_binary_mask.npy          # 이진 마스크 (선택)
│   └── sample_001_instance_mask.npy        # 인스턴스 마스크 (선택)
│
└── metadata/
    ├── sample_001_annotations.json         # JSON 어노테이션
    └── sample_001_annotations.xml          # XML 어노테이션 (호환성)
```

### NPZ 파일 내부 구조

```python
# sample_001_labeled.npz
{
    'intensity': np.ndarray,           # float32, (5137, 6400)
    'binary_mask': np.ndarray,         # uint8, (5137, 6400), 0 or 1
    'instance_mask': np.ndarray,       # uint16, (5137, 6400), 0-255
    'metadata': np.array(json_string)  # JSON string as numpy array
}
```

### JSON 메타데이터 구조

```json
{
  "source": {
    "file": "Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
    "extraction_date": "2025-11-04",
    "shape": [5137, 6400],
    "dtype": "float32"
  },
  "classes": {
    "0": "background",
    "1": "mine"
  },
  "annotations": [
    {
      "id": 1,
      "instance_id": 1,
      "category": "mine",
      "category_id": 1,
      "bbox": {
        "xmin": 4868,
        "ymin": 1070,
        "xmax": 5187,
        "ymax": 1119,
        "width": 319,
        "height": 49
      },
      "area": 15631,
      "segmentation": {
        "size": [5137, 6400],
        "counts": "..."  # RLE 인코딩 (선택)
      }
    }
  ],
  "statistics": {
    "total_pixels": 32876800,
    "mine_pixels": 390530,
    "background_pixels": 32486270,
    "mine_ratio": 0.011879
  }
}
```

---

## 5. 사용 예시

### 데이터 생성

```python
from mine_labeling.labeling import NPYLabeler

# 라벨러 초기화
labeler = NPYLabeler(config)

# NPY 데이터 + 어노테이션 → 라벨 생성
labeler.create_labels(
    npy_path="data/extracted/sample_001.npy",
    annotations_path="data/annotations/sample_001.json",
    output_path="data/labels/labeled_npy/sample_001_labeled.npz"
)
```

### 데이터 로드 (학습 시)

```python
import numpy as np
import json

# NPZ 로드
data = np.load("sample_001_labeled.npz", allow_pickle=True)

# 데이터 추출
intensity = data['intensity']              # (5137, 6400)
binary_mask = data['binary_mask']          # (5137, 6400)
instance_mask = data['instance_mask']      # (5137, 6400)

# 메타데이터 파싱
metadata = json.loads(data['metadata'].item())
annotations = metadata['annotations']

print(f"Mine 개수: {len(annotations)}")
print(f"첫 번째 mine bbox: {annotations[0]['bbox']}")
```

### 특정 객체 추출

```python
# 인스턴스 ID로 특정 기뢰 추출
mine_id = 5
mine_mask = (instance_mask == mine_id)

# 해당 기뢰 영역의 강도 데이터
mine_intensity = intensity[mine_mask]

# 바운딩 박스 크롭
bbox = annotations[mine_id - 1]['bbox']
mine_patch = intensity[
    bbox['ymin']:bbox['ymax'],
    bbox['xmin']:bbox['xmax']
]
```

---

## 6. 기존 이미지 라벨링 도구와의 호환성

### LabelImg XML → NPY 라벨 변환

```python
from mine_labeling.annotation import FormatConverter

converter = FormatConverter()

# XML → JSON
converter.xml_to_json(
    "annotations/sample_001.xml",
    "annotations/sample_001.json"
)

# JSON → NPY 라벨
labeler.create_labels(
    npy_path="extracted/sample_001.npy",
    annotations_path="annotations/sample_001.json",
    output_path="labels/sample_001_labeled.npz"
)
```

### NPY 라벨 → 이미지 + XML (시각화용)

```python
from mine_labeling.validation import VisualValidator

validator = VisualValidator()

# NPY 라벨 → 이미지 + 바운딩 박스
validator.export_to_image(
    labeled_npz="labels/sample_001_labeled.npz",
    output_image="validation/sample_001_annotated.png",
    output_xml="validation/sample_001.xml"  # LabelImg 호환
)
```

---

## 7. 핵심 차이점 요약

### 이미지 라벨링
```
✅ 단순: 이미지 + XML/JSON
✅ 도구: LabelImg, CVAT
✅ 표준: PASCAL VOC, COCO
❌ 메타정보: 이미지 파일과 분리
```

### NPY 라벨링 (본 프로젝트)
```
✅ 통합: NPZ에 데이터+라벨+메타정보
✅ 식별: 인스턴스 마스크 + 상세 어노테이션
✅ 호환: JSON/XML 형식도 지원
⚠️  복잡: 구조화된 데이터 관리 필요
```

---

## 결론

**NPY 데이터 라벨링은 이미지 라벨링과 달리**:

1. **통합 저장 필요**: 데이터와 라벨을 NPZ로 묶어야 함
2. **메타데이터 필수**: JSON으로 상세 정보 저장
3. **인스턴스 식별**: 각 객체에 고유 ID 부여
4. **양방향 호환**: XML/JSON ↔ NPZ 변환 지원

이를 통해 이미지 기반 도구의 편의성을 유지하면서도, NPY 배열 데이터의 효율적인 라벨링이 가능합니다.
