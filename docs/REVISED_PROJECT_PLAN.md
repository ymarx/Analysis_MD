# 기뢰 라벨링 독립 프로젝트 - 수정된 계획

## 핵심 원칙 수정

### ⚠️ 중요: Y축 Flip 처리 방침

**이전 이해 (수정됨)**:
- ~~모든 NPY → 이미지 변환 시 Y축 flip 필요~~

**올바른 이해**:
- **NPY → 이미지 변환**: Flip 하지 않음 (방향 보존!)
- **Flip이 필요한 이유**: 기존 제공받은 BMP가 이미 flip된 상태였음
- **새로운 처리기**: 방향 보존을 잘 하면 flip 불필요

### 처리 방식

1. **새로운 이미지 생성 시**:
   - NPY 데이터를 그대로 이미지로 변환 (flip 없음)
   - `origin='upper'` 사용하여 top-down 렌더링
   - 원본 XTF의 방향 그대로 유지

2. **기존 flipped 데이터 활용**:
   - `flipped/` 폴더의 검증된 좌표는 그대로 사용
   - 픽셀 좌표값 그대로 적용
   - JSON/XML 파일 생성 시 이 좌표 사용

3. **조건부 Flip 처리**:
   - 사용자가 이미 flip된 이미지로 어노테이션한 경우에만 적용
   - 설정 파일에서 `apply_flip: true/false` 옵션 제공

---

## 프로젝트 구조

```
mine_labeling_project/
├── README.md                          # 프로젝트 개요
├── INSTALL.md                         # 설치 가이드
├── USAGE.md                           # 사용 방법
├── requirements.txt                   # Python 의존성
├── setup.py                           # 패키지 설치 스크립트
│
├── config/
│   ├── default_config.yaml           # 기본 설정
│   └── example_config.yaml           # 예제 설정
│
├── data/
│   ├── raw/                          # 원본 XTF
│   ├── extracted/                    # 추출된 NPY
│   ├── images/                       # 생성된 이미지
│   ├── annotations/                  # 어노테이션
│   ├── labels/                       # 라벨 데이터
│   │   ├── masks/                   # 이진 마스크
│   │   ├── labeled_npy/             # 라벨 정보 포함 NPY
│   │   └── metadata/                # 라벨 메타데이터
│   └── samples/                      # 샘플 및 증강
│       ├── mine/
│       ├── background/
│       └── augmented/
│
├── outputs/                          # 타임스탬프별 출력
│   └── YYYYMMDD_HHMMSS/
│       ├── extracted/
│       ├── images/
│       ├── annotations/
│       ├── labels/
│       ├── samples/
│       └── validation/
│
├── verified_data/                    # 검증된 기존 데이터 보존
│   └── flipped_20251104/
│       ├── flipped_mine_label_mask.npy
│       ├── flipped_labeled_intensity_data.npz
│       └── flipped_mapped_annotations.json
│
├── src/
│   └── mine_labeling/
│       ├── __init__.py
│       ├── extractors/
│       │   ├── __init__.py
│       │   ├── xtf_extractor.py     # Full XTF 추출 (방향 보존)
│       │   └── metadata_handler.py
│       │
│       ├── converters/
│       │   ├── __init__.py
│       │   ├── npy_to_image.py      # NPY → 이미지 (방향 보존!)
│       │   └── image_enhancer.py
│       │
│       ├── annotation/
│       │   ├── __init__.py
│       │   ├── interactive_tool.py
│       │   └── format_converter.py  # JSON ↔ XML 변환
│       │
│       ├── labeling/
│       │   ├── __init__.py
│       │   ├── npy_labeler.py       # NPY 라벨링 (식별 가능)
│       │   ├── label_structure.py   # 라벨 구조 정의
│       │   └── coordinate_mapper.py # 좌표 매핑 (조건부 flip)
│       │
│       ├── sampling/
│       │   ├── __init__.py
│       │   ├── mine_sampler.py
│       │   └── background_sampler.py
│       │
│       ├── augmentation/
│       │   ├── __init__.py
│       │   └── techniques/
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   └── validators.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logger.py
│           └── file_manager.py
│
├── scripts/
│   ├── 01_extract_xtf.py
│   ├── 02_generate_images.py
│   ├── 03_annotate.py
│   ├── 04_create_labels.py
│   ├── 05_sample_and_augment.py
│   ├── migrate_verified_data.py      # 기존 데이터 마이그레이션
│   └── run_pipeline.py
│
├── tests/
│   ├── test_extractors.py
│   ├── test_converters.py
│   └── test_labeling.py
│
└── docs/
    ├── API.md
    ├── COORDINATE_SYSTEM.md
    └── TROUBLESHOOTING.md
```

---

## NPY 라벨 구조 설계

### 식별 가능한 라벨 구조

#### 방법 1: 별도 라벨 파일 (채택)

```python
# 구조
labeled_data = {
    'intensity': np.ndarray,        # 원본 강도 데이터 (5137, 6400)
    'labels': np.ndarray,           # 라벨 마스크 (5137, 6400)
    'label_info': {
        'classes': {
            0: 'background',
            1: 'mine'
        },
        'annotations': [
            {
                'id': 1,
                'class': 'mine',
                'bbox': {'xmin': 4868, 'ymin': 1070, 'xmax': 5187, 'ymax': 1119},
                'pixels': [...],  # 라벨 영역의 픽셀 좌표 리스트
                'metadata': {...}
            },
            ...
        ]
    }
}

# 저장
np.savez('labeled_data.npz', **labeled_data)
```

#### 방법 2: 인스턴스 세그멘테이션 스타일

```python
# 각 기뢰에 고유 ID 부여
instance_mask = np.zeros((5137, 6400), dtype=np.uint16)
# 0: background
# 1-25: mine instance IDs

# 메타데이터 별도 저장
metadata = {
    'instances': {
        1: {'class': 'mine', 'bbox': {...}, 'area': 15631},
        2: {'class': 'mine', 'bbox': {...}, 'area': 15631},
        ...
    }
}
```

### JSON 어노테이션 형식

```json
{
  "image_info": {
    "filename": "Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04",
    "width": 6400,
    "height": 5137,
    "source": "XTF"
  },
  "annotations": [
    {
      "id": 1,
      "category": "mine",
      "bbox": {
        "xmin": 4868,
        "ymin": 1070,
        "xmax": 5187,
        "ymax": 1119
      },
      "area": 15631,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "mine", "supercategory": "object"}
  ]
}
```

### XML 어노테이션 형식 (PASCAL VOC)

```xml
<annotation>
  <folder>extracted</folder>
  <filename>Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.npy</filename>
  <size>
    <width>6400</width>
    <height>5137</height>
    <depth>1</depth>
  </size>
  <object>
    <name>mine</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>4868</xmin>
      <ymin>1070</ymin>
      <xmax>5187</xmax>
      <ymax>1119</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## 설정 파일 구조

### `config/default_config.yaml`

```yaml
project:
  name: "mine_labeling"
  version: "1.0.0"

paths:
  raw_data: "data/raw"
  extracted: "data/extracted"
  images: "data/images"
  annotations: "data/annotations"
  labels: "data/labels"
  samples: "data/samples"
  outputs: "outputs"
  verified_data: "verified_data"

extraction:
  sample_rate: 1.0  # Full extraction
  channels: ["port", "starboard", "combined"]
  normalize: true

image_generation:
  preserve_orientation: true  # 방향 보존!
  flip_y: false               # Flip 하지 않음
  target_width: 1024
  enhance_contrast: true
  output_format: "png"

coordinate_mapping:
  source_flipped: false       # 새로운 이미지는 flip 안됨
  apply_flip_transform: false # 좌표 변환 시 flip 안함
  scale_x: 6.25
  scale_y: 1.0

labeling:
  label_format: "npz"         # npz, json, xml
  mask_dtype: "uint8"
  instance_segmentation: true # 인스턴스별 ID 부여
  save_metadata: true

sampling:
  mine_to_background_ratio: [1, 5]
  patch_size: [128, 128]
  overlap: 0.0

augmentation:
  enabled: true
  exclude_original: true
  techniques:
    - horizontal_flip
    - vertical_flip
    - rotation_90
    - rotation_180
    - rotation_270
    - gaussian_noise
    - gaussian_blur
    - contrast_adjust
    - brightness_adjust
  num_augmentations_per_sample: 9

validation:
  visual_check: true
  statistics_report: true
  save_comparison_images: true
```

---

## 설치 및 사용 문서

### INSTALL.md 구조

```markdown
# 설치 가이드

## 시스템 요구사항
- Python 3.8+
- 8GB+ RAM
- 10GB+ 디스크 공간

## 의존성 설치

### 1. 기본 패키지
pip install -r requirements.txt

### 2. 선택적 패키지
# XTF 처리
pip install pyxtf

# 이미지 처리
pip install opencv-python Pillow

# 인터랙티브 어노테이션
pip install labelImg  # 또는 자체 도구

## 설치 확인
python scripts/check_installation.py
```

### USAGE.md 구조

```markdown
# 사용 가이드

## 빠른 시작

### 1. 설정 파일 준비
cp config/default_config.yaml config/my_config.yaml

### 2. 전체 파이프라인 실행
python scripts/run_pipeline.py --config config/my_config.yaml

## 단계별 실행

### Step 1: XTF 추출
python scripts/01_extract_xtf.py --input data/raw/file.xtf --output data/extracted

### Step 2: 이미지 생성
python scripts/02_generate_images.py --input data/extracted/*.npy --output data/images

### Step 3: 어노테이션
python scripts/03_annotate.py --images data/images/*.png

### Step 4: 라벨 생성
python scripts/04_create_labels.py --annotations data/annotations/*.json

### Step 5: 샘플링 및 증강
python scripts/05_sample_and_augment.py --labels data/labels/*.npz

## 기존 데이터 마이그레이션

python scripts/migrate_verified_data.py \
  --source analysis_results/npy_labeling/flipped \
  --dest verified_data/flipped_20251104
```

---

## requirements.txt

```txt
# Core
numpy>=1.21.0
scipy>=1.7.0

# Image Processing
Pillow>=9.0.0
opencv-python>=4.5.0
scikit-image>=0.19.0

# XTF Processing
pyxtf>=1.0.0

# Data Processing
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Configuration
PyYAML>=6.0
python-dotenv>=0.19.0

# Augmentation
albumentations>=1.1.0
imgaug>=0.4.0

# CLI
click>=8.0.0
tqdm>=4.62.0

# Optional: Interactive Annotation
# labelImg>=1.8.0  # 또는 자체 구현

# Development
pytest>=6.2.0
black>=21.0
flake8>=3.9.0
```

---

## 다음 구현 단계

### Phase 1: 기본 인프라 (승인 후 진행)

1. **디렉토리 구조 생성**
   - 모든 폴더 생성
   - 기존 flipped 데이터 복사

2. **설정 파일 작성**
   - default_config.yaml
   - requirements.txt
   - setup.py

3. **문서 작성**
   - INSTALL.md
   - USAGE.md
   - API.md

### Phase 2: Core 모듈 구현

1. **XTF Extractor** (방향 보존)
2. **NPY to Image Converter** (방향 보존)
3. **Coordinate Mapper** (조건부 flip)
4. **NPY Labeler** (식별 가능한 구조)

### Phase 3: 유틸리티 구현

1. **Format Converter** (JSON ↔ XML)
2. **Data Migrator** (기존 데이터 이관)
3. **Validator** (검증 도구)

### Phase 4: 파이프라인 스크립트

1. **개별 스크립트** (01-05)
2. **통합 파이프라인**
3. **테스트 코드**

---

## 승인 요청

다음 사항 승인 부탁드립니다:

1. ✅ Y축 flip 처리 방침 (방향 보존 원칙)
2. ✅ NPY 라벨 구조 (npz + metadata)
3. ✅ JSON/XML 어노테이션 형식
4. ✅ 설정 파일 구조
5. ✅ 문서 구성 (INSTALL, USAGE)
6. ✅ 디렉토리 구조

승인하시면 Phase 1부터 구현 시작하겠습니다.
