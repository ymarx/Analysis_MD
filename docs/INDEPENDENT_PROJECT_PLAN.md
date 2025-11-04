# 독립 기뢰 라벨링 프로젝트 - 구조 설계

## 프로젝트 목표

XTF 소나 데이터의 전체 처리 파이프라인을 독립적으로 실행 가능한 프로젝트로 구축

---

## 디렉토리 구조

```
mine_labeling_project/
├── README.md
├── requirements.txt
│
├── config/
│   ├── extractor_config.yaml      # XTF 추출 설정
│   ├── labeling_config.yaml       # 라벨링 설정
│   └── augmentation_config.yaml   # 증강 설정
│
├── data/
│   ├── raw/                        # 원본 XTF 파일
│   ├── extracted/                  # 추출된 NPY + 메타데이터
│   │   ├── full/                   # Full extraction 결과
│   │   └── metadata/               # 메타데이터 JSON
│   │
│   ├── images/                     # 생성된 이미지
│   │   ├── original_resolution/   # 원본 해상도 이미지
│   │   ├── display/                # 디스플레이용 고대비 이미지
│   │   └── annotated/              # 바운딩 박스 표시된 이미지
│   │
│   ├── annotations/                # 어노테이션 파일
│   │   ├── xml/                    # LabelImg XML 파일
│   │   └── json/                   # JSON 좌표 파일
│   │
│   ├── labels/                     # 라벨링 결과
│   │   ├── masks/                  # 이진 마스크 (.npy)
│   │   ├── labeled_data/           # 통합 데이터 (.npz)
│   │   └── mapped_coords/          # 매핑된 좌표 (JSON)
│   │
│   └── samples/                    # 샘플링 및 증강
│       ├── mine_samples/           # 기뢰 샘플
│       ├── background_samples/     # 배경 샘플
│       └── augmented/              # 증강된 샘플
│
├── outputs/                        # 처리 결과 (날짜별)
│   ├── 20251104_001/
│   │   ├── extracted/
│   │   ├── labeled/
│   │   ├── samples/
│   │   └── validation/
│   └── 20251104_002/
│
├── src/
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── xtf_full_extractor.py       # Full XTF 추출기
│   │   └── metadata_extractor.py       # 메타데이터 추출
│   │
│   ├── converters/
│   │   ├── __init__.py
│   │   ├── npy_to_image.py             # NPY → 이미지 변환 (방향 보존)
│   │   ├── contrast_enhancer.py         # 대비 향상
│   │   └── resolution_scaler.py         # 해상도 조절
│   │
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── interactive_annotator.py     # 인터랙티브 바운딩 박스 도구
│   │   ├── coordinate_transformer.py    # 좌표 변환기
│   │   └── annotation_validator.py      # 어노테이션 검증
│   │
│   ├── labeling/
│   │   ├── __init__.py
│   │   ├── npy_labeler.py              # NPY 데이터 라벨러
│   │   ├── mask_generator.py           # 마스크 생성기
│   │   └── label_merger.py             # 라벨 병합
│   │
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── mine_sampler.py             # 기뢰 샘플 추출
│   │   ├── background_sampler.py       # 배경 샘플 추출
│   │   └── sample_balancer.py          # 샘플 밸런싱 (1:5)
│   │
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── augmentation_engine.py      # 증강 엔진
│   │   └── techniques/                 # 9가지 증강 기법
│   │       ├── flip.py
│   │       ├── rotation.py
│   │       ├── noise.py
│   │       ├── blur.py
│   │       ├── contrast.py
│   │       ├── brightness.py
│   │       ├── elastic.py
│   │       ├── scale.py
│   │       └── crop.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── coordinate_validator.py     # 좌표 검증
│   │   ├── visual_validator.py         # 시각적 검증
│   │   └── statistics_reporter.py      # 통계 보고서
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_manager.py             # 파일 관리
│       ├── logger.py                   # 로깅
│       └── config_loader.py            # 설정 로드
│
├── scripts/
│   ├── 01_extract_full_xtf.py          # Step 1: XTF 전체 추출
│   ├── 02_generate_display_images.py   # Step 2: 디스플레이 이미지 생성
│   ├── 03_annotate_interactive.py      # Step 3: 인터랙티브 어노테이션
│   ├── 04_transform_coordinates.py     # Step 4: 좌표 변환 및 검증
│   ├── 05_label_npy_data.py            # Step 5: NPY 데이터 라벨링
│   ├── 06_sample_extraction.py         # Step 6: 샘플 추출
│   ├── 07_data_augmentation.py         # Step 7: 데이터 증강
│   └── run_full_pipeline.py            # 전체 파이프라인 실행
│
├── tests/
│   ├── test_extractors.py
│   ├── test_converters.py
│   ├── test_labeling.py
│   └── test_augmentation.py
│
└── docs/
    ├── USER_GUIDE.md
    ├── API_REFERENCE.md
    ├── COORDINATE_SYSTEM.md
    └── TROUBLESHOOTING.md
```

---

## 핵심 처리 흐름

### Pipeline 1: 추출 및 이미지 생성

```
XTF 파일
  ↓
[01_extract_full_xtf.py]
  ↓
NPY 데이터 (5137 × 6400) + 메타데이터
  ↓
[02_generate_display_images.py]
  ↓
고대비 디스플레이 이미지 (1024 × 5137)
  ※ 방향 보존 필수!
```

### Pipeline 2: 어노테이션 및 변환

```
디스플레이 이미지 (1024 × 5137)
  ↓
[03_annotate_interactive.py] - 사용자 인터랙션
  ↓
바운딩 박스 좌표 (XML/JSON)
  ↓
[04_transform_coordinates.py]
  ↓
NPY 좌표 (6400 × 5137, Y-flipped)
  ↓
시각적 검증 이미지
```

### Pipeline 3: 라벨링

```
NPY 원본 데이터 + 변환된 좌표
  ↓
[05_label_npy_data.py]
  ↓
라벨 마스크 (.npy) + 통합 데이터 (.npz)
```

### Pipeline 4: 샘플링 및 증강

```
라벨된 NPY 데이터
  ↓
[06_sample_extraction.py]
  ├─→ 기뢰 샘플 추출 (25개 × patch_size)
  └─→ 배경 샘플 추출 (125개, 1:5 비율)
  ↓
[07_data_augmentation.py]
  ↓
증강된 샘플 (기뢰 225개, 배경 125개)
  ※ 원본 제외 9가지 기법 적용
```

---

## 주요 컴포넌트 상세

### 1. Full XTF Extractor

**입력**: XTF 파일
**출력**:
- `*_combined_intensity.npy` (전체 데이터)
- `*_port_intensity.npy`
- `*_starboard_intensity.npy`
- `*_metadata.json`
- `*_navigation.npz`

**핵심 기능**:
- sample_rate=1.0 (전체 추출)
- 메타데이터 보존
- 좌표 정보 저장

### 2. NPY to Image Converter

**입력**: NPY 강도 데이터
**출력**: BMP/PNG 이미지

**핵심 기능**:
- 방향 보존 (origin='upper')
- 고대비 변환 (CLAHE)
- 해상도 조정 (선택적)

**중요 코드**:
```python
def npy_to_image(npy_data, output_path, enhance=True, target_width=1024):
    """
    NPY → 이미지 변환 (방향 보존)

    주의: Y축 flip 하지 않음!
    """
    # 정규화
    normalized = (npy_data - npy_data.min()) / (npy_data.max() - npy_data.min())

    # 대비 향상 (선택)
    if enhance:
        normalized = apply_clahe(normalized)

    # 해상도 조정
    if target_width and target_width != npy_data.shape[1]:
        normalized = resize_preserve_aspect(normalized, target_width)

    # 이미지 저장 (방향 보존!)
    img = (normalized * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(output_path)
```

### 3. Coordinate Transformer

**입력**:
- 디스플레이 이미지 바운딩 박스 좌표
- 원본 NPY 데이터 shape

**출력**:
- NPY 좌표계로 변환된 바운딩 박스

**핵심 로직**:
```python
def transform_bbox_to_npy(bbox_display, display_shape, npy_shape):
    """
    디스플레이 이미지 좌표 → NPY 좌표 변환

    핵심: Y축 flip 적용!
    """
    display_h, display_w = display_shape[:2]
    npy_h, npy_w = npy_shape

    # X축 스케일
    scale_x = npy_w / display_w

    # Y축: Flip + 1:1
    xmin_npy = int(bbox_display['xmin'] * scale_x)
    xmax_npy = int(bbox_display['xmax'] * scale_x)

    # 주의: ymin과 ymax 교환 + flip
    ymin_npy = (npy_h - 1) - bbox_display['ymax']
    ymax_npy = (npy_h - 1) - bbox_display['ymin']

    return {
        'xmin': xmin_npy,
        'ymin': ymin_npy,
        'xmax': xmax_npy,
        'ymax': ymax_npy
    }
```

### 4. NPY Labeler

**입력**:
- NPY 강도 데이터
- 변환된 바운딩 박스 좌표

**출력**:
- 이진 라벨 마스크 (0=배경, 1=기뢰)

### 5. Sample Extractor & Augmenter

**샘플 추출**:
- 기뢰: 바운딩 박스 중심 기준 patch 추출
- 배경: 기뢰가 없는 영역에서 랜덤 샘플링 (1:5 비율)

**증강 기법** (9가지):
1. Horizontal Flip
2. Vertical Flip
3. Rotation (90°, 180°, 270°)
4. Gaussian Noise
5. Gaussian Blur
6. Contrast Adjustment
7. Brightness Adjustment
8. Elastic Deformation
9. Random Scale

---

## 출력 관리 시스템

### 타임스탬프 기반 출력 디렉토리

```python
from datetime import datetime

def create_output_dir(base_dir='outputs'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 하위 디렉토리 생성
    (output_dir / 'extracted').mkdir()
    (output_dir / 'labeled').mkdir()
    (output_dir / 'samples').mkdir()
    (output_dir / 'validation').mkdir()

    return output_dir
```

### 결과 복사 및 보존

```python
def archive_flipped_results(output_dir):
    """
    검증된 flipped 결과를 출력 디렉토리에 복사
    """
    source = Path('analysis_results/npy_labeling/flipped')
    dest = output_dir / 'labeled' / 'flipped_verified'

    shutil.copytree(source, dest)
```

---

## 설정 파일 예시

### `config/extractor_config.yaml`

```yaml
xtf_extractor:
  sample_rate: 1.0  # 전체 추출
  channels:
    - port
    - starboard
  normalize: true
  output_format: npy
```

### `config/labeling_config.yaml`

```yaml
labeling:
  coordinate_system:
    y_flip: true  # NPY 좌표 변환 시 Y축 flip
    origin: upper

  bbox_validation:
    min_width: 200
    min_height: 30

  output:
    save_mask: true
    save_npz: true
    save_json: true
```

### `config/augmentation_config.yaml`

```yaml
sampling:
  mine_to_background_ratio: [1, 5]
  patch_size: [128, 128]

augmentation:
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

  exclude_original: true  # 원본 제외하고 증강만
```

---

## 다음 구현 단계

1. **디렉토리 구조 생성**
2. **Core 모듈 구현**:
   - XTF Full Extractor
   - NPY to Image Converter (방향 보존)
   - Coordinate Transformer (Y-flip 처리)
3. **Interactive Annotation Tool**
4. **NPY Labeling Processor**
5. **Sampling & Augmentation Pipeline**
6. **검증 및 테스트**

---

## 승인 요청 사항

다음 사항에 대한 승인을 요청합니다:

1. ✅ 위의 디렉토리 구조
2. ✅ Pipeline 흐름 (4단계)
3. ✅ 타임스탬프 기반 출력 관리
4. ✅ Y축 flip 처리 방식
5. ✅ 샘플링 비율 1:5
6. ✅ 증강 기법 9가지 (원본 제외)

승인하시면 구현을 시작하겠습니다.
