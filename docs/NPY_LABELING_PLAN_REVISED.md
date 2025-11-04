# .npy 기반 기뢰 라벨링 수정 계획서

## 검토 결과

### ✅ 1. 기존 라벨 무시 확인
- **기존**: `analysis_results/labeling/mine_labels.npy` (200×6832, pipeline 버전)
- **조치**: 무시하고 새로 생성
- **이유**: Pipeline 다운샘플링 버전이 아닌 원본 해상도 사용

### ✅ 2. DATA_AUGMENTATION_STRATEGY_GUIDE.md 검토

**문서 권장사항 (섹션 2.3)**:
```
NPY 형식 (권장)
- Shape: (7974, 6832) - 원본 해상도
- Dtype: float32
- Range: 0.0 - 1.0
- 용도: 증강, 특징 추출, 학습
- 장점: 높은 정밀도, 정보 손실 없음, ML 최적화
```

**현재 파일 확인**:
```
✓ data/processed/xtf_extracted/
  Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_combined_intensity.npy
  - Shape: (7974, 6832) ← 문서 권장과 일치!
  - Dtype: float32
  - Range: [0.0, 1.0]
  - Size: 207.8 MB

△ data/processed/xtf_extracted/pipeline/
  ..._pipeline_combined_intensity.npy
  - Shape: (200, 6832) ← 다운샘플링됨
  - 학습용으로는 부적합
```

**결론**:
- ✅ **일반 추출 파일이 가이드 권장사항과 완벽히 일치**
- ✅ **XTF extractor 개선 불필요**
- ✅ **원본 해상도 (7974×6832) 사용**

### ✅ 3. 작업 폴더 구조

```
analysis_results/
└── npy_labeling/              ← 새로운 작업 폴더
    ├── raw/                   # 원본 데이터 복사/링크
    │   ├── intensity.npy      # 7974×6832 원본
    │   ├── annotation.json    # bbox 정보
    │   └── bmp_original.bmp   # 참조용
    │
    ├── labels/                # 생성된 라벨
    │   ├── mine_labels_full.npy           # 7974×6832 라벨
    │   ├── mine_positions.csv             # npy 좌표
    │   └── mine_metadata.json             # 라벨 메타데이터
    │
    ├── validation/            # 검증 결과
    │   ├── overlay_full_resolution.png    # 원본 해상도 오버레이
    │   ├── comparison_with_bmp.png        # BMP 비교
    │   ├── position_accuracy.csv          # 위치 정확도
    │   └── validation_report.md           # 검증 리포트
    │
    └── scripts/               # 작업 스크립트
        ├── 01_coordinate_transform.py     # 좌표 변환
        ├── 02_create_labels.py            # 라벨 생성
        └── 03_validate_labels.py          # 검증
```

**기존 파일과 분리**:
- ❌ `analysis_results/labeling/` - 건드리지 않음 (기존 pipeline 작업)
- ✅ `analysis_results/npy_labeling/` - 새로운 전용 폴더

---

## 수정된 작업 계획

### Phase 1: 좌표 변환 함수 작성

**입력**:
- Annotation 좌표: (1024×3862 이미지 공간)
- BMP 원본: (1024×7974)
- npy 원본: (7974×6832)

**좌표 변환 수식**:
```python
def annotation_to_npy_full(ann_x, ann_y):
    """
    Annotation (1024×3862) → npy (7974×6832)

    전제:
    - Annotation = BMP 하단 3862행을 180° 회전
    - BMP와 npy는 높이 동일 (7974)
    - 너비만 다름: 1024 → 6832
    """
    # 1. 180° 역회전
    bmp_x = 1024 - ann_x - 1
    bmp_y = 3862 - ann_y - 1

    # 2. BMP 하단 영역 → 전체 BMP 좌표
    bmp_y_full = 7974 - 3862 + bmp_y  # = 4112 + bmp_y

    # 3. BMP → npy 변환 (높이 동일, 너비만 스케일)
    npy_y = bmp_y_full  # 높이 그대로
    npy_x = int(bmp_x * 6832 / 1024)  # 너비만 6.67배

    return npy_x, npy_y
```

**검증 방법**:
- 역변환 후 원본 annotation과 오차 < 2픽셀

---

### Phase 2: 라벨 생성

**입력**:
- `annotation_bboxes_manual.json`: 25개 기뢰 bbox
- npy intensity: (7974, 6832)

**출력**:
- `mine_labels_full.npy`: (7974, 6832) uint8 배열
  - 0 = 배경
  - 1 = 기뢰

**Bounding box 크기**:
```python
# Annotation bbox 평균 크기 측정
ann_bbox_avg = (34픽셀 × 37픽셀)  # 예시

# npy 해상도로 변환
npy_bbox_width = int(34 * 6832 / 1024)   # ≈ 227 픽셀
npy_bbox_height = 37  # 높이 동일

# 보수적으로 약간 키움 (여유 포함)
npy_bbox = (40 × 250)  # (H × W)
```

**GPS 매핑**:
- Navigation 데이터 사용
- 7974 pings → 각 ping별 GPS 좌표 (보간 필요)

---

### Phase 3: 검증

**3.1 역변환 검증**:
```python
npy 좌표 → BMP 좌표 → Annotation 좌표
원본 annotation과 비교
평균 오차 < 2픽셀 (원본 해상도 기준)
```

**3.2 시각적 검증**:
- npy intensity + 라벨 오버레이
- BMP 원본 + 업스케일 라벨 오버레이
- Annotation과 side-by-side 비교

**3.3 정량적 검증**:
- 라벨된 픽셀 수
- 기뢰당 평균 픽셀
- 위치 정확도 (픽셀 오차)

---

## 작업 순서

### Step 1: 폴더 구조 생성
```bash
mkdir -p analysis_results/npy_labeling/{raw,labels,validation,scripts}
```

### Step 2: 데이터 준비
- 원본 npy (7974×6832) 링크
- annotation json 복사
- BMP 참조용 복사

### Step 3: 좌표 변환 스크립트
- `01_coordinate_transform.py` 작성
- 변환 함수 구현
- 단위 테스트

### Step 4: 라벨 생성
- `02_create_labels.py` 작성
- 25개 기뢰 라벨링
- metadata 저장

### Step 5: 검증
- `03_validate_labels.py` 작성
- 역변환 검증
- 시각화 생성
- 리포트 작성

---

## 예상 결과

### 데이터 형식
```python
# mine_labels_full.npy
Shape: (7974, 6832)
Dtype: uint8
Values: 0 (배경), 1 (기뢰)
Size: ~52 MB
라벨된 픽셀: ~370,000 (25개 × ~15,000 픽셀/개)
비율: ~0.68%
```

### 정확도 목표
- 평균 픽셀 오차: < 2픽셀 (npy 기준)
- 역변환 오차: < 2픽셀 (annotation 기준)
- 모든 기뢰 npy 범위 내 포함

### 활용 방안
1. **증강 데이터셋 생성**: DATA_AUGMENTATION_STRATEGY_GUIDE.md 3장 따름
2. **특징 추출**: 원본 해상도로 정밀한 특징 추출
3. **모델 학습**: 고해상도 입력으로 성능 향상

---

## 승인 요청

수정된 계획:
1. ✅ 기존 라벨 무시
2. ✅ 원본 해상도 npy (7974×6832) 사용
3. ✅ 별도 폴더 `npy_labeling/` 생성
4. ✅ XTF extractor 개선 불필요

**진행 가능 여부**: 승인 시 즉시 시작
**예상 시간**: 1-2시간
**출력 파일**: ~10개 (npy, csv, png, md)
