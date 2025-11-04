# 기뢰 라벨링 프로젝트 - 진행 내용 및 유의사항

## 프로젝트 개요

XTF 사이드 스캔 소나 데이터에서 추출한 강도 데이터(.npy)에 기뢰 위치를 정확하게 라벨링하는 프로젝트

**데이터셋**: Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04
**소나**: Klein 3900, 900kHz

---

## 진행 경과

### 1차 시도: 원본 매핑 (2025-11-04 10:47-10:52)

**접근 방법**:
- BMP 이미지 (1024 × 5137) 좌표를 NPY 데이터 (6400 × 5137)로 직접 스케일링
- X축: 6.25배 스케일 (1024 → 6400)
- Y축: 1:1 매핑 (5137 → 5137)

**결과**:
- 기술적으로는 정확한 변환이었으나, 시각적 검증 시 위치 불일치 발견
- 원인: NPY 데이터와 BMP 이미지의 Y축 방향 차이

### 2차 시도: Y축 Flip 매핑 (2025-11-04 11:12)

**접근 방법**:
- Y축 좌표를 상하 반전: `Y_npy = (height - 1) - Y_bmp`
- X축: 동일하게 6.25배 스케일

**결과**: ✅ **성공**
- BMP 이미지의 기뢰 위치와 NPY 데이터의 기뢰 위치가 정확히 일치
- 25개 모든 기뢰 정확하게 매핑됨

---

## 핵심 발견 사항

### ⚠️ 중요: Y축 방향 차이

**문제**:
- Original BMP (1024 × 5137): 위에서 아래로 증가하는 좌표계
- NPY intensity data (6400 × 5137): XTF 추출 시 좌표계가 상하 반전됨

**해결책**:
- 바운딩 박스 Y좌표를 flip해서 매핑: `Y_npy = (5137 - 1) - Y_bmp`
- 이렇게 해야 BMP의 기뢰 위치와 NPY의 기뢰 위치가 정확히 일치

### 좌표 변환 공식 (최종)

```python
# BMP (1024 × 5137) → NPY (6400 × 5137)

# X축: 스케일만 적용
X_npy = X_bmp × 6.25

# Y축: Flip + 1:1 매핑
Y_npy = (5137 - 1) - Y_bmp

# 바운딩 박스의 경우
xmin_npy = int(xmin_bmp * 6.25)
xmax_npy = int(xmax_bmp * 6.25)
ymin_npy = (5137 - 1) - ymax_bmp  # 주의: ymax와 ymin 교환됨
ymax_npy = (5137 - 1) - ymin_bmp
```

---

## 검증된 결과물

### 위치: `analysis_results/npy_labeling/flipped/`

1. **flipped_mine_label_mask.npy** (32,876,928 bytes)
   - Shape: (5137, 6400)
   - 이진 마스크: 0=배경, 1=기뢰
   - 기뢰 픽셀: 390,530개 (1.19%)

2. **flipped_labeled_intensity_data.npz** (164MB)
   - intensity: (5137, 6400) 강도 데이터
   - labels: (5137, 6400) 라벨 마스크
   - metadata: JSON 형태의 주석 정보

3. **flipped_mapped_annotations.json** (8.5KB)
   - 25개 기뢰의 원본 BMP 좌표
   - Flip된 BMP 좌표
   - 매핑된 NPY 좌표

4. **시각화 파일**:
   - `flipped_coordinate_mapping_visualization.png`: 전체 매핑 시각화
   - `flipped_mine_01_comparison.png`: 기뢰 #1 상세 비교
   - `flipped_mine_13_comparison.png`: 기뢰 #13 상세 비교
   - `flipped_mine_25_comparison.png`: 기뢰 #25 상세 비교

---

## 통계 정보

### 데이터 차원
- NPY 강도 데이터: (5137, 6400)
- BMP 이미지: (1024, 5137, 3)
- 라벨 마스크: (5137, 6400)

### 기뢰 주석
- 총 기뢰 개수: 25개
- 바운딩 박스 크기 (NPY 공간):
  - 너비: 318-319 샘플 (평균 318.80)
  - 높이: 49 핑 (고정)

### 라벨 분포
- 총 픽셀: 32,876,800
- 기뢰 픽셀: 390,530 (1.19%)
- 배경 픽셀: 32,486,270 (98.81%)

---

## 유의사항

### 1. 이미지 생성 시 방향 보존

**문제**:
- NPY 데이터를 BMP/PNG로 변환할 때 Y축이 자동으로 flip될 수 있음
- matplotlib의 `imshow()`는 기본적으로 top-down 렌더링

**해결**:
```python
# 올바른 방향 보존
plt.imshow(npy_data, origin='upper', cmap='gray')

# 또는 명시적 flip
plt.imshow(np.flipud(npy_data), cmap='gray')
```

### 2. 좌표계 일관성

**핵심**:
- BMP 어노테이션 좌표는 항상 원본 BMP 좌표계 기준
- NPY 매핑 시에만 flip 적용
- 새로운 이미지 생성 시 원본 방향과 동일하게 유지

### 3. 바운딩 박스 검증

**반드시 확인**:
1. BMP와 생성된 이미지의 기뢰 위치가 육안으로 일치하는지
2. 바운딩 박스가 기뢰 객체를 정확히 포함하는지
3. Y좌표의 min/max가 올바르게 변환되었는지

---

## 다음 단계

### 독립 프로젝트 구성 요소

1. **Full XTF Extractor**
   - 샘플링 없이 전체 데이터 추출
   - 메타데이터 보존
   - 원본 방향 유지한 이미지 변환

2. **Interactive Annotation Tool**
   - 사람이 보기 쉬운 고대비 이미지 생성
   - 바운딩 박스 인터랙티브 표시
   - 픽셀 좌표 자동 추출

3. **Coordinate Transformer**
   - 축소 이미지 좌표 → 원본 NPY 좌표 변환
   - Y축 flip 자동 처리
   - 검증 시각화 자동 생성

4. **NPY Labeling Processor**
   - NPY 배열에 직접 라벨 적용
   - 이진 마스크 생성
   - 통합 데이터 파일 생성

5. **Data Augmentation Pipeline**
   - 기뢰 샘플 증강 (9가지 기법)
   - 배경 샘플링 (1:5 비율)
   - 증강 샘플 별도 관리

---

## 파일 경로 참조

### 검증된 데이터
```
analysis_results/npy_labeling/flipped/
├── flipped_mine_label_mask.npy
├── flipped_labeled_intensity_data.npz
├── flipped_mapped_annotations.json
└── [visualization files]
```

### 원본 데이터
```
data/processed/xtf_extracted/
└── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_combined_intensity.npy

datasets/
└── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/
    └── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_IMG_00.BMP

KleinLabeling/
└── Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xml
```

---

## 작성일
2025-11-04

## 작성자
AI Assistant (Claude) + 사용자 협업
