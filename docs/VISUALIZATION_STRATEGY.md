# npy 라벨 시각화 전략

## 기존 Annotation 스타일 분석

### PH_annotation.png 특징
- **크기**: 1024×3862
- **형식**: RGB 컬러 이미지
- **기뢰 표시**: 빨간색 박스 (255, 0, 0)
- **스타일**:
  - Grayscale 소나 배경
  - 25개 기뢰에 빨간 박스
  - 박스 선명하게 표시
  - 소나 이미지 잘 보임

---

## npy 시각화 방식

### 방법 1: 빨간 박스 오버레이 ⭐ (메인)

**목적**: 기존 annotation과 동일한 느낌

```python
def create_red_box_overlay(intensity_npy, labels_npy, mine_positions):
    """
    기존 PH_annotation.png 스타일 재현

    Args:
        intensity_npy: (7974, 6832) float32, [0.0, 1.0]
        labels_npy: (7974, 6832) uint8, {0, 1}
        mine_positions: DataFrame with npy_x, npy_y, target_id

    Returns:
        RGB image with red bounding boxes
    """
    # 1. Intensity를 grayscale 이미지로
    img_gray = (intensity_npy * 255).astype(np.uint8)

    # 2. RGB로 변환
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # 3. 각 기뢰에 빨간 박스
    for _, mine in mine_positions.iterrows():
        x, y = mine['npy_x'], mine['npy_y']
        target_id = mine['target_id']

        # Bounding box 크기 (예: 40×250)
        h, w = 40, 250
        x1, y1 = int(x - w//2), int(y - h//2)
        x2, y2 = int(x + w//2), int(y + h//2)

        # 빨간 박스 그리기
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2),
                     color=(255, 0, 0),  # 빨강
                     thickness=3)

        # ID 번호 표시
        cv2.putText(img_rgb, target_id,
                   (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5,
                   color=(255, 255, 0),  # 노란색 텍스트
                   thickness=1)

    return img_rgb
```

**출력**:
- `npy_labeled_red_boxes_full.png` (7974×6832, ~180MB)
- `npy_labeled_red_boxes_preview.png` (축소 버전, ~5MB)

---

### 방법 2: Side-by-side 비교

**목적**: 원본과 라벨 동시 비교

```python
def create_sidebyside_comparison(intensity_npy, labeled_rgb):
    """
    원본 | 라벨 비교
    """
    # 원본 grayscale → RGB
    original_rgb = cv2.cvtColor(
        (intensity_npy * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )

    # 좌우 배치
    comparison = np.hstack([original_rgb, labeled_rgb])

    # 중앙 구분선
    h, w = comparison.shape[:2]
    cv2.line(comparison, (w//2, 0), (w//2, h),
            color=(0, 255, 0), thickness=5)

    # 상단에 제목
    cv2.putText(comparison, "Original", (50, 50), ...)
    cv2.putText(comparison, "Labeled", (w//2 + 50, 50), ...)

    return comparison
```

**출력**:
- `npy_comparison_sidebyside.png` (7974×13664)

---

### 방법 3: 확대 타일 그리드

**목적**: 각 기뢰 상세 확인

```python
def create_mine_tiles_grid(intensity_npy, mine_positions, tile_size=200):
    """
    25개 기뢰를 5×5 그리드로 배치
    """
    tiles = []

    for _, mine in mine_positions.iterrows():
        x, y = mine['npy_x'], mine['npy_y']

        # 기뢰 중심 ±tile_size//2 영역 추출
        x1 = max(0, x - tile_size//2)
        y1 = max(0, y - tile_size//2)
        x2 = min(intensity_npy.shape[1], x + tile_size//2)
        y2 = min(intensity_npy.shape[0], y + tile_size//2)

        tile = intensity_npy[y1:y2, x1:x2]

        # 고정 크기로 패딩
        tile_padded = np.zeros((tile_size, tile_size))
        h, w = tile.shape
        tile_padded[:h, :w] = tile

        # RGB 변환 + 중심점 표시
        tile_rgb = cv2.cvtColor((tile_padded * 255).astype(np.uint8),
                               cv2.COLOR_GRAY2RGB)
        cv2.circle(tile_rgb, (tile_size//2, tile_size//2),
                  radius=10, color=(255, 0, 0), thickness=-1)

        # ID 표시
        cv2.putText(tile_rgb, mine['target_id'], ...)

        tiles.append(tile_rgb)

    # 5×5 그리드로 배치
    rows = []
    for i in range(0, 25, 5):
        row = np.hstack(tiles[i:i+5])
        rows.append(row)

    grid = np.vstack(rows)

    # 상단에 전체 이미지 축소 버전
    overview = cv2.resize(labeled_rgb, (tile_size*5, ...))

    final = np.vstack([overview, grid])

    return final
```

**출력**:
- `npy_mine_tiles_grid.png` (5×5 타일 + 상단 전체 뷰)

---

### 방법 4: BMP 스타일 재현

**목적**: BMP 원본과 직접 비교

```python
def create_bmp_style_comparison(intensity_npy, labels_npy, bmp_original):
    """
    npy를 BMP 해상도로 맞춰서 비교
    """
    # npy (7974×6832) → BMP 크기 (7974×1024)
    npy_downsampled = cv2.resize(
        intensity_npy,
        (1024, 7974),
        interpolation=cv2.INTER_AREA
    )

    # 라벨도 다운샘플
    labels_downsampled = cv2.resize(
        labels_npy.astype(float),
        (1024, 7974),
        interpolation=cv2.INTER_NEAREST
    ) > 0.5

    # 빨간 박스 표시
    npy_rgb = cv2.cvtColor((npy_downsampled * 255).astype(np.uint8),
                          cv2.COLOR_GRAY2RGB)
    npy_rgb[labels_downsampled] = [255, 0, 0]

    # BMP 원본 RGB
    bmp_rgb = cv2.cvtColor(bmp_original, cv2.COLOR_GRAY2RGB)

    # 좌우 배치
    comparison = np.hstack([bmp_rgb, npy_rgb])

    return comparison
```

**출력**:
- `npy_vs_bmp_comparison.png`

---

### 방법 5: 인터랙티브 HTML (선택)

**목적**: 웹 브라우저에서 확대/탐색

```html
<!DOCTYPE html>
<html>
<head>
    <title>Mine Labeling Viewer</title>
    <style>
        #canvas { border: 1px solid black; }
        .mine-info { position: absolute; background: white; padding: 5px; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script>
        // OpenSeadragon 또는 Leaflet.js 사용
        // 고해상도 이미지 타일링
        // 클릭 시 기뢰 정보 팝업
    </script>
</body>
</html>
```

**출력**:
- `interactive_viewer.html` + 이미지 타일들

---

## 최종 출력 파일

### 필수 (방법 1, 2, 3)
1. **npy_labeled_red_boxes_full.png** (7974×6832)
   - 전체 해상도 빨간 박스 오버레이
   - 기존 annotation과 동일 스타일

2. **npy_labeled_red_boxes_preview.png** (1500×6000 축소)
   - 미리보기용 축소 버전

3. **npy_comparison_sidebyside.png**
   - 원본 | 라벨 비교

4. **npy_mine_tiles_grid.png**
   - 25개 기뢰 5×5 그리드

### 선택 (방법 4, 5)
5. **npy_vs_bmp_comparison.png**
   - BMP 원본과 비교

6. **interactive_viewer.html** (선택)
   - 웹 기반 인터랙티브 뷰어

---

## 구현 우선순위

### Phase 1 (필수)
- ✅ 방법 1: 빨간 박스 오버레이 (메인)
- ✅ 방법 2: Side-by-side 비교

### Phase 2 (권장)
- ✅ 방법 3: 확대 타일 그리드
- ✅ 방법 4: BMP 비교

### Phase 3 (선택)
- ⭕ 방법 5: 인터랙티브 (필요시)

---

## 색상 및 스타일 가이드

### 색상 팔레트
```python
COLORS = {
    'mine_box': (255, 0, 0),      # 빨강 - 기뢰 박스
    'mine_id': (255, 255, 0),     # 노랑 - ID 텍스트
    'center_dot': (0, 255, 0),    # 초록 - 중심점
    'separator': (0, 255, 0),     # 초록 - 구분선
    'title': (255, 255, 255),     # 흰색 - 제목
}
```

### 선 두께
```python
THICKNESS = {
    'bbox': 3,          # 기뢰 박스
    'separator': 5,     # 구분선
    'center_dot': -1,   # 중심점 (filled)
}
```

### 폰트
```python
FONT = {
    'face': cv2.FONT_HERSHEY_SIMPLEX,
    'scale_small': 0.5,   # ID
    'scale_large': 1.0,   # 제목
    'thickness': 1,
}
```

---

## 검증 체크리스트

### 시각적 검증
- [ ] 25개 기뢰 모두 빨간 박스 표시
- [ ] ID 번호 명확히 보임 (PH_01 ~ PH_25)
- [ ] 소나 이미지 배경 잘 보임
- [ ] 박스 위치가 기뢰 중심과 일치
- [ ] 전체 이미지 크기 정확 (7974×6832)

### 비교 검증
- [ ] BMP 원본과 위치 일치
- [ ] Annotation과 패턴 일치
- [ ] 역변환 시 원본 좌표 복원

### 품질 검증
- [ ] 이미지 선명도 유지
- [ ] 색상 왜곡 없음
- [ ] 파일 크기 적절 (<200MB)
