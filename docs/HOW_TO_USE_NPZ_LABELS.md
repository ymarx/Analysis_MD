# NPZ 라벨 데이터 사용법

## 라벨의 형태

### ✅ 두 가지 형태가 모두 포함되어 있습니다!

```python
# NPZ 파일 구조
{
    'intensity': (5137, 6400) float32 배열  # 원본 강도 데이터
    'labels': (5137, 6400) uint8 배열       # 픽셀별 마스크 (0 또는 1)
    'metadata': JSON 문자열                  # 바운딩 박스 픽셀 좌표
}
```

### 1️⃣ 바운딩 박스 픽셀 좌표 (metadata에 저장)

```python
metadata[0]['mapped_npy'] = {
    'xmin': 4868,    # 좌상단 X 픽셀 좌표
    'ymin': 1070,    # 좌상단 Y 픽셀 좌표
    'xmax': 5187,    # 우하단 X 픽셀 좌표
    'ymax': 1119,    # 우하단 Y 픽셀 좌표
    'width': 319,
    'height': 49
}
```
- **25개 기뢰** 각각의 바운딩 박스 좌표
- NPY 데이터 상의 **픽셀 좌표값**

### 2️⃣ 픽셀별 마스크 (labels 배열)

```python
labels[y, x] = 0  # 배경 픽셀
labels[y, x] = 1  # 기뢰 픽셀
```
- 바운딩 박스 좌표를 사용해 생성한 **마스크**
- (5137, 6400) 크기의 전체 배열

### 🔗 두 형태의 관계

```python
# metadata의 좌표로 labels 마스크를 만들었음
bbox = metadata[0]['mapped_npy']  # 좌표 가져오기
labels[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = 1  # 마킹

# 확인
print(labels[1070, 4868])  # 1 (기뢰 - bbox 안)
print(labels[0, 0])        # 0 (배경 - bbox 밖)
```

### 💡 언제 무엇을 사용?

| 사용 목적 | 사용할 데이터 |
|----------|-------------|
| 바운딩 박스 좌표 필요 | `metadata` |
| 픽셀 단위 마스킹 필요 | `labels` 배열 |
| 영역 크롭 | `metadata` 좌표 사용 |
| 기뢰 픽셀만 추출 | `labels` 마스크 사용 |
| Object Detection | `metadata` 좌표 |
| Semantic Segmentation | `labels` 배열 |

---

## 사용 예시

### 1. 기본 로드

```python
import numpy as np
import json

# NPZ 파일 로드
data = np.load('flipped_labeled_intensity_data.npz', allow_pickle=True)

# 데이터 추출
intensity = data['intensity']  # (5137, 6400) - 강도 데이터
labels = data['labels']        # (5137, 6400) - 0 또는 1
metadata = json.loads(str(data['metadata']))  # 25개 기뢰 정보
```

### 2. 특정 위치가 기뢰인지 확인

```python
# 방법 1: 직접 인덱싱
y, x = 1070, 4868
is_mine = (labels[y, x] == 1)
print(f"({x}, {y}) 위치는 기뢰? {is_mine}")  # True

# 방법 2: 여러 위치 확인
positions = [(1070, 4868), (0, 0), (1100, 5000)]
for y, x in positions:
    is_mine = (labels[y, x] == 1)
    print(f"({x}, {y}): {'기뢰' if is_mine else '배경'}")
```

### 3. 기뢰 영역만 추출

```python
# 모든 기뢰 픽셀의 강도값
mine_pixels = intensity[labels == 1]
print(f"기뢰 픽셀 개수: {len(mine_pixels)}")
print(f"기뢰 평균 강도: {mine_pixels.mean()}")

# 배경 픽셀의 강도값
background_pixels = intensity[labels == 0]
print(f"배경 픽셀 개수: {len(background_pixels)}")
print(f"배경 평균 강도: {background_pixels.mean()}")
```

### 4. 바운딩 박스 좌표 사용

```python
# 첫 번째 기뢰의 바운딩 박스 좌표
mine_1 = metadata[0]
bbox = mine_1['mapped_npy']

print(f"기뢰 #1 바운딩 박스:")
print(f"  좌상단: ({bbox['xmin']}, {bbox['ymin']})")
print(f"  우하단: ({bbox['xmax']}, {bbox['ymax']})")
print(f"  크기: {bbox['width']} × {bbox['height']}")

# 바운딩 박스 영역 크롭
mine_intensity = intensity[
    bbox['ymin']:bbox['ymax'],
    bbox['xmin']:bbox['xmax']
]
mine_label = labels[
    bbox['ymin']:bbox['ymax'],
    bbox['xmin']:bbox['xmax']
]

print(f"크롭된 강도 데이터: {mine_intensity.shape}")  # (49, 319)
print(f"크롭된 라벨: {mine_label.shape}")  # (49, 319)
```

### 5. 모든 기뢰 순회

```python
for i, mine_info in enumerate(metadata):
    bbox = mine_info['mapped_npy']

    # 바운딩 박스 영역 추출
    mine_patch = intensity[
        bbox['ymin']:bbox['ymax'],
        bbox['xmin']:bbox['xmax']
    ]

    # 라벨 영역 추출
    label_patch = labels[
        bbox['ymin']:bbox['ymax'],
        bbox['xmin']:bbox['xmax']
    ]

    # 통계
    mine_pixel_count = (label_patch == 1).sum()

    print(f"기뢰 #{i+1}:")
    print(f"  위치: ({bbox['xmin']}, {bbox['ymin']})")
    print(f"  크기: {mine_patch.shape}")
    print(f"  기뢰 픽셀: {mine_pixel_count}")
    print(f"  평균 강도: {mine_patch[label_patch == 1].mean():.4f}")
```

---

## 머신러닝 학습용 데이터셋 만들기

### 예시 1: Patch 추출

```python
import numpy as np

def extract_mine_patches(intensity, labels, metadata, patch_size=128):
    """기뢰 중심으로 patch 추출"""
    patches = []
    patch_labels = []

    for mine_info in metadata:
        bbox = mine_info['mapped_npy']

        # 중심점
        center_y = (bbox['ymin'] + bbox['ymax']) // 2
        center_x = (bbox['xmin'] + bbox['xmax']) // 2

        # Patch 영역
        half = patch_size // 2
        y_start = max(0, center_y - half)
        y_end = min(intensity.shape[0], center_y + half)
        x_start = max(0, center_x - half)
        x_end = min(intensity.shape[1], center_x + half)

        # 추출
        patch = intensity[y_start:y_end, x_start:x_end]
        label_patch = labels[y_start:y_end, x_start:x_end]

        # 크기 조정 (필요시)
        if patch.shape != (patch_size, patch_size):
            # 패딩 또는 리사이즈
            pass

        patches.append(patch)
        patch_labels.append(label_patch)

    return np.array(patches), np.array(patch_labels)

# 사용
mine_patches, mine_labels = extract_mine_patches(intensity, labels, metadata)
print(f"추출된 패치: {mine_patches.shape}")
```

### 예시 2: PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset

class MineDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.intensity = data['intensity']
        self.labels = data['labels']
        self.metadata = json.loads(str(data['metadata']))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        mine_info = self.metadata[idx]
        bbox = mine_info['mapped_npy']

        # 바운딩 박스 영역 추출
        patch = self.intensity[
            bbox['ymin']:bbox['ymax'],
            bbox['xmin']:bbox['xmax']
        ]
        label = self.labels[
            bbox['ymin']:bbox['ymax'],
            bbox['xmin']:bbox['xmax']
        ]

        # Tensor 변환
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(label).long()  # (H, W)

        return {
            'image': patch,
            'label': label,
            'bbox': bbox,
            'mine_id': idx
        }

# 사용
dataset = MineDataset('flipped_labeled_intensity_data.npz')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    images = batch['image']  # (4, 1, H, W)
    labels = batch['label']  # (4, H, W)
    # 학습...
```

---

## 시각화

### 라벨 오버레이

```python
import matplotlib.pyplot as plt

# 전체 이미지에 라벨 오버레이
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 강도
axes[0].imshow(intensity, cmap='gray', aspect='auto')
axes[0].set_title('원본 강도 데이터')

# 라벨 마스크
axes[1].imshow(labels, cmap='hot', aspect='auto')
axes[1].set_title('라벨 마스크 (0=배경, 1=기뢰)')

# 오버레이
axes[2].imshow(intensity, cmap='gray', aspect='auto')
axes[2].imshow(labels, cmap='Reds', alpha=0.3, aspect='auto')
axes[2].set_title('강도 데이터 + 라벨 오버레이')

plt.tight_layout()
plt.savefig('label_visualization.png', dpi=150)
```

### 개별 기뢰 시각화

```python
# 첫 번째 기뢰
mine_info = metadata[0]
bbox = mine_info['mapped_npy']

# 영역 추출
mine_intensity = intensity[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
mine_label = labels[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(mine_intensity, cmap='gray')
axes[0].set_title('기뢰 강도 데이터')

axes[1].imshow(mine_intensity, cmap='gray')
axes[1].imshow(mine_label, cmap='Reds', alpha=0.5)
axes[1].set_title('강도 + 라벨 오버레이')

plt.tight_layout()
plt.savefig('mine_detail.png', dpi=150)
```

---

## 요약

### 라벨 형태

| 항목 | 설명 |
|------|------|
| **labels 배열** | (5137, 6400) 크기의 픽셀별 마스크 |
| **값** | 0 (배경) 또는 1 (기뢰) |
| **좌표 정보** | metadata JSON에 바운딩 박스 좌표 |
| **사용법** | `labels[y, x]`로 픽셀 클래스 확인 |

### 데이��� 접근 방법

```python
# 1. 픽셀별 클래스 확인
is_mine = (labels[y, x] == 1)

# 2. 기뢰 영역 추출 (마스크 기반)
mine_pixels = intensity[labels == 1]

# 3. 바운딩 박스 크롭 (좌표 기반)
bbox = metadata[0]['mapped_npy']
mine_patch = intensity[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]

# 4. 두 방법 결합
mine_patch = intensity[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
mine_only = mine_patch[labels[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] == 1]
```

**핵심**: 라벨은 좌표가 아니라 **픽셀별 클래스 정보**이며, 좌표는 **metadata에 별도 저장**되어 있습니다.
