# 사용 가이드

## 빠른 시작

### 검증된 데이터 로드 및 확인

프로젝트에는 이미 검증된 라벨 데이터가 포함되어 있습니다:

```python
import numpy as np
import json

# 검증된 flipped 데이터 로드
data_path = 'verified_data/flipped_20251104/flipped_labeled_intensity_data.npz'
data = np.load(data_path, allow_pickle=True)

# 데이터 추출
intensity = data['intensity']  # (5137, 6400) 강도 데이터
labels = data['labels']        # (5137, 6400) 라벨 마스크
metadata = json.loads(str(data['metadata']))  # 25개 기뢰 정보

print(f"✓ 강도 데이터: {intensity.shape}")
print(f"✓ 라벨 마스크: {labels.shape}")
print(f"✓ 기뢰 개수: {len(metadata)}")
print(f"✓ 기뢰 픽셀: {(labels == 1).sum():,} / {labels.size:,}")
```

---

## 데이터 사용법

### 1. NPZ 라벨 데이터 로드

```python
import numpy as np
import json
from pathlib import Path

def load_labeled_data(npz_path):
    """라벨 데이터 로드"""
    data = np.load(npz_path, allow_pickle=True)

    return {
        'intensity': data['intensity'],
        'labels': data['labels'],
        'metadata': json.loads(str(data['metadata']))
    }

# 사용
data = load_labeled_data('verified_data/flipped_20251104/flipped_labeled_intensity_data.npz')
```

### 2. 바운딩 박스 좌표 사용

```python
# 첫 번째 기뢰의 좌표
mine_1 = data['metadata'][0]
bbox = mine_1['mapped_npy']

print(f"기뢰 #1:")
print(f"  위치: ({bbox['xmin']}, {bbox['ymin']})")
print(f"  크기: {bbox['width']} × {bbox['height']}")

# 바운딩 박스 영역 추출
mine_intensity = data['intensity'][
    bbox['ymin']:bbox['ymax'],
    bbox['xmin']:bbox['xmax']
]

print(f"  추출된 영역: {mine_intensity.shape}")
```

### 3. 픽셀 마스크 사용

```python
# 특정 위치가 기뢰인지 확인
y, x = 1070, 4868
is_mine = (data['labels'][y, x] == 1)
print(f"({x}, {y}) 위치: {'기뢰' if is_mine else '배경'}")

# 모든 기뢰 픽셀 추출
mine_pixels = data['intensity'][data['labels'] == 1]
background_pixels = data['intensity'][data['labels'] == 0]

print(f"기뢰 평균 강도: {mine_pixels.mean():.4f}")
print(f"배경 평균 강도: {background_pixels.mean():.4f}")
```

### 4. 시각화

```python
import matplotlib.pyplot as plt

# 전체 데이터 + 라벨 오버레이
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 강도
axes[0].imshow(data['intensity'], cmap='gray', aspect='auto')
axes[0].set_title('강도 데이터')

# 라벨 마스크
axes[1].imshow(data['labels'], cmap='hot', aspect='auto')
axes[1].set_title('라벨 마스크')

# 오버레이
axes[2].imshow(data['intensity'], cmap='gray', aspect='auto')
axes[2].imshow(data['labels'], cmap='Reds', alpha=0.3, aspect='auto')
axes[2].set_title('오버레이')

plt.tight_layout()
plt.savefig('visualization.png', dpi=150)
print("✓ 시각화 저장: visualization.png")
```

---

## 머신러닝 학습용 데이터셋

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class MineDataset(Dataset):
    def __init__(self, npz_path, patch_size=128):
        data = np.load(npz_path, allow_pickle=True)
        self.intensity = data['intensity']
        self.labels = data['labels']
        self.metadata = json.loads(str(data['metadata']))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        mine_info = self.metadata[idx]
        bbox = mine_info['mapped_npy']

        # 중심점 계산
        center_y = (bbox['ymin'] + bbox['ymax']) // 2
        center_x = (bbox['xmin'] + bbox['xmax']) // 2

        # Patch 추출
        half = self.patch_size // 2
        y_start = max(0, center_y - half)
        y_end = min(self.intensity.shape[0], center_y + half)
        x_start = max(0, center_x - half)
        x_end = min(self.intensity.shape[1], center_x + half)

        patch = self.intensity[y_start:y_end, x_start:x_end]
        label = self.labels[y_start:y_end, x_start:x_end]

        # Tensor 변환
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # (1, H, W)
        label = torch.from_numpy(label).long()  # (H, W)

        return {'image': patch, 'label': label, 'mine_id': idx}

# 사용
dataset = MineDataset('verified_data/flipped_20251104/flipped_labeled_intensity_data.npz')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"✓ 데이터셋 크기: {len(dataset)}")

for batch in dataloader:
    images = batch['image']  # (4, 1, 128, 128)
    labels = batch['label']  # (4, 128, 128)
    print(f"  배치 shape: {images.shape}")
    break
```

### TensorFlow/Keras Dataset

```python
import tensorflow as tf
import numpy as np
import json

def create_tf_dataset(npz_path, batch_size=4):
    """TensorFlow Dataset 생성"""
    data = np.load(npz_path, allow_pickle=True)
    intensity = data['intensity']
    labels = data['labels']
    metadata = json.loads(str(data['metadata']))

    # Patch 추출
    patches = []
    patch_labels = []

    for mine_info in metadata:
        bbox = mine_info['mapped_npy']

        patch = intensity[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
        label = labels[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]

        patches.append(patch[..., np.newaxis])  # (H, W, 1)
        patch_labels.append(label)

    patches = np.array(patches)
    patch_labels = np.array(patch_labels)

    # TF Dataset 생성
    dataset = tf.data.Dataset.from_tensor_slices((patches, patch_labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# 사용
dataset = create_tf_dataset('verified_data/flipped_20251104/flipped_labeled_intensity_data.npz')

for images, labels in dataset:
    print(f"✓ 배치 shape: {images.shape}")
    break
```

---

## 고급 사용법

### 1. 특정 영역 샘플링

```python
def extract_background_samples(intensity, labels, num_samples=100, patch_size=128):
    """배경 영역 랜덤 샘플링"""
    h, w = intensity.shape
    half = patch_size // 2

    samples = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(samples) < num_samples and attempts < max_attempts:
        # 랜덤 위치
        y = np.random.randint(half, h - half)
        x = np.random.randint(half, w - half)

        # 해당 영역이 배경인지 확인
        patch_labels = labels[y-half:y+half, x-half:x+half]

        if patch_labels.sum() == 0:  # 기뢰 픽셀이 없으면
            patch = intensity[y-half:y+half, x-half:x+half]
            samples.append(patch)

        attempts += 1

    return np.array(samples)

# 사용
background_samples = extract_background_samples(
    data['intensity'],
    data['labels'],
    num_samples=125  # 기뢰 25개 × 5
)
print(f"✓ 배경 샘플: {background_samples.shape}")
```

### 2. 데이터 증강

```python
import albumentations as A

# 증강 파이프라인 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),
])

def augment_sample(image, mask):
    """샘플 증강"""
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# 사용
mine_1 = data['metadata'][0]
bbox = mine_1['mapped_npy']

original_image = data['intensity'][bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
original_mask = data['labels'][bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]

aug_image, aug_mask = augment_sample(original_image, original_mask)

print(f"✓ 원본: {original_image.shape}")
print(f"✓ 증강: {aug_image.shape}")
```

### 3. 통계 분석

```python
def analyze_dataset(labeled_data):
    """데이터셋 통계 분석"""
    intensity = labeled_data['intensity']
    labels = labeled_data['labels']
    metadata = labeled_data['metadata']

    stats = {
        'total_pixels': labels.size,
        'mine_pixels': (labels == 1).sum(),
        'background_pixels': (labels == 0).sum(),
        'mine_ratio': (labels == 1).sum() / labels.size,
        'num_mines': len(metadata),
        'intensity_stats': {
            'mine': {
                'mean': intensity[labels == 1].mean(),
                'std': intensity[labels == 1].std(),
                'min': intensity[labels == 1].min(),
                'max': intensity[labels == 1].max(),
            },
            'background': {
                'mean': intensity[labels == 0].mean(),
                'std': intensity[labels == 0].std(),
                'min': intensity[labels == 0].min(),
                'max': intensity[labels == 0].max(),
            }
        },
        'bbox_stats': {
            'width': [m['mapped_npy']['width'] for m in metadata],
            'height': [m['mapped_npy']['height'] for m in metadata],
        }
    }

    return stats

# 사용
stats = analyze_dataset(data)

print("=== 데이터셋 통계 ===")
print(f"총 픽셀: {stats['total_pixels']:,}")
print(f"기뢰 픽셀: {stats['mine_pixels']:,} ({stats['mine_ratio']*100:.2f}%)")
print(f"기뢰 개수: {stats['num_mines']}")
print(f"\n기뢰 강도 - 평균: {stats['intensity_stats']['mine']['mean']:.4f}, "
      f"표준편차: {stats['intensity_stats']['mine']['std']:.4f}")
print(f"배경 강도 - 평균: {stats['intensity_stats']['background']['mean']:.4f}, "
      f"표준편차: {stats['intensity_stats']['background']['std']:.4f}")
```

---

## 문제 해결

### Q1: NPZ 파일 로드 오류

**증상**: `ValueError: Object arrays cannot be loaded when allow_pickle=False`

**해결**:
```python
# allow_pickle=True 추가
data = np.load('path/to/file.npz', allow_pickle=True)
```

### Q2: 메타데이터 파싱 오류

**증상**: JSON 파싱 실패

**해결**:
```python
# str() 변환 추가
metadata = json.loads(str(data['metadata']))

# 또는 item() 사용
if data['metadata'].ndim == 0:
    metadata = json.loads(data['metadata'].item())
```

### Q3: 메모리 부족

**증상**: 전체 데이터 로드 시 메모리 부족

**해결**:
```python
# 필요한 부분만 로드
data = np.load('file.npz', allow_pickle=True)
labels = data['labels']  # 라벨만 로드
# intensity는 나중에 필요할 때 로드
```

---

## 다음 단계

- 데이터 증강 및 샘플링 가이드: (작성 예정)
- 모델 학습 예제: (작성 예정)
- API 레퍼런스: `docs/API.md`

---

## 지원

사용 중 문의사항은 프로젝트 관리자에게 연락하세요.

상세한 라벨 구조 설명: [docs/HOW_TO_USE_NPZ_LABELS.md](docs/HOW_TO_USE_NPZ_LABELS.md)

---

**최종 업데이트**: 2025-11-04
