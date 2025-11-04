# Feature Extraction Cross-Validation Pipeline
## 검증된 방법론의 특징 추출 적용 가이드

### 핵심 원칙

**데이터 누수 방지**: 모든 전처리와 특징 추출은 Train 데이터로만 학습되고, Val/Test에는 변환만 적용

```
원칙 1: Split FIRST (원본 단위로 분할)
원칙 2: Feature Extraction WITHIN each fold (각 폴드에서 독립적으로)
원칙 3: Fit on Train, Transform on Val/Test (학습/변환 분리)
```

---

## 1. 전체 파이프라인 구조

### 1.1 데이터 흐름도

```
원본 25개 기뢰 샘플
  ↓
[1차 분할] Train(20) | Test(5)
  ↓
[2차 분할] 5-Fold CV on Train(20)
  ↓
Fold 1: Train_원본(16) | Val_원본(4)
  ↓
[증강] Train(16) → Train_증강(160)
  ↓
[배경 샘플링] Background(400) - 독립적
  ↓
[특징 추출] Train(560 patches) → Features
  ↓  HOG_extractor.fit(Train_features)
  ↓  LBP_extractor.fit(Train_features)
  ↓  Gabor_extractor.fit(Train_features)
  ↓
[변환] Val_원본(4) → Val_features (fit 없이 transform만)
  ↓
[학습] SVM.fit(Train_features, y_train)
  ↓
[평가] SVM.predict(Val_features)
```

---

## 2. 코드 구현

### 2.1 필수 임포트

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

# 특징 추출 모듈 (프로젝트 기존 코드)
from src.feature_extraction.multi_scale_hog import MultiScaleHOGExtractor
from src.feature_extraction.comprehensive_lbp import ComprehensiveLBPExtractor
from src.feature_extraction.gabor_features import GaborFeatureExtractor
from src.feature_extraction.shape_from_shading import ShapeFromShadingExtractor

# 증강 엔진 (프로젝트 기존 코드)
from src.data_augmentation.augmentation_engine import SonarAugmentationEngine
```

### 2.2 올바른 Cross-Validation Pipeline

```python
def feature_extraction_cv_pipeline(
    mine_patches: np.ndarray,  # (25, 64, 64) - 원본 기뢰 패치
    background_pool: np.ndarray,  # (10000, 64, 64) - 배경 풀
    feature_extractors: dict,  # {'HOG': extractor, 'LBP': extractor, ...}
    n_folds: int = 5,
    augmentation_factor: int = 10,
    mine_bg_ratio: float = 2.5  # 1:2.5 권장
):
    """
    올바른 특징 추출 Cross-Validation 파이프라인

    핵심:
    1. 원본 단위로 분할 (augmentation 전)
    2. 각 폴드에서 독립적으로 특징 추출기 학습
    3. Val/Test는 원본 그대로 + fit 없이 transform만
    """

    # === STEP 1: Test Set 분리 (원본 단위) ===
    n_total = len(mine_patches)
    n_test = int(n_total * 0.2)  # 5개 (20%)

    np.random.seed(42)
    test_indices = np.random.choice(n_total, n_test, replace=False)
    train_val_indices = np.setdiff1d(np.arange(n_total), test_indices)

    train_val_patches = mine_patches[train_val_indices]  # (20, 64, 64)
    test_patches = mine_patches[test_indices]  # (5, 64, 64)

    print(f"✅ Test set 분리: Train/Val={len(train_val_patches)}, Test={len(test_patches)}")

    # === STEP 2: K-Fold Cross-Validation ===
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_dummy = np.ones(len(train_val_patches))  # Stratified를 위한 더미 레이블

    fold_results = []

    for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(
        skf.split(train_val_patches, y_dummy)
    ):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        # --- 2.1: 원본 추출 ---
        fold_train_patches = train_val_patches[cv_train_idx]  # (16, 64, 64)
        fold_val_patches = train_val_patches[cv_val_idx]  # (4, 64, 64)

        print(f"원본: Train={len(fold_train_patches)}, Val={len(fold_val_patches)}")

        # --- 2.2: Train 증강 (Val은 증강 안 함!) ---
        augmenter = SonarAugmentationEngine(
            rotation_range=(-15, 15),
            gaussian_noise_std=0.05,
            brightness_range=(0.9, 1.1),
            gaussian_blur_sigma=(0, 0.5)
        )

        fold_train_augmented = augment_mine_samples_diverse(
            mine_patches=fold_train_patches,  # (16, 64, 64)
            augmenter=augmenter,
            augmentation_factor=augmentation_factor
        )  # → (160, 64, 64)

        print(f"증강 후: Train={len(fold_train_augmented)}")

        # --- 2.3: 배경 샘플링 (독립적) ---
        n_bg_needed = int(len(fold_train_augmented) * mine_bg_ratio)  # 160 * 2.5 = 400

        fold_bg_train = sample_hard_negatives(
            background_pool=background_pool,
            mine_patches=fold_train_patches,  # Hard Negative Mining용
            n_samples=n_bg_needed
        )  # → (400, 64, 64)

        # Val 배경 샘플링 (Val 기뢰 4개 기준)
        fold_bg_val = sample_hard_negatives(
            background_pool=background_pool,
            mine_patches=fold_val_patches,
            n_samples=int(len(fold_val_patches) * 5)  # 4 * 5 = 20개 (1:5 비율)
        )  # → (20, 64, 64)

        # --- 2.4: Train/Val 데이터셋 구성 ---
        X_train_patches = np.vstack([fold_train_augmented, fold_bg_train])  # (560, 64, 64)
        y_train = np.array([1]*len(fold_train_augmented) + [0]*len(fold_bg_train))

        X_val_patches = np.vstack([fold_val_patches, fold_bg_val])  # (24, 64, 64)
        y_val = np.array([1]*len(fold_val_patches) + [0]*len(fold_bg_val))

        print(f"Train: {X_train_patches.shape}, Val: {X_val_patches.shape}")

        # --- 2.5: 특징 추출 (핵심!) ---
        X_train_features_list = []
        X_val_features_list = []

        for feature_name, extractor in feature_extractors.items():
            print(f"\n  [{feature_name}] 특징 추출 중...")

            # ✅ Train에서 fit + transform
            train_features = extractor.fit_transform(X_train_patches)  # (560, n_features)
            print(f"    Train features: {train_features.shape}")

            # ✅ Val에서 transform ONLY (fit 절대 안 함!)
            val_features = extractor.transform(X_val_patches)  # (24, n_features)
            print(f"    Val features: {val_features.shape}")

            X_train_features_list.append(train_features)
            X_val_features_list.append(val_features)

        # 모든 특징 결합 (HOG + LBP + Gabor + SFS)
        X_train_features = np.hstack(X_train_features_list)  # (560, total_features)
        X_val_features = np.hstack(X_val_features_list)  # (24, total_features)

        print(f"\n결합된 특징: Train={X_train_features.shape}, Val={X_val_features.shape}")

        # --- 2.6: 특징 정규화 (StandardScaler) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)  # Train에서 fit
        X_val_scaled = scaler.transform(X_val_features)  # Val에서 transform만

        # --- 2.7: 모델 학습 및 평가 ---
        svm = SVC(
            kernel='rbf',
            class_weight='balanced',  # 클래스 불균형 대응
            random_state=42
        )

        svm.fit(X_train_scaled, y_train)

        # 평가
        y_val_pred = svm.predict(X_val_scaled)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='binary'
        )

        fold_results.append({
            'fold': fold_idx + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(f"\n✅ Fold {fold_idx + 1} 결과:")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # === STEP 3: 평균 성능 계산 ===
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])

    std_f1 = np.std([r['f1'] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"5-Fold CV 평균 성능:")
    print(f"  Precision: {avg_precision:.3f} ± {np.std([r['precision'] for r in fold_results]):.3f}")
    print(f"  Recall: {avg_recall:.3f} ± {np.std([r['recall'] for r in fold_results]):.3f}")
    print(f"  F1-Score: {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"{'='*60}")

    return fold_results, test_patches
```

### 2.3 특징 추출기 초기화

```python
# 프로젝트의 4가지 특징 추출기 초기화
feature_extractors = {
    'HOG': MultiScaleHOGExtractor(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        scales=[1.0, 1.5, 2.0]
    ),

    'LBP': ComprehensiveLBPExtractor(
        n_points=8,
        radius=1,
        method='uniform'
    ),

    'Gabor': GaborFeatureExtractor(
        frequencies=[0.1, 0.2, 0.3],
        orientations=[0, 45, 90, 135]
    ),

    'SFS': ShapeFromShadingExtractor(
        lambda_param=0.1,
        max_iter=100
    )
}
```

### 2.4 실행 예시

```python
# 데이터 로드 (프로젝트 기존 코드)
mine_patches = load_mine_patches_from_npy()  # (25, 64, 64)
background_pool = load_background_pool()  # (10000, 64, 64)

# 파이프라인 실행
fold_results, test_patches = feature_extraction_cv_pipeline(
    mine_patches=mine_patches,
    background_pool=background_pool,
    feature_extractors=feature_extractors,
    n_folds=5,
    augmentation_factor=10,
    mine_bg_ratio=2.5
)
```

---

## 3. 주요 포인트 정리

### 3.1 왜 이 순서가 중요한가?

| 단계 | 이유 |
|------|------|
| **Split FIRST** | 증강 전 분할로 원본 간 독립성 보장 |
| **Augment WITHIN fold** | 각 폴드마다 다른 증강 → 과적합 방지 |
| **Fit on Train ONLY** | Val/Test 정보가 Train에 누수되면 성능 과대평가 |
| **Val은 원본 그대로** | 실제 성능 평가 (증강된 데이터로 평가하면 의미 없음) |

### 3.2 잘못된 방법 (❌)

```python
# ❌ 잘못된 방법 1: 증강 후 분할
X_augmented = augment(mine_patches)  # (250, 64, 64)
train, val = split(X_augmented)  # 같은 원본의 증강본이 train/val에 섞임!

# ❌ 잘못된 방법 2: Val에서 fit
hog_extractor.fit(X_train)
hog_extractor.fit(X_val)  # Val 정보 누수!

# ❌ 잘못된 방법 3: Scaler fit을 전체 데이터로
scaler.fit(np.vstack([X_train, X_val]))  # Val 정보 누수!
```

### 3.3 올바른 방법 (✅)

```python
# ✅ 올바른 방법
train_patches, val_patches = split(mine_patches)  # 원본 단위 분할
train_augmented = augment(train_patches)  # Train만 증강

# Train에서만 fit
hog_extractor.fit(train_augmented)
scaler.fit(train_features)

# Val에서는 transform만
val_features = hog_extractor.transform(val_patches)  # fit 없음!
val_scaled = scaler.transform(val_features)  # fit 없음!
```

---

## 4. 프로젝트 적용 시 고려사항

### 4.1 데이터 형식

- **NPY 형식 권장**: float32, (7974, 6832), 0.0-1.0 정규화
- **BMP 형식**: uint8, (7974, 1024), 0-255 → ~6% 정보 손실

### 4.2 증강 기법 (Sonar-Specific)

**✅ 안전한 증강**:
- Rotation: ±15°
- Gaussian Noise: std=0.05
- Brightness: 0.9-1.1
- Gaussian Blur: σ=0-0.5

**❌ 피해야 할 증강**:
- Mixup (소나 물리 위반)
- Extreme Brightness (±50%)
- Color Transform (grayscale 소나 이미지)

### 4.3 배경 샘플링 전략

```python
def sample_hard_negatives(background_pool, mine_patches, n_samples):
    """
    Hard Negative Mining: 기뢰와 유사한 어려운 배경 샘플 우선 선택

    70%: 기뢰 근처 패치 (어려운 샘플)
    30%: 무작위 배경 (쉬운 샘플)
    """
    n_hard = int(n_samples * 0.7)
    n_easy = n_samples - n_hard

    # Hard negatives: 기뢰와 높은 유사도
    hard_negatives = find_similar_backgrounds(background_pool, mine_patches, n_hard)

    # Easy negatives: 무작위
    easy_negatives = np.random.choice(background_pool, n_easy, replace=False)

    return np.vstack([hard_negatives, easy_negatives])
```

---

## 5. 검증 체크리스트

### 실행 전 확인사항

- [ ] Test set은 원본 단위로 분리했는가? (증강 전)
- [ ] 각 CV fold마다 독립적으로 증강하는가?
- [ ] Val/Test는 원본 그대로 유지하는가?
- [ ] 특징 추출기는 Train에서만 fit하는가?
- [ ] Scaler는 Train 특징으로만 fit하는가?
- [ ] 배경 샘플링은 각 fold마다 독립적인가?

### 결과 해석

**오버피팅 확인**:
```python
# Train F1 vs Val F1 차이가 10% 이상이면 오버피팅 의심
train_f1 = 0.95
val_f1 = 0.72  # 차이 23% → 오버피팅!

# 해결책:
# 1. 증강 배수 줄이기 (10x → 5x)
# 2. 정규화 강화 (SVM C 값 감소)
# 3. 특징 차원 축소 (PCA)
```

**일반화 성능 확인**:
```python
# Test set 평가 (최종 1회만)
test_features = extract_features(test_patches)  # fit 없이 transform만
test_f1 = evaluate(model, test_features)

# Val F1과 Test F1이 유사하면 일반화 성공
# 차이가 15% 이상이면 Val set 운이 좋았던 것
```

---

## 6. 참고 자료

**검증된 방법론 출처**:
1. Scikit-learn Pipeline Documentation: Feature extraction and CV best practices
2. Cross Validated (Stack Exchange): "Data augmentation and cross-validation order"
3. Computer Vision Best Practices: "Preventing data leakage in CV pipelines"

**핵심 원칙 재확인**:
> "Feature selection must be performed independently within each cross-validation fold to prevent data leakage" - Scikit-learn Docs

> "Split your data first, then do data augmentation on the training set" - Cross Validated

> "Never call fit() on test data - this applies to all preprocessing steps" - ML Best Practices
