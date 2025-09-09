# Phase 2 개발 계획: 특징 추출 및 기물 탐지 모델 구축

## 📋 개요

Phase 1에서 구축한 기본 시스템을 바탕으로 고급 특징 추출 및 기물 탐지 모델을 개발합니다. 
샘플 데이터 분석 결과에 따라 맞춤형 개발 전략을 제시합니다.

---

## 🎯 Phase 2 목표

### 주요 목표
1. **특징 추출 시스템 구축**: HOG, LBP, Gabor, SfS 알고리즘 구현
2. **데이터 증강 시스템**: 불균형 데이터 해결을 위한 증강 기법
3. **탐지 모델 개발**: 전통적 ML부터 딥러닝까지 다양한 모델
4. **성능 최적화**: 실시간 처리 가능한 효율적 시스템
5. **실 데이터 검증**: 모의 데이터와 실 데이터 정합성 검증

### 성능 지표 목표
- **탐지 정확도**: 85% 이상
- **False Positive Rate**: 10% 이하  
- **처리 속도**: 실시간 처리 가능
- **메모리 효율성**: 대용량 데이터 안정 처리

---

## 🔄 샘플 분석 결과별 계획 분기

### 시나리오 A: 고품질 데이터 (SNR > 15dB, 좌표 매핑 정확)
**→ 고급 딥러닝 중심 개발**

### 시나리오 B: 중간 품질 데이터 (SNR 10-15dB, 부분적 매핑 이슈)
**→ 하이브리드 접근법**

### 시나리오 C: 저품질 데이터 (SNR < 10dB, 매핑 문제)
**→ 기초 안정화 우선**

---

## 📊 시나리오 A: 고급 딥러닝 중심 개발

### Phase 2A-1: 고급 특징 추출 시스템 (2주)

#### 구현 모듈
```python
# 1. HOG 특징 추출기 고도화
class AdvancedHOGExtractor:
    def __init__(self):
        self.multi_scale_configs = [
            {'orientations': 9, 'pixels_per_cell': (8, 8)},
            {'orientations': 12, 'pixels_per_cell': (16, 16)},
            {'orientations': 6, 'pixels_per_cell': (4, 4)}
        ]
    
    def extract_multiscale_features(self, image):
        """다중 스케일 HOG 특징 추출"""
        pass

# 2. 적응형 LBP 추출기
class AdaptiveLBPExtractor:
    def __init__(self):
        self.terrain_adaptive_configs = {
            'sand': {'radius': 1, 'n_points': 8},
            'mud': {'radius': 2, 'n_points': 16}, 
            'rock': {'radius': 1, 'n_points': 8}
        }
    
    def extract_terrain_adaptive_features(self, image, terrain_type):
        """지형별 적응형 LBP 특징 추출"""
        pass

# 3. 최적화된 Gabor 필터 뱅크
class OptimizedGaborBank:
    def __init__(self):
        self.frequency_range = np.logspace(-2, -0.5, 6)  # 6개 주파수
        self.orientation_range = np.arange(0, 180, 22.5)  # 8방향
    
    def extract_gabor_responses(self, image):
        """최적화된 Gabor 응답 추출"""
        pass
```

#### 주요 작업
- [ ] **다중 스케일 HOG**: 다양한 크기의 기물 탐지
- [ ] **적응형 LBP**: 지형별 최적화된 텍스처 분석
- [ ] **Gabor 필터 뱅크**: 방향성 특징 강화
- [ ] **SfS 고도화**: 3D 형상 정보 활용
- [ ] **특징 융합**: PCA, t-SNE 기반 차원 축소

#### 예상 결과
- 특징 벡터 차원: ~500-1000
- 클래스 분리도: Silhouette Score > 0.6
- 특징 추출 속도: <0.1초/패치

### Phase 2A-2: 딥러닝 모델 구축 (3주)

#### CNN 기반 탐지 모델
```python
class SidescanCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1: Low-level features
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: Mid-level features  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: High-level features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

#### 주요 작업
- [ ] **CNN 아키텍처 설계**: ResNet, EfficientNet 기반 백본
- [ ] **어텐션 메커니즘**: 중요한 영역에 집중
- [ ] **데이터 증강**: 회전, 노이즈, 밝기 조정 등
- [ ] **손실 함수 최적화**: Focal Loss로 불균형 해결
- [ ] **앙상블 모델**: 여러 모델 결합으로 성능 향상

#### 성능 목표
- **정확도**: 90% 이상
- **Precision**: 88% 이상
- **Recall**: 85% 이상
- **추론 속도**: <50ms/이미지

### Phase 2A-3: 실시간 처리 시스템 (1주)

#### 최적화된 파이프라인
```python
class RealTimeDetectionPipeline:
    def __init__(self, model_path, device='cuda'):
        self.model = self.load_optimized_model(model_path)
        self.device = device
        self.batch_processor = BatchProcessor(batch_size=32)
        
    def process_stream(self, xtf_stream):
        """스트리밍 XTF 데이터 실시간 처리"""
        for batch in self.batch_processor.get_batches(xtf_stream):
            # 전처리
            preprocessed = self.preprocess_batch(batch)
            
            # 추론
            with torch.no_grad():
                predictions = self.model(preprocessed)
            
            # 후처리
            results = self.postprocess_results(predictions)
            
            yield results
```

---

## 🔀 시나리오 B: 하이브리드 접근법

### Phase 2B-1: 안정화된 특징 추출 (3주)

#### 검증된 알고리즘 우선
- **HOG + LBP 조합**: 가장 안정적인 성능
- **단순 Gabor**: 주요 방향성만 추출
- **통계적 특징**: Mean, Std, Skewness 등
- **형태학적 특징**: 면적, 둘레, 원형도 등

```python
class StableFeatureExtractor:
    def __init__(self):
        self.hog_extractor = HOGExtractor(orientations=9)
        self.lbp_extractor = LBPExtractor(radius=1, n_points=8)
        
    def extract_combined_features(self, image_patch):
        """안정적인 특징 조합 추출"""
        features = []
        
        # HOG 특징
        hog_features = self.hog_extractor.extract(image_patch)
        features.extend(hog_features)
        
        # LBP 특징
        lbp_features = self.lbp_extractor.extract(image_patch)
        features.extend(lbp_features)
        
        # 통계적 특징
        stat_features = self.extract_statistical_features(image_patch)
        features.extend(stat_features)
        
        return np.array(features)
```

### Phase 2B-2: 전통적 머신러닝 모델 (2주)

#### 앙상블 분류기
```python
class EnsembleClassifier:
    def __init__(self):
        self.models = {
            'svm': SVC(kernel='rbf', probability=True),
            'rf': RandomForestClassifier(n_estimators=200),
            'gb': GradientBoostingClassifier(n_estimators=100),
            'xgb': XGBClassifier()
        }
        
    def fit_ensemble(self, X, y):
        """앙상블 모델 학습"""
        for name, model in self.models.items():
            model.fit(X, y)
            
    def predict_ensemble(self, X):
        """앙상블 예측"""
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # 평균 앙상블
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int), ensemble_pred
```

#### 성능 목표
- **정확도**: 80-85%
- **안정성**: 다양한 데이터에서 일관된 성능
- **해석가능성**: 특징 중요도 분석 가능

### Phase 2B-3: 점진적 딥러닝 도입 (3주)

#### 단순한 CNN부터 시작
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
```

---

## ⚠️ 시나리오 C: 기초 안정화 우선

### Phase 2C-1: 데이터 품질 개선 (4주)

#### 고급 전처리 기법
```python
class AdvancedPreprocessor:
    def __init__(self):
        self.noise_reducers = {
            'wiener': self.wiener_filter,
            'bilateral': self.bilateral_filter,
            'non_local_means': self.non_local_means_filter
        }
        
    def adaptive_enhancement(self, data):
        """적응형 품질 향상"""
        # 1. 노이즈 레벨 추정
        noise_level = self.estimate_noise_level(data)
        
        # 2. 노이즈 레벨에 따른 적응형 처리
        if noise_level > 0.3:
            data = self.aggressive_denoising(data)
        elif noise_level > 0.1:
            data = self.moderate_denoising(data)
        else:
            data = self.light_denoising(data)
            
        return data
```

### Phase 2C-2: 좌표 시스템 재구축 (3주)

#### 강건한 매핑 시스템
```python
class RobustCoordinateMapper:
    def __init__(self):
        self.projection_methods = ['utm', 'mercator', 'local_tangent']
        self.validation_threshold = 100  # 100m 오차 허용
        
    def auto_calibrate_mapping(self, sonar_data, reference_points):
        """자동 좌표 매핑 보정"""
        best_method = None
        best_error = float('inf')
        
        for method in self.projection_methods:
            mapper = self.create_mapper(method)
            error = self.validate_mapping(mapper, reference_points)
            
            if error < best_error:
                best_error = error
                best_method = method
                
        return best_method, best_error
```

### Phase 2C-3: 단순 탐지 알고리즘 (3주)

#### 임계값 기반 탐지
```python
class SimpleDetector:
    def __init__(self):
        self.adaptive_threshold = AdaptiveThreshold()
        self.morphological_ops = MorphologicalOperations()
        
    def detect_targets(self, intensity_data):
        """단순하지만 안정적인 탐지"""
        # 1. 적응형 임계값 적용
        binary_mask = self.adaptive_threshold.apply(intensity_data)
        
        # 2. 형태학적 연산으로 노이즈 제거
        cleaned_mask = self.morphological_ops.open(binary_mask)
        cleaned_mask = self.morphological_ops.close(cleaned_mask)
        
        # 3. 연결 성분 분석
        components = self.find_connected_components(cleaned_mask)
        
        # 4. 크기 기반 필터링
        targets = self.filter_by_size(components, min_area=50, max_area=500)
        
        return targets
```

---

## 📅 상세 개발 일정

### 공통 Phase 2 일정표

| 주차 | 시나리오 A | 시나리오 B | 시나리오 C |
|------|-----------|-----------|-----------|
| 1-2주 | 고급 특징 추출 | 안정화 특징 추출 | 데이터 품질 개선 |
| 3-4주 | CNN 모델 구축 | 머신러닝 모델 | 데이터 품질 개선 |
| 5-6주 | 딥러닝 최적화 | 점진적 CNN 도입 | 좌표 시스템 재구축 |
| 7-8주 | 실시간 시스템 | 점진적 CNN 도입 | 좌표 시스템 재구축 |
| 9-10주 | 성능 평가 | 시스템 통합 | 단순 탐지 알고리즘 |
| 11-12주 | 최종 검증 | 성능 평가 | 단순 탐지 알고리즘 |

### 주간별 상세 계획

#### 1-2주차: 기반 시스템 구축
```bash
Week 1:
- [ ] 특징 추출 모듈 아키텍처 설계
- [ ] HOG 추출기 구현 및 테스트
- [ ] LBP 추출기 구현 및 테스트
- [ ] 기본 성능 벤치마크

Week 2:
- [ ] Gabor 필터 구현
- [ ] SfS 알고리즘 구현 (시나리오 A만)
- [ ] 특징 융합 시스템
- [ ] 단위 테스트 완료
```

#### 3-4주차: 모델 개발
```bash
Week 3:
- [ ] 데이터 증강 시스템 구축
- [ ] 불균형 데이터 처리 전략
- [ ] 모델 아키텍처 구현
- [ ] 초기 학습 실험

Week 4:
- [ ] 하이퍼파라미터 최적화
- [ ] 교차 검증 시스템
- [ ] 성능 메트릭 분석
- [ ] 중간 평가 리포트
```

---

## 🔍 성능 평가 계획

### 평가 지표
```python
class PerformanceEvaluator:
    def __init__(self):
        self.metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'auc_roc', 'average_precision',
            'intersection_over_union',
            'false_positive_rate', 'false_negative_rate'
        ]
        
    def comprehensive_evaluation(self, model, test_data):
        """종합적인 성능 평가"""
        results = {}
        
        # 기본 분류 메트릭
        predictions = model.predict(test_data.X)
        results.update(self.calculate_classification_metrics(
            test_data.y, predictions
        ))
        
        # 공간적 정확도 (바운딩 박스가 있는 경우)
        if hasattr(test_data, 'bboxes'):
            spatial_metrics = self.calculate_spatial_metrics(
                test_data.bboxes, model.predict_bboxes(test_data.X)
            )
            results.update(spatial_metrics)
            
        # 속도 벤치마크
        speed_metrics = self.benchmark_speed(model, test_data.X)
        results.update(speed_metrics)
        
        return results
```

### 벤치마크 데이터셋
- **샘플 데이터**: 초기 개발 및 디버깅
- **검증 데이터**: 하이퍼파라미터 튜닝
- **테스트 데이터**: 최종 성능 평가
- **실제 운용 데이터**: 실전 성능 검증

---

## 🚀 배포 준비

### Phase 2 말기 (11-12주차)

#### Docker 컨테이너화
```dockerfile
FROM python:3.8-slim

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY src/ /app/src/
COPY models/ /app/models/

# 실행 환경 설정
WORKDIR /app
EXPOSE 8000

CMD ["python", "-m", "src.api.main"]
```

#### REST API 구현
```python
from fastapi import FastAPI, File, UploadFile
from src.models.detector import SidescanDetector

app = FastAPI(title="Sidescan Sonar Target Detection API")
detector = SidescanDetector.load_from_checkpoint("models/best_model.pth")

@app.post("/detect")
async def detect_targets(file: UploadFile = File(...)):
    """XTF 파일에서 기물 탐지"""
    # XTF 파일 처리
    xtf_data = await process_xtf_file(file)
    
    # 탐지 수행
    detections = detector.detect(xtf_data)
    
    return {
        "num_detections": len(detections),
        "detections": detections,
        "confidence_scores": [d.confidence for d in detections]
    }
```

---

## 📊 위험 관리 및 대안 계획

### 주요 위험 요소
1. **데이터 품질 이슈**: 예상보다 낮은 SNR
2. **좌표 매핑 실패**: 부정확한 위치 정보
3. **성능 목표 미달**: 탐지 정확도 부족
4. **처리 속도 문제**: 실시간 처리 불가

### 대응 전략
```python
class RiskMitigationPlan:
    def __init__(self):
        self.fallback_strategies = {
            'low_snr': self.apply_aggressive_denoising,
            'mapping_failure': self.use_manual_calibration,
            'low_accuracy': self.reduce_complexity_increase_data,
            'slow_processing': self.optimize_inference_pipeline
        }
        
    def assess_and_mitigate(self, current_performance):
        """현재 성능 평가 후 위험 대응"""
        risks = self.identify_risks(current_performance)
        
        for risk in risks:
            mitigation_strategy = self.fallback_strategies.get(risk)
            if mitigation_strategy:
                mitigation_strategy()
```

### 성공 기준 및 체크포인트

#### 2주차 체크포인트
- [ ] 특징 추출 파이프라인 완성
- [ ] 기본 분류 정확도 > 70%
- [ ] 처리 속도 < 1초/패치

#### 4주차 체크포인트  
- [ ] 모델 학습 완료
- [ ] 검증 정확도 > 75%
- [ ] 메모리 사용량 < 4GB

#### 6주차 체크포인트
- [ ] 최적화 완료
- [ ] 테스트 정확도 > 목표치
- [ ] 실시간 처리 가능

#### 최종 검증 (12주차)
- [ ] 모든 성능 목표 달성
- [ ] 실제 데이터 검증 완료
- [ ] 배포 준비 완료

---

## 🎓 학습 자료 및 참고문헌

### 추천 논문
1. "Deep Learning for Side-scan Sonar Image Classification" (2020)
2. "Automatic Target Recognition in Side-scan Sonar Images" (2019)
3. "Feature Fusion for Underwater Object Detection" (2021)

### 오픈소스 참고 프로젝트
- **PyTorch Sonar**: 소나 데이터 처리 라이브러리
- **OpenCV Marine**: 해양 이미지 처리 도구
- **SciKit-Sonar**: 전통적 특징 추출 알고리즘

### 개발 도구
- **MLflow**: 실험 관리 및 모델 버전 관리
- **Weights & Biases**: 하이퍼파라미터 최적화
- **TensorBoard**: 학습 과정 시각화
- **Docker**: 컨테이너 기반 배포

---

이 Phase 2 개발 계획은 샘플 데이터 분석 결과에 따라 최적화된 전략을 제시하며, 각 시나리오별로 현실적이고 달성 가능한 목표를 설정했습니다. 정기적인 체크포인트를 통해 진행 상황을 점검하고 필요시 전략을 조정할 수 있도록 구성되어 있습니다.