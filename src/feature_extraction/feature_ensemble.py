"""
특징 앙상블 및 융합 시스템

다중 특징 추출 방법의 결과를 효과적으로 결합하여 
더 강력하고 로버스트한 특징 벡터를 생성합니다.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    # 특징 결합 방법
    use_concatenation: bool = True
    use_weighted_fusion: bool = True
    use_stacking: bool = True
    use_attention_fusion: bool = False
    
    # 차원 축소 설정
    enable_pca: bool = True
    pca_variance_ratio: float = 0.95
    max_features: int = 1000
    
    # 특징 선택 설정
    feature_selection_method: str = 'mutual_info'  # 'f_test', 'mutual_info', 'random_forest'
    selection_k: int = 500
    
    # 정규화 설정
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # 가중치 학습 설정
    weight_learning_method: str = 'performance_based'  # 'uniform', 'performance_based', 'meta_learning'
    
    # 메타 학습 설정
    meta_learner_type: str = 'logistic'  # 'logistic', 'random_forest', 'svm'


class FeatureNormalizer:
    """특징 정규화 클래스"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
    
    def fit_transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """특징별 정규화 학습 및 적용"""
        normalized_features = {}
        
        for feature_name, features in features_dict.items():
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            normalized = scaler.fit_transform(features)
            normalized_features[feature_name] = normalized
            self.scalers[feature_name] = scaler
            
            logger.info(f"{feature_name} 정규화 완료: {features.shape} -> {normalized.shape}")
        
        return normalized_features
    
    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """학습된 정규화 적용"""
        normalized_features = {}
        
        for feature_name, features in features_dict.items():
            if feature_name in self.scalers:
                normalized = self.scalers[feature_name].transform(features)
                normalized_features[feature_name] = normalized
            else:
                logger.warning(f"{feature_name}에 대한 스케일러가 없습니다")
                normalized_features[feature_name] = features
        
        return normalized_features


class DimensionReducer:
    """차원 축소 클래스"""
    
    def __init__(self, method: str = 'pca', target_variance: float = 0.95):
        self.method = method
        self.target_variance = target_variance
        self.reducers = {}
        self.feature_dims = {}
    
    def fit_transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """특징별 차원 축소 학습 및 적용"""
        reduced_features = {}
        
        for feature_name, features in features_dict.items():
            original_dim = features.shape[1]
            
            if self.method == 'pca':
                pca = PCA()
                pca.fit(features)
                
                # 목표 분산 비율을 만족하는 컴포넌트 수 결정
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.target_variance) + 1
                n_components = min(n_components, min(features.shape) - 1)
                
                # 재학습
                pca_final = PCA(n_components=n_components)
                reduced = pca_final.fit_transform(features)
                
                self.reducers[feature_name] = pca_final
                self.feature_dims[feature_name] = n_components
                
                logger.info(f"{feature_name} PCA: {original_dim} -> {n_components} "
                           f"(분산 비율: {cumsum_variance[n_components-1]:.3f})")
            
            else:
                # 다른 차원 축소 방법 (향후 확장)
                reduced = features
                self.feature_dims[feature_name] = original_dim
            
            reduced_features[feature_name] = reduced
        
        return reduced_features
    
    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """학습된 차원 축소 적용"""
        reduced_features = {}
        
        for feature_name, features in features_dict.items():
            if feature_name in self.reducers:
                reduced = self.reducers[feature_name].transform(features)
                reduced_features[feature_name] = reduced
            else:
                logger.warning(f"{feature_name}에 대한 차원 축소기가 없습니다")
                reduced_features[feature_name] = features
        
        return reduced_features


class FeatureSelector:
    """특징 선택 클래스"""
    
    def __init__(self, method: str = 'mutual_info', k: int = 500):
        self.method = method
        self.k = k
        self.selectors = {}
        self.selected_indices = {}
    
    def fit_transform(self, features_dict: Dict[str, np.ndarray], 
                     labels: np.ndarray) -> Dict[str, np.ndarray]:
        """특징별 특징 선택 학습 및 적용"""
        selected_features = {}
        
        for feature_name, features in features_dict.items():
            n_features = features.shape[1]
            k = min(self.k, n_features)
            
            if self.method == 'f_test':
                selector = SelectKBest(score_func=f_classif, k=k)
            elif self.method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif self.method == 'random_forest':
                # Random Forest 기반 특징 중요도
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(features, labels)
                importance_indices = np.argsort(rf.feature_importances_)[-k:]
                
                selected = features[:, importance_indices]
                selected_features[feature_name] = selected
                self.selected_indices[feature_name] = importance_indices
                
                logger.info(f"{feature_name} Random Forest 특징 선택: {n_features} -> {k}")
                continue
            else:
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            selected = selector.fit_transform(features, labels)
            selected_features[feature_name] = selected
            self.selectors[feature_name] = selector
            
            logger.info(f"{feature_name} {self.method} 특징 선택: {n_features} -> {k}")
        
        return selected_features
    
    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """학습된 특징 선택 적용"""
        selected_features = {}
        
        for feature_name, features in features_dict.items():
            if feature_name in self.selectors:
                selected = self.selectors[feature_name].transform(features)
                selected_features[feature_name] = selected
            elif feature_name in self.selected_indices:
                selected = features[:, self.selected_indices[feature_name]]
                selected_features[feature_name] = selected
            else:
                logger.warning(f"{feature_name}에 대한 특징 선택기가 없습니다")
                selected_features[feature_name] = features
        
        return selected_features


class WeightLearner:
    """특징별 가중치 학습 클래스"""
    
    def __init__(self, method: str = 'performance_based'):
        self.method = method
        self.weights = {}
        self.meta_learner = None
    
    def learn_weights(self, features_dict: Dict[str, np.ndarray], 
                     labels: np.ndarray) -> Dict[str, float]:
        """특징별 가중치 학습"""
        
        if self.method == 'uniform':
            # 동일한 가중치
            n_features = len(features_dict)
            weights = {name: 1.0 / n_features for name in features_dict.keys()}
            
        elif self.method == 'performance_based':
            # 개별 성능 기반 가중치
            weights = {}
            performances = {}
            
            for feature_name, features in features_dict.items():
                try:
                    # 간단한 분류기로 개별 성능 측정
                    classifier = LogisticRegression(random_state=42, max_iter=1000)
                    
                    # 교차 검증으로 성능 측정
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(classifier, features, labels, cv=5, scoring='accuracy')
                    performance = np.mean(scores)
                    performances[feature_name] = performance
                    
                    logger.info(f"{feature_name} 개별 성능: {performance:.4f}")
                    
                except Exception as e:
                    logger.warning(f"{feature_name} 성능 측정 실패: {e}")
                    performances[feature_name] = 0.5
            
            # 성능에 비례한 가중치
            total_performance = sum(performances.values())
            weights = {name: perf / total_performance 
                      for name, perf in performances.items()}
            
        elif self.method == 'meta_learning':
            # 메타 학습 기반 가중치
            weights = self._learn_meta_weights(features_dict, labels)
        
        else:
            # 기본값: 동일 가중치
            n_features = len(features_dict)
            weights = {name: 1.0 / n_features for name in features_dict.keys()}
        
        self.weights = weights
        logger.info(f"학습된 가중치: {weights}")
        
        return weights
    
    def _learn_meta_weights(self, features_dict: Dict[str, np.ndarray], 
                           labels: np.ndarray) -> Dict[str, float]:
        """메타 학습으로 가중치 학습"""
        # 각 특징으로 예측 결과 생성
        predictions = {}
        
        for feature_name, features in features_dict.items():
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(features, labels)
            pred_proba = classifier.predict_proba(features)[:, 1]  # 양성 클래스 확률
            predictions[feature_name] = pred_proba
        
        # 메타 특징 구성 (각 특징의 예측 확률)
        meta_features = np.column_stack(list(predictions.values()))
        
        # 메타 학습기 훈련
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_learner.fit(meta_features, labels)
        
        # 가중치는 메타 학습기의 계수
        coefficients = np.abs(self.meta_learner.coef_[0])
        total_coef = np.sum(coefficients)
        
        weights = {}
        for i, feature_name in enumerate(features_dict.keys()):
            weights[feature_name] = coefficients[i] / total_coef
        
        return weights


class FeatureEnsemble:
    """특징 앙상블 메인 클래스"""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # 구성 요소 초기화
        self.normalizer = FeatureNormalizer(self.config.normalization_method)
        self.dimension_reducer = DimensionReducer('pca', self.config.pca_variance_ratio)
        self.feature_selector = FeatureSelector(self.config.feature_selection_method, 
                                               self.config.selection_k)
        self.weight_learner = WeightLearner(self.config.weight_learning_method)
        
        # 학습된 가중치
        self.feature_weights = {}
        
        # 결과 저장
        self.ensemble_methods = {}
        
        logger.info("특징 앙상블 시스템 초기화 완료")
    
    def fit(self, features_dict: Dict[str, np.ndarray], 
           labels: np.ndarray) -> 'FeatureEnsemble':
        """앙상블 시스템 학습"""
        logger.info("=== 특징 앙상블 학습 시작 ===")
        
        # 1. 특징 정규화
        logger.info("1. 특징 정규화")
        normalized_features = self.normalizer.fit_transform(features_dict)
        
        # 2. 차원 축소 (선택적)
        if self.config.enable_pca:
            logger.info("2. 차원 축소 (PCA)")
            normalized_features = self.dimension_reducer.fit_transform(normalized_features)
        
        # 3. 특징 선택 (선택적)
        if self.config.selection_k > 0:
            logger.info("3. 특징 선택")
            normalized_features = self.feature_selector.fit_transform(normalized_features, labels)
        
        # 4. 가중치 학습
        logger.info("4. 특징 가중치 학습")
        self.feature_weights = self.weight_learner.learn_weights(normalized_features, labels)
        
        # 5. 앙상블 방법별 준비
        self._prepare_ensemble_methods(normalized_features, labels)
        
        logger.info("=== 특징 앙상블 학습 완료 ===")
        return self
    
    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """학습된 앙상블로 특징 변환"""
        
        # 1. 정규화
        normalized_features = self.normalizer.transform(features_dict)
        
        # 2. 차원 축소
        if self.config.enable_pca:
            normalized_features = self.dimension_reducer.transform(normalized_features)
        
        # 3. 특징 선택
        if self.config.selection_k > 0:
            normalized_features = self.feature_selector.transform(normalized_features)
        
        # 4. 앙상블 적용
        ensemble_results = {}
        
        if self.config.use_concatenation:
            ensemble_results['concatenation'] = self._apply_concatenation(normalized_features)
        
        if self.config.use_weighted_fusion:
            ensemble_results['weighted_fusion'] = self._apply_weighted_fusion(normalized_features)
        
        if self.config.use_stacking:
            ensemble_results['stacking'] = self._apply_stacking(normalized_features)
        
        return ensemble_results
    
    def _prepare_ensemble_methods(self, features_dict: Dict[str, np.ndarray], 
                                 labels: np.ndarray):
        """앙상블 방법별 사전 준비"""
        
        # Stacking을 위한 메타 학습기 준비
        if self.config.use_stacking:
            self._prepare_stacking_meta_learner(features_dict, labels)
    
    def _apply_concatenation(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """단순 연결 앙상블"""
        feature_list = []
        
        for feature_name in sorted(features_dict.keys()):
            features = features_dict[feature_name]
            feature_list.append(features)
        
        if feature_list:
            concatenated = np.hstack(feature_list)
            logger.info(f"연결 앙상블: {[f.shape[1] for f in feature_list]} -> {concatenated.shape[1]}")
            return concatenated
        else:
            return np.array([])
    
    def _apply_weighted_fusion(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 융합 앙상블"""
        if not self.feature_weights:
            logger.warning("가중치가 없어 동일 가중치 사용")
            n_features = len(features_dict)
            weights = {name: 1.0 / n_features for name in features_dict.keys()}
        else:
            weights = self.feature_weights
        
        # 모든 특징을 같은 차원으로 맞추기 (평균 풀링)
        normalized_features = {}
        target_dim = min(f.shape[1] for f in features_dict.values()) if features_dict else 0
        
        if target_dim == 0:
            return np.array([])
        
        for feature_name, features in features_dict.items():
            if features.shape[1] > target_dim:
                # PCA로 차원 축소
                from sklearn.decomposition import PCA
                pca = PCA(n_components=target_dim)
                reduced = pca.fit_transform(features)
                normalized_features[feature_name] = reduced
            else:
                normalized_features[feature_name] = features
        
        # 가중합
        weighted_sum = None
        total_weight = 0
        
        for feature_name, features in normalized_features.items():
            weight = weights.get(feature_name, 0)
            if weighted_sum is None:
                weighted_sum = weight * features
            else:
                weighted_sum += weight * features
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        logger.info(f"가중 융합: {len(normalized_features)}개 특징 -> {weighted_sum.shape[1]}차원")
        return weighted_sum
    
    def _apply_stacking(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """스태킹 앙상블"""
        if not hasattr(self, 'stacking_meta_learner'):
            logger.warning("스태킹 메타 학습기가 없습니다")
            return self._apply_concatenation(features_dict)
        
        # 각 특징으로 예측 생성
        predictions = []
        
        for feature_name in sorted(features_dict.keys()):
            features = features_dict[feature_name]
            if feature_name in self.base_predictors:
                pred = self.base_predictors[feature_name].predict_proba(features)[:, 1]
                predictions.append(pred.reshape(-1, 1))
        
        if predictions:
            meta_features = np.hstack(predictions)
            # 메타 학습기로 최종 특징 생성 (여기서는 단순히 메타 특징 반환)
            logger.info(f"스태킹: {len(predictions)}개 예측 -> {meta_features.shape[1]}차원")
            return meta_features
        else:
            return np.array([])
    
    def _prepare_stacking_meta_learner(self, features_dict: Dict[str, np.ndarray], 
                                      labels: np.ndarray):
        """스태킹을 위한 베이스 예측기 및 메타 학습기 준비"""
        self.base_predictors = {}
        
        # 각 특징별 베이스 예측기 훈련
        for feature_name, features in features_dict.items():
            try:
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                classifier.fit(features, labels)
                self.base_predictors[feature_name] = classifier
                
                logger.info(f"스태킹 베이스 예측기 준비: {feature_name}")
                
            except Exception as e:
                logger.warning(f"{feature_name} 베이스 예측기 훈련 실패: {e}")
        
        # 메타 학습기 (여기서는 단순 구현)
        if self.config.meta_learner_type == 'logistic':
            self.stacking_meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif self.config.meta_learner_type == 'random_forest':
            self.stacking_meta_learner = RandomForestClassifier(random_state=42)
        else:
            self.stacking_meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    
    def evaluate_ensemble_methods(self, features_dict: Dict[str, np.ndarray], 
                                 labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """앙상블 방법별 성능 평가"""
        logger.info("앙상블 방법 성능 평가")
        
        # 앙상블 특징 생성
        ensemble_features = self.transform(features_dict)
        
        results = {}
        
        for method_name, features in ensemble_features.items():
            if features.size == 0:
                results[method_name] = {'accuracy': 0.0, 'error': 'Empty features'}
                continue
            
            try:
                # 교차 검증으로 성능 평가
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                
                # 교차 검증 점수
                cv_scores = cross_val_score(classifier, features, labels, cv=5, scoring='accuracy')
                
                # 평균 성능
                results[method_name] = {
                    'accuracy': np.mean(cv_scores),
                    'accuracy_std': np.std(cv_scores),
                    'feature_dim': features.shape[1]
                }
                
                logger.info(f"{method_name}: 정확도 {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"{method_name} 평가 실패: {e}")
                results[method_name] = {'accuracy': 0.0, 'error': str(e)}
        
        return results
    
    def get_best_ensemble_method(self, features_dict: Dict[str, np.ndarray], 
                                labels: np.ndarray) -> Tuple[str, np.ndarray]:
        """최고 성능의 앙상블 방법 반환"""
        
        # 성능 평가
        performance_results = self.evaluate_ensemble_methods(features_dict, labels)
        
        # 최고 성능 방법 찾기
        best_method = None
        best_accuracy = 0
        
        for method_name, metrics in performance_results.items():
            accuracy = metrics.get('accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method_name
        
        if best_method:
            # 최고 성능 방법으로 특징 생성
            ensemble_features = self.transform(features_dict)
            best_features = ensemble_features[best_method]
            
            logger.info(f"최고 성능 앙상블 방법: {best_method} (정확도: {best_accuracy:.4f})")
            return best_method, best_features
        else:
            logger.warning("유효한 앙상블 방법이 없습니다")
            return "concatenation", self._apply_concatenation(features_dict)
    
    def save_ensemble_model(self, save_path: Path):
        """앙상블 모델 저장"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 설정 저장
        config_path = save_path / 'ensemble_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # 모델 구성 요소 저장
        components = {
            'normalizer': self.normalizer,
            'dimension_reducer': self.dimension_reducer,
            'feature_selector': self.feature_selector,
            'weight_learner': self.weight_learner,
            'feature_weights': self.feature_weights
        }
        
        if hasattr(self, 'base_predictors'):
            components['base_predictors'] = self.base_predictors
        
        if hasattr(self, 'stacking_meta_learner'):
            components['stacking_meta_learner'] = self.stacking_meta_learner
        
        model_path = save_path / 'ensemble_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(components, f)
        
        logger.info(f"앙상블 모델 저장 완료: {save_path}")
    
    @classmethod
    def load_ensemble_model(cls, load_path: Path) -> 'FeatureEnsemble':
        """저장된 앙상블 모델 로드"""
        
        # 설정 로드
        config_path = load_path / 'ensemble_config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = EnsembleConfig(**config_dict)
        ensemble = cls(config)
        
        # 모델 구성 요소 로드
        model_path = load_path / 'ensemble_model.pkl'
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
        
        ensemble.normalizer = components['normalizer']
        ensemble.dimension_reducer = components['dimension_reducer']
        ensemble.feature_selector = components['feature_selector']
        ensemble.weight_learner = components['weight_learner']
        ensemble.feature_weights = components['feature_weights']
        
        if 'base_predictors' in components:
            ensemble.base_predictors = components['base_predictors']
        
        if 'stacking_meta_learner' in components:
            ensemble.stacking_meta_learner = components['stacking_meta_learner']
        
        logger.info(f"앙상블 모델 로드 완료: {load_path}")
        return ensemble


def create_sample_features() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """테스트용 샘플 특징 생성"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # 가상의 특징들
    features_dict = {
        'hog': np.random.randn(n_samples, 100),
        'lbp': np.random.randn(n_samples, 80),
        'gabor': np.random.randn(n_samples, 120),
        'sfs': np.random.randn(n_samples, 60)
    }
    
    # 가상의 라벨
    labels = np.random.randint(0, 2, n_samples)
    
    return features_dict, labels


def main():
    """앙상블 시스템 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 샘플 데이터 생성
    features_dict, labels = create_sample_features()
    
    print("=== 특징 앙상블 시스템 테스트 ===")
    print(f"샘플 데이터: {len(labels)}개 샘플")
    for name, features in features_dict.items():
        print(f"- {name}: {features.shape}")
    
    # 앙상블 설정
    config = EnsembleConfig(
        use_concatenation=True,
        use_weighted_fusion=True,
        use_stacking=True,
        enable_pca=True,
        pca_variance_ratio=0.95,
        selection_k=50,
        weight_learning_method='performance_based'
    )
    
    # 앙상블 시스템 생성 및 학습
    ensemble = FeatureEnsemble(config)
    ensemble.fit(features_dict, labels)
    
    # 성능 평가
    results = ensemble.evaluate_ensemble_methods(features_dict, labels)
    
    print("\n=== 앙상블 방법별 성능 ===")
    for method, metrics in results.items():
        accuracy = metrics.get('accuracy', 0)
        feature_dim = metrics.get('feature_dim', 0)
        print(f"{method}: 정확도 {accuracy:.4f}, 차원 {feature_dim}")
    
    # 최고 성능 방법
    best_method, best_features = ensemble.get_best_ensemble_method(features_dict, labels)
    print(f"\n최고 성능: {best_method} ({best_features.shape[1]}차원)")
    
    # 모델 저장
    save_path = Path("models/feature_ensemble")
    ensemble.save_ensemble_model(save_path)
    
    print(f"\n모델 저장 완료: {save_path}")


if __name__ == "__main__":
    main()