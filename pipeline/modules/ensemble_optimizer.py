"""
Ensemble Optimizer Module
=========================
앙상블 최적화 모듈 - 특징 조합 및 스태킹 최적화
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna

logger = logging.getLogger(__name__)


class EnsembleOptimizer:
    """앙상블 최적화 클래스"""

    def __init__(self,
                 optimization_method: str = "optuna",
                 cv_folds: int = 5,
                 max_trials: int = 100):
        """
        Initialize Ensemble Optimizer

        Args:
            optimization_method: 최적화 방법 ("optuna", "grid_search", "random_search")
            cv_folds: Cross-validation folds
            max_trials: 최대 시도 횟수
        """
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.max_trials = max_trials
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self,
                features: np.ndarray,
                labels: np.ndarray) -> Dict[str, Any]:
        """
        앙상블 최적화 실행

        Args:
            features: 특징 행렬
            labels: 레이블 배열

        Returns:
            최적화 결과
        """
        self.logger.info("Starting ensemble optimization")

        try:
            # Feature preprocessing
            features_processed = self._preprocess_features(features)

            # Individual model optimization
            individual_results = self._optimize_individual_models(
                features_processed, labels
            )

            # Ensemble optimization
            ensemble_results = self._optimize_ensemble_combination(
                features_processed, labels, individual_results
            )

            # Stacking optimization
            stacking_results = self._optimize_stacking(
                features_processed, labels, individual_results
            )

            # Select best approach
            best_config = self._select_best_configuration(
                individual_results, ensemble_results, stacking_results
            )

            return {
                'best_config': best_config,
                'individual_results': individual_results,
                'ensemble_results': ensemble_results,
                'stacking_results': stacking_results,
                'optimization_info': {
                    'method': self.optimization_method,
                    'cv_folds': self.cv_folds,
                    'max_trials': self.max_trials,
                    'feature_shape': features_processed.shape
                }
            }

        except Exception as e:
            self.logger.error(f"Ensemble optimization failed: {e}")
            raise

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """특징 전처리"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        return features_scaled

    def _optimize_individual_models(self,
                                  features: np.ndarray,
                                  labels: np.ndarray) -> Dict[str, Any]:
        """개별 모델 최적화"""
        self.logger.info("Optimizing individual models")

        models = {
            'random_forest': self._optimize_random_forest,
            'gradient_boosting': self._optimize_gradient_boosting,
            'svm': self._optimize_svm,
            'logistic_regression': self._optimize_logistic_regression,
            'mlp': self._optimize_mlp
        }

        results = {}
        for model_name, optimize_func in models.items():
            try:
                self.logger.debug(f"Optimizing {model_name}")
                results[model_name] = optimize_func(features, labels)
            except Exception as e:
                self.logger.warning(f"Failed to optimize {model_name}: {e}")
                results[model_name] = {'score': 0, 'params': {}, 'model': None}

        return results

    def _optimize_random_forest(self,
                               features: np.ndarray,
                               labels: np.ndarray) -> Dict[str, Any]:
        """Random Forest 최적화"""
        if self.optimization_method == "optuna":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }

                model = RandomForestClassifier(**params)
                scores = cross_val_score(model, features, labels,
                                       cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                       scoring='f1_weighted')
                return scores.mean()

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=self.max_trials // 5)

            best_params = study.best_params
            best_score = study.best_value

        else:
            # Simple grid search fallback
            best_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            model = RandomForestClassifier(**best_params)
            scores = cross_val_score(model, features, labels,
                                   cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                   scoring='f1_weighted')
            best_score = scores.mean()

        # Train final model
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(features, labels)

        return {
            'score': best_score,
            'params': best_params,
            'model': final_model
        }

    def _optimize_gradient_boosting(self,
                                   features: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, Any]:
        """Gradient Boosting 최적화"""
        if self.optimization_method == "optuna":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }

                model = GradientBoostingClassifier(**params)
                scores = cross_val_score(model, features, labels,
                                       cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                       scoring='f1_weighted')
                return scores.mean()

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=self.max_trials // 5)

            best_params = study.best_params
            best_score = study.best_value

        else:
            best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42}
            model = GradientBoostingClassifier(**best_params)
            scores = cross_val_score(model, features, labels,
                                   cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                   scoring='f1_weighted')
            best_score = scores.mean()

        final_model = GradientBoostingClassifier(**best_params)
        final_model.fit(features, labels)

        return {
            'score': best_score,
            'params': best_params,
            'model': final_model
        }

    def _optimize_svm(self,
                     features: np.ndarray,
                     labels: np.ndarray) -> Dict[str, Any]:
        """SVM 최적화"""
        if self.optimization_method == "optuna":
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'random_state': 42
                }

                model = SVC(**params)
                scores = cross_val_score(model, features, labels,
                                       cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                       scoring='f1_weighted')
                return scores.mean()

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=self.max_trials // 5)

            best_params = study.best_params
            best_score = study.best_value

        else:
            best_params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'random_state': 42}
            model = SVC(**best_params)
            scores = cross_val_score(model, features, labels,
                                   cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                   scoring='f1_weighted')
            best_score = scores.mean()

        final_model = SVC(**best_params)
        final_model.fit(features, labels)

        return {
            'score': best_score,
            'params': best_params,
            'model': final_model
        }

    def _optimize_logistic_regression(self,
                                    features: np.ndarray,
                                    labels: np.ndarray) -> Dict[str, Any]:
        """Logistic Regression 최적화"""
        best_params = {'C': 1.0, 'random_state': 42, 'max_iter': 1000}
        model = LogisticRegression(**best_params)
        scores = cross_val_score(model, features, labels,
                               cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                               scoring='f1_weighted')
        best_score = scores.mean()

        final_model = LogisticRegression(**best_params)
        final_model.fit(features, labels)

        return {
            'score': best_score,
            'params': best_params,
            'model': final_model
        }

    def _optimize_mlp(self,
                     features: np.ndarray,
                     labels: np.ndarray) -> Dict[str, Any]:
        """MLP 최적화"""
        best_params = {
            'hidden_layer_sizes': (100, 50),
            'learning_rate_init': 0.001,
            'random_state': 42,
            'max_iter': 500
        }
        model = MLPClassifier(**best_params)

        try:
            scores = cross_val_score(model, features, labels,
                                   cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                   scoring='f1_weighted')
            best_score = scores.mean()

            final_model = MLPClassifier(**best_params)
            final_model.fit(features, labels)

        except Exception as e:
            self.logger.warning(f"MLP optimization failed: {e}")
            best_score = 0
            final_model = None

        return {
            'score': best_score,
            'params': best_params,
            'model': final_model
        }

    def _optimize_ensemble_combination(self,
                                     features: np.ndarray,
                                     labels: np.ndarray,
                                     individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """앙상블 조합 최적화"""
        self.logger.info("Optimizing ensemble combinations")

        # Get best individual models
        valid_models = {
            name: result for name, result in individual_results.items()
            if result['model'] is not None and result['score'] > 0
        }

        if len(valid_models) < 2:
            return {'score': 0, 'combination': [], 'model': None}

        # Try different combinations
        best_score = 0
        best_combination = []
        best_model = None

        # Sort models by performance
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['score'], reverse=True)

        # Try top combinations
        for i in range(2, min(len(sorted_models) + 1, 6)):  # Up to 5 models
            try:
                model_names = [name for name, _ in sorted_models[:i]]
                models = [(name, result['model']) for name, result in sorted_models[:i]]

                voting_classifier = VotingClassifier(
                    estimators=models,
                    voting='hard'
                )

                scores = cross_val_score(voting_classifier, features, labels,
                                       cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                       scoring='f1_weighted')
                score = scores.mean()

                if score > best_score:
                    best_score = score
                    best_combination = model_names
                    best_model = voting_classifier
                    best_model.fit(features, labels)

            except Exception as e:
                self.logger.warning(f"Failed to create ensemble with {model_names}: {e}")

        return {
            'score': best_score,
            'combination': best_combination,
            'model': best_model
        }

    def _optimize_stacking(self,
                          features: np.ndarray,
                          labels: np.ndarray,
                          individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """스태킹 최적화"""
        self.logger.info("Optimizing stacking ensemble")

        try:
            from sklearn.ensemble import StackingClassifier

            # Get valid base models
            valid_models = {
                name: result for name, result in individual_results.items()
                if result['model'] is not None and result['score'] > 0
            }

            if len(valid_models) < 2:
                return {'score': 0, 'base_models': [], 'meta_model': None, 'model': None}

            # Create base estimators
            base_estimators = [(name, result['model']) for name, result in valid_models.items()]

            # Try different meta-learners
            meta_learners = [
                ('logistic', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('svm', SVC(kernel='linear', random_state=42))
            ]

            best_score = 0
            best_meta = None
            best_model = None

            for meta_name, meta_model in meta_learners:
                try:
                    stacking_classifier = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=meta_model,
                        cv=3  # Inner CV for stacking
                    )

                    scores = cross_val_score(stacking_classifier, features, labels,
                                           cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                                           scoring='f1_weighted')
                    score = scores.mean()

                    if score > best_score:
                        best_score = score
                        best_meta = meta_name
                        best_model = stacking_classifier
                        best_model.fit(features, labels)

                except Exception as e:
                    self.logger.warning(f"Stacking failed with {meta_name}: {e}")

            return {
                'score': best_score,
                'base_models': list(valid_models.keys()),
                'meta_model': best_meta,
                'model': best_model
            }

        except ImportError:
            self.logger.warning("StackingClassifier not available")
            return {'score': 0, 'base_models': [], 'meta_model': None, 'model': None}

    def _select_best_configuration(self,
                                 individual_results: Dict[str, Any],
                                 ensemble_results: Dict[str, Any],
                                 stacking_results: Dict[str, Any]) -> Dict[str, Any]:
        """최적 구성 선택"""
        configurations = {
            'individual': {
                'type': 'individual',
                'score': max([result['score'] for result in individual_results.values()]),
                'config': max(individual_results.items(), key=lambda x: x[1]['score'])
            },
            'ensemble': {
                'type': 'ensemble',
                'score': ensemble_results['score'],
                'config': ensemble_results
            },
            'stacking': {
                'type': 'stacking',
                'score': stacking_results['score'],
                'config': stacking_results
            }
        }

        # Select best configuration
        best_config = max(configurations.values(), key=lambda x: x['score'])

        self.logger.info(f"Best configuration: {best_config['type']} with score {best_config['score']:.4f}")

        return best_config

    def evaluate_model(self,
                      model: Any,
                      features: np.ndarray,
                      labels: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        try:
            predictions = model.predict(features)

            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(labels, predictions, average='weighted', zero_division=0)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}