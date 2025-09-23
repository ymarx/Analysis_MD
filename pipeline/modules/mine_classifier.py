"""
Mine Classifier Module
======================
기뢰 분류 모듈 - 최적화된 모델을 사용한 분류
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MineClassifier:
    """기뢰 분류 클래스"""

    def __init__(self, classifier_type: str = "ensemble"):
        """
        Initialize Mine Classifier

        Args:
            classifier_type: 분류기 타입 ("ensemble", "stacking", "individual")
        """
        self.classifier_type = classifier_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trained_model = None
        self.model_config = None

    def classify(self,
                features: np.ndarray,
                ensemble_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        분류 실행

        Args:
            features: 특징 행렬
            ensemble_config: 앙상블 구성 정보

        Returns:
            예측 결과 배열
        """
        self.logger.info(f"Classifying {features.shape[0]} samples using {self.classifier_type}")

        try:
            if ensemble_config is None:
                raise ValueError("Ensemble configuration required for classification")

            # Get best model from ensemble configuration
            best_config = ensemble_config.get('best_config')
            if best_config is None:
                raise ValueError("No best configuration found in ensemble config")

            model = self._get_model_from_config(best_config)
            if model is None:
                raise ValueError("No trained model available")

            # Make predictions
            predictions = model.predict(features)

            # Store model information
            self.trained_model = model
            self.model_config = best_config

            self.logger.info(f"Classification completed. Predictions shape: {predictions.shape}")

            return predictions

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            raise

    def _get_model_from_config(self, config: Dict[str, Any]) -> Any:
        """구성에서 모델 추출"""
        config_type = config.get('type')
        config_data = config.get('config', {})

        if config_type == 'individual':
            # Individual model
            model_name, model_data = config_data
            return model_data.get('model')

        elif config_type == 'ensemble':
            # Voting ensemble
            return config_data.get('model')

        elif config_type == 'stacking':
            # Stacking ensemble
            return config_data.get('model')

        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def train(self,
             features: np.ndarray,
             labels: np.ndarray,
             ensemble_config: Dict[str, Any],
             test_size: float = 0.2) -> Dict[str, Any]:
        """
        모델 훈련 및 평가

        Args:
            features: 특징 행렬
            labels: 레이블 배열
            ensemble_config: 앙상블 구성
            test_size: 테스트 데이터 비율

        Returns:
            훈련 결과
        """
        self.logger.info("Training classifier")

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42, stratify=labels
            )

            # Get best model configuration
            best_config = ensemble_config.get('best_config')
            if best_config is None:
                raise ValueError("No best configuration found")

            # Train model
            model = self._train_model_from_config(best_config, X_train, y_train)

            # Evaluate model
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_predictions)
            test_metrics = self._calculate_metrics(y_test, test_predictions)

            # Store trained model
            self.trained_model = model
            self.model_config = best_config

            training_results = {
                'model_type': best_config.get('type'),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'classification_report': classification_report(
                    y_test, test_predictions, output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test, test_predictions).tolist(),
                'feature_shape': features.shape
            }

            self.logger.info(f"Training completed. Test F1-score: {test_metrics['f1_score']:.4f}")

            return training_results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _train_model_from_config(self,
                                config: Dict[str, Any],
                                X_train: np.ndarray,
                                y_train: np.ndarray) -> Any:
        """구성에 따른 모델 훈련"""
        config_type = config.get('type')
        config_data = config.get('config', {})

        if config_type == 'individual':
            # Individual model
            model_name, model_data = config_data
            model_class = model_data.get('model').__class__
            model_params = model_data.get('params', {})

            model = model_class(**model_params)
            model.fit(X_train, y_train)
            return model

        elif config_type == 'ensemble':
            # Recreate voting ensemble
            from sklearn.ensemble import VotingClassifier
            individual_results = config_data.get('individual_results', {})
            combination = config_data.get('combination', [])

            estimators = []
            for model_name in combination:
                if model_name in individual_results:
                    model_data = individual_results[model_name]
                    model_class = model_data.get('model').__class__
                    model_params = model_data.get('params', {})
                    model = model_class(**model_params)
                    estimators.append((model_name, model))

            voting_classifier = VotingClassifier(estimators=estimators, voting='hard')
            voting_classifier.fit(X_train, y_train)
            return voting_classifier

        elif config_type == 'stacking':
            # Recreate stacking ensemble
            try:
                from sklearn.ensemble import StackingClassifier
                base_models = config_data.get('base_models', [])
                meta_model_name = config_data.get('meta_model')

                # Get base estimators from individual results
                individual_results = config_data.get('individual_results', {})
                base_estimators = []

                for model_name in base_models:
                    if model_name in individual_results:
                        model_data = individual_results[model_name]
                        model_class = model_data.get('model').__class__
                        model_params = model_data.get('params', {})
                        model = model_class(**model_params)
                        base_estimators.append((model_name, model))

                # Create meta-learner
                meta_model = self._create_meta_learner(meta_model_name)

                stacking_classifier = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=3
                )
                stacking_classifier.fit(X_train, y_train)
                return stacking_classifier

            except ImportError:
                self.logger.warning("StackingClassifier not available")
                raise

        else:
            raise ValueError(f"Unknown config type: {config_type}")

    def _create_meta_learner(self, meta_model_name: str) -> Any:
        """메타 학습기 생성"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        if meta_model_name == 'logistic':
            return LogisticRegression(random_state=42)
        elif meta_model_name == 'rf':
            return RandomForestClassifier(n_estimators=50, random_state=42)
        elif meta_model_name == 'svm':
            return SVC(kernel='linear', random_state=42)
        else:
            return LogisticRegression(random_state=42)  # Default

    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, float]:
        """평가 지표 계산"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if self.trained_model is None:
            raise ValueError("No trained model available. Train or load a model first.")

        try:
            if hasattr(self.trained_model, 'predict_proba'):
                return self.trained_model.predict_proba(features)
            else:
                # For models without predict_proba, return binary predictions
                predictions = self.trained_model.predict(features)
                n_samples = len(predictions)
                probabilities = np.zeros((n_samples, 2))
                probabilities[predictions == 0, 0] = 1.0
                probabilities[predictions == 1, 1] = 1.0
                return probabilities

        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """특징 중요도 반환"""
        if self.trained_model is None:
            return None

        try:
            if hasattr(self.trained_model, 'feature_importances_'):
                return self.trained_model.feature_importances_
            elif hasattr(self.trained_model, 'coef_'):
                return np.abs(self.trained_model.coef_[0])
            else:
                return None

        except Exception as e:
            self.logger.error(f"Feature importance extraction failed: {e}")
            return None

    def save_model(self, model_path: Path):
        """모델 저장"""
        if self.trained_model is None:
            raise ValueError("No trained model to save")

        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.trained_model,
                'config': self.model_config,
                'classifier_type': self.classifier_type
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: Path):
        """모델 로드"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.trained_model = model_data['model']
            self.model_config = model_data['config']
            self.classifier_type = model_data.get('classifier_type', 'ensemble')

            self.logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.trained_model is None:
            return {'status': 'No trained model'}

        info = {
            'classifier_type': self.classifier_type,
            'model_class': self.trained_model.__class__.__name__,
            'config': self.model_config,
            'has_feature_importance': hasattr(self.trained_model, 'feature_importances_') or hasattr(self.trained_model, 'coef_'),
            'has_predict_proba': hasattr(self.trained_model, 'predict_proba')
        }

        return info

    def evaluate_on_new_data(self,
                           features: np.ndarray,
                           labels: np.ndarray) -> Dict[str, Any]:
        """새로운 데이터에 대한 평가"""
        if self.trained_model is None:
            raise ValueError("No trained model available")

        try:
            predictions = self.trained_model.predict(features)
            probabilities = self.predict_proba(features)

            metrics = self._calculate_metrics(labels, predictions)

            evaluation_results = {
                'metrics': metrics,
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'classification_report': classification_report(
                    labels, predictions, output_dict=True
                ),
                'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
                'sample_count': len(labels)
            }

            self.logger.info(f"Evaluation completed. F1-score: {metrics['f1_score']:.4f}")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise