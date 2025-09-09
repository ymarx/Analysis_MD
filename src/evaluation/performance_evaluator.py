"""
성능 평가 및 최적화 시스템

모델 성능을 종합적으로 평가하고 최적화 방안을 제시합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import json
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 지표 데이터 클래스"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    specificity: float = 0.0
    
    # 추가 지표
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # 성능 지표
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'specificity': self.specificity,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb
        }


class ModelEvaluator:
    """모델 평가기"""
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """
        모델 평가기 초기화
        
        Args:
            model: 평가할 모델
            model_type: 모델 타입 ('sklearn', 'pytorch')
        """
        self.model = model
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type == 'pytorch' and hasattr(model, 'to'):
            self.model.to(self.device)
        
        logger.info(f"모델 평가기 초기화 - 타입: {model_type}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        예측 수행
        
        Args:
            X: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (예측값, 확률값)
        """
        if self.model_type == 'sklearn':
            predictions = self.model.predict(X)
            probabilities = None
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[:, 1]
            elif hasattr(self.model, 'decision_function'):
                probabilities = self.model.decision_function(X)
            
        elif self.model_type == 'pytorch':
            self.model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                # 배치 처리
                batch_size = 32
                for i in range(0, len(X), batch_size):
                    batch = X[i:i+batch_size]
                    
                    if isinstance(batch, np.ndarray):
                        batch_tensor = torch.from_numpy(batch).float().to(self.device)
                    else:
                        batch_tensor = batch.to(self.device)
                    
                    outputs = self.model(batch_tensor)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['classification']
                    else:
                        logits = outputs
                    
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    predictions.extend(preds)
                    probabilities.extend(probs)
            
            predictions = np.array(predictions)
            probabilities = np.array(probabilities)
        
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        return predictions, probabilities
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> PerformanceMetrics:
        """
        종합적인 모델 평가
        
        Args:
            X: 테스트 데이터
            y_true: 실제 레이블
            
        Returns:
            PerformanceMetrics: 성능 지표
        """
        # 메모리 사용량 측정 시작
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 추론 시간 측정
        start_time = time.time()
        predictions, probabilities = self.predict(X)
        end_time = time.time()
        
        # 메모리 사용량 측정 종료
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # 기본 성능 지표 계산
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        
        # AUC 점수
        auc = 0.0
        if probabilities is not None and len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, probabilities)
            except Exception as e:
                logger.warning(f"AUC 계산 실패: {e}")
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, predictions)
        
        # 이진 분류인 경우
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # 다중 분류인 경우 평균 특이도
            specificity = 0.0
            tp = fp = fn = tn = 0
        
        # 성능 메트릭 생성
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            specificity=specificity,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            inference_time_ms=(end_time - start_time) * 1000,
            memory_usage_mb=memory_after - memory_before
        )
        
        logger.info(f"모델 평가 완료 - 정확도: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics


class VisualizationEngine:
    """시각화 엔진"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정 (한국어 지원)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = None, title: str = "Confusion Matrix"):
        """혼동 행렬 플롯"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or ['Background', 'Target'],
                   yticklabels=class_names or ['Background', 'Target'])
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.output_dir / f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"혼동 행렬 저장: {save_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, title: str = "ROC Curve"):
        """ROC 곡선 플롯"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = self.output_dir / f"roc_curve_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC 곡선 저장: {save_path}")
            
        except Exception as e:
            logger.error(f"ROC 곡선 생성 실패: {e}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   title: str = "Precision-Recall Curve"):
        """정밀도-재현율 곡선 플롯"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.fill_between(recall, precision, alpha=0.2, color='blue')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = self.output_dir / f"pr_curve_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PR 곡선 저장: {save_path}")
            
        except Exception as e:
            logger.error(f"PR 곡선 생성 실패: {e}")
    
    def plot_model_comparison(self, model_results: Dict[str, PerformanceMetrics],
                             title: str = "Model Performance Comparison"):
        """모델 비교 플롯"""
        if not model_results:
            return
        
        # 메트릭 추출
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        data = []
        for model_name, result in model_results.items():
            for metric in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': getattr(result, metric, 0.0)
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.output_dir / f"model_comparison_{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"모델 비교 플롯 저장: {save_path}")
    
    def plot_training_history(self, history: Dict[str, List[float]], title: str = "Training History"):
        """훈련 히스토리 플롯"""
        if not history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 플롯
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
            axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 정확도 플롯
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='green')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        save_path = self.output_dir / f"training_history_{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"훈련 히스토리 플롯 저장: {save_path}")


class ModelOptimizer:
    """모델 최적화기"""
    
    def __init__(self):
        pass
    
    def optimize_hyperparameters(self, model, param_grid: Dict[str, List],
                                X_train: np.ndarray, y_train: np.ndarray,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화
        
        Args:
            model: 최적화할 모델
            param_grid: 매개변수 그리드
            X_train: 훈련 데이터
            y_train: 훈련 레이블
            cv_folds: 교차 검증 폴드 수
            
        Returns:
            Dict: 최적화 결과
        """
        logger.info("하이퍼파라미터 최적화 시작")
        
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"최적화 완료 - 최고 점수: {grid_search.best_score_:.4f}")
            logger.info(f"최적 매개변수: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"하이퍼파라미터 최적화 실패: {e}")
            return {'error': str(e)}
    
    def suggest_model_improvements(self, metrics: PerformanceMetrics) -> List[str]:
        """
        모델 개선 방안 제안
        
        Args:
            metrics: 성능 지표
            
        Returns:
            List[str]: 개선 방안 리스트
        """
        suggestions = []
        
        # 정확도 기반 제안
        if metrics.accuracy < 0.8:
            suggestions.append("모델 복잡도 증가 고려 (더 깊은 네트워크, 더 많은 특징)")
            
        if metrics.accuracy > 0.95 and metrics.f1_score < 0.9:
            suggestions.append("클래스 불균형 문제 해결 (SMOTE, 클래스 가중치 조정)")
        
        # 정밀도/재현율 기반 제안
        if metrics.precision < 0.8:
            suggestions.append("False Positive 감소 필요 - 더 보수적인 임계값 설정")
            
        if metrics.recall < 0.8:
            suggestions.append("False Negative 감소 필요 - 더 민감한 모델 사용")
        
        # 성능 기반 제안
        if metrics.inference_time_ms > 100:
            suggestions.append("추론 속도 최적화 - 모델 경량화, 양자화 고려")
            
        if metrics.memory_usage_mb > 500:
            suggestions.append("메모리 사용량 최적화 - 모델 압축, 프루닝 고려")
        
        # AUC 기반 제안
        if metrics.auc_score < 0.8:
            suggestions.append("특징 엔지니어링 개선 - 더 의미있는 특징 추가")
        
        # 기본 제안
        if not suggestions:
            suggestions.append("현재 성능이 양호합니다. 추가 데이터 수집으로 일반화 성능 향상 고려")
        
        return suggestions


class ComprehensiveEvaluator:
    """종합 평가 시스템"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = VisualizationEngine(output_dir)
        self.optimizer = ModelOptimizer()
        
        logger.info(f"종합 평가 시스템 초기화 - 출력 디렉토리: {output_dir}")
    
    def evaluate_single_model(self, model: Any, model_name: str, model_type: str,
                             X_test: np.ndarray, y_test: np.ndarray) -> PerformanceMetrics:
        """단일 모델 평가"""
        logger.info(f"{model_name} 모델 평가 시작")
        
        evaluator = ModelEvaluator(model, model_type)
        metrics = evaluator.evaluate(X_test, y_test)
        
        # 예측 결과 얻기
        predictions, probabilities = evaluator.predict(X_test)
        
        # 시각화 생성
        self.visualizer.plot_confusion_matrix(y_test, predictions, title=f"{model_name} Confusion Matrix")
        
        if probabilities is not None:
            self.visualizer.plot_roc_curve(y_test, probabilities, title=f"{model_name} ROC Curve")
            self.visualizer.plot_precision_recall_curve(y_test, probabilities, title=f"{model_name} PR Curve")
        
        logger.info(f"{model_name} 평가 완료")
        return metrics
    
    def evaluate_multiple_models(self, models_dict: Dict[str, Tuple[Any, str]],
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, PerformanceMetrics]:
        """다중 모델 평가"""
        logger.info(f"{len(models_dict)}개 모델 비교 평가 시작")
        
        results = {}
        
        for model_name, (model, model_type) in models_dict.items():
            try:
                metrics = self.evaluate_single_model(model, model_name, model_type, X_test, y_test)
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"{model_name} 평가 실패: {e}")
                results[model_name] = PerformanceMetrics()  # 기본값
        
        # 모델 비교 시각화
        self.visualizer.plot_model_comparison(results)
        
        return results
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 model_metrics: Dict[str, PerformanceMetrics]):
        """평가 리포트 생성"""
        report_path = self.output_dir / 'evaluation_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 사이드스캔 소나 기물 탐지 모델 평가 리포트\n\n")
            f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 전체 요약
            f.write("## 📊 평가 요약\n\n")
            if model_metrics:
                best_model = max(model_metrics.items(), key=lambda x: x[1].f1_score)
                f.write(f"- **최고 성능 모델**: {best_model[0]}\n")
                f.write(f"- **최고 F1 점수**: {best_model[1].f1_score:.4f}\n")
                f.write(f"- **평가된 모델 수**: {len(model_metrics)}개\n\n")
            
            # 모델별 상세 결과
            f.write("## 🤖 모델별 성능\n\n")
            for model_name, metrics in model_metrics.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"| 지표 | 값 |\n")
                f.write(f"|------|----|\n")
                f.write(f"| 정확도 | {metrics.accuracy:.4f} |\n")
                f.write(f"| 정밀도 | {metrics.precision:.4f} |\n")
                f.write(f"| 재현율 | {metrics.recall:.4f} |\n")
                f.write(f"| F1 점수 | {metrics.f1_score:.4f} |\n")
                f.write(f"| AUC | {metrics.auc_score:.4f} |\n")
                f.write(f"| 특이도 | {metrics.specificity:.4f} |\n")
                f.write(f"| 추론 시간 | {metrics.inference_time_ms:.2f}ms |\n")
                f.write(f"| 메모리 사용량 | {metrics.memory_usage_mb:.2f}MB |\n\n")
                
                # 개선 방안
                suggestions = self.optimizer.suggest_model_improvements(metrics)
                if suggestions:
                    f.write("**개선 방안:**\n")
                    for suggestion in suggestions:
                        f.write(f"- {suggestion}\n")
                    f.write("\n")
            
            # 결론 및 권장사항
            f.write("## 🎯 결론 및 권장사항\n\n")
            if model_metrics:
                avg_accuracy = np.mean([m.accuracy for m in model_metrics.values()])
                avg_f1 = np.mean([m.f1_score for m in model_metrics.values()])
                
                f.write(f"- **평균 정확도**: {avg_accuracy:.4f}\n")
                f.write(f"- **평균 F1 점수**: {avg_f1:.4f}\n\n")
                
                if avg_f1 > 0.9:
                    f.write("✅ **우수한 성능** - 실제 배포 고려 가능\n")
                elif avg_f1 > 0.8:
                    f.write("🔶 **양호한 성능** - 추가 최적화 후 배포 권장\n")
                else:
                    f.write("⚠️ **성능 개선 필요** - 모델 아키텍처 또는 데이터 품질 검토 권장\n")
        
        # JSON 결과도 저장
        json_results = {}
        for model_name, metrics in model_metrics.items():
            json_results[model_name] = metrics.to_dict()
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"평가 리포트 생성 완료: {report_path}")


# 사용 예제 함수
def run_comprehensive_evaluation(models_dict: Dict[str, Tuple[Any, str]],
                                X_test: np.ndarray, y_test: np.ndarray,
                                output_dir: Path) -> Dict[str, PerformanceMetrics]:
    """종합 평가 실행"""
    evaluator = ComprehensiveEvaluator(output_dir)
    
    # 모델 평가
    results = evaluator.evaluate_multiple_models(models_dict, X_test, y_test)
    
    # 리포트 생성
    evaluator.generate_evaluation_report({}, results)
    
    return results


if __name__ == "__main__":
    # 예제 실행
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from pathlib import Path
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 가상 데이터 생성
    np.random.seed(42)
    X_test = np.random.random((100, 50))
    y_test = np.random.randint(0, 2, 100)
    
    # 모델 생성
    models = {
        'RandomForest': (RandomForestClassifier(n_estimators=50, random_state=42), 'sklearn'),
        'SVM': (SVC(probability=True, random_state=42), 'sklearn')
    }
    
    # 모델 훈련 (간단한 예제)
    for name, (model, model_type) in models.items():
        X_train = np.random.random((200, 50))
        y_train = np.random.randint(0, 2, 200)
        model.fit(X_train, y_train)
    
    # 평가 실행
    output_dir = Path("data/results/evaluation_test")
    results = run_comprehensive_evaluation(models, X_test, y_test, output_dir)
    
    print("평가 완료!")
    print(f"결과는 {output_dir}에 저장되었습니다.")