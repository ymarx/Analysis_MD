"""
통합 학습 파이프라인

특징 추출, 데이터 증강, CNN 모델을 통합한 엔드투엔드 학습 시스템입니다.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from datetime import datetime

# 프로젝트 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))

from feature_extraction.hog_extractor import MultiScaleHOGExtractor
from feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from feature_extraction.gabor_extractor import GaborFeatureExtractor
from feature_extraction.sfs_extractor import EnhancedSfSExtractor
from data_augmentation.augmentation_engine import AdvancedAugmentationEngine, AugmentationConfig
from models.cnn_detector import SidescanTargetDetector, ModelConfig, ModelTrainer, SidescanDataset

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """통합 파이프라인 설정"""
    # 데이터 설정
    test_split_ratio: float = 0.2
    validation_split_ratio: float = 0.2
    random_seed: int = 42
    
    # 특징 추출 설정
    use_hog: bool = True
    use_lbp: bool = True
    use_gabor: bool = True
    use_sfs: bool = True
    
    # 데이터 증강 설정
    augmentation_strength: float = 0.6
    augmentations_per_positive: int = 5
    target_balance_ratio: float = 1.0
    
    # 모델 설정
    use_traditional_ml: bool = True
    use_deep_learning: bool = True
    ensemble_models: bool = True
    
    # 훈련 설정
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 출력 설정
    save_models: bool = True
    save_features: bool = True
    verbose: bool = True


class FeatureExtractorPipeline:
    """특징 추출 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # 특징 추출기 초기화
        self.extractors = {}
        
        if config.use_hog:
            self.extractors['hog'] = MultiScaleHOGExtractor()
        
        if config.use_lbp:
            self.extractors['lbp'] = ComprehensiveLBPExtractor()
        
        if config.use_gabor:
            self.extractors['gabor'] = GaborFeatureExtractor()
        
        if config.use_sfs:
            self.extractors['sfs'] = EnhancedSfSExtractor()
        
        logger.info(f"특징 추출 파이프라인 초기화 - {len(self.extractors)}개 추출기")
    
    def extract_features(self, images: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        이미지에서 모든 특징 추출
        
        Args:
            images: 입력 이미지 리스트
            
        Returns:
            Tuple[np.ndarray, Dict[str, int]]: (특징 행렬, 특징 차원 정보)
        """
        if not images:
            return np.array([]), {}
        
        all_features = []
        feature_dims = {}
        
        for name, extractor in self.extractors.items():
            logger.info(f"{name.upper()} 특징 추출 중...")
            
            batch_features = []
            for i, image in enumerate(images):
                try:
                    if name == 'hog':
                        features = extractor.extract_combined_features(image)
                    elif name == 'lbp':
                        features = extractor.extract_comprehensive_features(image)
                    elif name == 'gabor':
                        features = extractor.extract_comprehensive_features(image)
                    elif name == 'sfs':
                        features = extractor.extract_comprehensive_sfs_features(image)
                    
                    if len(features) == 0:
                        logger.warning(f"{name} 특징 추출 실패 - 이미지 {i}")
                        features = np.zeros(100, dtype=np.float32)  # 기본 크기
                    
                    batch_features.append(features)
                    
                except Exception as e:
                    logger.error(f"{name} 특징 추출 오류 - 이미지 {i}: {e}")
                    batch_features.append(np.zeros(100, dtype=np.float32))
            
            if batch_features:
                # 모든 특징 벡터를 같은 길이로 맞추기
                max_len = max(len(f) for f in batch_features)
                if max_len == 0:
                    max_len = 100  # 기본값
                
                padded_features = []
                for features in batch_features:
                    if len(features) < max_len:
                        padded = np.zeros(max_len, dtype=np.float32)
                        padded[:len(features)] = features
                        padded_features.append(padded)
                    else:
                        padded_features.append(features[:max_len])
                
                feature_matrix = np.array(padded_features)
                all_features.append(feature_matrix)
                feature_dims[name] = max_len
                
                logger.info(f"{name.upper()} 특징 추출 완료: {feature_matrix.shape}")
        
        if all_features:
            combined_features = np.hstack(all_features)
            logger.info(f"전체 특징 결합 완료: {combined_features.shape}")
        else:
            logger.warning("모든 특징 추출 실패")
            combined_features = np.zeros((len(images), 400), dtype=np.float32)
            feature_dims = {'default': 400}
        
        return combined_features, feature_dims


class DataAugmentationPipeline:
    """데이터 증강 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        aug_config = AugmentationConfig(
            augmentation_strength=config.augmentation_strength
        )
        
        self.augmentation_engine = AdvancedAugmentationEngine(aug_config)
        
        logger.info("데이터 증강 파이프라인 초기화 완료")
    
    def augment_dataset(self, images: List[np.ndarray], 
                       labels: List[int],
                       masks: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[int], Optional[List[np.ndarray]]]:
        """
        데이터셋 증강 및 균형화
        
        Args:
            images: 입력 이미지
            labels: 라벨
            masks: 마스크 (옵션)
            
        Returns:
            Tuple: (증강된 이미지, 증강된 라벨, 증강된 마스크)
        """
        # 양성/음성 샘플 분리
        positive_images = [img for img, label in zip(images, labels) if label == 1]
        negative_images = [img for img, label in zip(images, labels) if label == 0]
        
        positive_masks = None
        if masks:
            positive_masks = [mask for mask, label in zip(masks, labels) if label == 1 and mask is not None]
        
        logger.info(f"원본 데이터: {len(positive_images)} 양성, {len(negative_images)} 음성")
        
        # 데이터셋 균형화
        balanced_images, _, balanced_masks, balanced_labels = self.augmentation_engine.balance_dataset(
            positive_images=positive_images,
            negative_images=negative_images,
            positive_masks=positive_masks,
            target_ratio=self.config.target_balance_ratio
        )
        
        logger.info(f"균형화된 데이터: {len(balanced_images)} 총 샘플")
        
        return balanced_images, balanced_labels, balanced_masks


class TraditionalMLPipeline:
    """전통적 기계학습 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
        
        # 모델 초기화
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=config.random_seed,
            n_jobs=-1
        )
        
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=config.random_seed
        )
        
        logger.info("전통적 ML 파이프라인 초기화 완료")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        전통적 ML 모델 훈련
        
        Args:
            X_train: 훈련 특징
            y_train: 훈련 라벨
            X_val: 검증 특징
            y_val: 검증 라벨
            
        Returns:
            Dict: 모델별 성능 지표
        """
        # 특징 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"{name} 모델 훈련 중...")
            
            try:
                # 모델 훈련
                model.fit(X_train_scaled, y_train)
                
                # 예측
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 성능 평가
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                auc = roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc
                }
                
                logger.info(f"{name} 성능 - 정확도: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 모델 훈련 실패: {e}")
                results[name] = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc': 0.0
                }
        
        return results
    
    def save_models(self, save_dir: Path):
        """모델 저장"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 스케일러 저장
        with open(save_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 모델 저장
        for name, model in self.models.items():
            with open(save_dir / f'{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"전통적 ML 모델 저장 완료: {save_dir}")


class DeepLearningPipeline:
    """딥러닝 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # 모델 설정
        self.model_config = ModelConfig(
            backbone='resnet18',
            input_channels=1,
            num_classes=2,
            dropout_rate=0.3,
            use_attention=True
        )
        
        # 모델 및 훈련기 초기화
        self.model = SidescanTargetDetector(self.model_config)
        self.trainer = ModelTrainer(self.model, config.device)
        self.trainer.setup_optimizer(config.learning_rate)
        
        logger.info("딥러닝 파이프라인 초기화 완료")
    
    def prepare_data(self, images: List[np.ndarray], labels: List[int],
                    masks: Optional[List[np.ndarray]] = None) -> Tuple[DataLoader, DataLoader]:
        """
        데이터 준비
        
        Args:
            images: 이미지 리스트
            labels: 라벨 리스트
            masks: 마스크 리스트
            
        Returns:
            Tuple[DataLoader, DataLoader]: (훈련 로더, 검증 로더)
        """
        # 데이터셋 생성
        dataset = SidescanDataset(images, labels, masks)
        
        # 훈련/검증 분할
        val_size = int(self.config.validation_split_ratio * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        logger.info(f"데이터 준비 완료 - 훈련: {len(train_dataset)}, 검증: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """
        딥러닝 모델 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            save_path: 모델 저장 경로
            
        Returns:
            Dict: 훈련 히스토리
        """
        logger.info("딥러닝 모델 훈련 시작")
        
        save_path_str = str(save_path) if save_path else None
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.num_epochs,
            save_path=save_path_str
        )
        
        return history


class IntegratedPipeline:
    """통합 학습 파이프라인"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 하위 파이프라인 초기화
        self.feature_extractor = FeatureExtractorPipeline(self.config)
        self.data_augmentor = DataAugmentationPipeline(self.config)
        
        if self.config.use_traditional_ml:
            self.traditional_ml = TraditionalMLPipeline(self.config)
        
        if self.config.use_deep_learning:
            self.deep_learning = DeepLearningPipeline(self.config)
        
        # 결과 저장
        self.results = {}
        
        logger.info("통합 파이프라인 초기화 완료")
    
    def run_complete_pipeline(self, 
                            images: List[np.ndarray],
                            labels: List[int],
                            masks: Optional[List[np.ndarray]] = None,
                            output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Args:
            images: 입력 이미지
            labels: 라벨
            masks: 마스크 (옵션)
            output_dir: 결과 저장 디렉토리
            
        Returns:
            Dict: 전체 결과
        """
        start_time = datetime.now()
        logger.info("=== 통합 파이프라인 실행 시작 ===")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 데이터 증강 및 균형화
        logger.info("1. 데이터 증강 및 균형화")
        augmented_images, augmented_labels, augmented_masks = self.data_augmentor.augment_dataset(
            images, labels, masks
        )
        
        # 2. 특징 추출
        logger.info("2. 특징 추출")
        features, feature_dims = self.feature_extractor.extract_features(augmented_images)
        
        # 특징 저장
        if output_dir and self.config.save_features:
            np.save(output_dir / 'features.npy', features)
            np.save(output_dir / 'labels.npy', np.array(augmented_labels))
            
            with open(output_dir / 'feature_dims.json', 'w') as f:
                json.dump(feature_dims, f, indent=2)
        
        # 3. 데이터 분할
        logger.info("3. 데이터 분할")
        X_train, X_test, y_train, y_test = train_test_split(
            features, augmented_labels,
            test_size=self.config.test_split_ratio,
            random_state=self.config.random_seed,
            stratify=augmented_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split_ratio,
            random_state=self.config.random_seed,
            stratify=y_train
        )
        
        logger.info(f"데이터 분할 완료 - 훈련: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")
        
        # 4. 전통적 기계학습
        if self.config.use_traditional_ml:
            logger.info("4. 전통적 기계학습 모델 훈련")
            ml_results = self.traditional_ml.train(X_train, y_train, X_val, y_val)
            self.results['traditional_ml'] = ml_results
            
            if output_dir and self.config.save_models:
                self.traditional_ml.save_models(output_dir / 'traditional_models')
        
        # 5. 딥러닝
        if self.config.use_deep_learning:
            logger.info("5. 딥러닝 모델 훈련")
            
            # 이미지 인덱스 복구 (특징에서 이미지로)
            train_indices = range(len(X_train))
            val_indices = range(len(X_train), len(X_train) + len(X_val))
            
            train_images = [augmented_images[i] for i in train_indices if i < len(augmented_images)]
            val_images = [augmented_images[i] for i in val_indices if i < len(augmented_images)]
            
            # 충분한 데이터가 있을 때만 딥러닝 실행
            if len(train_images) >= self.config.batch_size and len(val_images) >= self.config.batch_size:
                train_masks = [augmented_masks[i] if augmented_masks else None for i in train_indices]
                val_masks = [augmented_masks[i] if augmented_masks else None for i in val_indices]
                
                train_loader, val_loader = self.deep_learning.prepare_data(
                    train_images[:len(X_train)], y_train.tolist(), 
                    train_masks[:len(X_train)] if any(train_masks) else None
                )
                
                val_loader = self.deep_learning.prepare_data(
                    val_images[:len(X_val)], y_val.tolist(),
                    val_masks[:len(X_val)] if any(val_masks) else None
                )[1]
                
                save_path = output_dir / 'deep_learning_model.pth' if output_dir else None
                dl_history = self.deep_learning.train(train_loader, val_loader, save_path)
                self.results['deep_learning'] = dl_history
            else:
                logger.warning("딥러닝을 위한 충분한 데이터가 없습니다")
                self.results['deep_learning'] = {'message': 'Insufficient data for deep learning'}
        
        # 6. 결과 요약
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        self.results['summary'] = {
            'execution_time_seconds': execution_time,
            'total_samples': len(augmented_images),
            'feature_dimensions': sum(feature_dims.values()),
            'config': self.config.__dict__
        }
        
        # 결과 저장
        if output_dir:
            with open(output_dir / 'pipeline_results.json', 'w') as f:
                # JSON 직렬화 가능한 형태로 변환
                json_results = self._convert_to_json_serializable(self.results)
                json.dump(json_results, f, indent=2)
        
        logger.info(f"=== 통합 파이프라인 실행 완료 ({execution_time:.1f}초) ===")
        
        return self.results
    
    def _convert_to_json_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def generate_report(self, output_dir: Path):
        """결과 리포트 생성"""
        if not self.results:
            logger.warning("생성할 결과가 없습니다")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 마크다운 리포트 생성
        report_path = output_dir / 'training_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 사이드스캔 소나 기물 탐지 모델 훈련 리포트\n\n")
            f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 요약 정보
            if 'summary' in self.results:
                summary = self.results['summary']
                f.write("## 📊 훈련 요약\n\n")
                f.write(f"- **실행 시간**: {summary.get('execution_time_seconds', 0):.1f}초\n")
                f.write(f"- **총 샘플 수**: {summary.get('total_samples', 0):,}개\n")
                f.write(f"- **특징 차원**: {summary.get('feature_dimensions', 0):,}차원\n\n")
            
            # 전통적 ML 결과
            if 'traditional_ml' in self.results:
                f.write("## 🤖 전통적 기계학습 결과\n\n")
                for model_name, metrics in self.results['traditional_ml'].items():
                    f.write(f"### {model_name.replace('_', ' ').title()}\n")
                    f.write(f"- **정확도**: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"- **정밀도**: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"- **재현율**: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"- **F1 점수**: {metrics.get('f1_score', 0):.4f}\n")
                    f.write(f"- **AUC**: {metrics.get('auc', 0):.4f}\n\n")
            
            # 딥러닝 결과
            if 'deep_learning' in self.results:
                f.write("## 🧠 딥러닝 결과\n\n")
                dl_results = self.results['deep_learning']
                if 'val_accuracy' in dl_results and dl_results['val_accuracy']:
                    best_accuracy = max(dl_results['val_accuracy'])
                    f.write(f"- **최고 검증 정확도**: {best_accuracy:.4f}\n")
                    f.write(f"- **총 에폭**: {len(dl_results['val_accuracy'])}회\n\n")
        
        logger.info(f"훈련 리포트 생성 완료: {report_path}")


class PipelineRunner:
    """파이프라인 실행기"""
    
    @staticmethod
    def run_with_sample_data(sample_data_dir: Path, output_dir: Path):
        """샘플 데이터로 파이프라인 실행"""
        logger.info("샘플 데이터로 파이프라인 실행")
        
        # 샘플 데이터 생성 (실제로는 데이터 로더에서 가져옴)
        np.random.seed(42)
        
        # 가상의 소나 이미지 데이터 생성
        images = []
        labels = []
        
        # 양성 샘플 (기물)
        for i in range(50):
            # 가상의 기물 이미지 (중앙에 밝은 영역)
            img = np.random.normal(0.3, 0.1, (128, 128))
            img[40:80, 40:80] += np.random.normal(0.4, 0.1, (40, 40))
            img = np.clip(img, 0, 1)
            images.append(img)
            labels.append(1)
        
        # 음성 샘플 (배경)
        for i in range(30):
            # 가상의 배경 이미지
            img = np.random.normal(0.2, 0.05, (128, 128))
            img = np.clip(img, 0, 1)
            images.append(img)
            labels.append(0)
        
        # 파이프라인 설정
        config = PipelineConfig(
            use_traditional_ml=True,
            use_deep_learning=False,  # 샘플 데이터로는 딥러닝 비활성화
            batch_size=16,
            num_epochs=10,
            augmentation_strength=0.3
        )
        
        # 파이프라인 실행
        pipeline = IntegratedPipeline(config)
        results = pipeline.run_complete_pipeline(images, labels, output_dir=output_dir)
        
        # 리포트 생성
        pipeline.generate_report(output_dir)
        
        return results


if __name__ == "__main__":
    # 예제 실행
    from pathlib import Path
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 출력 디렉토리
    output_dir = Path("data/results/pipeline_test")
    
    # 샘플 데이터로 실행
    runner = PipelineRunner()
    results = runner.run_with_sample_data(None, output_dir)
    
    print("파이프라인 실행 완료!")
    print(f"결과는 {output_dir}에 저장되었습니다.")