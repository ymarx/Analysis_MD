"""
고급 HOG (Histogram of Oriented Gradients) 특징 추출기

다중 스케일 HOG 특징을 추출하여 다양한 크기의 기물을 탐지합니다.
"""

import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from skimage.feature import hog
from skimage import exposure
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HOGConfig:
    """HOG 설정 파라미터"""
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = 'L2-Hys'
    visualize: bool = False
    transform_sqrt: bool = False
    feature_vector: bool = True


class MultiScaleHOGExtractor:
    """
    다중 스케일 HOG 특징 추출기
    
    다양한 크기의 기물을 탐지하기 위해 여러 스케일에서 HOG 특징을 추출합니다.
    """
    
    def __init__(self, scales: Optional[List[HOGConfig]] = None):
        """
        다중 스케일 HOG 추출기 초기화
        
        Args:
            scales: HOG 설정 리스트. None인 경우 기본 설정 사용
        """
        if scales is None:
            self.scales = [
                # Fine scale: 작은 기물 탐지
                HOGConfig(
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                ),
                # Medium scale: 중간 크기 기물
                HOGConfig(
                    orientations=9, 
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                ),
                # Coarse scale: 큰 기물 탐지
                HOGConfig(
                    orientations=12,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                )
            ]
        else:
            self.scales = scales
            
        logger.info(f"다중 스케일 HOG 추출기 초기화 - {len(self.scales)} 스케일")
    
    def extract_single_scale(self, image: np.ndarray, config: HOGConfig) -> np.ndarray:
        """
        단일 스케일 HOG 특징 추출
        
        Args:
            image: 입력 이미지 (grayscale)
            config: HOG 설정
            
        Returns:
            np.ndarray: HOG 특징 벡터
        """
        try:
            # 이미지 전처리
            if len(image.shape) == 3:
                if CV2_AVAILABLE:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    # OpenCV 없이 그레이스케일 변환
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            
            # 이미지 정규화
            image = exposure.equalize_adapthist(image)
            
            # HOG 특징 추출
            features = hog(
                image,
                orientations=config.orientations,
                pixels_per_cell=config.pixels_per_cell,
                cells_per_block=config.cells_per_block,
                block_norm=config.block_norm,
                visualize=config.visualize,
                transform_sqrt=config.transform_sqrt,
                feature_vector=config.feature_vector
            )
            
            # visualize=True인 경우 특징과 이미지가 튜플로 반환됨
            if config.visualize:
                features = features[0]
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"HOG 특징 추출 실패: {e}")
            return np.array([], dtype=np.float32)
    
    def extract_multiscale_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        다중 스케일 HOG 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict[str, np.ndarray]: 스케일별 HOG 특징 딕셔너리
        """
        features = {}
        
        for i, config in enumerate(self.scales):
            scale_name = f"scale_{i}"
            
            try:
                scale_features = self.extract_single_scale(image, config)
                features[scale_name] = scale_features
                
                logger.debug(f"{scale_name}: {len(scale_features)} 차원 특징 추출")
                
            except Exception as e:
                logger.warning(f"{scale_name} HOG 추출 실패: {e}")
                features[scale_name] = np.array([], dtype=np.float32)
        
        return features
    
    def extract_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        모든 스케일의 HOG 특징을 결합하여 반환
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 결합된 HOG 특징 벡터
        """
        multiscale_features = self.extract_multiscale_features(image)
        
        # 유효한 특징들만 결합
        valid_features = [features for features in multiscale_features.values() 
                         if len(features) > 0]
        
        if not valid_features:
            logger.warning("유효한 HOG 특징이 없습니다")
            return np.array([], dtype=np.float32)
        
        combined = np.concatenate(valid_features)
        
        logger.debug(f"결합된 HOG 특징 차원: {len(combined)}")
        
        return combined
    
    def extract_with_visualization(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        시각화와 함께 HOG 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: (특징 벡터, HOG 시각화 이미지 리스트)
        """
        features = []
        visualizations = []
        
        for i, config in enumerate(self.scales):
            # 시각화 설정 활성화
            config_viz = HOGConfig(
                orientations=config.orientations,
                pixels_per_cell=config.pixels_per_cell,
                cells_per_block=config.cells_per_block,
                block_norm=config.block_norm,
                visualize=True
            )
            
            try:
                # 이미지 전처리
                processed_image = image.copy()
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                
                processed_image = exposure.equalize_adapthist(processed_image)
                
                # HOG 특징과 시각화 추출
                hog_features, hog_image = hog(
                    processed_image,
                    orientations=config_viz.orientations,
                    pixels_per_cell=config_viz.pixels_per_cell,
                    cells_per_block=config_viz.cells_per_block,
                    block_norm=config_viz.block_norm,
                    visualize=True,
                    feature_vector=True
                )
                
                features.append(hog_features)
                visualizations.append(hog_image)
                
            except Exception as e:
                logger.warning(f"Scale {i} HOG 시각화 실패: {e}")
        
        # 특징 결합
        if features:
            combined_features = np.concatenate(features)
        else:
            combined_features = np.array([], dtype=np.float32)
        
        return combined_features, visualizations


class AdaptiveHOGExtractor:
    """
    적응형 HOG 특징 추출기
    
    이미지 특성에 따라 HOG 파라미터를 자동으로 조정합니다.
    """
    
    def __init__(self):
        self.base_extractor = MultiScaleHOGExtractor()
        
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        """
        이미지 특성 분석
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict[str, float]: 이미지 특성 딕셔너리
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 이미지 특성 계산
        characteristics = {
            'contrast': np.std(gray),
            'brightness': np.mean(gray),
            'edge_density': self._calculate_edge_density(gray),
            'texture_complexity': self._calculate_texture_complexity(gray)
        }
        
        return characteristics
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """엣지 밀도 계산"""
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _calculate_texture_complexity(self, image: np.ndarray) -> float:
        """텍스처 복잡도 계산"""
        # Laplacian variance를 이용한 텍스처 복잡도
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()
    
    def select_optimal_config(self, characteristics: Dict[str, float]) -> List[HOGConfig]:
        """
        이미지 특성에 따른 최적 HOG 설정 선택
        
        Args:
            characteristics: 이미지 특성
            
        Returns:
            List[HOGConfig]: 최적화된 HOG 설정 리스트
        """
        configs = []
        
        # 대비가 낮은 경우 더 많은 orientation 사용
        if characteristics['contrast'] < 0.1:
            orientations = 12
        else:
            orientations = 9
        
        # 엣지 밀도에 따른 cell 크기 조정
        if characteristics['edge_density'] > 0.1:
            # 엣지가 많은 경우 작은 cell 크기
            cell_sizes = [(4, 4), (8, 8)]
        else:
            # 엣지가 적은 경우 큰 cell 크기
            cell_sizes = [(8, 8), (16, 16)]
        
        # 설정 생성
        for cell_size in cell_sizes:
            config = HOGConfig(
                orientations=orientations,
                pixels_per_cell=cell_size,
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )
            configs.append(config)
        
        return configs
    
    def extract_adaptive_features(self, image: np.ndarray) -> np.ndarray:
        """
        적응형 HOG 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 적응형 HOG 특징
        """
        # 이미지 특성 분석
        characteristics = self.analyze_image_characteristics(image)
        
        # 최적 설정 선택
        optimal_configs = self.select_optimal_config(characteristics)
        
        # 적응형 추출기 생성
        adaptive_extractor = MultiScaleHOGExtractor(optimal_configs)
        
        # 특징 추출
        features = adaptive_extractor.extract_combined_features(image)
        
        logger.info(f"적응형 HOG 특징 추출 완료: {len(features)} 차원")
        
        return features


class HOGFeatureSelector:
    """
    HOG 특징 선택기
    
    가장 중요한 HOG 특징을 선택하여 차원을 줄입니다.
    """
    
    def __init__(self, selection_method: str = 'variance'):
        """
        특징 선택기 초기화
        
        Args:
            selection_method: 선택 방법 ('variance', 'mutual_info', 'correlation')
        """
        self.selection_method = selection_method
        self.selected_indices = None
        
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, top_k: int = 100):
        """
        특징 선택 학습
        
        Args:
            features: HOG 특징 행렬 [samples, features]
            labels: 레이블 (supervised 선택인 경우)
            top_k: 선택할 특징 수
        """
        if self.selection_method == 'variance':
            # 분산 기반 선택
            variances = np.var(features, axis=0)
            self.selected_indices = np.argsort(variances)[-top_k:]
            
        elif self.selection_method == 'mutual_info' and labels is not None:
            # 상호정보량 기반 선택
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(features, labels)
            self.selected_indices = np.argsort(scores)[-top_k:]
            
        elif self.selection_method == 'correlation' and labels is not None:
            # 상관관계 기반 선택
            correlations = []
            for i in range(features.shape[1]):
                corr = np.corrcoef(features[:, i], labels)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            self.selected_indices = np.argsort(correlations)[-top_k:]
            
        else:
            # 기본: 모든 특징 사용
            self.selected_indices = np.arange(min(top_k, features.shape[1]))
        
        logger.info(f"HOG 특징 선택 완료: {len(self.selected_indices)}개 특징 선택")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        특징 선택 적용
        
        Args:
            features: 원본 HOG 특징
            
        Returns:
            np.ndarray: 선택된 HOG 특징
        """
        if self.selected_indices is None:
            raise ValueError("fit() 메서드를 먼저 호출해야 합니다")
        
        return features[:, self.selected_indices]
    
    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
                     top_k: int = 100) -> np.ndarray:
        """
        특징 선택 학습 및 적용
        
        Args:
            features: HOG 특징 행렬
            labels: 레이블
            top_k: 선택할 특징 수
            
        Returns:
            np.ndarray: 선택된 HOG 특징
        """
        self.fit(features, labels, top_k)
        return self.transform(features)