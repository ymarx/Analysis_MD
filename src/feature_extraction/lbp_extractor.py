"""
적응형 LBP (Local Binary Pattern) 특징 추출기

지형별 최적화된 LBP 특징을 추출하여 해저면 텍스처를 분석합니다.
"""

import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from skimage.feature import local_binary_pattern
from skimage import exposure
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class LBPConfig:
    """LBP 설정 파라미터"""
    radius: float = 3
    n_points: int = 24
    method: str = 'uniform'
    use_rotation_invariant: bool = True
    use_uniform: bool = True


class TerrainAdaptiveLBP:
    """
    지형 적응형 LBP 특징 추출기
    
    해저면 지형(모래, 진흙, 암반)에 따라 최적화된 LBP 매개변수를 사용합니다.
    """
    
    def __init__(self):
        """지형별 LBP 설정 초기화"""
        self.terrain_configs = {
            'sand': LBPConfig(
                radius=1,
                n_points=8,
                method='uniform',
                use_rotation_invariant=True
            ),
            'mud': LBPConfig(
                radius=2,
                n_points=16,
                method='uniform',
                use_rotation_invariant=True
            ),
            'rock': LBPConfig(
                radius=3,
                n_points=24,
                method='uniform',
                use_rotation_invariant=True
            ),
            'mixed': LBPConfig(
                radius=2,
                n_points=16,
                method='uniform',
                use_rotation_invariant=True
            )
        }
        
        # 기본 설정
        self.default_config = self.terrain_configs['mixed']
        
        logger.info("지형 적응형 LBP 추출기 초기화 완료")
    
    def classify_terrain_type(self, image: np.ndarray) -> str:
        """
        이미지 특성에 따른 지형 분류
        
        Args:
            image: 입력 이미지 (grayscale)
            
        Returns:
            str: 지형 타입 ('sand', 'mud', 'rock', 'mixed')
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 이미지 통계 계산
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 엣지 밀도 계산
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 텍스처 복잡도 계산
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        texture_complexity = laplacian.var()
        
        # 규칙 기반 지형 분류
        if edge_density > 0.15 and texture_complexity > 0.05:
            return 'rock'  # 높은 엣지 밀도와 텍스처 복잡도
        elif mean_intensity < 0.3 and std_intensity < 0.1:
            return 'mud'   # 낮은 밝기와 균등한 분포
        elif mean_intensity > 0.5 and std_intensity > 0.15:
            return 'sand'  # 높은 밝기와 변동성
        else:
            return 'mixed' # 혼합 지형
    
    def extract_single_lbp(self, image: np.ndarray, config: LBPConfig) -> np.ndarray:
        """
        단일 설정으로 LBP 특징 추출
        
        Args:
            image: 입력 이미지 (grayscale)
            config: LBP 설정
            
        Returns:
            np.ndarray: LBP 히스토그램
        """
        try:
            # 이미지 전처리
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 이미지 정규화
            image = exposure.equalize_adapthist(image)
            
            # LBP 계산
            if config.use_rotation_invariant and config.use_uniform:
                method = 'uniform'
            elif config.use_rotation_invariant:
                method = 'ror'
            elif config.use_uniform:
                method = 'uniform'
            else:
                method = 'default'
            
            lbp = local_binary_pattern(
                image, 
                config.n_points,
                config.radius,
                method=method
            )
            
            # 히스토그램 생성
            if method == 'uniform':
                n_bins = config.n_points + 2  # uniform patterns + non-uniform
            else:
                n_bins = 2 ** config.n_points
            
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            
            # 정규화
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            
            return hist
            
        except Exception as e:
            logger.error(f"LBP 특징 추출 실패: {e}")
            return np.array([], dtype=np.float32)
    
    def extract_adaptive_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        적응형 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict[str, np.ndarray]: 지형별 LBP 특징 딕셔너리
        """
        # 지형 분류
        terrain_type = self.classify_terrain_type(image)
        
        features = {}
        
        # 분류된 지형에 대한 최적 설정 사용
        optimal_config = self.terrain_configs[terrain_type]
        features['optimal'] = self.extract_single_lbp(image, optimal_config)
        
        # 모든 지형 설정으로 추출 (robust features)
        for terrain, config in self.terrain_configs.items():
            features[terrain] = self.extract_single_lbp(image, config)
        
        # 메타데이터 추가
        features['metadata'] = np.array([
            self.terrain_configs[terrain_type].radius,
            self.terrain_configs[terrain_type].n_points,
            ord(terrain_type[0])  # 지형 타입의 첫 글자 ASCII
        ], dtype=np.float32)
        
        logger.debug(f"적응형 LBP 추출 완료 - 지형: {terrain_type}")
        
        return features
    
    def extract_multiscale_lbp(self, image: np.ndarray, 
                              scales: Optional[List[float]] = None) -> np.ndarray:
        """
        다중 스케일 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            scales: 스케일 리스트 (기본: [0.5, 1.0, 1.5])
            
        Returns:
            np.ndarray: 결합된 다중 스케일 LBP 특징
        """
        if scales is None:
            scales = [0.5, 1.0, 1.5]
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 지형 분류 및 최적 설정
        terrain_type = self.classify_terrain_type(image)
        base_config = self.terrain_configs[terrain_type]
        
        all_features = []
        
        for scale in scales:
            # 이미지 리사이즈
            h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h < 10 or new_w < 10:  # 너무 작은 이미지는 건너뜀
                continue
            
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # 스케일에 맞춰 설정 조정
            scaled_config = LBPConfig(
                radius=base_config.radius * scale,
                n_points=base_config.n_points,
                method=base_config.method,
                use_rotation_invariant=base_config.use_rotation_invariant,
                use_uniform=base_config.use_uniform
            )
            
            # 최소/최대 radius 제한
            scaled_config.radius = max(1.0, min(scaled_config.radius, 5.0))
            
            # LBP 특징 추출
            lbp_features = self.extract_single_lbp(scaled_image, scaled_config)
            all_features.append(lbp_features)
        
        # 특징 결합
        if all_features:
            combined = np.concatenate(all_features)
        else:
            combined = np.array([], dtype=np.float32)
        
        logger.debug(f"다중 스케일 LBP 특징 차원: {len(combined)}")
        
        return combined


class RotationInvariantLBP:
    """
    회전 불변 LBP 특징 추출기
    
    소나 데이터의 방향성 변화에 강한 특징을 제공합니다.
    """
    
    def __init__(self, radius_list: Optional[List[float]] = None,
                 n_points_list: Optional[List[int]] = None):
        """
        회전 불변 LBP 추출기 초기화
        
        Args:
            radius_list: 반지름 리스트
            n_points_list: 점 개수 리스트
        """
        self.radius_list = radius_list or [1, 2, 3]
        self.n_points_list = n_points_list or [8, 16, 24]
        
        logger.info(f"회전 불변 LBP 추출기 초기화 - {len(self.radius_list)} 반지름, {len(self.n_points_list)} 점 설정")
    
    def extract_ri_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        회전 불변 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 회전 불변 LBP 특징
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 이미지 정규화
        image = exposure.equalize_adapthist(image)
        
        all_features = []
        
        for radius in self.radius_list:
            for n_points in self.n_points_list:
                try:
                    # 회전 불변 uniform LBP
                    lbp = local_binary_pattern(
                        image, n_points, radius, method='uniform'
                    )
                    
                    # 히스토그램 생성
                    hist, _ = np.histogram(
                        lbp.ravel(), 
                        bins=n_points + 2, 
                        range=(0, n_points + 2)
                    )
                    
                    # 정규화
                    hist = hist.astype(np.float32)
                    if hist.sum() > 0:
                        hist = hist / hist.sum()
                    
                    all_features.append(hist)
                    
                except Exception as e:
                    logger.warning(f"R={radius}, P={n_points} LBP 실패: {e}")
                    continue
        
        if all_features:
            combined = np.concatenate(all_features)
        else:
            combined = np.array([], dtype=np.float32)
        
        return combined
    
    def extract_directional_features(self, image: np.ndarray, 
                                   n_directions: int = 8) -> np.ndarray:
        """
        방향성 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            n_directions: 방향 개수
            
        Returns:
            np.ndarray: 방향성 LBP 특징
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        directional_features = []
        
        # 각 방향에 대해 특징 추출
        angles = np.linspace(0, 360, n_directions, endpoint=False)
        
        for angle in angles:
            # 이미지 회전
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, 
                                   (image.shape[1], image.shape[0]))
            
            # LBP 특징 추출
            ri_features = self.extract_ri_lbp(rotated)
            directional_features.append(ri_features)
        
        # 방향별 특징의 통계량 계산
        if directional_features:
            directional_array = np.array(directional_features)
            
            # 평균, 표준편차, 최대, 최소
            mean_features = np.mean(directional_array, axis=0)
            std_features = np.std(directional_array, axis=0)
            max_features = np.max(directional_array, axis=0)
            min_features = np.min(directional_array, axis=0)
            
            combined = np.concatenate([
                mean_features, std_features, max_features, min_features
            ])
        else:
            combined = np.array([], dtype=np.float32)
        
        return combined


class LBPTextureAnalyzer:
    """
    LBP 기반 텍스처 분석기
    
    텍스처 복잡도, 균등성, 대조도 등을 분석합니다.
    """
    
    def __init__(self):
        self.terrain_lbp = TerrainAdaptiveLBP()
        self.ri_lbp = RotationInvariantLBP()
        
    def calculate_texture_uniformity(self, lbp_hist: np.ndarray) -> float:
        """텍스처 균등성 계산"""
        if len(lbp_hist) == 0 or lbp_hist.sum() == 0:
            return 0.0
        
        # Shannon entropy 기반 균등성
        normalized_hist = lbp_hist / lbp_hist.sum()
        entropy = -np.sum(normalized_hist * np.log2(normalized_hist + 1e-10))
        max_entropy = np.log2(len(lbp_hist))
        
        uniformity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return uniformity
    
    def calculate_texture_contrast(self, image: np.ndarray, 
                                 config: Optional[LBPConfig] = None) -> float:
        """텍스처 대조도 계산"""
        if config is None:
            config = LBPConfig()
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # LBP 계산
        lbp = local_binary_pattern(image, config.n_points, config.radius)
        
        # 대조도 계산 (LBP 값의 분산)
        contrast = np.var(lbp)
        
        return float(contrast)
    
    def analyze_texture_properties(self, image: np.ndarray) -> Dict[str, float]:
        """
        종합적인 텍스처 특성 분석
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict[str, float]: 텍스처 특성 딕셔너리
        """
        # 적응형 LBP 특징 추출
        adaptive_features = self.terrain_lbp.extract_adaptive_features(image)
        
        properties = {}
        
        # 각 지형별 텍스처 특성 분석
        for terrain, hist in adaptive_features.items():
            if terrain == 'metadata':
                continue
                
            if len(hist) > 0:
                # 균등성
                uniformity = self.calculate_texture_uniformity(hist)
                properties[f'{terrain}_uniformity'] = uniformity
                
                # 엔트로피
                if hist.sum() > 0:
                    normalized_hist = hist / hist.sum()
                    entropy = -np.sum(normalized_hist * np.log2(normalized_hist + 1e-10))
                    properties[f'{terrain}_entropy'] = entropy
                
                # 집중도 (dominant pattern의 비율)
                concentration = np.max(hist) / hist.sum() if hist.sum() > 0 else 0
                properties[f'{terrain}_concentration'] = concentration
        
        # 전체 이미지 대조도
        properties['image_contrast'] = self.calculate_texture_contrast(image)
        
        # 지형 분류 결과
        terrain_type = self.terrain_lbp.classify_terrain_type(image)
        properties['terrain_type'] = hash(terrain_type) % 1000  # 해시값으로 변환
        
        return properties


class ComprehensiveLBPExtractor:
    """
    종합적인 LBP 특징 추출기
    
    모든 LBP 변형을 결합한 강력한 특징 추출기입니다.
    """
    
    def __init__(self):
        self.terrain_lbp = TerrainAdaptiveLBP()
        self.ri_lbp = RotationInvariantLBP()
        self.texture_analyzer = LBPTextureAnalyzer()
        
        logger.info("종합적인 LBP 추출기 초기화 완료")
    
    def extract_comprehensive_features(self, image: np.ndarray) -> np.ndarray:
        """
        종합적인 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 결합된 LBP 특징 벡터
        """
        all_features = []
        
        # 1. 적응형 LBP 특징
        try:
            adaptive_features = self.terrain_lbp.extract_adaptive_features(image)
            for key, features in adaptive_features.items():
                if key != 'metadata' and len(features) > 0:
                    all_features.append(features)
        except Exception as e:
            logger.warning(f"적응형 LBP 추출 실패: {e}")
        
        # 2. 다중 스케일 LBP 특징
        try:
            multiscale_features = self.terrain_lbp.extract_multiscale_lbp(image)
            if len(multiscale_features) > 0:
                all_features.append(multiscale_features)
        except Exception as e:
            logger.warning(f"다중 스케일 LBP 추출 실패: {e}")
        
        # 3. 회전 불변 LBP 특징
        try:
            ri_features = self.ri_lbp.extract_ri_lbp(image)
            if len(ri_features) > 0:
                all_features.append(ri_features)
        except Exception as e:
            logger.warning(f"회전 불변 LBP 추출 실패: {e}")
        
        # 4. 텍스처 분석 결과
        try:
            texture_props = self.texture_analyzer.analyze_texture_properties(image)
            texture_features = np.array(list(texture_props.values()), dtype=np.float32)
            if len(texture_features) > 0:
                all_features.append(texture_features)
        except Exception as e:
            logger.warning(f"텍스처 분석 실패: {e}")
        
        # 특징 결합
        if all_features:
            combined = np.concatenate(all_features)
        else:
            logger.error("모든 LBP 특징 추출 실패")
            combined = np.array([], dtype=np.float32)
        
        logger.info(f"종합 LBP 특징 추출 완료: {len(combined)} 차원")
        
        return combined
    
    def extract_with_visualization(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        시각화와 함께 LBP 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: (특징 벡터, LBP 이미지 딕셔너리)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 이미지 정규화
        image = exposure.equalize_adapthist(image)
        
        visualizations = {}
        
        # 각 지형 설정별 LBP 시각화
        for terrain, config in self.terrain_lbp.terrain_configs.items():
            try:
                lbp_image = local_binary_pattern(
                    image, config.n_points, config.radius, method='uniform'
                )
                visualizations[terrain] = lbp_image
            except Exception as e:
                logger.warning(f"{terrain} LBP 시각화 실패: {e}")
        
        # 특징 추출
        features = self.extract_comprehensive_features(image)
        
        return features, visualizations