"""
향상된 SfS (Shape-from-Shading) 특징 추출기

음영 정보로부터 3D 형상을 복원하여 기물의 형태학적 특징을 추출합니다.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import minimize
from skimage import morphology, measure
from skimage.filters import gaussian
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class SfSConfig:
    """SfS 설정 파라미터"""
    light_direction: Tuple[float, float, float] = (0, 0, 1)  # (x, y, z)
    albedo: float = 0.5
    regularization_weight: float = 0.1
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    gradient_method: str = 'sobel'  # 'sobel', 'prewitt', 'scharr'
    
    def __post_init__(self):
        # 광원 방향 정규화
        norm = np.sqrt(sum(x**2 for x in self.light_direction))
        if norm > 0:
            self.light_direction = tuple(x/norm for x in self.light_direction)


class EnhancedSfSExtractor:
    """
    향상된 SfS 특징 추출기
    
    다중 광원과 적응적 알고리즘을 사용하여 강인한 3D 형상 복원을 수행합니다.
    """
    
    def __init__(self, config: Optional[SfSConfig] = None):
        """
        SfS 추출기 초기화
        
        Args:
            config: SfS 설정 (기본값 사용시 None)
        """
        self.config = config or SfSConfig()
        
        # 다중 광원 설정 (소나 데이터 특성 고려)
        self.light_sources = [
            (0, 0, 1),      # 수직 조명
            (0.5, 0, 0.866), # 30도 각도
            (-0.5, 0, 0.866), # -30도 각도
            (0, 0.5, 0.866), # y축 30도
            (0, -0.5, 0.866) # y축 -30도
        ]
        
        logger.info("향상된 SfS 추출기 초기화 완료")
    
    def calculate_image_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지 기울기 계산
        
        Args:
            image: 입력 이미지 (grayscale)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (x 방향 기울기, y 방향 기울기)
        """
        if self.config.gradient_method == 'sobel':
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        elif self.config.gradient_method == 'scharr':
            grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        else:  # prewitt
            prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
            prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
            grad_x = ndimage.convolve(image, prewitt_x)
            grad_y = ndimage.convolve(image, prewitt_y)
        
        return grad_x, grad_y
    
    def estimate_surface_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """
        깊이 맵으로부터 표면 법선 추정
        
        Args:
            depth_map: 깊이 맵
            
        Returns:
            np.ndarray: 표면 법선 벡터 (H, W, 3)
        """
        # 깊이 맵 기울기 계산
        grad_x, grad_y = self.calculate_image_gradients(depth_map)
        
        # 표면 법선 계산
        h, w = depth_map.shape
        normals = np.zeros((h, w, 3))
        
        # 법선 벡터 = (-dz/dx, -dz/dy, 1) 정규화
        normals[:, :, 0] = -grad_x
        normals[:, :, 1] = -grad_y  
        normals[:, :, 2] = 1
        
        # 정규화
        norm_magnitude = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        norm_magnitude = np.maximum(norm_magnitude, 1e-10)  # 0으로 나누기 방지
        normals = normals / norm_magnitude
        
        return normals
    
    def lambertian_reflectance_model(self, normals: np.ndarray, 
                                   light_dir: Tuple[float, float, float],
                                   albedo: float) -> np.ndarray:
        """
        Lambert 반사 모델
        
        Args:
            normals: 표면 법선
            light_dir: 광원 방향
            albedo: 알베도 값
            
        Returns:
            np.ndarray: 예측된 밝기
        """
        light_vector = np.array(light_dir).reshape(1, 1, 3)
        
        # 내적 계산 (코사인 값)
        dot_product = np.sum(normals * light_vector, axis=2)
        
        # Lambert 법칙: I = albedo * max(0, n · l)
        intensity = albedo * np.maximum(0, dot_product)
        
        return intensity
    
    def solve_sfs_iterative(self, image: np.ndarray, 
                          initial_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        반복적 SfS 해법
        
        Args:
            image: 입력 이미지
            initial_depth: 초기 깊이 추정값
            
        Returns:
            np.ndarray: 복원된 깊이 맵
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 이미지 정규화
        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        h, w = image.shape
        
        # 초기 깊이 추정
        if initial_depth is None:
            # 간단한 초기 추정: 밝기에 비례
            depth = image.copy() * 0.5
        else:
            depth = initial_depth.copy()
        
        prev_error = float('inf')
        
        for iteration in range(self.config.max_iterations):
            # 현재 깊이에서 표면 법선 계산
            normals = self.estimate_surface_normals(depth)
            
            # Lambert 모델로 밝기 예측
            predicted_intensity = self.lambertian_reflectance_model(
                normals, self.config.light_direction, self.config.albedo
            )
            
            # 오차 계산
            error = np.sum((predicted_intensity - image)**2)
            
            # 수렴 검사
            if abs(prev_error - error) < self.config.convergence_threshold:
                logger.debug(f"SfS 수렴: {iteration+1} 반복")
                break
            
            # 깊이 업데이트 (기울기 하강법)
            intensity_error = predicted_intensity - image
            
            # 깊이 조정
            learning_rate = 0.01
            depth_update = learning_rate * intensity_error
            
            # 정규화 항 추가 (평활화)
            laplacian = ndimage.laplace(depth)
            regularization = self.config.regularization_weight * laplacian
            
            depth = depth - depth_update - regularization
            
            # 깊이 범위 제한
            depth = np.clip(depth, 0, 1)
            
            prev_error = error
        
        return depth
    
    def extract_3d_features(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        3D 형상 특징 추출
        
        Args:
            depth_map: 깊이 맵
            
        Returns:
            Dict[str, float]: 3D 특징 딕셔너리
        """
        features = {}
        
        # 기본 통계량
        features['mean_depth'] = np.mean(depth_map)
        features['std_depth'] = np.std(depth_map)
        features['max_depth'] = np.max(depth_map)
        features['min_depth'] = np.min(depth_map)
        features['depth_range'] = features['max_depth'] - features['min_depth']
        
        # 기울기 분석
        grad_x, grad_y = self.calculate_image_gradients(depth_map)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['mean_gradient'] = np.mean(gradient_magnitude)
        features['max_gradient'] = np.max(gradient_magnitude)
        features['gradient_variance'] = np.var(gradient_magnitude)
        
        # 곡률 분석
        curvature = self._calculate_curvature(depth_map)
        features['mean_curvature'] = np.mean(curvature)
        features['curvature_variance'] = np.var(curvature)
        features['max_curvature'] = np.max(np.abs(curvature))
        
        # 표면적 추정
        surface_area = self._estimate_surface_area(depth_map)
        projected_area = depth_map.shape[0] * depth_map.shape[1]
        features['surface_roughness'] = surface_area / projected_area
        
        # 볼록도/오목도 분석
        convex_pixels = np.sum(curvature > 0)
        concave_pixels = np.sum(curvature < 0)
        total_pixels = curvature.size
        
        features['convexity_ratio'] = convex_pixels / total_pixels
        features['concavity_ratio'] = concave_pixels / total_pixels
        
        return features
    
    def _calculate_curvature(self, depth_map: np.ndarray) -> np.ndarray:
        """곡률 계산 (Laplacian 사용)"""
        # 2차 미분 계산
        laplacian = ndimage.laplace(depth_map.astype(np.float64))
        return laplacian
    
    def _estimate_surface_area(self, depth_map: np.ndarray) -> float:
        """표면적 추정"""
        grad_x, grad_y = self.calculate_image_gradients(depth_map)
        
        # 표면 요소의 크기 계산
        surface_elements = np.sqrt(1 + grad_x**2 + grad_y**2)
        
        # 전체 표면적
        surface_area = np.sum(surface_elements)
        
        return float(surface_area)
    
    def extract_morphological_features(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        형태학적 특징 추출
        
        Args:
            depth_map: 깊이 맵
            
        Returns:
            Dict[str, float]: 형태학적 특징
        """
        features = {}
        
        # 임계값으로 이진화
        threshold = np.percentile(depth_map, 70)  # 상위 30%
        binary_map = depth_map > threshold
        
        # 연결된 구성 요소 분석
        labeled = measure.label(binary_map)
        regions = measure.regionprops(labeled, intensity_image=depth_map)
        
        if regions:
            # 가장 큰 영역 분석
            largest_region = max(regions, key=lambda r: r.area)
            
            features['object_area'] = largest_region.area
            features['object_perimeter'] = largest_region.perimeter
            features['object_circularity'] = 4 * np.pi * largest_region.area / (largest_region.perimeter**2 + 1e-10)
            features['object_eccentricity'] = largest_region.eccentricity if hasattr(largest_region, 'eccentricity') else 0
            features['object_solidity'] = largest_region.solidity if hasattr(largest_region, 'solidity') else 0
            
            # 방향성 분석
            if hasattr(largest_region, 'orientation'):
                features['object_orientation'] = largest_region.orientation
            else:
                features['object_orientation'] = 0
            
            # 깊이 관련 특징
            features['object_mean_depth'] = largest_region.mean_intensity if hasattr(largest_region, 'mean_intensity') else 0
            features['object_max_depth'] = np.max(depth_map[labeled == largest_region.label])
            
        else:
            # 기본값
            for key in ['object_area', 'object_perimeter', 'object_circularity', 
                       'object_eccentricity', 'object_solidity', 'object_orientation',
                       'object_mean_depth', 'object_max_depth']:
                features[key] = 0.0
        
        # 전체 영역의 형태학적 연산
        # 열기/닫기 연산으로 노이즈 제거 효과 분석
        kernel = morphology.disk(3)
        opened = morphology.opening(binary_map, kernel)
        closed = morphology.closing(binary_map, kernel)
        
        features['opening_difference'] = np.sum(binary_map) - np.sum(opened)
        features['closing_difference'] = np.sum(closed) - np.sum(binary_map)
        
        return features
    
    def multi_light_sfs(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        다중 광원 SfS
        
        Args:
            image: 입력 이미지
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: (최종 깊이 맵, 광원별 깊이 맵)
        """
        depth_maps = {}
        
        # 각 광원에 대해 SfS 수행
        for i, light_dir in enumerate(self.light_sources):
            # 광원별 설정
            light_config = SfSConfig(
                light_direction=light_dir,
                albedo=self.config.albedo,
                regularization_weight=self.config.regularization_weight
            )
            
            # 임시 추출기 생성
            temp_config = self.config
            self.config = light_config
            
            try:
                depth_map = self.solve_sfs_iterative(image)
                depth_maps[f'light_{i}'] = depth_map
            except Exception as e:
                logger.warning(f"광원 {i} SfS 실패: {e}")
                depth_maps[f'light_{i}'] = np.zeros_like(image, dtype=np.float64)
            
            # 원래 설정 복구
            self.config = temp_config
        
        # 다중 깊이 맵 융합
        if depth_maps:
            depth_stack = np.stack(list(depth_maps.values()), axis=2)
            
            # 중간값 사용 (노이즈에 강함)
            final_depth = np.median(depth_stack, axis=2)
            
            # 신뢰도 기반 가중 평균 (선택적)
            weights = np.std(depth_stack, axis=2)
            weights = 1 / (weights + 1e-10)  # 낮은 분산 = 높은 신뢰도
            
            weighted_depth = np.average(depth_stack, axis=2, weights=weights)
            
            # 두 결과의 평균
            final_depth = 0.7 * final_depth + 0.3 * weighted_depth
        else:
            final_depth = np.zeros_like(image, dtype=np.float64)
        
        return final_depth, depth_maps
    
    def extract_comprehensive_sfs_features(self, image: np.ndarray) -> np.ndarray:
        """
        종합적인 SfS 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 결합된 SfS 특징 벡터
        """
        all_features = []
        
        try:
            # 다중 광원 SfS 수행
            final_depth, individual_depths = self.multi_light_sfs(image)
            
            # 1. 최종 깊이 맵에서 3D 특징 추출
            d3_features = self.extract_3d_features(final_depth)
            feature_values = list(d3_features.values())
            all_features.extend(feature_values)
            
            # 2. 형태학적 특징 추출
            morphological_features = self.extract_morphological_features(final_depth)
            feature_values = list(morphological_features.values())
            all_features.extend(feature_values)
            
            # 3. 광원별 깊이 맵 일관성 분석
            if len(individual_depths) > 1:
                depth_values = list(individual_depths.values())
                depth_stack = np.stack(depth_values, axis=2)
                
                # 깊이 맵 간 일관성
                depth_consistency = np.mean(np.std(depth_stack, axis=2))
                depth_correlation = self._calculate_depth_correlations(depth_values)
                
                all_features.extend([depth_consistency, depth_correlation])
            
        except Exception as e:
            logger.error(f"SfS 특징 추출 실패: {e}")
            # 기본 특징 벡터 반환
            all_features = [0.0] * 30  # 적절한 크기의 영벡터
        
        # numpy 배열로 변환
        features_array = np.array(all_features, dtype=np.float32)
        
        logger.info(f"SfS 특징 추출 완료: {len(features_array)} 차원")
        
        return features_array
    
    def _calculate_depth_correlations(self, depth_maps: List[np.ndarray]) -> float:
        """깊이 맵 간 상관관계 계산"""
        if len(depth_maps) < 2:
            return 1.0
        
        correlations = []
        for i in range(len(depth_maps)):
            for j in range(i+1, len(depth_maps)):
                corr = np.corrcoef(depth_maps[i].ravel(), depth_maps[j].ravel())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0


class SfSQualityAssessment:
    """
    SfS 품질 평가 도구
    
    복원된 깊이 맵의 품질을 평가합니다.
    """
    
    def __init__(self):
        pass
    
    def assess_depth_quality(self, depth_map: np.ndarray, 
                           original_image: np.ndarray) -> Dict[str, float]:
        """
        깊이 맵 품질 평가
        
        Args:
            depth_map: 복원된 깊이 맵
            original_image: 원본 이미지
            
        Returns:
            Dict[str, float]: 품질 지표
        """
        quality_metrics = {}
        
        # 1. 평활도 평가 (과도한 노이즈 검출)
        grad_x, grad_y = np.gradient(depth_map)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        quality_metrics['smoothness'] = 1.0 / (1.0 + np.mean(gradient_magnitude))
        quality_metrics['gradient_variance'] = np.var(gradient_magnitude)
        
        # 2. 일관성 평가 (밝기와 깊이의 일관성)
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original_image
        
        # 밝기와 깊이의 상관관계
        brightness_depth_corr = np.corrcoef(
            original_gray.ravel(), 
            depth_map.ravel()
        )[0, 1]
        
        quality_metrics['brightness_depth_correlation'] = brightness_depth_corr if not np.isnan(brightness_depth_corr) else 0
        
        # 3. 구조적 유사성
        # 엣지 일관성
        image_edges = cv2.Canny((original_gray * 255).astype(np.uint8), 50, 150)
        depth_edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
        
        edge_similarity = np.sum(image_edges & depth_edges) / (np.sum(image_edges | depth_edges) + 1e-10)
        quality_metrics['edge_consistency'] = edge_similarity
        
        # 4. 동적 범위
        depth_range = np.max(depth_map) - np.min(depth_map)
        quality_metrics['dynamic_range'] = depth_range
        
        # 5. 신호 대 잡음비 추정
        signal_power = np.mean(depth_map**2)
        noise_power = np.mean(gradient_magnitude**2)  # 기울기를 노이즈 지표로 사용
        
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_metrics['estimated_snr'] = snr
        
        return quality_metrics
    
    def recommend_parameters(self, quality_metrics: Dict[str, float]) -> SfSConfig:
        """
        품질 지표에 따른 매개변수 추천
        
        Args:
            quality_metrics: 품질 지표
            
        Returns:
            SfSConfig: 추천 설정
        """
        config = SfSConfig()
        
        # 평활도가 낮으면 정규화 강도 증가
        if quality_metrics['smoothness'] < 0.5:
            config.regularization_weight = 0.2
        
        # 엣지 일관성이 낮으면 반복 횟수 증가
        if quality_metrics['edge_consistency'] < 0.3:
            config.max_iterations = 150
        
        # SNR이 낮으면 수렴 임계값 완화
        if quality_metrics['estimated_snr'] < 10:
            config.convergence_threshold = 1e-5
        
        return config