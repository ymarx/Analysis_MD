"""
사이드스캔 소나 데이터 전처리 파이프라인

워터컬럼 처리, 강도 정규화, 노이즈 제거, 지형별 적응형 처리 등을 수행합니다.
"""

import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from scipy import ndimage, signal
from scipy.stats import zscore
from skimage import filters, restoration, morphology, exposure
from skimage.morphology import disk, rectangle
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """지형 유형 열거형"""
    SAND = "sand"
    MUD = "mud"
    ROCK = "rock"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class PreprocessingConfig:
    """전처리 설정을 저장하는 데이터클래스"""
    remove_water_column: bool = True
    water_column_width: int = 50
    normalize_intensity: bool = True
    normalization_method: str = 'minmax'  # 'minmax', 'zscore', 'histogram'
    apply_denoising: bool = True
    denoising_method: str = 'gaussian'  # 'gaussian', 'bilateral', 'tv'
    enhance_contrast: bool = True
    contrast_method: str = 'clahe'  # 'clahe', 'histogram_eq', 'adaptive'
    terrain_adaptive: bool = True


@dataclass
class ProcessingResult:
    """전처리 결과를 저장하는 데이터클래스"""
    processed_data: np.ndarray
    original_shape: Tuple[int, int]
    processing_steps: List[str]
    terrain_map: Optional[np.ndarray] = None
    quality_metrics: Optional[Dict[str, float]] = None


class WaterColumnProcessor:
    """
    워터컬럼(예인체 바로 아래 주사되지 않는 영역) 처리 클래스
    """
    
    def __init__(self, water_column_width: int = 50):
        """
        워터컬럼 프로세서 초기화
        
        Args:
            water_column_width: 워터컬럼 폭 (픽셀)
        """
        self.water_column_width = water_column_width
        logger.info(f"워터컬럼 프로세서 초기화 - 폭: {water_column_width}")
    
    def detect_water_column(self, data: np.ndarray) -> np.ndarray:
        """
        워터컬럼 영역 자동 검출
        
        Args:
            data: 입력 데이터 [pings, samples]
            
        Returns:
            np.ndarray: 워터컬럼 마스크 (True가 워터컬럼)
        """
        # 각 ping에서 중앙 부근의 낮은 강도 영역을 워터컬럼으로 판단
        num_pings, num_samples = data.shape
        center_col = num_samples // 2
        
        # 워터컬럼 마스크 초기화
        water_column_mask = np.zeros_like(data, dtype=bool)
        
        # 각 ping에 대해 워터컬럼 검출
        for ping_idx in range(num_pings):
            ping_data = data[ping_idx, :]
            
            # 중앙 부근에서 최소값 위치 찾기
            search_start = max(0, center_col - self.water_column_width)
            search_end = min(num_samples, center_col + self.water_column_width)
            
            search_region = ping_data[search_start:search_end]
            min_idx = np.argmin(search_region) + search_start
            
            # 워터컬럼 영역 마킹
            wc_start = max(0, min_idx - self.water_column_width // 2)
            wc_end = min(num_samples, min_idx + self.water_column_width // 2)
            
            water_column_mask[ping_idx, wc_start:wc_end] = True
        
        return water_column_mask
    
    def remove_water_column(self, data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        워터컬럼 영역 제거 또는 보간
        
        Args:
            data: 입력 데이터
            method: 제거 방법 ('zero', 'interpolate', 'median')
            
        Returns:
            np.ndarray: 워터컬럼이 처리된 데이터
        """
        processed_data = data.copy()
        water_column_mask = self.detect_water_column(data)
        
        if method == 'zero':
            # 워터컬럼을 0으로 설정
            processed_data[water_column_mask] = 0
            
        elif method == 'interpolate':
            # 선형 보간으로 채우기
            for ping_idx in range(data.shape[0]):
                mask_row = water_column_mask[ping_idx, :]
                if np.any(mask_row):
                    # 워터컬럼이 아닌 영역의 값들로 보간
                    valid_indices = np.where(~mask_row)[0]
                    invalid_indices = np.where(mask_row)[0]
                    
                    if len(valid_indices) > 1:
                        interpolated_values = np.interp(
                            invalid_indices,
                            valid_indices,
                            processed_data[ping_idx, valid_indices]
                        )
                        processed_data[ping_idx, invalid_indices] = interpolated_values
                        
        elif method == 'median':
            # 주변 픽셀의 중간값으로 채우기
            kernel_size = 5
            for ping_idx in range(data.shape[0]):
                for sample_idx in range(data.shape[1]):
                    if water_column_mask[ping_idx, sample_idx]:
                        # 주변 영역 추출
                        p_start = max(0, ping_idx - kernel_size // 2)
                        p_end = min(data.shape[0], ping_idx + kernel_size // 2 + 1)
                        s_start = max(0, sample_idx - kernel_size // 2)
                        s_end = min(data.shape[1], sample_idx + kernel_size // 2 + 1)
                        
                        neighborhood = data[p_start:p_end, s_start:s_end]
                        neighbor_mask = water_column_mask[p_start:p_end, s_start:s_end]
                        
                        # 워터컬럼이 아닌 이웃 픽셀의 중간값 사용
                        valid_neighbors = neighborhood[~neighbor_mask]
                        if len(valid_neighbors) > 0:
                            processed_data[ping_idx, sample_idx] = np.median(valid_neighbors)
        
        logger.info(f"워터컬럼 처리 완료 - 방법: {method}")
        
        return processed_data


class IntensityNormalizer:
    """
    강도(intensity) 데이터 정규화 클래스
    """
    
    def __init__(self):
        logger.info("강도 정규화기 초기화 완료")
    
    def minmax_normalize(self, data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Min-Max 정규화
        
        Args:
            data: 입력 데이터
            feature_range: 정규화 범위
            
        Returns:
            np.ndarray: 정규화된 데이터
        """
        min_val, max_val = feature_range
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max - data_min == 0:
            return np.full_like(data, min_val)
        
        normalized = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        
        return normalized.astype(np.float32)
    
    def zscore_normalize(self, data: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
        """
        Z-score 정규화
        
        Args:
            data: 입력 데이터
            clip_std: 클리핑할 표준편차 범위
            
        Returns:
            np.ndarray: 정규화된 데이터
        """
        normalized = zscore(data, axis=None, nan_policy='omit')
        
        # 극값 클리핑
        if clip_std > 0:
            normalized = np.clip(normalized, -clip_std, clip_std)
        
        # 0-1 범위로 재정규화
        normalized = (normalized + clip_std) / (2 * clip_std)
        
        return normalized.astype(np.float32)
    
    def histogram_equalize(self, data: np.ndarray, bins: int = 256) -> np.ndarray:
        """
        히스토그램 평활화
        
        Args:
            data: 입력 데이터
            bins: 히스토그램 bin 수
            
        Returns:
            np.ndarray: 평활화된 데이터
        """
        # 0-1 범위로 정규화
        normalized = self.minmax_normalize(data)
        
        # 히스토그램 평활화 적용
        equalized = exposure.equalize_hist(normalized, nbins=bins)
        
        return equalized.astype(np.float32)
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        지정된 방법으로 정규화
        
        Args:
            data: 입력 데이터
            method: 정규화 방법
            
        Returns:
            np.ndarray: 정규화된 데이터
        """
        if method == 'minmax':
            return self.minmax_normalize(data)
        elif method == 'zscore':
            return self.zscore_normalize(data)
        elif method == 'histogram':
            return self.histogram_equalize(data)
        else:
            logger.warning(f"알 수 없는 정규화 방법: {method}")
            return self.minmax_normalize(data)


class NoiseReducer:
    """
    노이즈 제거 클래스
    """
    
    def __init__(self):
        logger.info("노이즈 제거기 초기화 완료")
    
    def gaussian_filter(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        가우시안 필터 적용
        
        Args:
            data: 입력 데이터
            sigma: 가우시안 표준편차
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        return ndimage.gaussian_filter(data, sigma=sigma)
    
    def bilateral_filter(self, data: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        양방향 필터 적용 (엣지 보존)
        
        Args:
            data: 입력 데이터
            d: 필터 크기
            sigma_color: 색상 차이에 대한 시그마
            sigma_space: 거리에 대한 시그마
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        # 데이터를 0-255 범위로 변환
        data_uint8 = (data * 255).astype(np.uint8)
        
        # 양방향 필터 적용
        if CV2_AVAILABLE:
            filtered = cv2.bilateralFilter(data_uint8, d, sigma_color, sigma_space)
        else:
            # OpenCV 없을 때 가우시안 필터로 대체
            from scipy.ndimage import gaussian_filter
            filtered = gaussian_filter(data_uint8, sigma=1.0)
        
        # 다시 float 범위로 변환
        return (filtered / 255.0).astype(np.float32)
    
    def tv_denoise(self, data: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """
        Total Variation 노이즈 제거
        
        Args:
            data: 입력 데이터
            weight: 정규화 가중치
            
        Returns:
            np.ndarray: 노이즈가 제거된 데이터
        """
        return restoration.denoise_tv_chambolle(data, weight=weight)
    
    def median_filter(self, data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        중간값 필터 적용
        
        Args:
            data: 입력 데이터
            kernel_size: 커널 크기
            
        Returns:
            np.ndarray: 필터링된 데이터
        """
        return ndimage.median_filter(data, size=kernel_size)
    
    def reduce_noise(self, data: np.ndarray, method: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        노이즈 제거 적용
        
        Args:
            data: 입력 데이터
            method: 노이즈 제거 방법
            **kwargs: 방법별 추가 매개변수
            
        Returns:
            np.ndarray: 노이즈가 제거된 데이터
        """
        if method == 'gaussian':
            return self.gaussian_filter(data, **kwargs)
        elif method == 'bilateral':
            return self.bilateral_filter(data, **kwargs)
        elif method == 'tv':
            return self.tv_denoise(data, **kwargs)
        elif method == 'median':
            return self.median_filter(data, **kwargs)
        else:
            logger.warning(f"알 수 없는 노이즈 제거 방법: {method}")
            return self.gaussian_filter(data)


class ContrastEnhancer:
    """
    대비 향상 클래스
    """
    
    def __init__(self):
        logger.info("대비 향상기 초기화 완료")
    
    def clahe(self, data: np.ndarray, clip_limit: float = 2.0, 
             tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        
        Args:
            data: 입력 데이터
            clip_limit: 클리핑 한계
            tile_grid_size: 타일 격자 크기
            
        Returns:
            np.ndarray: 대비가 향상된 데이터
        """
        # 0-255 범위로 변환
        data_uint8 = (data * 255).astype(np.uint8)
        
        # CLAHE 객체 생성 및 적용
        if CV2_AVAILABLE:
            clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe_obj.apply(data_uint8)
        else:
            # OpenCV 없을 때 skimage의 adaptive histogram equalization 사용
            from skimage.exposure import equalize_adapthist
            enhanced = (equalize_adapthist(data) * 255).astype(np.uint8)
        
        # float 범위로 재변환
        return (enhanced / 255.0).astype(np.float32)
    
    def adaptive_histogram_eq(self, data: np.ndarray) -> np.ndarray:
        """
        적응형 히스토그램 평활화
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 평활화된 데이터
        """
        return exposure.equalize_adapthist(data)
    
    def gamma_correction(self, data: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        감마 보정
        
        Args:
            data: 입력 데이터
            gamma: 감마 값
            
        Returns:
            np.ndarray: 감마 보정된 데이터
        """
        return exposure.adjust_gamma(data, gamma=gamma)
    
    def enhance_contrast(self, data: np.ndarray, method: str = 'clahe', **kwargs) -> np.ndarray:
        """
        대비 향상 적용
        
        Args:
            data: 입력 데이터
            method: 향상 방법
            **kwargs: 방법별 추가 매개변수
            
        Returns:
            np.ndarray: 대비가 향상된 데이터
        """
        if method == 'clahe':
            return self.clahe(data, **kwargs)
        elif method == 'adaptive':
            return self.adaptive_histogram_eq(data)
        elif method == 'gamma':
            return self.gamma_correction(data, **kwargs)
        else:
            logger.warning(f"알 수 없는 대비 향상 방법: {method}")
            return self.clahe(data)


class TerrainClassifier:
    """
    지형 분류 클래스
    """
    
    def __init__(self):
        logger.info("지형 분류기 초기화 완료")
    
    def analyze_texture(self, data: np.ndarray, window_size: int = 32) -> Dict[str, np.ndarray]:
        """
        텍스처 특성 분석
        
        Args:
            data: 입력 데이터
            window_size: 분석 윈도우 크기
            
        Returns:
            Dict[str, np.ndarray]: 텍스처 특성 맵들
        """
        # 텍스처 특성 맵 초기화
        texture_maps = {
            'variance': np.zeros_like(data),
            'entropy': np.zeros_like(data),
            'contrast': np.zeros_like(data)
        }
        
        half_window = window_size // 2
        
        for i in range(half_window, data.shape[0] - half_window):
            for j in range(half_window, data.shape[1] - half_window):
                # 윈도우 영역 추출
                window = data[i-half_window:i+half_window, j-half_window:j+half_window]
                
                # 분산 계산
                texture_maps['variance'][i, j] = np.var(window)
                
                # 엔트로피 계산
                hist, _ = np.histogram(window, bins=32, range=(0, 1))
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # 0 제거
                texture_maps['entropy'][i, j] = -np.sum(hist * np.log2(hist))
                
                # 대비 계산 (표준편차)
                texture_maps['contrast'][i, j] = np.std(window)
        
        return texture_maps
    
    def classify_terrain(self, data: np.ndarray) -> np.ndarray:
        """
        지형 분류 수행
        
        Args:
            data: 입력 데이터
            
        Returns:
            np.ndarray: 지형 분류 맵
        """
        # 텍스처 특성 분석
        texture_maps = self.analyze_texture(data)
        
        variance_map = texture_maps['variance']
        entropy_map = texture_maps['entropy']
        contrast_map = texture_maps['contrast']
        
        # 지형 분류 맵 초기화
        terrain_map = np.full(data.shape, TerrainType.UNKNOWN.value, dtype=object)
        
        # 임계값 설정 (데이터에 따라 조정 필요)
        high_variance_thresh = np.percentile(variance_map, 75)
        low_variance_thresh = np.percentile(variance_map, 25)
        high_entropy_thresh = np.percentile(entropy_map, 75)
        
        # 지형 분류 규칙
        # 암반: 높은 분산, 높은 대비
        rock_mask = (variance_map > high_variance_thresh) & (contrast_map > np.percentile(contrast_map, 70))
        terrain_map[rock_mask] = TerrainType.ROCK.value
        
        # 모래: 중간 분산, 낮은 엔트로피
        sand_mask = ((variance_map > low_variance_thresh) & (variance_map < high_variance_thresh) & 
                    (entropy_map < np.percentile(entropy_map, 50)))
        terrain_map[sand_mask] = TerrainType.SAND.value
        
        # 뻘: 낮은 분산, 균일한 텍스처
        mud_mask = (variance_map < low_variance_thresh) & (contrast_map < np.percentile(contrast_map, 30))
        terrain_map[mud_mask] = TerrainType.MUD.value
        
        # 나머지는 혼합 지형
        unknown_mask = terrain_map == TerrainType.UNKNOWN.value
        terrain_map[unknown_mask] = TerrainType.MIXED.value
        
        logger.info("지형 분류 완료")
        
        return terrain_map


class Preprocessor:
    """
    통합 전처리 파이프라인
    """
    
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        """
        전처리기 초기화
        
        Args:
            config: 전처리 설정
        """
        self.config = config
        
        # 각 처리 모듈 초기화
        self.water_column_processor = WaterColumnProcessor(config.water_column_width)
        self.intensity_normalizer = IntensityNormalizer()
        self.noise_reducer = NoiseReducer()
        self.contrast_enhancer = ContrastEnhancer()
        self.terrain_classifier = TerrainClassifier()
        
        logger.info("통합 전처리기 초기화 완료")
    
    def process(self, data: np.ndarray) -> ProcessingResult:
        """
        전체 전처리 파이프라인 수행
        
        Args:
            data: 원본 사이드스캔 소나 데이터
            
        Returns:
            ProcessingResult: 전처리 결과
        """
        original_shape = data.shape
        processed_data = data.copy().astype(np.float32)
        processing_steps = []
        
        logger.info(f"전처리 시작 - 입력 크기: {original_shape}")
        
        # 1. 워터컬럼 처리
        if self.config.remove_water_column:
            processed_data = self.water_column_processor.remove_water_column(processed_data)
            processing_steps.append("water_column_removal")
            logger.info("워터컬럼 제거 완료")
        
        # 2. 강도 정규화
        if self.config.normalize_intensity:
            processed_data = self.intensity_normalizer.normalize(
                processed_data, 
                method=self.config.normalization_method
            )
            processing_steps.append(f"intensity_normalization_{self.config.normalization_method}")
            logger.info(f"강도 정규화 완료 - 방법: {self.config.normalization_method}")
        
        # 3. 노이즈 제거
        if self.config.apply_denoising:
            processed_data = self.noise_reducer.reduce_noise(
                processed_data,
                method=self.config.denoising_method
            )
            processing_steps.append(f"denoising_{self.config.denoising_method}")
            logger.info(f"노이즈 제거 완료 - 방법: {self.config.denoising_method}")
        
        # 4. 대비 향상
        if self.config.enhance_contrast:
            processed_data = self.contrast_enhancer.enhance_contrast(
                processed_data,
                method=self.config.contrast_method
            )
            processing_steps.append(f"contrast_enhancement_{self.config.contrast_method}")
            logger.info(f"대비 향상 완료 - 방법: {self.config.contrast_method}")
        
        # 5. 지형 분류 (옵션)
        terrain_map = None
        if self.config.terrain_adaptive:
            terrain_map = self.terrain_classifier.classify_terrain(processed_data)
            processing_steps.append("terrain_classification")
            logger.info("지형 분류 완료")
        
        # 6. 품질 메트릭 계산
        quality_metrics = self._calculate_quality_metrics(data, processed_data)
        
        result = ProcessingResult(
            processed_data=processed_data,
            original_shape=original_shape,
            processing_steps=processing_steps,
            terrain_map=terrain_map,
            quality_metrics=quality_metrics
        )
        
        logger.info("전처리 파이프라인 완료")
        
        return result
    
    def _calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        처리 품질 메트릭 계산
        
        Args:
            original: 원본 데이터
            processed: 처리된 데이터
            
        Returns:
            Dict[str, float]: 품질 메트릭
        """
        metrics = {}
        
        try:
            # 신호 대 잡음 비 (SNR)
            signal_power = np.mean(processed ** 2)
            noise_power = np.mean((original - processed) ** 2)
            if noise_power > 0:
                metrics['snr'] = 10 * np.log10(signal_power / noise_power)
            else:
                metrics['snr'] = float('inf')
            
            # 엣지 보존 지수
            original_edges = filters.sobel(original)
            processed_edges = filters.sobel(processed)
            edge_correlation = np.corrcoef(original_edges.flatten(), processed_edges.flatten())[0, 1]
            metrics['edge_preservation'] = edge_correlation if not np.isnan(edge_correlation) else 0.0
            
            # 대비 개선 비율
            original_contrast = np.std(original)
            processed_contrast = np.std(processed)
            metrics['contrast_improvement'] = processed_contrast / original_contrast if original_contrast > 0 else 1.0
            
            # 동적 범위
            metrics['dynamic_range'] = np.max(processed) - np.min(processed)
            
        except Exception as e:
            logger.warning(f"품질 메트릭 계산 실패: {e}")
            metrics = {'snr': 0.0, 'edge_preservation': 0.0, 'contrast_improvement': 1.0, 'dynamic_range': 1.0}
        
        return metrics
    
    def process_with_terrain_adaptation(self, data: np.ndarray) -> ProcessingResult:
        """
        지형별 적응형 전처리 수행
        
        Args:
            data: 원본 데이터
            
        Returns:
            ProcessingResult: 전처리 결과
        """
        # 먼저 기본 전처리 수행
        basic_result = self.process(data)
        
        if basic_result.terrain_map is None:
            return basic_result
        
        # 지형별 추가 처리
        processed_data = basic_result.processed_data.copy()
        terrain_map = basic_result.terrain_map
        
        # 각 지형 유형별 후처리
        for terrain_type in [TerrainType.SAND, TerrainType.MUD, TerrainType.ROCK]:
            mask = terrain_map == terrain_type.value
            
            if np.any(mask):
                if terrain_type == TerrainType.SAND:
                    # 모래: 추가 노이즈 제거
                    sand_data = processed_data[mask]
                    processed_data[mask] = ndimage.gaussian_filter(sand_data.reshape(-1, 1), sigma=0.5).flatten()
                    
                elif terrain_type == TerrainType.MUD:
                    # 뻘: 대비 향상
                    mud_region = processed_data * mask.astype(float)
                    mud_region = np.where(mask, self.contrast_enhancer.gamma_correction(mud_region, gamma=0.8), mud_region)
                    processed_data = np.where(mask, mud_region, processed_data)
                    
                elif terrain_type == TerrainType.ROCK:
                    # 암반: 엣지 보존 필터
                    rock_region = processed_data * mask.astype(float)
                    rock_region = np.where(mask, self.noise_reducer.bilateral_filter(rock_region), rock_region)
                    processed_data = np.where(mask, rock_region, processed_data)
        
        # 결과 업데이트
        basic_result.processed_data = processed_data
        basic_result.processing_steps.append("terrain_adaptive_processing")
        
        logger.info("지형별 적응형 전처리 완료")
        
        return basic_result