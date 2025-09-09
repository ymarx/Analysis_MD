"""
최적화된 Gabor 필터 뱅크 특징 추출기

다중 주파수와 방향성을 가진 Gabor 필터를 사용하여 방향성 텍스처 특징을 추출합니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from skimage.filters import gabor, gabor_kernel
    from skimage import exposure
    SKIMAGE_GABOR_AVAILABLE = True
except ImportError:
    SKIMAGE_GABOR_AVAILABLE = False

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


def create_gabor_kernel_pure(frequency: float, theta: float, 
                           sigma_x: float, sigma_y: float,
                           n_stds: float = 3, offset: float = 0,
                           kernel_size: Optional[int] = None) -> np.ndarray:
    """
    순수 Python으로 Gabor 커널 생성 (scikit-image 대체)
    
    Args:
        frequency: 주파수
        theta: 회전각 (라디안)
        sigma_x: X방향 표준편차
        sigma_y: Y방향 표준편차  
        n_stds: 커널 크기 결정 계수
        offset: 위상 오프셋
        kernel_size: 고정 커널 크기 (None이면 자동 계산)
        
    Returns:
        np.ndarray: 복소수 Gabor 커널
    """
    if kernel_size is None:
        # 커널 크기 자동 계산
        size_x = int(2 * n_stds * sigma_x + 1)
        size_y = int(2 * n_stds * sigma_y + 1)
        kernel_size = max(size_x, size_y)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # 좌표 그리드 생성
    center = kernel_size // 2
    x = np.arange(-center, center + 1, dtype=np.float64)
    y = np.arange(-center, center + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    
    # 회전 변환
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    x_theta = X * cos_theta + Y * sin_theta
    y_theta = -X * sin_theta + Y * cos_theta
    
    # Gaussian envelope
    gaussian = np.exp(-0.5 * ((x_theta / sigma_x) ** 2 + (y_theta / sigma_y) ** 2))
    
    # 복소 정현파
    sinusoid = np.exp(1j * (2 * math.pi * frequency * x_theta + offset))
    
    # Gabor 커널 = Gaussian × 복소 정현파
    gabor_kernel = gaussian * sinusoid
    
    return gabor_kernel


def apply_gabor_filter_pure(image: np.ndarray, gabor_kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    순수 Python으로 Gabor 필터 적용
    
    Args:
        image: 입력 이미지
        gabor_kernel: Gabor 커널 (복소수)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (magnitude, phase)
    """
    # 실수부와 허수부로 분리하여 컨볼루션
    real_response = convolve(image, gabor_kernel.real, mode='constant')
    imag_response = convolve(image, gabor_kernel.imag, mode='constant')
    
    # 크기와 위상 계산
    magnitude = np.sqrt(real_response**2 + imag_response**2)
    phase = np.arctan2(imag_response, real_response)
    
    return magnitude, phase


def rgb_to_grayscale_pure(image: np.ndarray) -> np.ndarray:
    """
    순수 Python으로 RGB를 그레이스케일로 변환 (OpenCV 대체)
    
    Args:
        image: RGB 이미지 (H, W, 3)
        
    Returns:
        np.ndarray: 그레이스케일 이미지 (H, W)
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # ITU-R BT.709 표준 가중치
        return np.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    else:
        return image


def adaptive_histogram_equalization_pure(image: np.ndarray, 
                                       clip_limit: float = 0.03,
                                       nbins: int = 256) -> np.ndarray:
    """
    간단한 히스토그램 균등화 (scikit-image exposure 대체)
    
    Args:
        image: 입력 이미지
        clip_limit: 클리핑 제한
        nbins: 히스토그램 빈 수
        
    Returns:
        np.ndarray: 균등화된 이미지
    """
    # 이미지를 0-1 범위로 정규화
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
    
    # 히스토그램 계산 및 균등화
    hist, bins = np.histogram(image_norm.flatten(), nbins, [0, 1])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # 이미지에 적용
    equalized = np.interp(image_norm.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(image.shape)
    
    return equalized.astype(np.float64)


@dataclass
class GaborConfig:
    """Gabor 필터 설정 파라미터"""
    frequency: float = 0.1
    theta: float = 0
    sigma_x: float = 2
    sigma_y: float = 2
    n_stds: float = 3
    offset: float = 0
    
    def __post_init__(self):
        # 파라미터 유효성 검사
        self.frequency = max(0.01, min(self.frequency, 1.0))
        self.theta = self.theta % (2 * np.pi)
        self.sigma_x = max(0.5, min(self.sigma_x, 10.0))
        self.sigma_y = max(0.5, min(self.sigma_y, 10.0))


class OptimizedGaborBank:
    """
    최적화된 Gabor 필터 뱅크
    
    다양한 주파수와 방향을 가진 Gabor 필터들을 효율적으로 관리합니다.
    """
    
    def __init__(self, 
                 n_frequencies: int = 6,
                 n_orientations: int = 8,
                 frequency_range: Tuple[float, float] = (0.01, 0.3),
                 sigma_range: Tuple[float, float] = (1, 4)):
        """
        Gabor 필터 뱅크 초기화
        
        Args:
            n_frequencies: 주파수 개수
            n_orientations: 방향 개수
            frequency_range: 주파수 범위 (min, max)
            sigma_range: 시그마 범위 (min, max)
        """
        self.n_frequencies = n_frequencies
        self.n_orientations = n_orientations
        self.frequency_range = frequency_range
        self.sigma_range = sigma_range
        
        # 주파수와 방향 생성
        self.frequencies = np.logspace(
            np.log10(frequency_range[0]), 
            np.log10(frequency_range[1]), 
            n_frequencies
        )
        
        self.orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
        
        # 시그마 값들 (주파수에 따라 적응적)
        self.sigmas = np.linspace(sigma_range[0], sigma_range[1], n_frequencies)
        
        # 필터 뱅크 생성
        self.filter_bank = self._create_filter_bank()
        
        logger.info(f"Gabor 필터 뱅크 생성 - {n_frequencies}개 주파수 × {n_orientations}개 방향 = {len(self.filter_bank)}개 필터")
    
    def _create_filter_bank(self) -> List[Tuple[np.ndarray, GaborConfig]]:
        """Gabor 필터 뱅크 생성"""
        filter_bank = []
        
        for i, freq in enumerate(self.frequencies):
            for theta in self.orientations:
                config = GaborConfig(
                    frequency=freq,
                    theta=theta,
                    sigma_x=self.sigmas[i],
                    sigma_y=self.sigmas[i]
                )
                
                # Gabor 커널 생성 (라이브러리 의존성에 따라)
                if SKIMAGE_GABOR_AVAILABLE:
                    try:
                        kernel = gabor_kernel(
                            frequency=config.frequency,
                            theta=config.theta,
                            sigma_x=config.sigma_x,
                            sigma_y=config.sigma_y,
                            n_stds=config.n_stds,
                            offset=config.offset
                        )
                        filter_bank.append((kernel, config))
                    except Exception as e:
                        logger.warning(f"scikit-image gabor_kernel 생성 실패: {e}, 순수 Python 사용")
                        kernel = create_gabor_kernel_pure(
                            frequency=config.frequency,
                            theta=config.theta,
                            sigma_x=config.sigma_x,
                            sigma_y=config.sigma_y,
                            n_stds=config.n_stds,
                            offset=config.offset
                        )
                        filter_bank.append((kernel, config))
                else:
                    # 순수 Python 구현 사용
                    kernel = create_gabor_kernel_pure(
                        frequency=config.frequency,
                        theta=config.theta,
                        sigma_x=config.sigma_x,
                        sigma_y=config.sigma_y,
                        n_stds=config.n_stds,
                        offset=config.offset
                    )
                    filter_bank.append((kernel, config))
        
        return filter_bank
    
    def apply_single_filter(self, image: np.ndarray, 
                          kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 Gabor 필터 적용
        
        Args:
            image: 입력 이미지
            kernel: Gabor 커널 (복소수 또는 실수)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (magnitude, phase)
        """
        try:
            # 커널이 복소수인지 확인
            if np.iscomplexobj(kernel):
                # 복소수 커널 - 순수 Python 구현
                return apply_gabor_filter_pure(image, kernel)
            else:
                # 실수 커널 - 기존 방식
                if hasattr(kernel, 'real') and hasattr(kernel, 'imag'):
                    real_response = convolve(image, kernel.real, mode='constant')
                    imag_response = convolve(image, kernel.imag, mode='constant')
                else:
                    # 실수 응답만 있는 경우
                    real_response = convolve(image, kernel, mode='constant')
                    imag_response = np.zeros_like(real_response)
                
                magnitude = np.sqrt(real_response**2 + imag_response**2)
                phase = np.arctan2(imag_response, real_response)
                
                return magnitude, phase
            
        except Exception as e:
            logger.warning(f"Gabor 필터 적용 실패: {e}")
            return np.zeros_like(image), np.zeros_like(image)
    
    def extract_filter_responses(self, image: np.ndarray, 
                               use_parallel: bool = True) -> Dict[str, np.ndarray]:
        """
        모든 필터에 대한 응답 추출
        
        Args:
            image: 입력 이미지
            use_parallel: 병렬 처리 사용 여부
            
        Returns:
            Dict[str, np.ndarray]: 필터 응답 딕셔너리
        """
        # 그레이스케일 변환 (OpenCV 의존성 없이)
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except:
                    image = rgb_to_grayscale_pure(image)
            else:
                image = rgb_to_grayscale_pure(image)
        
        # 이미지 정규화 (scikit-image 의존성 없이)
        if SKIMAGE_GABOR_AVAILABLE:
            try:
                image = exposure.equalize_adapthist(image)
            except:
                image = adaptive_histogram_equalization_pure(image)
        else:
            image = adaptive_histogram_equalization_pure(image)
        
        responses = {}
        
        if use_parallel and len(self.filter_bank) > 4:
            # 병렬 처리
            responses = self._extract_parallel(image)
        else:
            # 순차 처리
            responses = self._extract_sequential(image)
        
        return responses
    
    def _extract_sequential(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """순차적 필터 응답 추출"""
        responses = {}
        
        for i, (kernel, config) in enumerate(self.filter_bank):
            filter_name = f"f{config.frequency:.3f}_t{config.theta:.2f}"
            
            try:
                if SKIMAGE_GABOR_AVAILABLE:
                    try:
                        # scikit-image gabor 함수 사용
                        magnitude, phase = gabor(
                            image,
                            frequency=config.frequency,
                            theta=config.theta,
                            sigma_x=config.sigma_x,
                            sigma_y=config.sigma_y
                        )
                    except Exception as e:
                        logger.debug(f"scikit-image gabor 실패, 직접 적용 사용: {e}")
                        magnitude, phase = self.apply_single_filter(image, kernel)
                else:
                    # 직접 필터 적용
                    magnitude, phase = self.apply_single_filter(image, kernel)
                
                responses[f"{filter_name}_mag"] = magnitude
                responses[f"{filter_name}_phase"] = phase
                
            except Exception as e:
                logger.warning(f"필터 {filter_name} 적용 실패: {e}")
                responses[f"{filter_name}_mag"] = np.zeros_like(image)
                responses[f"{filter_name}_phase"] = np.zeros_like(image)
        
        return responses
    
    def _extract_parallel(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """병렬 필터 응답 추출"""
        responses = {}
        
        def process_filter(args):
            i, kernel, config = args
            filter_name = f"f{config.frequency:.3f}_t{config.theta:.2f}"
            
            try:
                if SKIMAGE_GABOR_AVAILABLE:
                    try:
                        magnitude, phase = gabor(
                            image,
                            frequency=config.frequency,
                            theta=config.theta,
                            sigma_x=config.sigma_x,
                            sigma_y=config.sigma_y
                        )
                    except Exception as e:
                        logger.debug(f"scikit-image gabor 실패, 직접 적용 사용: {e}")
                        magnitude, phase = self.apply_single_filter(image, kernel)
                else:
                    # 직접 필터 적용
                    magnitude, phase = self.apply_single_filter(image, kernel)
                    
                return filter_name, magnitude, phase
            except Exception as e:
                logger.warning(f"필터 {filter_name} 처리 실패: {e}")
                return filter_name, np.zeros_like(image), np.zeros_like(image)
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_filter, (i, kernel, config))
                for i, (kernel, config) in enumerate(self.filter_bank)
            ]
            
            for future in as_completed(futures):
                filter_name, magnitude, phase = future.result()
                responses[f"{filter_name}_mag"] = magnitude
                responses[f"{filter_name}_phase"] = phase
        
        return responses


class GaborFeatureExtractor:
    """
    Gabor 특징 추출기
    
    Gabor 필터 응답으로부터 다양한 통계적 특징을 추출합니다.
    """
    
    def __init__(self, 
                 n_frequencies: int = 6,
                 n_orientations: int = 8,
                 patch_size: int = 32):
        """
        Gabor 특징 추출기 초기화
        
        Args:
            n_frequencies: 주파수 개수
            n_orientations: 방향 개수
            patch_size: 패치 크기 (지역 특징 계산용)
        """
        self.gabor_bank = OptimizedGaborBank(n_frequencies, n_orientations)
        self.patch_size = patch_size
        
        logger.info(f"Gabor 특징 추출기 초기화 - 패치 크기: {patch_size}")
    
    def extract_statistical_features(self, responses: Dict[str, np.ndarray]) -> np.ndarray:
        """
        통계적 특징 추출
        
        Args:
            responses: Gabor 필터 응답 딕셔너리
            
        Returns:
            np.ndarray: 통계적 특징 벡터
        """
        features = []
        
        for name, response in responses.items():
            if response.size == 0:
                # 빈 응답에 대한 기본 특징
                stats = np.zeros(8, dtype=np.float32)
            else:
                # 기본 통계량
                mean_val = np.mean(response)
                std_val = np.std(response)
                max_val = np.max(response)
                min_val = np.min(response)
                
                # 고차 모멘트
                skewness = self._calculate_skewness(response)
                kurtosis = self._calculate_kurtosis(response)
                
                # 에너지 및 엔트로피
                energy = np.sum(response**2)
                entropy = self._calculate_entropy(response)
                
                stats = np.array([
                    mean_val, std_val, max_val, min_val,
                    skewness, kurtosis, energy, entropy
                ], dtype=np.float32)
            
            features.append(stats)
        
        if features:
            combined = np.concatenate(features)
        else:
            combined = np.array([], dtype=np.float32)
        
        return combined
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도 계산"""
        if data.size < 2 or np.std(data) == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        
        return float(skew)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        if data.size < 2 or np.std(data) == 0:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
        
        return float(kurt)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """엔트로피 계산"""
        if data.size == 0:
            return 0.0
        
        # 히스토그램 생성
        hist, _ = np.histogram(data, bins=64, density=True)
        
        # 엔트로피 계산
        hist = hist + 1e-10  # log(0) 방지
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def extract_spatial_features(self, responses: Dict[str, np.ndarray]) -> np.ndarray:
        """
        공간적 특징 추출 (패치 기반)
        
        Args:
            responses: Gabor 필터 응답 딕셔너리
            
        Returns:
            np.ndarray: 공간적 특징 벡터
        """
        spatial_features = []
        
        for name, response in responses.items():
            if response.size == 0:
                continue
            
            h, w = response.shape
            
            # 패치별 평균과 표준편차 계산
            patch_means = []
            patch_stds = []
            
            for i in range(0, h, self.patch_size):
                for j in range(0, w, self.patch_size):
                    patch = response[i:i+self.patch_size, j:j+self.patch_size]
                    
                    if patch.size > 0:
                        patch_means.append(np.mean(patch))
                        patch_stds.append(np.std(patch))
            
            if patch_means:
                # 패치 통계량의 통계량
                mean_of_means = np.mean(patch_means)
                std_of_means = np.std(patch_means)
                mean_of_stds = np.mean(patch_stds)
                std_of_stds = np.std(patch_stds)
                
                spatial_stats = np.array([
                    mean_of_means, std_of_means, mean_of_stds, std_of_stds
                ], dtype=np.float32)
                
                spatial_features.append(spatial_stats)
        
        if spatial_features:
            combined = np.concatenate(spatial_features)
        else:
            combined = np.array([], dtype=np.float32)
        
        return combined
    
    def extract_directional_features(self, responses: Dict[str, np.ndarray]) -> np.ndarray:
        """
        방향성 특징 추출
        
        Args:
            responses: Gabor 필터 응답 딕셔너리
            
        Returns:
            np.ndarray: 방향성 특징 벡터
        """
        # 방향별 응답 그룹화
        orientation_groups = {}
        
        for name, response in responses.items():
            if '_mag' not in name or response.size == 0:
                continue
            
            # 방향 정보 추출
            try:
                parts = name.split('_')
                theta_str = parts[1][1:]  # 't' 제거
                theta = float(theta_str)
                
                if theta not in orientation_groups:
                    orientation_groups[theta] = []
                
                orientation_groups[theta].append(response)
                
            except (ValueError, IndexError):
                continue
        
        directional_features = []
        
        # 각 방향별 통계량 계산
        for theta, responses_list in orientation_groups.items():
            if not responses_list:
                continue
            
            # 방향별 평균 응답
            avg_response = np.mean(np.stack(responses_list), axis=0)
            
            # 방향성 에너지
            directional_energy = np.sum(avg_response**2)
            
            # 방향성 집중도 (최대값 비율)
            max_response = np.max(avg_response)
            concentration = max_response / (np.mean(avg_response) + 1e-10)
            
            direction_stats = np.array([
                directional_energy, concentration, np.mean(avg_response), np.std(avg_response)
            ], dtype=np.float32)
            
            directional_features.append(direction_stats)
        
        if directional_features:
            combined = np.concatenate(directional_features)
        else:
            combined = np.array([], dtype=np.float32)
        
        return combined
    
    def extract_comprehensive_features(self, image: np.ndarray) -> np.ndarray:
        """
        종합적인 Gabor 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 결합된 Gabor 특징 벡터
        """
        # Gabor 필터 응답 추출
        responses = self.gabor_bank.extract_filter_responses(image)
        
        all_features = []
        
        # 1. 통계적 특징
        try:
            statistical_features = self.extract_statistical_features(responses)
            if len(statistical_features) > 0:
                all_features.append(statistical_features)
        except Exception as e:
            logger.warning(f"통계적 특징 추출 실패: {e}")
        
        # 2. 공간적 특징
        try:
            spatial_features = self.extract_spatial_features(responses)
            if len(spatial_features) > 0:
                all_features.append(spatial_features)
        except Exception as e:
            logger.warning(f"공간적 특징 추출 실패: {e}")
        
        # 3. 방향성 특징
        try:
            directional_features = self.extract_directional_features(responses)
            if len(directional_features) > 0:
                all_features.append(directional_features)
        except Exception as e:
            logger.warning(f"방향성 특징 추출 실패: {e}")
        
        # 특징 결합
        if all_features:
            combined = np.concatenate(all_features)
        else:
            logger.error("모든 Gabor 특징 추출 실패")
            combined = np.array([], dtype=np.float32)
        
        logger.info(f"종합 Gabor 특징 추출 완료: {len(combined)} 차원")
        
        return combined


class AdaptiveGaborExtractor:
    """
    적응형 Gabor 특징 추출기
    
    이미지 특성에 따라 Gabor 필터 매개변수를 자동으로 조정합니다.
    """
    
    def __init__(self):
        self.base_extractor = GaborFeatureExtractor()
        
    def analyze_image_frequency_content(self, image: np.ndarray) -> Dict[str, float]:
        """
        이미지 주파수 내용 분석
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict[str, float]: 주파수 특성 딕셔너리
        """
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except:
                    image = rgb_to_grayscale_pure(image)
            else:
                image = rgb_to_grayscale_pure(image)
        
        # FFT 분석
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 중심에서의 거리 계산
        h, w = image.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 주파수 대역별 에너지 계산
        low_freq_energy = np.sum(magnitude_spectrum[distance < min(h, w) * 0.1])
        mid_freq_energy = np.sum(magnitude_spectrum[(distance >= min(h, w) * 0.1) & 
                                                   (distance < min(h, w) * 0.3)])
        high_freq_energy = np.sum(magnitude_spectrum[distance >= min(h, w) * 0.3])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            characteristics = {
                'low_freq_ratio': low_freq_energy / total_energy,
                'mid_freq_ratio': mid_freq_energy / total_energy,
                'high_freq_ratio': high_freq_energy / total_energy,
                'dominant_frequency': self._find_dominant_frequency(magnitude_spectrum, distance),
                'frequency_spread': np.std(distance[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)])
            }
        else:
            characteristics = {
                'low_freq_ratio': 0.0,
                'mid_freq_ratio': 0.0, 
                'high_freq_ratio': 0.0,
                'dominant_frequency': 0.0,
                'frequency_spread': 0.0
            }
        
        return characteristics
    
    def _find_dominant_frequency(self, magnitude_spectrum: np.ndarray, 
                               distance: np.ndarray) -> float:
        """주요 주파수 찾기"""
        # 거리별 평균 크기 계산
        max_distance = int(np.max(distance))
        freq_profile = []
        
        for d in range(1, max_distance):
            mask = (distance >= d-0.5) & (distance < d+0.5)
            if np.any(mask):
                avg_magnitude = np.mean(magnitude_spectrum[mask])
                freq_profile.append(avg_magnitude)
            else:
                freq_profile.append(0)
        
        if freq_profile:
            dominant_distance = np.argmax(freq_profile) + 1
            # 정규화된 주파수로 변환
            dominant_freq = dominant_distance / max_distance
            return min(dominant_freq, 0.5)  # Nyquist 제한
        else:
            return 0.1
    
    def select_optimal_gabor_params(self, characteristics: Dict[str, float]) -> Tuple[int, int, Tuple[float, float]]:
        """
        최적 Gabor 매개변수 선택
        
        Args:
            characteristics: 이미지 주파수 특성
            
        Returns:
            Tuple[int, int, Tuple[float, float]]: (n_frequencies, n_orientations, frequency_range)
        """
        # 주파수 특성에 따른 매개변수 조정
        if characteristics['high_freq_ratio'] > 0.4:
            # 고주파 성분이 많은 경우
            n_frequencies = 8
            frequency_range = (0.05, 0.5)
        elif characteristics['low_freq_ratio'] > 0.6:
            # 저주파 성분이 많은 경우
            n_frequencies = 4
            frequency_range = (0.01, 0.2)
        else:
            # 균형잡힌 경우
            n_frequencies = 6
            frequency_range = (0.02, 0.3)
        
        # 텍스처 복잡도에 따른 방향 개수 조정
        if characteristics['frequency_spread'] > 10:
            n_orientations = 12  # 복잡한 텍스처
        elif characteristics['frequency_spread'] < 5:
            n_orientations = 4   # 단순한 텍스처
        else:
            n_orientations = 8   # 기본값
        
        return n_frequencies, n_orientations, frequency_range
    
    def extract_adaptive_features(self, image: np.ndarray) -> np.ndarray:
        """
        적응형 Gabor 특징 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 적응형 Gabor 특징
        """
        # 이미지 특성 분석
        characteristics = self.analyze_image_frequency_content(image)
        
        # 최적 매개변수 선택
        n_freq, n_orient, freq_range = self.select_optimal_gabor_params(characteristics)
        
        # 적응형 추출기 생성
        adaptive_extractor = GaborFeatureExtractor(
            n_frequencies=n_freq,
            n_orientations=n_orient
        )
        
        # 주파수 범위 업데이트
        adaptive_extractor.gabor_bank.frequencies = np.logspace(
            np.log10(freq_range[0]), 
            np.log10(freq_range[1]), 
            n_freq
        )
        adaptive_extractor.gabor_bank.filter_bank = adaptive_extractor.gabor_bank._create_filter_bank()
        
        # 특징 추출
        features = adaptive_extractor.extract_comprehensive_features(image)
        
        logger.info(f"적응형 Gabor 특징 추출 완료: {len(features)} 차원 "
                   f"(freq={n_freq}, orient={n_orient})")
        
        return features