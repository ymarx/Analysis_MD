"""
소나 데이터 전용 데이터 증강 엔진

불균형 데이터셋 문제 해결을 위한 전문화된 증강 기법들을 제공합니다.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import transform, exposure, util
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """데이터 증강 설정"""
    # 회전 관련
    rotation_range: Tuple[float, float] = (-180, 180)
    rotation_probability: float = 0.7
    
    # 스케일링
    scale_range: Tuple[float, float] = (0.8, 1.2)
    scale_probability: float = 0.5
    
    # 번역/이동
    translation_range: Tuple[float, float] = (-0.1, 0.1)  # 이미지 크기 비율
    translation_probability: float = 0.4
    
    # 노이즈
    noise_std_range: Tuple[float, float] = (0.01, 0.05)
    noise_probability: float = 0.6
    
    # 밝기/대비
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    brightness_probability: float = 0.5
    contrast_probability: float = 0.5
    
    # 소나 전용
    beam_angle_variation: float = 5.0  # 도
    range_distortion: float = 0.05     # 거리 왜곡 비율
    acoustic_shadow_probability: float = 0.3
    
    # 증강 강도
    augmentation_strength: float = 0.5  # 0~1


class BaseAugmentation(ABC):
    """기본 증강 클래스"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    @abstractmethod
    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """증강 적용"""
        pass
    
    def should_apply(self, probability: float) -> bool:
        """확률적 적용 여부 결정"""
        return random.random() < probability


class GeometricAugmentation(BaseAugmentation):
    """기하학적 변환 증강"""
    
    def apply_rotation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """회전 변환"""
        if not self.should_apply(self.config.rotation_probability):
            return image, mask
        
        angle = random.uniform(*self.config.rotation_range)
        
        # 이미지 회전
        rotated_image = transform.rotate(image, angle, preserve_range=True, mode='reflect')
        
        # 마스크 회전 (있는 경우)
        rotated_mask = None
        if mask is not None:
            rotated_mask = transform.rotate(mask, angle, preserve_range=True, mode='constant', cval=0)
            rotated_mask = (rotated_mask > 0.5).astype(mask.dtype)
        
        return rotated_image.astype(image.dtype), rotated_mask
    
    def apply_scaling(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """스케일링 변환"""
        if not self.should_apply(self.config.scale_probability):
            return image, mask
        
        scale_factor = random.uniform(*self.config.scale_range)
        
        # 중앙에서 스케일링
        h, w = image.shape[:2]
        scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
        
        # 이미지 리사이즈
        scaled_image = cv2.resize(image, (scaled_w, scaled_h))
        
        # 원본 크기로 크롭/패딩
        if scale_factor > 1:
            # 크롭
            start_h = (scaled_h - h) // 2
            start_w = (scaled_w - w) // 2
            result_image = scaled_image[start_h:start_h+h, start_w:start_w+w]
        else:
            # 패딩
            pad_h = (h - scaled_h) // 2
            pad_w = (w - scaled_w) // 2
            
            if len(image.shape) == 3:
                result_image = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
                result_image[pad_h:pad_h+scaled_h, pad_w:pad_w+scaled_w] = scaled_image
            else:
                result_image = np.zeros((h, w), dtype=image.dtype)
                result_image[pad_h:pad_h+scaled_h, pad_w:pad_w+scaled_w] = scaled_image
        
        # 마스크 처리
        result_mask = None
        if mask is not None:
            scaled_mask = cv2.resize(mask.astype(np.float32), (scaled_w, scaled_h))
            
            if scale_factor > 1:
                result_mask = scaled_mask[start_h:start_h+h, start_w:start_w+w]
            else:
                result_mask = np.zeros((h, w), dtype=mask.dtype)
                result_mask[pad_h:pad_h+scaled_h, pad_w:pad_w+scaled_w] = scaled_mask
            
            result_mask = (result_mask > 0.5).astype(mask.dtype)
        
        return result_image, result_mask
    
    def apply_translation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """평행이동 변환"""
        if not self.should_apply(self.config.translation_probability):
            return image, mask
        
        h, w = image.shape[:2]
        
        # 이동량 계산
        tx = int(random.uniform(*self.config.translation_range) * w)
        ty = int(random.uniform(*self.config.translation_range) * h)
        
        # 변환 행렬
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 이미지 이동
        translated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # 마스크 이동
        translated_mask = None
        if mask is not None:
            translated_mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            translated_mask = (translated_mask > 0.5).astype(mask.dtype)
        
        return translated_image, translated_mask
    
    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """모든 기하학적 변환 적용"""
        result_image, result_mask = image.copy(), mask
        
        # 순차적으로 변환 적용
        result_image, result_mask = self.apply_rotation(result_image, result_mask)
        result_image, result_mask = self.apply_scaling(result_image, result_mask)
        result_image, result_mask = self.apply_translation(result_image, result_mask)
        
        return result_image, result_mask


class PhotometricAugmentation(BaseAugmentation):
    """광도 변환 증강"""
    
    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """가우시안 노이즈 추가"""
        if not self.should_apply(self.config.noise_probability):
            return image
        
        noise_std = random.uniform(*self.config.noise_std_range)
        
        if image.dtype == np.uint8:
            noise = np.random.normal(0, noise_std * 255, image.shape)
        else:
            noise = np.random.normal(0, noise_std, image.shape)
        
        noisy_image = image.astype(np.float64) + noise
        
        # 값 범위 클리핑
        if image.dtype == np.uint8:
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        else:
            noisy_image = np.clip(noisy_image, 0, 1).astype(image.dtype)
        
        return noisy_image
    
    def apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """밝기 조정"""
        if not self.should_apply(self.config.brightness_probability):
            return image
        
        brightness_factor = random.uniform(*self.config.brightness_range)
        
        if image.dtype == np.uint8:
            adjusted = image.astype(np.float64) * brightness_factor
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        else:
            adjusted = np.clip(image * brightness_factor, 0, 1)
        
        return adjusted
    
    def apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 조정"""
        if not self.should_apply(self.config.contrast_probability):
            return image
        
        contrast_factor = random.uniform(*self.config.contrast_range)
        
        if image.dtype == np.uint8:
            # 중간값을 기준으로 대비 조정
            mean_val = 127.5
            adjusted = (image.astype(np.float64) - mean_val) * contrast_factor + mean_val
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        else:
            mean_val = 0.5
            adjusted = (image - mean_val) * contrast_factor + mean_val
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """감마 보정"""
        gamma_range = (0.7, 1.3)
        gamma = random.uniform(*gamma_range)
        
        if image.dtype == np.uint8:
            corrected = exposure.adjust_gamma(image, gamma)
        else:
            corrected = np.power(image, gamma)
            corrected = np.clip(corrected, 0, 1)
        
        return corrected
    
    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """모든 광도 변환 적용"""
        result_image = image.copy()
        
        result_image = self.apply_noise(result_image)
        result_image = self.apply_brightness(result_image)
        result_image = self.apply_contrast(result_image)
        result_image = self.apply_gamma_correction(result_image)
        
        return result_image, mask


class SonarSpecificAugmentation(BaseAugmentation):
    """소나 데이터 전용 증강"""
    
    def simulate_acoustic_shadow(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """음향 그림자 시뮬레이션"""
        if not self.should_apply(self.config.acoustic_shadow_probability):
            return image
        
        h, w = image.shape[:2]
        
        # 그림자 영역 생성
        shadow_length = random.randint(h//10, h//3)
        shadow_width = random.randint(w//20, w//8)
        
        # 랜덤 위치에서 시작
        start_x = random.randint(0, w - shadow_width)
        start_y = random.randint(0, h - shadow_length)
        
        # 그림자 강도
        shadow_intensity = random.uniform(0.2, 0.7)
        
        # 그림자 적용
        shadow_mask = np.ones_like(image)
        shadow_mask[start_y:start_y+shadow_length, start_x:start_x+shadow_width] = shadow_intensity
        
        # 가장자리 블러 처리
        shadow_mask = gaussian_filter(shadow_mask, sigma=2)
        
        shadowed_image = image * shadow_mask
        
        return shadowed_image.astype(image.dtype)
    
    def simulate_beam_angle_variation(self, image: np.ndarray) -> np.ndarray:
        """빔 각도 변화 시뮬레이션"""
        h, w = image.shape[:2]
        
        # 각도 변화량
        angle_change = random.uniform(-self.config.beam_angle_variation, 
                                     self.config.beam_angle_variation)
        
        # 사다리꼴 변형으로 빔 각도 변화 모사
        # 상단과 하단의 폭 차이 생성
        width_ratio = 1 + (angle_change / 180.0)  # 각도를 비율로 변환
        
        # 변환 포인트 정의
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        width_offset = int(w * (width_ratio - 1) / 2)
        dst_points = np.float32([
            [max(0, -width_offset), 0],
            [min(w, w + width_offset), 0], 
            [w, h],
            [0, h]
        ])
        
        # 원근 변환 적용
        if width_offset != 0:
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            transformed = cv2.warpPerspective(image, M, (w, h), 
                                            borderMode=cv2.BORDER_REFLECT)
        else:
            transformed = image
        
        return transformed
    
    def simulate_range_distortion(self, image: np.ndarray) -> np.ndarray:
        """거리 왜곡 시뮬레이션"""
        h, w = image.shape[:2]
        
        # 거리 방향(세로 방향) 왜곡
        distortion_strength = random.uniform(-self.config.range_distortion, 
                                           self.config.range_distortion)
        
        # 사인파 형태의 왜곡 생성
        y_indices, x_indices = np.ogrid[:h, :w]
        
        # y 방향 왜곡
        wave_amplitude = distortion_strength * h
        wave_freq = 2 * np.pi / w
        
        y_distorted = y_indices + wave_amplitude * np.sin(wave_freq * x_indices)
        y_distorted = np.clip(y_distorted, 0, h-1)
        
        # 보간을 위한 좌표 생성
        coords = np.array([y_distorted, x_indices])
        
        # 이미지 변형
        if len(image.shape) == 3:
            distorted = np.zeros_like(image)
            for c in range(image.shape[2]):
                distorted[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c], coords, order=1, mode='reflect'
                )
        else:
            distorted = ndimage.map_coordinates(
                image, coords, order=1, mode='reflect'
            )
        
        return distorted.astype(image.dtype)
    
    def simulate_multipath_interference(self, image: np.ndarray) -> np.ndarray:
        """다중 경로 간섭 시뮬레이션"""
        # 약한 고스트 이미지 생성
        ghost_offset_x = random.randint(-10, 10)
        ghost_offset_y = random.randint(5, 20)  # 거리 방향으로 주로 오프셋
        ghost_intensity = random.uniform(0.1, 0.3)
        
        h, w = image.shape[:2]
        
        # 고스트 이미지 생성
        ghost_image = np.zeros_like(image)
        
        # 오프셋 적용하여 고스트 복사
        y_start = max(0, ghost_offset_y)
        y_end = min(h, h + ghost_offset_y)
        x_start = max(0, ghost_offset_x)
        x_end = min(w, w + ghost_offset_x)
        
        src_y_start = max(0, -ghost_offset_y)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -ghost_offset_x)
        src_x_end = src_x_start + (x_end - x_start)
        
        if len(image.shape) == 3:
            ghost_image[y_start:y_end, x_start:x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end] * ghost_intensity
        else:
            ghost_image[y_start:y_end, x_start:x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end] * ghost_intensity
        
        # 원본과 고스트 합성
        interfered = image.astype(np.float64) + ghost_image.astype(np.float64)
        
        # 값 범위 정규화
        if image.dtype == np.uint8:
            interfered = np.clip(interfered, 0, 255).astype(np.uint8)
        else:
            interfered = np.clip(interfered, 0, 1).astype(image.dtype)
        
        return interfered
    
    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """모든 소나 전용 증강 적용"""
        result_image = image.copy()
        
        # 확률적으로 각 효과 적용
        if random.random() < 0.3:
            result_image = self.simulate_acoustic_shadow(result_image, mask)
        
        if random.random() < 0.4:
            result_image = self.simulate_beam_angle_variation(result_image)
        
        if random.random() < 0.2:
            result_image = self.simulate_range_distortion(result_image)
        
        if random.random() < 0.2:
            result_image = self.simulate_multipath_interference(result_image)
        
        return result_image, mask


class AdvancedAugmentationEngine:
    """
    고급 데이터 증강 엔진
    
    여러 증강 기법을 조합하여 균형잡힌 데이터셋을 생성합니다.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        증강 엔진 초기화
        
        Args:
            config: 증강 설정 (기본값 사용시 None)
        """
        self.config = config or AugmentationConfig()
        
        # 증강기 초기화
        self.geometric = GeometricAugmentation(self.config)
        self.photometric = PhotometricAugmentation(self.config)
        self.sonar_specific = SonarSpecificAugmentation(self.config)
        
        logger.info("고급 데이터 증강 엔진 초기화 완료")
    
    def augment_single(self, image: np.ndarray, 
                      mask: Optional[np.ndarray] = None,
                      augmentation_types: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        단일 이미지 증강
        
        Args:
            image: 입력 이미지
            mask: 마스크 (옵션)
            augmentation_types: 적용할 증강 타입 리스트
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (증강된 이미지, 증강된 마스크)
        """
        if augmentation_types is None:
            augmentation_types = ['geometric', 'photometric', 'sonar']
        
        result_image, result_mask = image.copy(), mask
        
        # 강도에 따른 적용 확률 조정
        strength_factor = self.config.augmentation_strength
        
        for aug_type in augmentation_types:
            # 강도에 따라 적용 여부 결정
            if random.random() > strength_factor:
                continue
            
            if aug_type == 'geometric':
                result_image, result_mask = self.geometric.apply(result_image, result_mask)
            elif aug_type == 'photometric':
                result_image, result_mask = self.photometric.apply(result_image, result_mask)
            elif aug_type == 'sonar':
                result_image, result_mask = self.sonar_specific.apply(result_image, result_mask)
        
        return result_image, result_mask
    
    def augment_batch(self, images: List[np.ndarray],
                     masks: Optional[List[np.ndarray]] = None,
                     augmentations_per_image: int = 3) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """
        배치 이미지 증강
        
        Args:
            images: 입력 이미지 리스트
            masks: 마스크 리스트 (옵션)
            augmentations_per_image: 이미지당 생성할 증강 수
            
        Returns:
            Tuple[List[np.ndarray], Optional[List[np.ndarray]]]: (증강된 이미지 리스트, 증강된 마스크 리스트)
        """
        augmented_images = []
        augmented_masks = [] if masks is not None else None
        
        for i, image in enumerate(images):
            current_mask = masks[i] if masks is not None else None
            
            # 원본 추가
            augmented_images.append(image.copy())
            if augmented_masks is not None:
                augmented_masks.append(current_mask.copy() if current_mask is not None else None)
            
            # 증강 버전 생성
            for _ in range(augmentations_per_image):
                aug_image, aug_mask = self.augment_single(image, current_mask)
                
                augmented_images.append(aug_image)
                if augmented_masks is not None:
                    augmented_masks.append(aug_mask)
        
        logger.info(f"배치 증강 완료: {len(images)} → {len(augmented_images)} 이미지")
        
        return augmented_images, augmented_masks
    
    def balance_dataset(self, positive_images: List[np.ndarray],
                       negative_images: List[np.ndarray],
                       positive_masks: Optional[List[np.ndarray]] = None,
                       target_ratio: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                          Optional[List[np.ndarray]], List[int]]:
        """
        클래스 불균형 해결을 위한 데이터셋 균형화
        
        Args:
            positive_images: 양성 샘플 (기물)
            negative_images: 음성 샘플 (배경)
            positive_masks: 양성 샘플 마스크
            target_ratio: 목표 양성:음성 비율
            
        Returns:
            Tuple: (균형화된 이미지, 원본 음성 이미지, 마스크, 라벨)
        """
        n_positive = len(positive_images)
        n_negative = len(negative_images)
        
        target_positive = max(n_positive, int(n_negative * target_ratio))
        target_negative = max(n_negative, int(target_positive / target_ratio))
        
        logger.info(f"데이터셋 균형화: {n_positive}(+) + {n_negative}(-) → {target_positive}(+) + {target_negative}(-)")
        
        balanced_images = []
        balanced_masks = [] if positive_masks is not None else None
        labels = []
        
        # 양성 샘플 증강
        if target_positive > n_positive:
            augmentations_needed = target_positive - n_positive
            augmentations_per_image = augmentations_needed // n_positive + 1
            
            # 원본 추가
            for i, pos_img in enumerate(positive_images):
                balanced_images.append(pos_img)
                labels.append(1)
                if balanced_masks is not None:
                    balanced_masks.append(positive_masks[i] if positive_masks else None)
            
            # 증강 샘플 생성
            for i, pos_img in enumerate(positive_images):
                current_mask = positive_masks[i] if positive_masks else None
                
                for _ in range(augmentations_per_image):
                    if len(balanced_images) >= target_positive:
                        break
                    
                    aug_img, aug_mask = self.augment_single(pos_img, current_mask)
                    balanced_images.append(aug_img)
                    labels.append(1)
                    if balanced_masks is not None:
                        balanced_masks.append(aug_mask)
                
                if len(balanced_images) >= target_positive:
                    break
        else:
            # 양성 샘플 다운샘플링
            selected_indices = random.sample(range(n_positive), target_positive)
            for idx in selected_indices:
                balanced_images.append(positive_images[idx])
                labels.append(1)
                if balanced_masks is not None:
                    balanced_masks.append(positive_masks[idx] if positive_masks else None)
        
        # 음성 샘플 처리
        if target_negative > n_negative:
            # 음성 샘플 증강
            augmentations_needed = target_negative - n_negative
            augmentations_per_image = augmentations_needed // n_negative + 1
            
            # 원본 추가
            for neg_img in negative_images:
                balanced_images.append(neg_img)
                labels.append(0)
                if balanced_masks is not None:
                    balanced_masks.append(None)
            
            # 증강 샘플 생성
            for neg_img in negative_images:
                for _ in range(augmentations_per_image):
                    if sum(1 for l in labels if l == 0) >= target_negative:
                        break
                    
                    # 음성 샘플은 마스크 없이 증강
                    aug_img, _ = self.augment_single(neg_img, None)
                    balanced_images.append(aug_img)
                    labels.append(0)
                    if balanced_masks is not None:
                        balanced_masks.append(None)
                
                if sum(1 for l in labels if l == 0) >= target_negative:
                    break
        else:
            # 음성 샘플 다운샘플링
            selected_indices = random.sample(range(n_negative), target_negative)
            for idx in selected_indices:
                balanced_images.append(negative_images[idx])
                labels.append(0)
                if balanced_masks is not None:
                    balanced_masks.append(None)
        
        logger.info(f"균형화 완료: {len(balanced_images)} 총 샘플 "
                   f"({sum(labels)} 양성, {len(labels) - sum(labels)} 음성)")
        
        return balanced_images, negative_images, balanced_masks, labels
    
    def create_augmentation_pipeline(self, 
                                   strength_schedule: Optional[Dict[str, float]] = None) -> Callable:
        """
        사용자 정의 증강 파이프라인 생성
        
        Args:
            strength_schedule: 증강 타입별 강도 스케줄
            
        Returns:
            Callable: 증강 함수
        """
        if strength_schedule is None:
            strength_schedule = {
                'geometric': 0.7,
                'photometric': 0.5, 
                'sonar': 0.3
            }
        
        def augmentation_pipeline(image: np.ndarray, 
                                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            """사용자 정의 증강 파이프라인"""
            result_image, result_mask = image.copy(), mask
            
            for aug_type, strength in strength_schedule.items():
                if random.random() < strength:
                    if aug_type == 'geometric':
                        result_image, result_mask = self.geometric.apply(result_image, result_mask)
                    elif aug_type == 'photometric':
                        result_image, result_mask = self.photometric.apply(result_image, result_mask)
                    elif aug_type == 'sonar':
                        result_image, result_mask = self.sonar_specific.apply(result_image, result_mask)
            
            return result_image, result_mask
        
        return augmentation_pipeline


class AugmentationValidator:
    """
    데이터 증강 품질 검증기
    
    증강된 데이터의 품질을 평가하고 문제점을 식별합니다.
    """
    
    def __init__(self):
        pass
    
    def validate_augmentation_quality(self, 
                                    original_image: np.ndarray,
                                    augmented_image: np.ndarray) -> Dict[str, float]:
        """
        증강 품질 평가
        
        Args:
            original_image: 원본 이미지
            augmented_image: 증강된 이미지
            
        Returns:
            Dict[str, float]: 품질 지표
        """
        metrics = {}
        
        # 구조적 유사성 (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if len(original_image.shape) == 3:
                orig_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
                aug_gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original_image
                aug_gray = augmented_image
            
            ssim_score = ssim(orig_gray, aug_gray, data_range=orig_gray.max() - orig_gray.min())
            metrics['structural_similarity'] = ssim_score
            
        except ImportError:
            logger.warning("scikit-image SSIM 사용 불가")
            metrics['structural_similarity'] = 0.0
        
        # 히스토그램 유사성
        orig_hist = cv2.calcHist([original_image.ravel()], [0], None, [256], [0, 256])
        aug_hist = cv2.calcHist([augmented_image.ravel()], [0], None, [256], [0, 256])
        
        hist_correlation = cv2.compareHist(orig_hist, aug_hist, cv2.HISTCMP_CORREL)
        metrics['histogram_similarity'] = hist_correlation
        
        # 에너지 보존
        orig_energy = np.sum(original_image.astype(np.float64) ** 2)
        aug_energy = np.sum(augmented_image.astype(np.float64) ** 2)
        
        energy_ratio = aug_energy / (orig_energy + 1e-10)
        metrics['energy_preservation'] = 1.0 - abs(1.0 - energy_ratio)
        
        # 기울기 보존
        orig_grad_x, orig_grad_y = np.gradient(original_image.astype(np.float64))
        aug_grad_x, aug_grad_y = np.gradient(augmented_image.astype(np.float64))
        
        orig_grad_mag = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
        aug_grad_mag = np.sqrt(aug_grad_x**2 + aug_grad_y**2)
        
        grad_correlation = np.corrcoef(orig_grad_mag.ravel(), aug_grad_mag.ravel())[0, 1]
        metrics['gradient_preservation'] = grad_correlation if not np.isnan(grad_correlation) else 0.0
        
        return metrics
    
    def assess_dataset_diversity(self, images: List[np.ndarray]) -> Dict[str, float]:
        """
        데이터셋 다양성 평가
        
        Args:
            images: 이미지 리스트
            
        Returns:
            Dict[str, float]: 다양성 지표
        """
        metrics = {}
        
        if len(images) < 2:
            return {'diversity_score': 0.0}
        
        # 이미지 간 평균 SSIM (낮을수록 다양함)
        ssim_scores = []
        n_comparisons = min(100, len(images) * (len(images) - 1) // 2)  # 계산량 제한
        
        compared_pairs = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if compared_pairs >= n_comparisons:
                    break
                
                try:
                    if len(images[i].shape) == 3:
                        img1 = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
                        img2 = cv2.cvtColor(images[j], cv2.COLOR_RGB2GRAY)
                    else:
                        img1 = images[i]
                        img2 = images[j]
                    
                    from skimage.metrics import structural_similarity as ssim
                    ssim_score = ssim(img1, img2, data_range=img1.max() - img1.min())
                    ssim_scores.append(ssim_score)
                    compared_pairs += 1
                    
                except:
                    continue
            
            if compared_pairs >= n_comparisons:
                break
        
        if ssim_scores:
            avg_similarity = np.mean(ssim_scores)
            metrics['diversity_score'] = 1.0 - avg_similarity  # 낮은 유사성 = 높은 다양성
            metrics['similarity_std'] = np.std(ssim_scores)
        else:
            metrics['diversity_score'] = 0.0
            metrics['similarity_std'] = 0.0
        
        # 히스토그램 분산
        histograms = []
        for img in images[:50]:  # 샘플링하여 계산량 제한
            hist = cv2.calcHist([img.ravel()], [0], None, [64], [0, 256])
            histograms.append(hist.ravel())
        
        if histograms:
            hist_matrix = np.array(histograms)
            hist_variance = np.mean(np.var(hist_matrix, axis=0))
            metrics['histogram_diversity'] = hist_variance
        else:
            metrics['histogram_diversity'] = 0.0
        
        return metrics