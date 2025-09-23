"""
Label Generator Module
======================
레이블 생성 모듈 - 좌표 매핑 기반 객체 레이블 생성
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class LabelGenerator:
    """레이블 생성 클래스"""

    def __init__(self,
                 patch_size: Tuple[int, int] = (64, 64),
                 background_ratio: float = 2.0):
        """
        Initialize Label Generator

        Args:
            patch_size: 패치 크기 (width, height)
            background_ratio: 배경 샘플 비율 (객체 대비)
        """
        self.patch_size = patch_size
        self.background_ratio = background_ratio
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self,
                coordinate_mapping: Dict[str, Any],
                intensity_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        좌표 매핑을 기반으로 레이블 생성

        Args:
            coordinate_mapping: 좌표 매핑 데이터
            intensity_data: 강도 데이터 (선택적)

        Returns:
            레이블 데이터
        """
        self.logger.info("Generating labels from coordinate mapping")

        try:
            mappings = coordinate_mapping.get('mappings', [])
            image_size = coordinate_mapping.get('image_size', (1024, 3862))

            # Extract positive samples (objects)
            positive_samples = self._extract_positive_samples(mappings, image_size)

            # Generate negative samples (background)
            negative_samples = self._generate_negative_samples(
                positive_samples, image_size, intensity_data
            )

            # Combine samples
            all_samples = positive_samples + negative_samples

            # Create labels array
            labels = self._create_labels_array(all_samples)

            # Extract patch locations
            patch_locations = self._extract_patch_locations(all_samples)

            return {
                'samples': all_samples,
                'labels': labels,
                'patch_locations': patch_locations,
                'positive_count': len(positive_samples),
                'negative_count': len(negative_samples),
                'class_distribution': self._calculate_class_distribution(labels),
                'generation_info': {
                    'patch_size': self.patch_size,
                    'background_ratio': self.background_ratio,
                    'image_size': image_size
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate labels: {e}")
            raise

    def _extract_positive_samples(self,
                                mappings: List[Dict],
                                image_size: Tuple[int, int]) -> List[Dict]:
        """객체 위치에서 positive 샘플 추출"""
        positive_samples = []

        for mapping in mappings:
            # Get object center
            center_x = mapping['pixel_x']
            center_y = mapping['pixel_y']
            bbox = mapping['bbox']

            # Calculate patch bounds
            patch_bounds = self._calculate_patch_bounds(
                center_x, center_y, image_size
            )

            sample = {
                'type': 'positive',
                'label': 1,
                'center_x': center_x,
                'center_y': center_y,
                'patch_bounds': patch_bounds,
                'bbox': bbox,
                'object_id': mapping['object_id'],
                'gps_point_id': mapping['gps_point_id'],
                'latitude': mapping['latitude'],
                'longitude': mapping['longitude'],
                'confidence': mapping.get('mapping_confidence', 1.0)
            }

            positive_samples.append(sample)

        self.logger.debug(f"Extracted {len(positive_samples)} positive samples")
        return positive_samples

    def _generate_negative_samples(self,
                                 positive_samples: List[Dict],
                                 image_size: Tuple[int, int],
                                 intensity_data: Optional[np.ndarray] = None) -> List[Dict]:
        """Negative 샘플 생성 (배경 영역)"""
        num_negatives = int(len(positive_samples) * self.background_ratio)
        negative_samples = []

        # Create exclusion mask for positive regions
        exclusion_mask = self._create_exclusion_mask(positive_samples, image_size)

        # Generate random background samples
        attempts = 0
        max_attempts = num_negatives * 10

        while len(negative_samples) < num_negatives and attempts < max_attempts:
            attempts += 1

            # Random location
            x = np.random.randint(self.patch_size[0]//2,
                                image_size[0] - self.patch_size[0]//2)
            y = np.random.randint(self.patch_size[1]//2,
                                image_size[1] - self.patch_size[1]//2)

            # Check if location is valid (not overlapping with objects)
            if self._is_valid_background_location(x, y, exclusion_mask):
                patch_bounds = self._calculate_patch_bounds(x, y, image_size)

                # Additional quality check with intensity data
                if intensity_data is not None:
                    if not self._is_good_background_patch(patch_bounds, intensity_data):
                        continue

                sample = {
                    'type': 'negative',
                    'label': 0,
                    'center_x': x,
                    'center_y': y,
                    'patch_bounds': patch_bounds,
                    'bbox': None,
                    'object_id': None,
                    'gps_point_id': None,
                    'latitude': None,
                    'longitude': None,
                    'confidence': 1.0
                }

                negative_samples.append(sample)

        self.logger.debug(f"Generated {len(negative_samples)} negative samples")
        return negative_samples

    def _calculate_patch_bounds(self,
                              center_x: int,
                              center_y: int,
                              image_size: Tuple[int, int]) -> Dict[str, int]:
        """패치 경계 계산"""
        half_width = self.patch_size[0] // 2
        half_height = self.patch_size[1] // 2

        x_min = max(0, center_x - half_width)
        x_max = min(image_size[0], center_x + half_width)
        y_min = max(0, center_y - half_height)
        y_max = min(image_size[1], center_y + half_height)

        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }

    def _create_exclusion_mask(self,
                             positive_samples: List[Dict],
                             image_size: Tuple[int, int]) -> np.ndarray:
        """Positive 샘플 주변 제외 마스크 생성"""
        mask = np.zeros(image_size[::-1], dtype=bool)  # (height, width)

        for sample in positive_samples:
            bounds = sample['patch_bounds']

            # Expand exclusion area around objects
            margin = max(self.patch_size) // 2
            y_min = max(0, bounds['y_min'] - margin)
            y_max = min(image_size[1], bounds['y_max'] + margin)
            x_min = max(0, bounds['x_min'] - margin)
            x_max = min(image_size[0], bounds['x_max'] + margin)

            mask[y_min:y_max, x_min:x_max] = True

        return mask

    def _is_valid_background_location(self,
                                    x: int,
                                    y: int,
                                    exclusion_mask: np.ndarray) -> bool:
        """배경 위치 유효성 검사"""
        # Check if location is within exclusion mask
        if y < exclusion_mask.shape[0] and x < exclusion_mask.shape[1]:
            return not exclusion_mask[y, x]
        return True

    def _is_good_background_patch(self,
                                patch_bounds: Dict[str, int],
                                intensity_data: np.ndarray) -> bool:
        """배경 패치 품질 검사"""
        try:
            # Extract patch from intensity data
            y_min, y_max = patch_bounds['y_min'], patch_bounds['y_max']
            x_min, x_max = patch_bounds['x_min'], patch_bounds['x_max']

            if (y_max <= intensity_data.shape[0] and
                x_max <= intensity_data.shape[1]):

                patch = intensity_data[y_min:y_max, x_min:x_max]

                # Check for sufficient variation (not empty water)
                std_dev = np.std(patch)
                return std_dev > 0.01  # Threshold for variation

            return True

        except Exception:
            return True  # Default to valid if check fails

    def _create_labels_array(self, samples: List[Dict]) -> np.ndarray:
        """레이블 배열 생성"""
        labels = np.array([sample['label'] for sample in samples])
        return labels

    def _extract_patch_locations(self, samples: List[Dict]) -> List[Dict]:
        """패치 위치 정보 추출"""
        locations = []

        for i, sample in enumerate(samples):
            location = {
                'sample_id': i,
                'center_x': sample['center_x'],
                'center_y': sample['center_y'],
                'patch_bounds': sample['patch_bounds'],
                'label': sample['label'],
                'type': sample['type']
            }
            locations.append(location)

        return locations

    def _calculate_class_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """클래스 분포 계산"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        distribution = {
            'total_samples': total,
            'classes': {}
        }

        for cls, count in zip(unique, counts):
            class_name = 'positive' if cls == 1 else 'negative'
            distribution['classes'][class_name] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }

        return distribution

    def visualize_labels(self,
                        label_data: Dict[str, Any],
                        image_size: Tuple[int, int],
                        output_path: Optional[str] = None) -> Image.Image:
        """레이블 시각화"""
        # Create blank image
        img = Image.new('RGB', image_size, color='black')
        draw = ImageDraw.Draw(img)

        samples = label_data.get('samples', [])

        for sample in samples:
            bounds = sample['patch_bounds']
            color = 'red' if sample['label'] == 1 else 'blue'

            # Draw rectangle
            draw.rectangle([
                bounds['x_min'], bounds['y_min'],
                bounds['x_max'], bounds['y_max']
            ], outline=color, width=2)

            # Draw center point
            center_x, center_y = sample['center_x'], sample['center_y']
            draw.ellipse([
                center_x - 3, center_y - 3,
                center_x + 3, center_y + 3
            ], fill=color)

        if output_path:
            img.save(output_path)
            self.logger.info(f"Label visualization saved to {output_path}")

        return img

    def get_label_statistics(self, label_data: Dict[str, Any]) -> Dict[str, Any]:
        """레이블 통계 반환"""
        samples = label_data.get('samples', [])

        # Patch size statistics
        patch_sizes = []
        for sample in samples:
            bounds = sample['patch_bounds']
            patch_sizes.append(bounds['width'] * bounds['height'])

        # Confidence statistics for positive samples
        positive_confidences = [
            sample['confidence'] for sample in samples
            if sample['label'] == 1
        ]

        stats = {
            'total_samples': len(samples),
            'class_distribution': label_data.get('class_distribution', {}),
            'patch_statistics': {
                'target_size': self.patch_size,
                'actual_sizes': {
                    'mean': np.mean(patch_sizes) if patch_sizes else 0,
                    'std': np.std(patch_sizes) if patch_sizes else 0,
                    'min': np.min(patch_sizes) if patch_sizes else 0,
                    'max': np.max(patch_sizes) if patch_sizes else 0
                }
            },
            'confidence_statistics': {
                'positive_samples': {
                    'mean': np.mean(positive_confidences) if positive_confidences else 0,
                    'std': np.std(positive_confidences) if positive_confidences else 0,
                    'min': np.min(positive_confidences) if positive_confidences else 0,
                    'max': np.max(positive_confidences) if positive_confidences else 0
                }
            }
        }

        return stats