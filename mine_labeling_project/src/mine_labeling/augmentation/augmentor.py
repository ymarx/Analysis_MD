"""
Data Augmentation Module

Applies 9 augmentation techniques to mine and background samples.
Uses albumentations library for efficient transformations.

Techniques:
1. Horizontal flip
2. Vertical flip
3. Rotation (±15°)
4. Scale (±10%)
5. Brightness (±20%)
6. Contrast (±20%)
7. Gaussian blur
8. Gaussian noise
9. Elastic transform

Author: Mine Detection Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Using basic augmentations only.")


class Augmentor:
    """
    Applies data augmentation to intensity-label pairs

    Supports 9 augmentation techniques with configurable parameters
    """

    def __init__(self,
                 augmentation_factor: int = 9,
                 random_seed: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentor

        Args:
            augmentation_factor: Number of augmented versions per sample (default: 9)
            random_seed: Random seed for reproducibility
            config: Configuration dictionary with technique parameters
        """
        self.augmentation_factor = augmentation_factor
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Load configuration
        self.config = config or {}

        # Create augmentation pipelines
        self.pipelines = self._create_augmentation_pipelines()

    def _create_augmentation_pipelines(self) -> List:
        """
        Create augmentation pipelines for each technique

        Returns:
            List of albumentations Compose objects
        """
        if not ALBUMENTATIONS_AVAILABLE:
            return []

        techniques = self.config.get('techniques', {})

        pipelines = []

        # 1. Horizontal flip
        if techniques.get('horizontal_flip', {}).get('enabled', True):
            pipelines.append(
                A.Compose([
                    A.HorizontalFlip(p=1.0)
                ])
            )

        # 2. Vertical flip
        if techniques.get('vertical_flip', {}).get('enabled', True):
            pipelines.append(
                A.Compose([
                    A.VerticalFlip(p=1.0)
                ])
            )

        # 3. Rotation
        if techniques.get('rotation', {}).get('enabled', True):
            limit = techniques.get('rotation', {}).get('limit', 15)
            pipelines.append(
                A.Compose([
                    A.Rotate(limit=limit, p=1.0, border_mode=0)
                ])
            )

        # 4. Scale
        if techniques.get('scale', {}).get('enabled', True):
            scale_limit = techniques.get('scale', {}).get('scale_limit', 0.1)
            pipelines.append(
                A.Compose([
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=scale_limit,
                        rotate_limit=0,
                        p=1.0,
                        border_mode=0
                    )
                ])
            )

        # 5. Brightness
        if techniques.get('brightness', {}).get('enabled', True):
            limit = techniques.get('brightness', {}).get('limit', 0.2)
            pipelines.append(
                A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=limit,
                        contrast_limit=0,
                        p=1.0
                    )
                ])
            )

        # 6. Contrast
        if techniques.get('contrast', {}).get('enabled', True):
            limit = techniques.get('contrast', {}).get('limit', 0.2)
            pipelines.append(
                A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=0,
                        contrast_limit=limit,
                        p=1.0
                    )
                ])
            )

        # 7. Gaussian blur
        if techniques.get('gaussian_blur', {}).get('enabled', True):
            blur_limit = techniques.get('gaussian_blur', {}).get('blur_limit', [3, 7])
            pipelines.append(
                A.Compose([
                    A.GaussianBlur(blur_limit=blur_limit, p=1.0)
                ])
            )

        # 8. Gaussian noise
        if techniques.get('gaussian_noise', {}).get('enabled', True):
            var_limit = techniques.get('gaussian_noise', {}).get('var_limit', [10, 50])
            pipelines.append(
                A.Compose([
                    A.GaussNoise(var_limit=var_limit, p=1.0)
                ])
            )

        # 9. Elastic transform
        if techniques.get('elastic_transform', {}).get('enabled', True):
            alpha = techniques.get('elastic_transform', {}).get('alpha', 1)
            sigma = techniques.get('elastic_transform', {}).get('sigma', 50)
            pipelines.append(
                A.Compose([
                    A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0, border_mode=0)
                ])
            )

        return pipelines

    def augment_sample(self,
                      intensity: np.ndarray,
                      label: np.ndarray,
                      technique_idx: int) -> Dict[str, np.ndarray]:
        """
        Apply single augmentation technique to sample

        Args:
            intensity: Intensity patch (H, W)
            label: Label mask (H, W)
            technique_idx: Index of augmentation technique

        Returns:
            Dictionary with augmented intensity and label
        """
        if not ALBUMENTATIONS_AVAILABLE or technique_idx >= len(self.pipelines):
            # Return original if augmentation not available
            return {
                'intensity': intensity.copy(),
                'label': label.copy()
            }

        pipeline = self.pipelines[technique_idx]

        # Convert to uint8 for albumentations
        if intensity.dtype == np.float32 or intensity.dtype == np.float64:
            intensity_uint8 = (intensity * 255).astype(np.uint8)
        else:
            intensity_uint8 = intensity.astype(np.uint8)

        label_uint8 = label.astype(np.uint8)

        # Apply augmentation
        augmented = pipeline(image=intensity_uint8, mask=label_uint8)

        # Convert back to original dtype
        if intensity.dtype == np.float32 or intensity.dtype == np.float64:
            augmented_intensity = augmented['image'].astype(np.float32) / 255.0
        else:
            augmented_intensity = augmented['image']

        augmented_label = augmented['mask']

        return {
            'intensity': augmented_intensity,
            'label': augmented_label
        }

    def augment_all(self,
                   intensity: np.ndarray,
                   label: np.ndarray,
                   num_augmentations: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        """
        Apply all augmentation techniques to sample

        Args:
            intensity: Intensity patch
            label: Label mask
            num_augmentations: Number of augmentations (default: self.augmentation_factor)

        Returns:
            List of augmented samples
        """
        if num_augmentations is None:
            num_augmentations = self.augmentation_factor

        augmented_samples = []

        num_techniques = len(self.pipelines)

        for i in range(num_augmentations):
            technique_idx = i % num_techniques

            augmented = self.augment_sample(intensity, label, technique_idx)
            augmented['technique_idx'] = technique_idx
            augmented['technique_name'] = self.get_technique_name(technique_idx)

            augmented_samples.append(augmented)

        return augmented_samples

    def get_technique_name(self, idx: int) -> str:
        """
        Get technique name from index

        Args:
            idx: Technique index

        Returns:
            Technique name
        """
        technique_names = [
            'horizontal_flip',
            'vertical_flip',
            'rotation',
            'scale',
            'brightness',
            'contrast',
            'gaussian_blur',
            'gaussian_noise',
            'elastic_transform'
        ]

        if idx < len(technique_names):
            return technique_names[idx]
        else:
            return f'technique_{idx}'

    def process_dataset(self,
                       samples: List[Dict[str, Any]],
                       save_original: bool = False) -> List[Dict[str, Any]]:
        """
        Process entire dataset with augmentation

        Args:
            samples: List of original samples
            save_original: Include original samples in output (default: False)

        Returns:
            List of augmented samples
        """
        augmented_dataset = []

        for sample in samples:
            intensity = sample['intensity']
            label = sample['label']

            # Apply all augmentations
            augmented_samples = self.augment_all(intensity, label)

            # Add metadata
            for aug_sample in augmented_samples:
                aug_data = {
                    'intensity': aug_sample['intensity'],
                    'label': aug_sample['label'],
                    'original_id': sample.get('mine_id') or sample.get('sample_id', 0),
                    'technique': aug_sample['technique_name'],
                    'is_augmented': True
                }

                # Copy additional metadata
                if 'name' in sample:
                    aug_data['name'] = sample['name']

                augmented_dataset.append(aug_data)

            # Optionally include original
            if save_original:
                original_data = {
                    'intensity': intensity.copy(),
                    'label': label.copy(),
                    'original_id': sample.get('mine_id') or sample.get('sample_id', 0),
                    'technique': 'original',
                    'is_augmented': False
                }

                if 'name' in sample:
                    original_data['name'] = sample['name']

                augmented_dataset.append(original_data)

        return augmented_dataset

    def save_augmented_samples(self,
                              augmented_samples: List[Dict[str, Any]],
                              output_dir: Path,
                              separate_folders: bool = True) -> None:
        """
        Save augmented samples to disk

        Args:
            augmented_samples: List of augmented samples
            output_dir: Output directory
            separate_folders: Create separate folder per technique (default: True)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(augmented_samples):
            technique = sample['technique']
            original_id = sample['original_id']

            if separate_folders:
                # Create technique-specific folder
                technique_dir = output_dir / technique
                technique_dir.mkdir(exist_ok=True)
                output_path = technique_dir / f'sample_{original_id:03d}_{technique}.npz'
            else:
                output_path = output_dir / f'sample_{original_id:03d}_{technique}.npz'

            np.savez_compressed(
                output_path,
                intensity=sample['intensity'],
                label=sample['label'],
                technique=technique,
                original_id=original_id
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Augmentor':
        """
        Create Augmentor from configuration

        Args:
            config: Configuration dictionary

        Returns:
            Augmentor instance
        """
        aug_config = config.get('augmentation', {})

        return cls(
            augmentation_factor=aug_config.get('augmentation_factor', 9),
            random_seed=aug_config.get('random_seed', None),
            config=aug_config
        )

    def __repr__(self) -> str:
        return (f"Augmentor("
                f"factor={self.augmentation_factor}, "
                f"techniques={len(self.pipelines)})")
