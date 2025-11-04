"""
Background Sampler Module

Extracts background samples (no mines) from labeled NPY data for training.
Maintains 1:5 ratio (mine:background) for class balance.

Author: Mine Detection Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class BackgroundSampler:
    """
    Samples background regions from labeled intensity data

    Features:
    - Random sampling from background regions
    - Maintain minimum distance from mines
    - Ensure no overlap between samples
    - Support configurable mine:background ratio
    """

    def __init__(self,
                 patch_size: int = 128,
                 num_samples: int = 125,
                 min_distance_from_mine: int = 50,
                 ensure_no_overlap: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize background sampler

        Args:
            patch_size: Size of extracted patches (default: 128)
            num_samples: Number of background samples to extract (default: 125)
            min_distance_from_mine: Minimum distance from mines in pixels (default: 50)
            ensure_no_overlap: Prevent overlapping samples (default: True)
            random_seed: Random seed for reproducibility
        """
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.min_distance_from_mine = min_distance_from_mine
        self.ensure_no_overlap = ensure_no_overlap
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def create_background_mask(self,
                              label_mask: np.ndarray,
                              annotations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create mask of valid background sampling regions

        Args:
            label_mask: Label mask (H, W)
            annotations: List of mine annotations

        Returns:
            Binary mask where 1 = valid background, 0 = invalid
        """
        height, width = label_mask.shape

        # Start with all non-mine regions as valid
        background_mask = (label_mask == 0).astype(np.uint8)

        # Exclude regions near mines
        if self.min_distance_from_mine > 0:
            for ann in annotations:
                bbox = ann['mapped_npy']

                # Expand bbox by min_distance
                x1 = max(0, bbox['xmin'] - self.min_distance_from_mine)
                y1 = max(0, min(bbox['ymin'], bbox['ymax']) - self.min_distance_from_mine)
                x2 = min(width, bbox['xmax'] + self.min_distance_from_mine)
                y2 = min(height, max(bbox['ymin'], bbox['ymax']) + self.min_distance_from_mine)

                # Mark as invalid
                background_mask[y1:y2, x1:x2] = 0

        return background_mask

    def find_valid_patch_locations(self,
                                   background_mask: np.ndarray,
                                   num_samples: int) -> List[Tuple[int, int]]:
        """
        Find valid locations for background patches

        Args:
            background_mask: Valid background regions mask
            num_samples: Number of samples to find

        Returns:
            List of (y, x) top-left corner coordinates
        """
        height, width = background_mask.shape
        half_patch = self.patch_size // 2

        # Create valid sampling region (ensure patches fit within bounds)
        valid_region = background_mask.copy()
        valid_region[:half_patch, :] = 0
        valid_region[-half_patch:, :] = 0
        valid_region[:, :half_patch] = 0
        valid_region[:, -half_patch:] = 0

        # Get all valid pixel coordinates
        valid_coords = np.argwhere(valid_region > 0)

        if len(valid_coords) < num_samples:
            print(f"Warning: Only {len(valid_coords)} valid locations found, "
                  f"requested {num_samples}")
            num_samples = len(valid_coords)

        # Randomly sample locations
        sampled_indices = np.random.choice(
            len(valid_coords),
            size=num_samples,
            replace=False
        )

        locations = []
        used_mask = np.zeros_like(background_mask)

        for idx in sampled_indices:
            y_center, x_center = valid_coords[idx]

            # Calculate patch bounds
            y1 = y_center - half_patch
            x1 = x_center - half_patch
            y2 = y1 + self.patch_size
            x2 = x1 + self.patch_size

            # Check overlap if required
            if self.ensure_no_overlap:
                if np.any(used_mask[y1:y2, x1:x2] > 0):
                    continue  # Skip overlapping location

                # Mark as used
                used_mask[y1:y2, x1:x2] = 1

            locations.append((y1, x1))

            if len(locations) >= num_samples:
                break

        return locations

    def extract_background_patch(self,
                                intensity: np.ndarray,
                                label_mask: np.ndarray,
                                location: Tuple[int, int],
                                sample_id: int) -> Dict[str, Any]:
        """
        Extract single background patch

        Args:
            intensity: Intensity array
            label_mask: Label mask
            location: (y, x) top-left corner
            sample_id: Sample identifier

        Returns:
            Dictionary with patch data
        """
        y1, x1 = location
        y2 = y1 + self.patch_size
        x2 = x1 + self.patch_size

        # Extract patches
        intensity_patch = intensity[y1:y2, x1:x2].copy()
        label_patch = label_mask[y1:y2, x1:x2].copy()

        # Verify it's actually background
        mine_pixels = np.sum(label_patch > 0)

        return {
            'sample_id': sample_id,
            'intensity': intensity_patch,
            'label': label_patch,
            'location': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            },
            'mine_pixels': int(mine_pixels),
            'is_pure_background': (mine_pixels == 0)
        }

    def extract_background_samples(self,
                                   intensity: np.ndarray,
                                   label_mask: np.ndarray,
                                   annotations: List[Dict[str, Any]],
                                   num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract all background samples

        Args:
            intensity: Intensity array
            label_mask: Label mask
            annotations: List of mine annotations
            num_samples: Number of samples (default: self.num_samples)

        Returns:
            List of extracted background samples
        """
        if num_samples is None:
            num_samples = self.num_samples

        # Create background mask
        background_mask = self.create_background_mask(label_mask, annotations)

        # Find valid locations
        locations = self.find_valid_patch_locations(background_mask, num_samples)

        # Extract patches
        samples = []
        for i, location in enumerate(locations):
            sample_id = i + 1

            sample = self.extract_background_patch(
                intensity, label_mask, location, sample_id
            )

            samples.append(sample)

        return samples

    def get_sampling_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate background sampling statistics

        Args:
            samples: List of extracted samples

        Returns:
            Dictionary with statistics
        """
        total_samples = len(samples)
        pure_background = sum(1 for s in samples if s['is_pure_background'])
        contaminated = total_samples - pure_background

        if contaminated > 0:
            mine_pixel_counts = [s['mine_pixels'] for s in samples if not s['is_pure_background']]
            max_contamination = max(mine_pixel_counts) if mine_pixel_counts else 0
        else:
            max_contamination = 0

        stats = {
            'total_samples': total_samples,
            'pure_background': pure_background,
            'contaminated_samples': contaminated,
            'purity_rate': float(pure_background / total_samples) if total_samples > 0 else 0,
            'max_mine_pixels': int(max_contamination),
            'patch_size': self.patch_size,
            'min_distance_from_mine': self.min_distance_from_mine
        }

        return stats

    def save_samples(self,
                    samples: List[Dict[str, Any]],
                    output_dir: Path,
                    format: str = 'npz') -> None:
        """
        Save extracted background samples to disk

        Args:
            samples: List of extracted samples
            output_dir: Output directory
            format: 'npz' or 'npy' (default: 'npz')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            sample_id = sample['sample_id']

            if format == 'npz':
                output_path = output_dir / f'background_{sample_id:03d}.npz'

                np.savez_compressed(
                    output_path,
                    intensity=sample['intensity'],
                    label=sample['label'],
                    location=np.array([
                        sample['location']['x1'],
                        sample['location']['y1'],
                        sample['location']['x2'],
                        sample['location']['y2']
                    ]),
                    is_pure_background=sample['is_pure_background']
                )

            elif format == 'npy':
                intensity_path = output_dir / f'background_{sample_id:03d}_intensity.npy'
                label_path = output_dir / f'background_{sample_id:03d}_label.npy'

                np.save(intensity_path, sample['intensity'])
                np.save(label_path, sample['label'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BackgroundSampler':
        """
        Create BackgroundSampler from configuration

        Args:
            config: Configuration dictionary

        Returns:
            BackgroundSampler instance
        """
        sampling_config = config.get('sampling', {})

        return cls(
            patch_size=sampling_config.get('patch_size', 128),
            num_samples=sampling_config.get('background_samples', 125),
            min_distance_from_mine=sampling_config.get('min_distance_from_mine', 50),
            ensure_no_overlap=sampling_config.get('ensure_no_overlap', True),
            random_seed=sampling_config.get('random_seed', None)
        )

    def __repr__(self) -> str:
        return (f"BackgroundSampler("
                f"patch_size={self.patch_size}, "
                f"num_samples={self.num_samples}, "
                f"min_distance={self.min_distance_from_mine})")
