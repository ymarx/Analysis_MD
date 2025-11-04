"""
NPY to BMP Converter

Converts NPY intensity data (6400×5137) to human-readable BMP (1024×5137)
for interactive labeling.

Key Features:
- Preserves orientation (no flip, origin='upper')
- Contrast enhancement (CLAHE)
- Width reduction: 6400 → 1024 (6.25x)
- Height preserved: 5137 → 5137

Author: Mine Detection Team
Date: 2025-11-04
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class NpyToBmpConverter:
    """
    Converts NPY intensity data to BMP format for labeling

    Coordinate transformation:
    - NPY: (6400, 5137) → BMP: (1024, 5137)
    - Scale factor: 6.25x (width reduction)
    - Direction: PRESERVED (no flip)
    """

    def __init__(self,
                 target_width: int = 1024,
                 apply_clahe: bool = True,
                 clip_limit: float = 2.0,
                 tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize converter

        Args:
            target_width: Target BMP width (default: 1024)
            apply_clahe: Apply CLAHE contrast enhancement (default: True)
            clip_limit: CLAHE clip limit (default: 2.0)
            tile_grid_size: CLAHE tile grid size (default: (8, 8))
        """
        self.target_width = target_width
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def normalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """
        Normalize intensity data to [0, 255] uint8

        Args:
            intensity: Input intensity array (H, W) float32

        Returns:
            Normalized uint8 array
        """
        # Handle different input ranges
        if intensity.dtype == np.float32 or intensity.dtype == np.float64:
            if intensity.max() <= 1.0:
                # Already normalized to [0, 1]
                normalized = (intensity * 255).astype(np.uint8)
            else:
                # Scale to [0, 255]
                normalized = ((intensity - intensity.min()) /
                            (intensity.max() - intensity.min()) * 255).astype(np.uint8)
        else:
            # Already uint8
            normalized = intensity.astype(np.uint8)

        return normalized

    def apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: Input grayscale image uint8

        Returns:
            Enhanced image
        """
        if not self.apply_clahe:
            return image

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

        enhanced = clahe.apply(image)

        return enhanced

    def resize_width(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """
        Resize image width while preserving height

        Args:
            image: Input image (H, W)
            target_width: Target width

        Returns:
            Resized image (H, target_width)
        """
        height, width = image.shape

        # Use cv2.INTER_AREA for downsampling (best quality)
        resized = cv2.resize(
            image,
            (target_width, height),
            interpolation=cv2.INTER_AREA
        )

        return resized

    def convert_to_bmp(self,
                      npy_data: np.ndarray,
                      output_path: Optional[Path] = None) -> np.ndarray:
        """
        Convert NPY intensity data to BMP format

        Args:
            npy_data: NPY intensity array (H, W)
            output_path: Optional path to save BMP file

        Returns:
            BMP image array (H, target_width) uint8
        """
        # Step 1: Normalize to uint8
        normalized = self.normalize_intensity(npy_data)

        # Step 2: Apply contrast enhancement
        enhanced = self.apply_contrast_enhancement(normalized)

        # Step 3: Resize width
        resized = self.resize_width(enhanced, self.target_width)

        # Step 4: Convert to 3-channel BGR for BMP
        bmp_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        # Step 5: Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as BMP
            cv2.imwrite(str(output_path), bmp_image)

            print(f"✓ BMP saved: {output_path}")
            print(f"  - Original NPY: {npy_data.shape}")
            print(f"  - BMP output: {bmp_image.shape}")
            print(f"  - Scale factor: {npy_data.shape[1] / bmp_image.shape[1]:.2f}x")

        return bmp_image

    def get_scale_factor(self, npy_width: int) -> float:
        """
        Calculate scale factor from NPY to BMP

        Args:
            npy_width: NPY data width

        Returns:
            Scale factor (NPY_width / BMP_width)
        """
        return npy_width / self.target_width

    def bmp_to_npy_coordinates(self,
                               bmp_bbox: Dict[str, int],
                               npy_width: int) -> Dict[str, int]:
        """
        Convert BMP coordinates to NPY coordinates

        Args:
            bmp_bbox: BMP bounding box {xmin, ymin, xmax, ymax}
            npy_width: NPY data width

        Returns:
            NPY bounding box coordinates
        """
        scale_factor = self.get_scale_factor(npy_width)

        npy_bbox = {
            'xmin': int(bmp_bbox['xmin'] * scale_factor),
            'ymin': bmp_bbox['ymin'],  # Y unchanged
            'xmax': int(bmp_bbox['xmax'] * scale_factor),
            'ymax': bmp_bbox['ymax']   # Y unchanged
        }

        # Calculate dimensions
        npy_bbox['width'] = npy_bbox['xmax'] - npy_bbox['xmin']
        npy_bbox['height'] = npy_bbox['ymax'] - npy_bbox['ymin']

        return npy_bbox

    def npy_to_bmp_coordinates(self,
                               npy_bbox: Dict[str, int],
                               npy_width: int) -> Dict[str, int]:
        """
        Convert NPY coordinates to BMP coordinates

        Args:
            npy_bbox: NPY bounding box {xmin, ymin, xmax, ymax}
            npy_width: NPY data width

        Returns:
            BMP bounding box coordinates
        """
        scale_factor = self.get_scale_factor(npy_width)

        bmp_bbox = {
            'xmin': int(npy_bbox['xmin'] / scale_factor),
            'ymin': npy_bbox['ymin'],  # Y unchanged
            'xmax': int(npy_bbox['xmax'] / scale_factor),
            'ymax': npy_bbox['ymax']   # Y unchanged
        }

        # Calculate dimensions
        bmp_bbox['width'] = bmp_bbox['xmax'] - bmp_bbox['xmin']
        bmp_bbox['height'] = bmp_bbox['ymax'] - bmp_bbox['ymin']

        return bmp_bbox

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NpyToBmpConverter':
        """
        Create converter from configuration

        Args:
            config: Configuration dictionary

        Returns:
            NpyToBmpConverter instance
        """
        viz_config = config.get('visualization', {})

        return cls(
            target_width=viz_config.get('target_width', 1024),
            apply_clahe=viz_config.get('apply_clahe', True),
            clip_limit=viz_config.get('clip_limit', 2.0),
            tile_grid_size=tuple(viz_config.get('tile_grid_size', [8, 8]))
        )

    def __repr__(self) -> str:
        return (f"NpyToBmpConverter("
                f"target_width={self.target_width}, "
                f"clahe={self.apply_clahe})")
