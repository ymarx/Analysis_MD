#!/usr/bin/env python3
"""
Test NPY to BMP Converter

Tests the conversion from NPY intensity data to BMP format
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
from mine_labeling.visualization import NpyToBmpConverter
from mine_labeling.utils import load_npz_data


def main():
    print("="*60)
    print("NPY to BMP Converter Test")
    print("="*60)

    # Load verified NPY data
    npz_path = project_root / 'verified_data/flipped_20251104/flipped_labeled_intensity_data.npz'

    print(f"\n1. Loading NPY data from: {npz_path.name}")
    data = load_npz_data(npz_path)

    intensity = data['intensity']
    print(f"   - Intensity shape: {intensity.shape}")
    print(f"   - Data type: {intensity.dtype}")
    print(f"   - Value range: [{intensity.min():.3f}, {intensity.max():.3f}]")

    # Create converter
    print(f"\n2. Creating BMP converter")
    converter = NpyToBmpConverter(
        target_width=1024,
        apply_clahe=True,
        clip_limit=2.0
    )
    print(f"   - {converter}")

    # Convert to BMP
    output_dir = project_root / 'data/visualization'
    output_path = output_dir / 'intensity_visualization.bmp'

    print(f"\n3. Converting NPY → BMP")
    bmp_image = converter.convert_to_bmp(intensity, output_path)

    # Show scale factor
    scale_factor = converter.get_scale_factor(intensity.shape[1])
    print(f"\n4. Coordinate transformation info:")
    print(f"   - NPY width: {intensity.shape[1]} → BMP width: {bmp_image.shape[1]}")
    print(f"   - Scale factor: {scale_factor:.4f}x")
    print(f"   - Height: {intensity.shape[0]} (preserved)")

    # Test coordinate conversion
    print(f"\n5. Testing coordinate conversion:")

    # Example: First mine from verified data
    test_npy_bbox = {
        'xmin': 4868,
        'ymin': 1070,
        'xmax': 5187,
        'ymax': 1119
    }

    print(f"   Original NPY bbox: {test_npy_bbox}")

    bmp_bbox = converter.npy_to_bmp_coordinates(test_npy_bbox, intensity.shape[1])
    print(f"   → BMP bbox: {bmp_bbox}")

    # Convert back
    npy_bbox_back = converter.bmp_to_npy_coordinates(bmp_bbox, intensity.shape[1])
    print(f"   → NPY bbox (back): {npy_bbox_back}")

    # Verify
    print(f"\n6. Verification:")
    x_error = abs(test_npy_bbox['xmin'] - npy_bbox_back['xmin'])
    print(f"   - X error: {x_error} pixels")
    if x_error < 10:
        print(f"   ✓ Coordinate conversion accurate!")
    else:
        print(f"   ⚠️  Large error detected")

    print("\n" + "="*60)
    print("✓ Test completed!")
    print("="*60)
    print(f"\nBMP file saved at:")
    print(f"  {output_path}")
    print(f"\nYou can now use this BMP for interactive labeling.")


if __name__ == '__main__':
    main()
