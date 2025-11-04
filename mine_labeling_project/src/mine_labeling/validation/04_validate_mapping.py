#!/usr/bin/env python3
"""
Validate Coordinate Mapping Accuracy
Performs detailed validation to ensure mapping correctness
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data():
    """Load all required data"""
    base_dir = Path('/Users/YMARX/Dropbox/2025_ECMiner/C_P02_기뢰전대/03_진행/Analysis_MD')

    # Load NPY data
    npy_path = base_dir / 'data/processed/xtf_extracted/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_combined_intensity.npy'
    npy_data = np.load(npy_path)

    # Load BMP image
    bmp_path = base_dir / 'datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_IMG_00.BMP'
    bmp_image = np.array(Image.open(bmp_path))

    # Load mapped annotations
    annotations_path = base_dir / 'analysis_results/npy_labeling/mapped_annotations.json'
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Load label mask
    mask_path = base_dir / 'analysis_results/npy_labeling/mine_label_mask.npy'
    label_mask = np.load(mask_path)

    return {
        'npy_data': npy_data,
        'bmp_image': bmp_image,
        'annotations': annotations,
        'label_mask': label_mask
    }

def validate_dimensions(data):
    """Validate data dimensions match expected values"""
    print("="*60)
    print("DIMENSION VALIDATION")
    print("="*60)

    npy_shape = data['npy_data'].shape
    bmp_shape = data['bmp_image'].shape
    mask_shape = data['label_mask'].shape

    print(f"NPY intensity: {npy_shape}")
    print(f"BMP image: {bmp_shape}")
    print(f"Label mask: {mask_shape}")

    # Check NPY and mask match
    assert npy_shape == mask_shape, f"NPY and mask shape mismatch!"
    print("✓ NPY and label mask dimensions match")

    # Check height consistency
    assert npy_shape[0] == bmp_shape[0], f"Height mismatch: NPY={npy_shape[0]}, BMP={bmp_shape[0]}"
    print(f"✓ Height consistent: {npy_shape[0]} pings")

    # Check width scaling
    bmp_width = bmp_shape[1]
    npy_width = npy_shape[1]
    expected_ratio = 6.25
    actual_ratio = npy_width / bmp_width

    print(f"Width ratio (NPY/BMP): {actual_ratio:.4f} (expected: {expected_ratio})")
    assert abs(actual_ratio - expected_ratio) < 0.01, "Width ratio incorrect!"
    print("✓ Width scaling correct")

    return True

def validate_scaling_factor(data):
    """Validate scaling factor is correctly applied"""
    print("\n" + "="*60)
    print("SCALING FACTOR VALIDATION")
    print("="*60)

    annotations = data['annotations']
    scale_x = 6.25
    scale_y = 1.0

    errors = []
    for i, ann in enumerate(annotations):
        orig = ann['original_bmp']
        mapped = ann['mapped_npy']

        # Calculate expected mapped coordinates
        expected_xmin = int(orig['xmin'] * scale_x)
        expected_ymin = int(orig['ymin'] * scale_y)
        expected_xmax = int(orig['xmax'] * scale_x)
        expected_ymax = int(orig['ymax'] * scale_y)

        # Check if mapping is correct
        if (mapped['xmin'] != expected_xmin or
            mapped['ymin'] != expected_ymin or
            mapped['xmax'] != expected_xmax or
            mapped['ymax'] != expected_ymax):
            errors.append({
                'mine_id': i+1,
                'expected': (expected_xmin, expected_ymin, expected_xmax, expected_ymax),
                'actual': (mapped['xmin'], mapped['ymin'], mapped['xmax'], mapped['ymax'])
            })

    if errors:
        print(f"❌ Found {len(errors)} mapping errors:")
        for err in errors[:5]:
            print(f"  Mine {err['mine_id']}: Expected {err['expected']}, Got {err['actual']}")
        return False
    else:
        print(f"✓ All {len(annotations)} annotations correctly scaled")
        return True

def validate_label_mask(data):
    """Validate label mask covers correct regions"""
    print("\n" + "="*60)
    print("LABEL MASK VALIDATION")
    print("="*60)

    label_mask = data['label_mask']
    annotations = data['annotations']

    # Check each annotation has corresponding mask
    for i, ann in enumerate(annotations):
        bbox = ann['mapped_npy']
        ymin, ymax = bbox['ymin'], bbox['ymax']
        xmin, xmax = bbox['xmin'], bbox['xmax']

        # Extract mask region
        mask_region = label_mask[ymin:ymax, xmin:xmax]

        # Check if region is labeled
        if mask_region.sum() == 0:
            print(f"❌ Mine {i+1}: No labels in bounding box!")
            return False
        elif mask_region.sum() != mask_region.size:
            coverage = 100 * mask_region.sum() / mask_region.size
            print(f"⚠️  Mine {i+1}: Partial labeling ({coverage:.1f}% coverage)")

    print(f"✓ All {len(annotations)} bounding boxes have labels in mask")

    # Calculate statistics
    total_pixels = label_mask.size
    mine_pixels = label_mask.sum()
    percentage = 100 * mine_pixels / total_pixels

    print(f"\nLabel mask statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Mine pixels: {mine_pixels:,}")
    print(f"  Coverage: {percentage:.4f}%")

    return True

def validate_intensity_values(data):
    """Validate intensity values in NPY data"""
    print("\n" + "="*60)
    print("INTENSITY DATA VALIDATION")
    print("="*60)

    npy_data = data['npy_data']

    print(f"Data type: {npy_data.dtype}")
    print(f"Value range: [{npy_data.min():.4f}, {npy_data.max():.4f}]")
    print(f"Mean: {npy_data.mean():.4f}")
    print(f"Std: {npy_data.std():.4f}")

    # Check normalization
    if npy_data.min() >= 0 and npy_data.max() <= 1:
        print("✓ Data normalized to [0, 1] range")
    else:
        print("⚠️  Data not normalized")

    return True

def create_detailed_comparison(data, output_dir):
    """Create detailed side-by-side comparison of selected mines"""
    print("\n" + "="*60)
    print("CREATING DETAILED COMPARISON")
    print("="*60)

    npy_data = data['npy_data']
    bmp_image = data['bmp_image']
    annotations = data['annotations']

    # Select mines to compare (first, middle, last)
    indices = [0, len(annotations)//2, len(annotations)-1]

    for idx in indices:
        ann = annotations[idx]
        mine_id = idx + 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # BMP view
        orig = ann['original_bmp']
        bmp_y_min = max(0, orig['ymin'] - 100)
        bmp_y_max = min(bmp_image.shape[0], orig['ymax'] + 100)
        bmp_x_min = max(0, orig['xmin'] - 100)
        bmp_x_max = min(bmp_image.shape[1], orig['xmax'] + 100)

        bmp_crop = bmp_image[bmp_y_min:bmp_y_max, bmp_x_min:bmp_x_max]
        axes[0].imshow(bmp_crop)
        axes[0].set_title(f'BMP Image - Mine {mine_id}\n'
                         f'Bbox: ({orig["xmin"]}, {orig["ymin"]}) -> ({orig["xmax"]}, {orig["ymax"]})')

        # Draw bbox relative to crop
        rect = patches.Rectangle(
            (orig['xmin'] - bmp_x_min, orig['ymin'] - bmp_y_min),
            orig['xmax'] - orig['xmin'],
            orig['ymax'] - orig['ymin'],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0].add_patch(rect)

        # NPY view
        mapped = ann['mapped_npy']
        npy_y_min = max(0, mapped['ymin'] - 100)
        npy_y_max = min(npy_data.shape[0], mapped['ymax'] + 100)
        npy_x_min = max(0, mapped['xmin'] - 625)  # Scale the margin
        npy_x_max = min(npy_data.shape[1], mapped['xmax'] + 625)

        npy_crop = npy_data[npy_y_min:npy_y_max, npy_x_min:npy_x_max]
        axes[1].imshow(npy_crop, cmap='gray', aspect='auto')
        axes[1].set_title(f'NPY Intensity - Mine {mine_id}\n'
                         f'Mapped Bbox: ({mapped["xmin"]}, {mapped["ymin"]}) -> ({mapped["xmax"]}, {mapped["ymax"]})')

        # Draw bbox relative to crop
        rect = patches.Rectangle(
            (mapped['xmin'] - npy_x_min, mapped['ymin'] - npy_y_min),
            mapped['width'],
            mapped['height'],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[1].add_patch(rect)

        plt.tight_layout()
        output_path = output_dir / f'mine_{mine_id:02d}_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def main():
    print("="*60)
    print("COORDINATE MAPPING VALIDATION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data = load_data()

    # Output directory
    output_dir = Path('/Users/YMARX/Dropbox/2025_ECMiner/C_P02_기뢰전대/03_진행/Analysis_MD/analysis_results/npy_labeling/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validations
    validations = [
        ("Dimensions", validate_dimensions),
        ("Scaling Factor", validate_scaling_factor),
        ("Label Mask", validate_label_mask),
        ("Intensity Values", validate_intensity_values)
    ]

    results = {}
    for name, func in validations:
        try:
            results[name] = func(data)
        except Exception as e:
            print(f"\n❌ {name} validation failed: {e}")
            results[name] = False

    # Create detailed comparisons
    create_detailed_comparison(data, output_dir)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("Coordinate mapping is accurate and verified!")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please review the results above.")
    print("="*60)

    return all_passed

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
