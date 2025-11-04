"""
Mapping Validation Module

Validates coordinate mapping accuracy, label masks, and data integrity.

Verified: 2025-11-04
Based on: scripts/mine_labeling/04_validate_mapping.py
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class MappingValidator:
    """
    Validates coordinate mapping and labeling accuracy

    Checks:
    - Dimension consistency
    - Scaling factor accuracy
    - Label mask coverage
    - Data value ranges
    """

    def __init__(self,
                 expected_scale_x: float = 6.25,
                 expected_scale_y: float = 1.0,
                 min_coverage_threshold: float = 0.0001):
        """
        Initialize mapping validator

        Args:
            expected_scale_x: Expected X scaling factor (default: 6.25)
            expected_scale_y: Expected Y scaling factor (default: 1.0)
            min_coverage_threshold: Minimum label coverage percentage (default: 0.0001)
        """
        self.expected_scale_x = expected_scale_x
        self.expected_scale_y = expected_scale_y
        self.min_coverage_threshold = min_coverage_threshold

        self.validation_results = {}

    def validate_dimensions(self,
                          npy_shape: Tuple[int, int],
                          bmp_shape: Tuple[int, int],
                          mask_shape: Tuple[int, int]) -> bool:
        """
        Validate data dimensions match expected values

        Args:
            npy_shape: NPY intensity data shape (H, W)
            bmp_shape: BMP image shape (H, W) or (H, W, C)
            mask_shape: Label mask shape (H, W)

        Returns:
            True if dimensions are valid
        """
        result = {
            'npy_shape': npy_shape,
            'bmp_shape': bmp_shape[:2] if len(bmp_shape) == 3 else bmp_shape,
            'mask_shape': mask_shape,
            'errors': []
        }

        # Check NPY and mask match
        if npy_shape != mask_shape:
            result['errors'].append(
                f"NPY shape {npy_shape} != Mask shape {mask_shape}"
            )

        # Check height consistency
        bmp_height = bmp_shape[0]
        npy_height = npy_shape[0]

        if npy_height != bmp_height:
            result['errors'].append(
                f"Height mismatch: NPY={npy_height}, BMP={bmp_height}"
            )

        # Check width scaling
        bmp_width = bmp_shape[1] if len(bmp_shape) >= 2 else bmp_shape[0]
        npy_width = npy_shape[1]
        actual_ratio = npy_width / bmp_width

        if abs(actual_ratio - self.expected_scale_x) > 0.01:
            result['errors'].append(
                f"Width ratio {actual_ratio:.4f} != Expected {self.expected_scale_x}"
            )

        result['passed'] = len(result['errors']) == 0
        self.validation_results['dimensions'] = result

        return result['passed']

    def validate_scaling_factor(self,
                               annotations: List[Dict[str, Any]]) -> bool:
        """
        Validate scaling factor is correctly applied to all annotations

        Args:
            annotations: List of mapped annotations

        Returns:
            True if all scaling is correct
        """
        errors = []

        for i, ann in enumerate(annotations):
            orig = ann['original_bmp']
            mapped = ann['mapped_npy']

            # Calculate expected mapped coordinates
            expected = {
                'xmin': int(orig['xmin'] * self.expected_scale_x),
                'ymin': int(orig['ymin'] * self.expected_scale_y),
                'xmax': int(orig['xmax'] * self.expected_scale_x),
                'ymax': int(orig['ymax'] * self.expected_scale_y)
            }

            # Check if mapping is correct
            if (mapped['xmin'] != expected['xmin'] or
                mapped['ymin'] != expected['ymin'] or
                mapped['xmax'] != expected['xmax'] or
                mapped['ymax'] != expected['ymax']):

                errors.append({
                    'mine_id': i + 1,
                    'expected': expected,
                    'actual': {
                        'xmin': mapped['xmin'],
                        'ymin': mapped['ymin'],
                        'xmax': mapped['xmax'],
                        'ymax': mapped['ymax']
                    }
                })

        result = {
            'total_annotations': len(annotations),
            'errors': errors,
            'error_count': len(errors),
            'passed': len(errors) == 0
        }

        self.validation_results['scaling'] = result

        return result['passed']

    def validate_label_mask(self,
                          label_mask: np.ndarray,
                          annotations: List[Dict[str, Any]]) -> bool:
        """
        Validate label mask covers correct regions

        Args:
            label_mask: Label mask array
            annotations: List of mapped annotations

        Returns:
            True if all annotations have corresponding labels
        """
        missing_labels = []
        partial_labels = []

        for i, ann in enumerate(annotations):
            bbox = ann['mapped_npy']
            ymin = bbox['ymin']
            ymax = bbox['ymax']
            xmin = bbox['xmin']
            xmax = bbox['xmax']

            # Ensure valid bounds
            ymin = max(0, min(ymin, ymax))
            ymax = min(label_mask.shape[0], max(ymin, ymax))
            xmin = max(0, xmin)
            xmax = min(label_mask.shape[1], xmax)

            # Extract mask region
            if ymin < ymax and xmin < xmax:
                mask_region = label_mask[ymin:ymax, xmin:xmax]

                # Check if region is labeled
                if mask_region.sum() == 0:
                    missing_labels.append(i + 1)
                elif mask_region.sum() != mask_region.size:
                    coverage = mask_region.sum() / mask_region.size
                    partial_labels.append({
                        'mine_id': i + 1,
                        'coverage': coverage
                    })

        # Calculate statistics
        total_pixels = label_mask.size
        mine_pixels = (label_mask > 0).sum()
        percentage = mine_pixels / total_pixels

        result = {
            'total_annotations': len(annotations),
            'missing_labels': missing_labels,
            'partial_labels': partial_labels,
            'total_pixels': int(total_pixels),
            'mine_pixels': int(mine_pixels),
            'coverage_percentage': float(percentage),
            'passed': len(missing_labels) == 0 and percentage >= self.min_coverage_threshold
        }

        self.validation_results['label_mask'] = result

        return result['passed']

    def validate_intensity_values(self, npy_data: np.ndarray,
                                 check_normalization: bool = True) -> bool:
        """
        Validate intensity values in NPY data

        Args:
            npy_data: NPY intensity array
            check_normalization: Check if data is normalized to [0, 1]

        Returns:
            True if intensity values are valid
        """
        result = {
            'dtype': str(npy_data.dtype),
            'min': float(npy_data.min()),
            'max': float(npy_data.max()),
            'mean': float(npy_data.mean()),
            'std': float(npy_data.std()),
            'normalized': False,
            'passed': True
        }

        # Check normalization
        if check_normalization:
            if result['min'] >= 0 and result['max'] <= 1:
                result['normalized'] = True
            else:
                result['normalized'] = False
                if check_normalization:
                    result['passed'] = False

        self.validation_results['intensity'] = result

        return result['passed']

    def validate_bounding_boxes(self,
                               annotations: List[Dict[str, Any]],
                               max_width: int,
                               max_height: int) -> bool:
        """
        Validate bounding boxes are within bounds

        Args:
            annotations: List of mapped annotations
            max_width: Maximum width (NPY width)
            max_height: Maximum height (NPY height)

        Returns:
            True if all bboxes are valid
        """
        out_of_bounds = []
        invalid_sizes = []

        for i, ann in enumerate(annotations):
            bbox = ann['mapped_npy']

            # Check bounds
            if (bbox['xmin'] < 0 or bbox['xmax'] > max_width or
                bbox['ymin'] < 0 or bbox['ymax'] > max_height):
                out_of_bounds.append({
                    'mine_id': i + 1,
                    'bbox': bbox,
                    'bounds': (max_width, max_height)
                })

            # Check size validity
            if bbox['width'] <= 0 or bbox['height'] <= 0:
                invalid_sizes.append({
                    'mine_id': i + 1,
                    'width': bbox['width'],
                    'height': bbox['height']
                })

        result = {
            'total_annotations': len(annotations),
            'out_of_bounds': out_of_bounds,
            'invalid_sizes': invalid_sizes,
            'passed': len(out_of_bounds) == 0 and len(invalid_sizes) == 0
        }

        self.validation_results['bounding_boxes'] = result

        return result['passed']

    def run_all_validations(self,
                           npy_data: np.ndarray,
                           bmp_shape: Tuple[int, int],
                           label_mask: np.ndarray,
                           annotations: List[Dict[str, Any]]) -> bool:
        """
        Run all validation checks

        Args:
            npy_data: NPY intensity data
            bmp_shape: BMP image shape
            label_mask: Label mask
            annotations: Mapped annotations

        Returns:
            True if all validations pass
        """
        validations = [
            ('dimensions', lambda: self.validate_dimensions(
                npy_data.shape, bmp_shape, label_mask.shape)),
            ('scaling', lambda: self.validate_scaling_factor(annotations)),
            ('label_mask', lambda: self.validate_label_mask(label_mask, annotations)),
            ('intensity', lambda: self.validate_intensity_values(npy_data)),
            ('bounding_boxes', lambda: self.validate_bounding_boxes(
                annotations, npy_data.shape[1], npy_data.shape[0]))
        ]

        all_passed = True

        for name, validation_func in validations:
            try:
                passed = validation_func()
                all_passed = all_passed and passed
            except Exception as e:
                self.validation_results[name] = {
                    'passed': False,
                    'error': str(e)
                }
                all_passed = False

        return all_passed

    def get_summary(self) -> Dict[str, Any]:
        """
        Get validation summary

        Returns:
            Dictionary with summary of all validations
        """
        summary = {
            'total_checks': len(self.validation_results),
            'passed_checks': sum(1 for r in self.validation_results.values()
                                if r.get('passed', False)),
            'failed_checks': sum(1 for r in self.validation_results.values()
                                if not r.get('passed', True)),
            'all_passed': all(r.get('passed', False)
                            for r in self.validation_results.values()),
            'details': self.validation_results
        }

        return summary

    def print_summary(self) -> None:
        """Print validation summary to console"""
        summary = self.get_summary()

        print("="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        for name, result in summary['details'].items():
            status = "✓ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"{name.upper():20s}: {status}")

            if not result.get('passed', False):
                if 'errors' in result and result['errors']:
                    for error in result['errors'][:3]:
                        print(f"  - {error}")

        print("="*60)
        if summary['all_passed']:
            print("✓ ALL VALIDATIONS PASSED")
        else:
            print(f"❌ {summary['failed_checks']}/{summary['total_checks']} VALIDATIONS FAILED")
        print("="*60)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MappingValidator':
        """
        Create MappingValidator from configuration dictionary

        Args:
            config: Configuration dictionary

        Returns:
            MappingValidator instance
        """
        coord_config = config.get('coordinate_mapping', {})
        val_config = config.get('validation', {})

        return cls(
            expected_scale_x=coord_config.get('scale_x', 6.25),
            expected_scale_y=coord_config.get('scale_y', 1.0),
            min_coverage_threshold=val_config.get('min_coverage_threshold', 0.0001)
        )
