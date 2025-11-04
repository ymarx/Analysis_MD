#!/usr/bin/env python3
"""
Flip Bounding Box Y-coordinates for NPY Mapping
Apply vertical flip to bounding boxes while keeping intensity data unchanged
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlippedCoordinateMapper:
    """Maps coordinates with Y-axis flip"""

    def __init__(self, npy_path, bmp_path, xml_path):
        self.npy_path = Path(npy_path)
        self.bmp_path = Path(bmp_path)
        self.xml_path = Path(xml_path)

        # Load data
        print("Loading data...")
        self.npy_data = np.load(self.npy_path)
        self.bmp_image = Image.open(self.bmp_path)
        self.bmp_array = np.array(self.bmp_image)

        # Parse XML annotations
        self.annotations = self._parse_xml_annotations()

        print(f"NPY shape: {self.npy_data.shape}")
        print(f"BMP shape: {self.bmp_array.shape}")
        print(f"Number of mines: {len(self.annotations)}")

    def _parse_xml_annotations(self):
        """Parse XML annotation file"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        size_elem = root.find('size')
        img_width = int(size_elem.find('width').text)
        img_height = int(size_elem.find('height').text)

        annotations = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            annotations.append({
                'name': obj.find('name').text,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'center_x': (xmin + xmax) / 2,
                'center_y': (ymin + ymax) / 2,
                'width': xmax - xmin,
                'height': ymax - ymin
            })

        return annotations

    def map_coordinates_with_flip(self):
        """
        Map coordinates with Y-axis flip

        X-axis: Scale by 6.25 (1024 -> 6400)
        Y-axis: Flip + 1:1 mapping (5137 -> 5137)

        Y_npy = (height - 1) - Y_bmp
        """
        npy_height, npy_width = self.npy_data.shape
        bmp_width, bmp_height = self.bmp_image.size

        print(f"\nCoordinate mapping with Y-flip:")
        print(f"  BMP: {bmp_width} x {bmp_height}")
        print(f"  NPY: {npy_width} x {npy_height}")

        # Calculate scaling factors
        scale_x = npy_width / bmp_width
        scale_y = 1.0  # Height 1:1 after flip

        print(f"  Scale factors: X={scale_x:.4f}, Y={scale_y:.4f}")
        print(f"  Y-axis: FLIPPED (top -> bottom, bottom -> top)")

        mapped_annotations = []
        for ann in self.annotations:
            # Flip Y coordinates
            ymin_flipped = (bmp_height - 1) - ann['ymax']
            ymax_flipped = (bmp_height - 1) - ann['ymin']

            # Map coordinates
            mapped = {
                'name': ann['name'],
                'original_bmp': {
                    'xmin': ann['xmin'],
                    'ymin': ann['ymin'],
                    'xmax': ann['xmax'],
                    'ymax': ann['ymax']
                },
                'flipped_bmp': {
                    'xmin': ann['xmin'],
                    'ymin': ymin_flipped,
                    'xmax': ann['xmax'],
                    'ymax': ymax_flipped
                },
                'mapped_npy': {
                    'xmin': int(ann['xmin'] * scale_x),
                    'ymin': int(ymin_flipped * scale_y),
                    'xmax': int(ann['xmax'] * scale_x),
                    'ymax': int(ymax_flipped * scale_y)
                },
                'center_npy': {
                    'x': int(ann['center_x'] * scale_x),
                    'y': int(((bmp_height - 1) - ann['center_y']) * scale_y)
                }
            }

            # Calculate dimensions
            mapped['mapped_npy']['width'] = mapped['mapped_npy']['xmax'] - mapped['mapped_npy']['xmin']
            mapped['mapped_npy']['height'] = mapped['mapped_npy']['ymax'] - mapped['mapped_npy']['ymin']

            mapped_annotations.append(mapped)

        return mapped_annotations

    def create_label_mask(self, mapped_annotations):
        """Create binary label mask"""
        npy_height, npy_width = self.npy_data.shape
        label_mask = np.zeros((npy_height, npy_width), dtype=np.uint8)

        for ann in mapped_annotations:
            bbox = ann['mapped_npy']
            ymin = max(0, min(bbox['ymin'], bbox['ymax']))
            ymax = min(npy_height, max(bbox['ymin'], bbox['ymax']))
            xmin = max(0, bbox['xmin'])
            xmax = min(npy_width, bbox['xmax'])

            label_mask[ymin:ymax, xmin:xmax] = 1

        return label_mask

    def visualize_flipped_mapping(self, mapped_annotations, label_mask, output_path):
        """Visualize flipped coordinate mapping"""

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # 1. Original BMP with annotations
        axes[0].imshow(self.bmp_array)
        axes[0].set_title(f'Original BMP ({self.bmp_array.shape[1]}x{self.bmp_array.shape[0]})\n{len(mapped_annotations)} mines (original positions)')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pings)')

        for ann in self.annotations:
            rect = patches.Rectangle(
                (ann['xmin'], ann['ymin']),
                ann['width'], ann['height'],
                linewidth=1, edgecolor='r', facecolor='none'
            )
            axes[0].add_patch(rect)

        # 2. NPY intensity with flipped bounding boxes
        axes[1].imshow(self.npy_data, cmap='gray', aspect='auto')
        axes[1].set_title(f'NPY Intensity ({self.npy_data.shape[1]}x{self.npy_data.shape[0]})\nFlipped bounding boxes')
        axes[1].set_xlabel('X (samples)')
        axes[1].set_ylabel('Y (pings) - FLIPPED')

        for ann in mapped_annotations:
            bbox = ann['mapped_npy']
            ymin = min(bbox['ymin'], bbox['ymax'])
            ymax = max(bbox['ymin'], bbox['ymax'])
            rect = patches.Rectangle(
                (bbox['xmin'], ymin),
                bbox['width'], abs(bbox['height']),
                linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[1].add_patch(rect)

        # 3. Label mask
        axes[2].imshow(label_mask, cmap='hot', aspect='auto')
        axes[2].set_title(f'Label Mask ({label_mask.shape[1]}x{label_mask.shape[0]})\nFlipped binary labels')
        axes[2].set_xlabel('X (samples)')
        axes[2].set_ylabel('Y (pings) - FLIPPED')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved: {output_path}")
        plt.close()

    def create_detailed_comparison(self, mapped_annotations, output_dir):
        """Create detailed comparison of selected mines"""

        # Select mines to compare
        indices = [0, len(mapped_annotations)//2, len(mapped_annotations)-1]

        for idx in indices:
            ann = mapped_annotations[idx]
            mine_id = idx + 1

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # BMP view
            orig = ann['original_bmp']
            bmp_y_min = max(0, orig['ymin'] - 100)
            bmp_y_max = min(self.bmp_array.shape[0], orig['ymax'] + 100)
            bmp_x_min = max(0, orig['xmin'] - 100)
            bmp_x_max = min(self.bmp_array.shape[1], orig['xmax'] + 100)

            bmp_crop = self.bmp_array[bmp_y_min:bmp_y_max, bmp_x_min:bmp_x_max]
            axes[0].imshow(bmp_crop)
            axes[0].set_title(f'BMP Image - Mine {mine_id}\n'
                             f'Original: ({orig["xmin"]}, {orig["ymin"]}) -> ({orig["xmax"]}, {orig["ymax"]})')

            rect = patches.Rectangle(
                (orig['xmin'] - bmp_x_min, orig['ymin'] - bmp_y_min),
                orig['xmax'] - orig['xmin'],
                orig['ymax'] - orig['ymin'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[0].add_patch(rect)

            # NPY view with flipped bbox
            mapped = ann['mapped_npy']
            npy_y_min = max(0, min(mapped['ymin'], mapped['ymax']) - 100)
            npy_y_max = min(self.npy_data.shape[0], max(mapped['ymin'], mapped['ymax']) + 100)
            npy_x_min = max(0, mapped['xmin'] - 625)
            npy_x_max = min(self.npy_data.shape[1], mapped['xmax'] + 625)

            npy_crop = self.npy_data[npy_y_min:npy_y_max, npy_x_min:npy_x_max]
            axes[1].imshow(npy_crop, cmap='gray', aspect='auto')

            flipped = ann['flipped_bmp']
            axes[1].set_title(f'NPY Intensity - Mine {mine_id}\n'
                             f'Flipped BMP: ({flipped["xmin"]}, {flipped["ymin"]}) -> ({flipped["xmax"]}, {flipped["ymax"]})\n'
                             f'Mapped NPY: ({mapped["xmin"]}, {mapped["ymin"]}) -> ({mapped["xmax"]}, {mapped["ymax"]})')

            ymin_plot = min(mapped['ymin'], mapped['ymax']) - npy_y_min
            rect = patches.Rectangle(
                (mapped['xmin'] - npy_x_min, ymin_plot),
                mapped['width'], abs(mapped['height']),
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)

            plt.tight_layout()
            output_path = output_dir / f'flipped_mine_{mine_id:02d}_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()

def main():
    base_dir = Path('/Users/YMARX/Dropbox/2025_ECMiner/C_P02_기뢰전대/03_진행/Analysis_MD')

    npy_path = base_dir / 'data/processed/xtf_extracted/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_combined_intensity.npy'
    bmp_path = base_dir / 'datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_IMG_00.BMP'
    xml_path = base_dir / 'KleinLabeling/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xml'

    output_dir = base_dir / 'analysis_results/npy_labeling/flipped'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Flipped Coordinate Mapping: XML/BMP -> NPY (Y-axis flipped)")
    print("="*60)

    # Initialize mapper
    mapper = FlippedCoordinateMapper(npy_path, bmp_path, xml_path)

    # Map with flip
    print("\n" + "="*60)
    print("Mapping coordinates with Y-flip...")
    print("="*60)
    mapped_annotations = mapper.map_coordinates_with_flip()

    # Print examples
    print(f"\nFirst 3 mapped annotations (with flip):")
    for i, ann in enumerate(mapped_annotations[:3]):
        print(f"\nMine {i+1}:")
        print(f"  Original BMP: ({ann['original_bmp']['xmin']}, {ann['original_bmp']['ymin']}) -> "
              f"({ann['original_bmp']['xmax']}, {ann['original_bmp']['ymax']})")
        print(f"  Flipped BMP:  ({ann['flipped_bmp']['xmin']}, {ann['flipped_bmp']['ymin']}) -> "
              f"({ann['flipped_bmp']['xmax']}, {ann['flipped_bmp']['ymax']})")
        print(f"  Mapped NPY:   ({ann['mapped_npy']['xmin']}, {ann['mapped_npy']['ymin']}) -> "
              f"({ann['mapped_npy']['xmax']}, {ann['mapped_npy']['ymax']})")

    # Create label mask
    print("\n" + "="*60)
    print("Creating flipped label mask...")
    print("="*60)
    label_mask = mapper.create_label_mask(mapped_annotations)
    print(f"Label mask shape: {label_mask.shape}")
    print(f"Mine pixels: {label_mask.sum()} / {label_mask.size} ({100*label_mask.sum()/label_mask.size:.4f}%)")

    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)

    annotations_path = output_dir / 'flipped_mapped_annotations.json'
    with open(annotations_path, 'w') as f:
        json.dump(mapped_annotations, f, indent=2)
    print(f"Saved: {annotations_path}")

    label_mask_path = output_dir / 'flipped_mine_label_mask.npy'
    np.save(label_mask_path, label_mask)
    print(f"Saved: {label_mask_path}")

    labeled_path = output_dir / 'flipped_labeled_intensity_data.npz'
    np.savez(labeled_path,
             intensity=mapper.npy_data,
             labels=label_mask,
             metadata=json.dumps(mapped_annotations))
    print(f"Saved: {labeled_path}")

    # Visualize
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    viz_path = output_dir / 'flipped_coordinate_mapping_visualization.png'
    mapper.visualize_flipped_mapping(mapped_annotations, label_mask, viz_path)

    mapper.create_detailed_comparison(mapped_annotations, output_dir)

    print("\n" + "="*60)
    print("Flipped coordinate mapping completed!")
    print("="*60)

if __name__ == '__main__':
    main()
