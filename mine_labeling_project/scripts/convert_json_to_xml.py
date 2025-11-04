#!/usr/bin/env python3
"""
JSON to PASCAL VOC XML Converter

Converts flipped_mapped_annotations.json to NPY coordinate XML format
for use with NPY intensity data (6400×5137)
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path


def create_pascal_voc_xml(annotations, image_info, output_path):
    """
    Create PASCAL VOC format XML from annotations

    Args:
        annotations: List of annotation dictionaries
        image_info: Dictionary with image metadata
        output_path: Path to save XML file
    """
    # Create root element
    root = ET.Element('annotation')

    # Add folder
    folder = ET.SubElement(root, 'folder')
    folder.text = image_info.get('folder', 'verified_data')

    # Add filename
    filename = ET.SubElement(root, 'filename')
    filename.text = image_info.get('filename', 'flipped_intensity_data.npy')

    # Add path
    path = ET.SubElement(root, 'path')
    path.text = image_info.get('path', str(output_path.parent / image_info.get('filename', 'flipped_intensity_data.npy')))

    # Add source
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Mine Detection Dataset'

    # Add size (NPY dimensions)
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_info.get('width', 6400))
    height = ET.SubElement(size, 'height')
    height.text = str(image_info.get('height', 5137))
    depth = ET.SubElement(size, 'depth')
    depth.text = str(image_info.get('depth', 1))  # Single channel intensity

    # Add segmented
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    # Add objects (mines)
    for ann in annotations:
        obj = ET.SubElement(root, 'object')

        # Name
        name = ET.SubElement(obj, 'name')
        name.text = ann.get('name', 'mine')

        # Pose
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        # Truncated
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'

        # Difficult
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'

        # Bounding box (NPY coordinates)
        bndbox = ET.SubElement(obj, 'bndbox')

        # Get mapped NPY coordinates
        mapped = ann['mapped_npy']

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(mapped['xmin'])

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(mapped['ymin'])

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(mapped['xmax'])

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(mapped['ymax'])

    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='\t')

    # Remove extra blank lines
    xml_lines = [line for line in xml_str.split('\n') if line.strip()]
    xml_str = '\n'.join(xml_lines)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"✓ XML saved: {output_path}")
    print(f"  - Total mines: {len(annotations)}")
    print(f"  - Image dimensions: {image_info['width']}×{image_info['height']}")


def main():
    # Paths
    json_path = Path('/Users/YMARX/Dropbox/2025_ECMiner/C_P02_기뢰전대/03_진행/Analysis_MD/mine_labeling_project/verified_data/flipped_20251104/flipped_mapped_annotations.json')

    xml_path = json_path.parent / 'flipped_npy_annotations.xml'

    print("="*60)
    print("JSON to PASCAL VOC XML Converter")
    print("="*60)
    print(f"Input:  {json_path.name}")
    print(f"Output: {xml_path.name}")
    print()

    # Load JSON
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations from JSON")

    # Image info (NPY dimensions)
    image_info = {
        'folder': 'flipped_20251104',
        'filename': 'flipped_labeled_intensity_data.npy',
        'width': 6400,
        'height': 5137,
        'depth': 1
    }

    # Convert to XML
    create_pascal_voc_xml(annotations, image_info, xml_path)

    print()
    print("="*60)
    print("✓ Conversion completed successfully!")
    print("="*60)

    # Verification
    print("\nFirst 3 bounding boxes:")
    for i, ann in enumerate(annotations[:3]):
        mapped = ann['mapped_npy']
        print(f"  Mine {i+1}: ({mapped['xmin']}, {mapped['ymin']}) → ({mapped['xmax']}, {mapped['ymax']})")


if __name__ == '__main__':
    main()
