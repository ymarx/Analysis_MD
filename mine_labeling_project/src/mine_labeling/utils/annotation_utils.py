"""
Annotation Format Conversion Utilities
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any


def parse_xml_annotations(xml_path: str) -> Dict[str, Any]:
    """
    Parse PASCAL VOC XML annotation file

    Args:
        xml_path: Path to XML file

    Returns:
        Dictionary with parsed annotations
    """
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse image metadata
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    # Parse objects
    annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')

        annotation = {
            'name': name,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }

        annotations.append(annotation)

    result = {
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'annotations': annotations
    }

    return result


def convert_to_json(xml_data: Dict[str, Any],
                    output_format: str = 'simple') -> Dict[str, Any]:
    """
    Convert XML annotation data to JSON format

    Args:
        xml_data: Parsed XML data
        output_format: 'simple' or 'coco'

    Returns:
        JSON-formatted dictionary
    """
    if output_format == 'simple':
        return _convert_to_simple_json(xml_data)
    elif output_format == 'coco':
        return _convert_to_coco_json(xml_data)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def _convert_to_simple_json(xml_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert to simple JSON format

    Args:
        xml_data: Parsed XML data

    Returns:
        Simple JSON dictionary
    """
    result = {
        'image': {
            'filename': xml_data['filename'],
            'width': xml_data['width'],
            'height': xml_data['height'],
            'depth': xml_data['depth']
        },
        'annotations': []
    }

    for i, ann in enumerate(xml_data['annotations']):
        annotation = {
            'id': i + 1,
            'category': ann['name'],
            'bbox': {
                'xmin': ann['xmin'],
                'ymin': ann['ymin'],
                'xmax': ann['xmax'],
                'ymax': ann['ymax'],
                'width': ann['xmax'] - ann['xmin'],
                'height': ann['ymax'] - ann['ymin']
            }
        }

        result['annotations'].append(annotation)

    return result


def _convert_to_coco_json(xml_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert to COCO format JSON

    Args:
        xml_data: Parsed XML data

    Returns:
        COCO-formatted JSON dictionary
    """
    # COCO format structure
    result = {
        'info': {
            'description': 'Mine Detection Dataset',
            'version': '1.0',
            'year': 2025
        },
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Add image
    image_id = 1
    result['images'].append({
        'id': image_id,
        'file_name': xml_data['filename'],
        'width': xml_data['width'],
        'height': xml_data['height']
    })

    # Add categories (unique object names)
    categories = list(set([ann['name'] for ann in xml_data['annotations']]))
    category_map = {}

    for i, category in enumerate(categories):
        category_id = i + 1
        category_map[category] = category_id

        result['categories'].append({
            'id': category_id,
            'name': category,
            'supercategory': 'object'
        })

    # Add annotations
    for i, ann in enumerate(xml_data['annotations']):
        annotation_id = i + 1
        category_id = category_map[ann['name']]

        x = ann['xmin']
        y = ann['ymin']
        width = ann['xmax'] - ann['xmin']
        height = ann['ymax'] - ann['ymin']

        result['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': [x, y, width, height],  # COCO format: [x, y, width, height]
            'area': width * height,
            'iscrowd': 0
        })

    return result


def xml_to_json_file(xml_path: str,
                     output_path: str,
                     output_format: str = 'simple') -> None:
    """
    Convert XML annotation file to JSON file

    Args:
        xml_path: Path to XML file
        output_path: Output JSON file path
        output_format: 'simple' or 'coco'
    """
    import json

    # Parse XML
    xml_data = parse_xml_annotations(xml_path)

    # Convert to JSON
    json_data = convert_to_json(xml_data, output_format)

    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
