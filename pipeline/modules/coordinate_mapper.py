"""
Coordinate Mapper Module
========================
좌표 매핑 모듈 - 픽셀과 GPS 좌표 간 변환
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from PIL import Image
import json

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """좌표 매핑 클래스"""

    def __init__(self, transform_type: str = "rotate_flip"):
        """
        Initialize Coordinate Mapper

        Args:
            transform_type: 이미지 변환 타입 ("rotate_flip", "none", "rotate_only", "flip_only")
        """
        self.transform_type = transform_type
        self.logger = logging.getLogger(self.__class__.__name__)

    def map(self,
            extracted_data: Dict[str, Any],
            gps_file: Path,
            annotation_image: Path) -> Dict[str, Any]:
        """
        픽셀 좌표와 GPS 좌표 매핑

        Args:
            extracted_data: 추출된 XTF 데이터
            gps_file: GPS 좌표 파일 (Excel)
            annotation_image: 어노테이션 이미지

        Returns:
            매핑 결과 딕셔너리
        """
        self.logger.info("Mapping pixel coordinates to GPS coordinates")

        try:
            # Load GPS data
            gps_data = self._load_gps_data(gps_file)

            # Load annotation image
            annotation_img = self._load_annotation_image(annotation_image)

            # Extract object locations from annotation
            object_locations = self._extract_object_locations(annotation_img)

            # Create coordinate mapping
            mapping = self._create_coordinate_mapping(
                object_locations,
                gps_data,
                extracted_data,
                annotation_img.size
            )

            return {
                'mappings': mapping,
                'gps_data': gps_data.to_dict('records'),
                'image_size': annotation_img.size,
                'transform_type': self.transform_type,
                'object_count': len(object_locations),
                'mapping_statistics': self._calculate_mapping_statistics(mapping)
            }

        except Exception as e:
            self.logger.error(f"Failed to create coordinate mapping: {e}")
            raise

    def _load_gps_data(self, gps_file: Path) -> pd.DataFrame:
        """GPS 데이터 로드 및 파싱"""
        if not gps_file.exists():
            raise FileNotFoundError(f"GPS file not found: {gps_file}")

        self.logger.debug(f"Loading GPS data from {gps_file}")

        # Load Excel file
        df = pd.read_excel(gps_file)

        # Parse coordinates
        df['lat'] = df['위도'].apply(self._parse_latitude)
        df['lon'] = df['경도'].apply(self._parse_longitude)

        # Sort by point ID
        df = df.sort_values('정점').reset_index(drop=True)

        self.logger.debug(f"Loaded {len(df)} GPS points")
        return df

    def _parse_latitude(self, coord_str: str) -> float:
        """위도 파싱 (예: '36.5933983 N')"""
        try:
            parts = coord_str.strip().split()
            if len(parts) >= 2 and 'N' in parts[1]:
                return float(parts[0])
            return 0.0
        except:
            return 0.0

    def _parse_longitude(self, coord_str: str) -> float:
        """경도 파싱 (예: '129 30.557773 E')"""
        try:
            parts = coord_str.strip().split()
            if len(parts) >= 3 and 'E' in parts[2]:
                deg = float(parts[0])
                min_val = float(parts[1])
                return deg + min_val / 60
            return 0.0
        except:
            return 0.0

    def _load_annotation_image(self, image_path: Path) -> Image.Image:
        """어노테이션 이미지 로드"""
        if not image_path.exists():
            raise FileNotFoundError(f"Annotation image not found: {image_path}")

        self.logger.debug(f"Loading annotation image from {image_path}")
        return Image.open(image_path)

    def _extract_object_locations(self, annotation_img: Image.Image) -> List[Dict]:
        """어노테이션 이미지에서 객체 위치 추출"""
        # Convert to numpy array
        img_array = np.array(annotation_img)

        # Find red bounding boxes (annotation markers)
        # Red channel이 높고 다른 채널이 낮은 픽셀 찾기
        red_mask = (img_array[:, :, 0] > 200) & \
                   (img_array[:, :, 1] < 100) & \
                   (img_array[:, :, 2] < 100)

        # Find connected components
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(red_mask)

        objects = []
        for i in range(1, num_features + 1):
            # Find bounding box of each component
            component_mask = labeled_array == i
            y_coords, x_coords = np.where(component_mask)

            if len(y_coords) > 10:  # 최소 크기 필터
                bbox = {
                    'x': int(np.min(x_coords)),
                    'y': int(np.min(y_coords)),
                    'width': int(np.max(x_coords) - np.min(x_coords)),
                    'height': int(np.max(y_coords) - np.min(y_coords))
                }

                # Calculate center
                center_x = bbox['x'] + bbox['width'] // 2
                center_y = bbox['y'] + bbox['height'] // 2

                objects.append({
                    'object_id': len(objects) + 1,
                    'pixel_x': center_x,
                    'pixel_y': center_y,
                    'bbox': bbox
                })

        # Sort by Y coordinate (top to bottom)
        objects.sort(key=lambda x: x['pixel_y'])

        # Reassign IDs after sorting
        for i, obj in enumerate(objects):
            obj['object_id'] = i + 1

        self.logger.debug(f"Extracted {len(objects)} objects from annotation")
        return objects

    def _create_coordinate_mapping(self,
                                 object_locations: List[Dict],
                                 gps_data: pd.DataFrame,
                                 extracted_data: Dict[str, Any],
                                 image_size: Tuple[int, int]) -> List[Dict]:
        """픽셀 좌표와 GPS 좌표 매핑 생성"""
        mappings = []

        # GPS 데이터 순서대로 객체와 매핑
        for i, (_, gps_row) in enumerate(gps_data.iterrows()):
            if i < len(object_locations):
                obj = object_locations[i]

                # Transform coordinates if needed
                transformed_coords = self._apply_coordinate_transform(
                    obj['pixel_x'], obj['pixel_y'], image_size
                )

                # Calculate mapping confidence
                confidence = self._calculate_mapping_confidence(
                    obj, gps_row, i, len(object_locations)
                )

                mapping = {
                    'object_id': obj['object_id'],
                    'pixel_x': obj['pixel_x'],
                    'pixel_y': obj['pixel_y'],
                    'transformed_x': transformed_coords[0],
                    'transformed_y': transformed_coords[1],
                    'bbox': obj['bbox'],
                    'gps_point_id': gps_row['정점'],
                    'latitude': gps_row['lat'],
                    'longitude': gps_row['lon'],
                    'mapping_confidence': confidence
                }

                mappings.append(mapping)

        return mappings

    def _apply_coordinate_transform(self,
                                  pixel_x: int,
                                  pixel_y: int,
                                  image_size: Tuple[int, int]) -> Tuple[int, int]:
        """좌표 변환 적용 (180도 회전 + 좌우 반전)"""
        width, height = image_size

        if self.transform_type == "rotate_flip":
            # 180도 회전 + 좌우 반전
            # 1. 180도 회전: (x, y) -> (width-x, height-y)
            # 2. 좌우 반전: (x, y) -> (width-x, y)
            # 결합: (x, y) -> (x, height-y)
            transformed_x = pixel_x
            transformed_y = height - pixel_y

        elif self.transform_type == "rotate_only":
            # 180도 회전만
            transformed_x = width - pixel_x
            transformed_y = height - pixel_y

        elif self.transform_type == "flip_only":
            # 좌우 반전만
            transformed_x = width - pixel_x
            transformed_y = pixel_y

        else:  # "none"
            transformed_x = pixel_x
            transformed_y = pixel_y

        return (transformed_x, transformed_y)

    def _calculate_mapping_confidence(self,
                                    obj: Dict,
                                    gps_row: pd.Series,
                                    index: int,
                                    total_objects: int) -> float:
        """매핑 신뢰도 계산"""
        # 기본 신뢰도
        base_confidence = 0.8

        # 객체 크기 기반 조정
        bbox_area = obj['bbox']['width'] * obj['bbox']['height']
        size_factor = min(1.0, bbox_area / 1000)  # 1000픽셀을 기준

        # 순서 기반 조정 (가장자리 객체는 신뢰도 낮음)
        if index == 0 or index == total_objects - 1:
            order_factor = 0.7
        else:
            order_factor = 1.0

        confidence = base_confidence * size_factor * order_factor
        return min(1.0, confidence)

    def _calculate_mapping_statistics(self, mappings: List[Dict]) -> Dict[str, Any]:
        """매핑 통계 계산"""
        if not mappings:
            return {}

        confidences = [m['mapping_confidence'] for m in mappings]
        latitudes = [m['latitude'] for m in mappings]
        longitudes = [m['longitude'] for m in mappings]

        return {
            'total_mappings': len(mappings),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'latitude_range': [np.min(latitudes), np.max(latitudes)],
            'longitude_range': [np.min(longitudes), np.max(longitudes)],
            'coordinate_span': {
                'lat_span': np.max(latitudes) - np.min(latitudes),
                'lon_span': np.max(longitudes) - np.min(longitudes)
            }
        }

    def save_mapping(self, mapping_data: Dict[str, Any], output_path: Path):
        """매핑 결과 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(mapping_data, f, indent=2, default=str)

        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        if mapping_data.get('mappings'):
            df = pd.DataFrame(mapping_data['mappings'])
            df.to_csv(csv_path, index=False)

        self.logger.info(f"Mapping saved to {json_path} and {csv_path}")

    def load_mapping(self, mapping_path: Path) -> Dict[str, Any]:
        """저장된 매핑 로드"""
        if mapping_path.suffix == '.json':
            with open(mapping_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Only JSON format supported for loading")