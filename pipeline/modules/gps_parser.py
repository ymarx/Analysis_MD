"""
GPS Coordinate Parser Module
===========================
다양한 형식의 GPS 좌표를 파싱하는 유틸리티 모듈
"""

import re
import logging
from typing import Tuple, Optional, Union
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class GPSParser:
    """GPS 좌표 파서"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def parse_coordinate_string(self, coord_str: str) -> Optional[float]:
        """
        다양한 형식의 좌표 문자열을 float로 변환

        지원 형식:
        - "36.5933983 N" -> 36.5933983
        - "129.5151167 E" -> 129.5151167
        - "36°35'36.23" -> 36.593397
        - "129°30'54.42" -> 129.515117
        - "36.5933983" -> 36.5933983
        """
        if not coord_str or pd.isna(coord_str):
            return None

        coord_str = str(coord_str).strip()

        try:
            # Pattern 1: "36.5933983 N" or "129.5151167 E"
            decimal_with_direction = re.match(r'^([\d.-]+)\s*([NSEW])$', coord_str, re.IGNORECASE)
            if decimal_with_direction:
                value = float(decimal_with_direction.group(1))
                direction = decimal_with_direction.group(2).upper()

                # Apply direction (S and W are negative)
                if direction in ['S', 'W']:
                    value = -value

                return value

            # Pattern 2: "36°35'36.23" (degrees, minutes, seconds)
            dms_pattern = re.match(r'^(\d+)°(\d+)\'([\d.]+)"?\s*([NSEW])?$', coord_str, re.IGNORECASE)
            if dms_pattern:
                degrees = int(dms_pattern.group(1))
                minutes = int(dms_pattern.group(2))
                seconds = float(dms_pattern.group(3))
                direction = dms_pattern.group(4)

                # Convert to decimal degrees
                decimal = degrees + minutes/60 + seconds/3600

                # Apply direction if present
                if direction and direction.upper() in ['S', 'W']:
                    decimal = -decimal

                return decimal

            # Pattern 3: "36°35.604'" (degrees, decimal minutes)
            dm_pattern = re.match(r'^(\d+)°([\d.]+)\'?\s*([NSEW])?$', coord_str, re.IGNORECASE)
            if dm_pattern:
                degrees = int(dm_pattern.group(1))
                minutes = float(dm_pattern.group(2))
                direction = dm_pattern.group(3)

                # Convert to decimal degrees
                decimal = degrees + minutes/60

                # Apply direction if present
                if direction and direction.upper() in ['S', 'W']:
                    decimal = -decimal

                return decimal

            # Pattern 4: "129 30.557773 E" (degrees space decimal_minutes direction)
            space_dm_pattern = re.match(r'^(\d+)\s+([\d.]+)\s*([NSEW])$', coord_str, re.IGNORECASE)
            if space_dm_pattern:
                degrees = int(space_dm_pattern.group(1))
                minutes = float(space_dm_pattern.group(2))
                direction = space_dm_pattern.group(3).upper()

                # Convert to decimal degrees
                decimal = degrees + minutes/60

                # Apply direction (S and W are negative)
                if direction in ['S', 'W']:
                    decimal = -decimal

                return decimal

            # Pattern 5: Simple decimal number
            decimal_pattern = re.match(r'^([\d.-]+)$', coord_str)
            if decimal_pattern:
                return float(decimal_pattern.group(1))

            # If none of the patterns match, try direct float conversion
            return float(coord_str)

        except (ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse coordinate '{coord_str}': {e}")
            return None

    def parse_gps_file(self, file_path: Path) -> list:
        """
        GPS 파일을 파싱하여 좌표 리스트 반환

        Args:
            file_path: GPS 데이터 파일 경로 (.xlsx, .csv 지원)

        Returns:
            좌표 딕셔너리 리스트
        """
        self.logger.info(f"Parsing GPS file: {file_path}")

        try:
            # Load file based on extension
            if file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            self.logger.info(f"Loaded GPS file with {len(df)} rows and columns: {list(df.columns)}")

            # Try to identify coordinate columns
            lat_col, lon_col, id_col = self._identify_coordinate_columns(df)

            if not lat_col or not lon_col:
                raise ValueError("Could not identify latitude and longitude columns")

            self.logger.info(f"Identified columns - ID: {id_col}, Lat: {lat_col}, Lon: {lon_col}")

            # Parse coordinates
            locations = []
            parsed_count = 0

            for idx, row in df.iterrows():
                try:
                    # Parse latitude
                    lat_raw = row[lat_col]
                    lat = self.parse_coordinate_string(lat_raw)

                    # Parse longitude
                    lon_raw = row[lon_col]
                    lon = self.parse_coordinate_string(lon_raw)

                    if lat is not None and lon is not None:
                        # Get ID
                        point_id = row[id_col] if id_col else f'Point_{idx+1:02d}'

                        location = {
                            'id': str(point_id),
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'raw_latitude': str(lat_raw),
                            'raw_longitude': str(lon_raw),
                            'row_index': idx
                        }

                        locations.append(location)
                        parsed_count += 1

                    else:
                        self.logger.warning(f"Failed to parse coordinates at row {idx}: lat='{lat_raw}', lon='{lon_raw}'")

                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")

            self.logger.info(f"Successfully parsed {parsed_count} out of {len(df)} GPS locations")

            return locations

        except Exception as e:
            self.logger.error(f"Failed to parse GPS file: {e}")
            raise

    def _identify_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """GPS 데이터프레임에서 좌표 컬럼 식별"""

        columns = [col.lower() for col in df.columns]

        # Latitude column patterns
        lat_patterns = ['lat', 'latitude', '위도', 'y', 'north']
        lat_col = None
        for pattern in lat_patterns:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                lat_col = matches[0]
                break

        # Longitude column patterns
        lon_patterns = ['lon', 'lng', 'longitude', '경도', 'x', 'east']
        lon_col = None
        for pattern in lon_patterns:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                lon_col = matches[0]
                break

        # ID column patterns
        id_patterns = ['id', 'point_id', 'name', 'point', '지점', 'no', 'number']
        id_col = None
        for pattern in id_patterns:
            matches = [col for col in df.columns if pattern in col.lower()]
            if matches:
                id_col = matches[0]
                break

        return lat_col, lon_col, id_col

    def validate_coordinates(self, locations: list) -> dict:
        """좌표 유효성 검증"""

        validation_results = {
            'total_count': len(locations),
            'valid_count': 0,
            'invalid_count': 0,
            'validation_errors': [],
            'coordinate_ranges': {
                'latitude': {'min': None, 'max': None},
                'longitude': {'min': None, 'max': None}
            }
        }

        valid_lats = []
        valid_lons = []

        for location in locations:
            try:
                lat = location['latitude']
                lon = location['longitude']

                # Check latitude range (-90 to 90)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    validation_results['valid_count'] += 1
                    valid_lats.append(lat)
                    valid_lons.append(lon)
                else:
                    validation_results['invalid_count'] += 1
                    validation_results['validation_errors'].append({
                        'id': location['id'],
                        'error': f'Coordinates out of range: lat={lat}, lon={lon}'
                    })

            except Exception as e:
                validation_results['invalid_count'] += 1
                validation_results['validation_errors'].append({
                    'id': location.get('id', 'unknown'),
                    'error': f'Validation error: {e}'
                })

        # Calculate ranges for valid coordinates
        if valid_lats and valid_lons:
            validation_results['coordinate_ranges'] = {
                'latitude': {'min': min(valid_lats), 'max': max(valid_lats)},
                'longitude': {'min': min(valid_lons), 'max': max(valid_lons)}
            }

        self.logger.info(f"GPS validation: {validation_results['valid_count']}/{validation_results['total_count']} valid coordinates")

        return validation_results


def test_gps_parser():
    """GPS 파서 테스트"""
    parser = GPSParser()

    # Test coordinate parsing
    test_coordinates = [
        "36.5933983 N",
        "129.5151167 E",
        "36°35'36.23\"",
        "129°30'54.42\"",
        "36.5933983",
        "-129.5151167"
    ]

    print("Testing coordinate parsing:")
    for coord in test_coordinates:
        result = parser.parse_coordinate_string(coord)
        print(f"  '{coord}' -> {result}")

    # Test GPS file parsing if file exists
    gps_file = Path("datasets/Location_MDGPS.xlsx")
    if gps_file.exists():
        print(f"\nTesting GPS file parsing: {gps_file}")
        try:
            locations = parser.parse_gps_file(gps_file)
            print(f"  Parsed {len(locations)} locations")
            if locations:
                print(f"  Sample: {locations[0]}")

            validation = parser.validate_coordinates(locations)
            print(f"  Validation: {validation['valid_count']}/{validation['total_count']} valid")

        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"\nGPS file not found: {gps_file}")


if __name__ == "__main__":
    test_gps_parser()