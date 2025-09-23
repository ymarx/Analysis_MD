"""
XTF Reader Module
=================
XTF 파일 읽기 모듈 - 검증된 working 방식 사용
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import working components
try:
    from src.data_processing.xtf_reader import XTFReader as WorkingXTFReader
    from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor, IntensityPing
    WORKING_READER_AVAILABLE = True
except ImportError:
    WORKING_READER_AVAILABLE = False

logger = logging.getLogger(__name__)


class XTFReader:
    """XTF 파일 읽기 클래스 - 검증된 working 방식 사용"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize working components if available
        if WORKING_READER_AVAILABLE:
            self.extractor = XTFIntensityExtractor()
            self.logger.info("Working XTF components initialized successfully")
        else:
            self.extractor = None
            self.logger.warning("Working XTF components not available")

    def read(self, xtf_path: Path) -> Dict[str, Any]:
        """
        검증된 방식으로 XTF 파일 읽기

        Args:
            xtf_path: XTF 파일 경로

        Returns:
            XTF 데이터 딕셔너리
        """
        self.logger.info(f"Reading XTF file with working reader: {xtf_path}")

        if not WORKING_READER_AVAILABLE:
            raise RuntimeError("Working XTF reader components not available")

        try:
            # Step 1: Create XTF reader with filepath
            self.logger.info("Step 1: Creating XTF reader with filepath")
            reader = WorkingXTFReader(str(xtf_path), max_pings=200)  # Limit for speed

            # Load file
            if not reader.load_file():
                raise ValueError("Failed to load XTF file")

            # Parse pings using the proven working method
            ping_data = reader.parse_pings()

            if not ping_data:
                raise ValueError("Failed to parse pings from XTF data")

            self.logger.info(f"Successfully parsed {len(ping_data)} pings")

            # Step 2: Convert ping_data to IntensityPing objects (as in working code)
            self.logger.info("Step 2: Converting ping data to IntensityPing objects")

            intensity_pings = []

            for i, ping in enumerate(ping_data):
                # Convert Reader's PingData to Extractor's IntensityPing format
                # Split ping.data into port and starboard channels (as in process_edgetech_complete.py)
                data_len = len(ping.data) if ping.data is not None else 0
                if data_len > 0:
                    mid_point = data_len // 2
                    port_intensity = ping.data[:mid_point]
                    starboard_intensity = ping.data[mid_point:]
                else:
                    port_intensity = np.array([], dtype=np.float32)
                    starboard_intensity = np.array([], dtype=np.float32)

                intensity_ping = IntensityPing(
                    ping_number=ping.ping_number,
                    timestamp=ping.timestamp.timestamp() if hasattr(ping.timestamp, 'timestamp') else float(ping.timestamp),
                    latitude=ping.latitude,
                    longitude=ping.longitude,
                    heading=0.0,  # Reader에서는 heading 정보가 없음, 기본값 사용
                    port_intensity=port_intensity,
                    starboard_intensity=starboard_intensity,
                    port_range=np.arange(len(port_intensity), dtype=np.float32),
                    starboard_range=np.arange(len(starboard_intensity), dtype=np.float32)
                )
                intensity_pings.append(intensity_ping)

            self.logger.info(f"Converted {len(intensity_pings)} ping objects")

            # Step 3: Extract intensity data using working extractor methods
            self.logger.info("Step 3: Extracting intensity data")

            # Create intensity images
            images = self.extractor._create_intensity_images(intensity_pings)

            # Extract navigation data
            nav_data = self.extractor._extract_navigation_data(intensity_pings)

            # Combine into intensity_data format
            intensity_data = {
                'combined_shape': images.get('combined', np.array([])).shape,
                'port_shape': images.get('port', np.array([])).shape,
                'starboard_shape': images.get('starboard', np.array([])).shape,
                'intensity_images': images,
                'navigation': nav_data,
                'ping_data': intensity_pings
            }

            if not intensity_data:
                raise ValueError("Failed to extract intensity data")

            # Combine data into unified format
            combined_data = {
                'summary': {
                    'filename': xtf_path.name,
                    'total_pings': len(ping_data),
                    'processed_pings': len(ping_data),
                    'num_sonar_channels': 2,  # Edgetech 4205 has port/starboard
                    'frequency_info': {},
                    'coordinate_bounds': self._extract_coordinate_bounds(intensity_data),
                    'time_range': [None, None],  # TODO: Extract from pings if needed
                    'is_loaded': True,
                    'is_parsed': True
                },
                'ping_count': len(ping_data),
                'intensity_matrix_shape': intensity_data.get('combined_shape', [0, 0]),
                'coordinate_stats': self._calculate_coordinate_stats(intensity_data),
                'intensity_data': intensity_data,
                'raw_pings': ping_data[:10] if len(ping_data) > 10 else ping_data  # Save sample for debugging
            }

            self.logger.info("XTF data extraction completed successfully")
            self.logger.info(f"Combined intensity shape: {intensity_data.get('combined_shape', 'N/A')}")

            return combined_data

        except Exception as e:
            self.logger.error(f"Failed to read XTF file: {e}")
            raise

    def _extract_coordinate_bounds(self, intensity_data: Dict) -> Dict:
        """Extract coordinate bounds from intensity data"""
        try:
            navigation = intensity_data.get('navigation', {})
            latitudes = navigation.get('latitudes', [])
            longitudes = navigation.get('longitudes', [])

            if latitudes and longitudes:
                return {
                    'lat': [float(min(latitudes)), float(max(latitudes))],
                    'lon': [float(min(longitudes)), float(max(longitudes))]
                }
            else:
                return {'lat': [0, 0], 'lon': [0, 0]}

        except Exception as e:
            self.logger.warning(f"Failed to extract coordinate bounds: {e}")
            return {'lat': [0, 0], 'lon': [0, 0]}

    def _calculate_coordinate_stats(self, intensity_data: Dict) -> Dict:
        """Calculate coordinate statistics"""
        try:
            navigation = intensity_data.get('navigation', {})
            latitudes = navigation.get('latitudes', [])
            longitudes = navigation.get('longitudes', [])

            if latitudes and longitudes:
                return {
                    'latitude_range': [float(min(latitudes)), float(max(latitudes))],
                    'longitude_range': [float(min(longitudes)), float(max(longitudes))],
                    'total_records': len(latitudes)
                }
            else:
                return {
                    'latitude_range': [0, 0],
                    'longitude_range': [0, 0],
                    'total_records': 0
                }

        except Exception as e:
            self.logger.warning(f"Failed to calculate coordinate stats: {e}")
            return {
                'latitude_range': [0, 0],
                'longitude_range': [0, 0],
                'total_records': 0
            }

    def validate_xtf_file(self, xtf_path: Path) -> bool:
        """XTF 파일 유효성 검사"""
        try:
            if not xtf_path.exists():
                return False

            # 파일 크기 확인
            if xtf_path.stat().st_size < 1024:  # 최소 1KB
                return False

            # 확장자 확인
            if xtf_path.suffix.lower() not in ['.xtf']:
                return False

            # 간단한 읽기 테스트
            data = self.read(xtf_path)
            return data is not None

        except Exception as e:
            self.logger.error(f"XTF validation failed: {e}")
            return False

    def get_file_info(self, xtf_path: Path) -> Dict[str, Any]:
        """XTF 파일 정보 반환"""
        if not xtf_path.exists():
            return {}

        try:
            data = self.read(xtf_path)
            summary = data.get('summary', {})
            intensity_data = data.get('intensity_data', {})
            navigation = intensity_data.get('navigation', {})

            return {
                'file_path': str(xtf_path),
                'file_size_mb': xtf_path.stat().st_size / (1024 * 1024),
                'ping_count': data.get('ping_count', 0),
                'intensity_matrix_shape': data.get('intensity_matrix_shape', [0, 0]),
                'navigation_points': len(navigation.get('latitudes', [])),
                'coordinate_bounds': summary.get('coordinate_bounds', {}),
                'system_type': summary.get('system_type', 'Unknown')
            }

        except Exception as e:
            self.logger.error(f"Failed to get file info: {e}")
            return {'error': str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Get reader summary information"""
        return {
            'reader_type': 'XTFReader',
            'components_available': WORKING_READER_AVAILABLE,
            'description': 'Uses proven working XTF processing from process_edgetech_complete.py'
        }