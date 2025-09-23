#!/usr/bin/env python3
"""
좌표 검증 분석
===============
original XTF 파일들과 Location_MDGPS.xlsx의 좌표 범위를 정밀 비교하여
기존 분석에서 "다른 위치"로 나온 원인을 규명하고 정확한 일치성을 판단

Author: YMARX
Date: 2025-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.modules.gps_parser import GPSParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import XTF libraries with fallback
try:
    import pyxtf
    PYXTF_AVAILABLE = True
    logger.info("pyxtf is available for XTF processing")
except ImportError:
    PYXTF_AVAILABLE = False
    logger.warning("pyxtf not available - using alternative methods")


class CoordinateVerificationAnalyzer:
    """좌표 검증 분석 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.original_xtf_files = []
        self.gps_data = None
        self.analysis_results = {}

    def find_original_xtf_files(self):
        """original 폴더의 XTF 파일들 찾기"""
        self.logger.info("Finding original XTF files")

        datasets_dir = Path("datasets")
        self.original_xtf_files = []

        for item in datasets_dir.iterdir():
            if item.is_dir():
                original_dir = item / "original"
                if original_dir.exists():
                    for xtf_file in original_dir.glob("*.xtf"):
                        self.original_xtf_files.append(xtf_file)

        self.logger.info(f"Found {len(self.original_xtf_files)} original XTF files:")
        for xtf_file in self.original_xtf_files:
            self.logger.info(f"  - {xtf_file}")

        return self.original_xtf_files

    def load_gps_data(self):
        """Location_MDGPS.xlsx 로드"""
        self.logger.info("Loading GPS data from Location_MDGPS.xlsx")

        gps_file = Path("datasets/Location_MDGPS.xlsx")

        if not gps_file.exists():
            self.logger.error(f"GPS file not found: {gps_file}")
            return None

        try:
            gps_parser = GPSParser()
            mine_locations = gps_parser.parse_gps_file(gps_file)
            validation = gps_parser.validate_coordinates(mine_locations)

            self.gps_data = {
                'locations': mine_locations,
                'count': len(mine_locations),
                'validation': validation,
                'bounds': self._calculate_gps_bounds(mine_locations)
            }

            self.logger.info(f"Loaded {len(mine_locations)} GPS mine locations")
            self.logger.info(f"GPS bounds: {self.gps_data['bounds']}")

            return self.gps_data

        except Exception as e:
            self.logger.error(f"Error loading GPS data: {e}")
            return None

    def _calculate_gps_bounds(self, locations):
        """GPS 좌표 범위 계산"""
        if not locations:
            return None

        lats = [loc['latitude'] for loc in locations]
        lons = [loc['longitude'] for loc in locations]

        return {
            'latitude': {
                'min': min(lats),
                'max': max(lats),
                'center': sum(lats) / len(lats),
                'range': max(lats) - min(lats)
            },
            'longitude': {
                'min': min(lons),
                'max': max(lons),
                'center': sum(lons) / len(lons),
                'range': max(lons) - min(lons)
            }
        }

    def extract_xtf_metadata_robust(self, xtf_path: Path) -> Dict[str, Any]:
        """강화된 XTF 메타데이터 추출 (여러 방법 시도)"""
        self.logger.info(f"Extracting metadata from: {xtf_path.name}")

        metadata = {
            'file_path': str(xtf_path),
            'file_name': xtf_path.name,
            'file_size_mb': xtf_path.stat().st_size / (1024 * 1024),
            'extraction_method': None,
            'coordinates': {},
            'timing': {},
            'sonar_params': {},
            'ping_summary': {},
            'errors': []
        }

        # Method 1: pyxtf 사용
        if PYXTF_AVAILABLE:
            try:
                self.logger.info("Attempting extraction with pyxtf")
                pyxtf_result = self._extract_with_pyxtf(xtf_path)
                if pyxtf_result and pyxtf_result.get('success', False):
                    metadata.update(pyxtf_result)
                    metadata['extraction_method'] = 'pyxtf'
                    return metadata
            except Exception as e:
                self.logger.warning(f"pyxtf extraction failed: {e}")
                metadata['errors'].append(f"pyxtf: {e}")

        # Method 2: 바이너리 헤더 분석
        try:
            self.logger.info("Attempting binary header analysis")
            binary_result = self._extract_binary_header(xtf_path)
            if binary_result and binary_result.get('success', False):
                metadata.update(binary_result)
                metadata['extraction_method'] = 'binary_header'
                return metadata
        except Exception as e:
            self.logger.warning(f"Binary header analysis failed: {e}")
            metadata['errors'].append(f"binary_header: {e}")

        # Method 3: 파일명에서 정보 추출
        try:
            self.logger.info("Extracting info from filename")
            filename_result = self._extract_from_filename(xtf_path)
            metadata.update(filename_result)
            metadata['extraction_method'] = 'filename_only'
        except Exception as e:
            self.logger.warning(f"Filename analysis failed: {e}")
            metadata['errors'].append(f"filename: {e}")

        return metadata

    def _extract_with_pyxtf(self, xtf_path: Path) -> Dict[str, Any]:
        """pyxtf를 사용한 메타데이터 추출"""
        try:
            # Read XTF file
            (fh, p) = pyxtf.xtf_read(str(xtf_path))

            result = {
                'success': True,
                'coordinates': {},
                'timing': {},
                'sonar_params': {},
                'ping_summary': {}
            }

            # Extract file header information
            if hasattr(fh, 'FileFormat'):
                result['file_format'] = fh.FileFormat
            if hasattr(fh, 'SystemType'):
                result['system_type'] = fh.SystemType
            if hasattr(fh, 'SonarName'):
                result['sonar_name'] = fh.SonarName.decode('utf-8') if isinstance(fh.SonarName, bytes) else str(fh.SonarName)
            if hasattr(fh, 'PipeX100'):
                result['sonar_params']['frequency'] = fh.PipeX100 / 100.0

            # Extract navigation data
            nav_packets = [packet for packet in p if hasattr(packet, 'Latitude')]

            if nav_packets:
                latitudes = []
                longitudes = []
                timestamps = []

                for packet in nav_packets:
                    if hasattr(packet, 'Latitude') and hasattr(packet, 'Longitude'):
                        # Convert from XTF coordinate format
                        lat = self._convert_xtf_coordinate(packet.Latitude)
                        lon = self._convert_xtf_coordinate(packet.Longitude)

                        if self._is_valid_coordinate(lat, lon):
                            latitudes.append(lat)
                            longitudes.append(lon)

                        if hasattr(packet, 'TimeTag'):
                            timestamps.append(packet.TimeTag)

                if latitudes and longitudes:
                    result['coordinates'] = {
                        'latitude': {
                            'min': min(latitudes),
                            'max': max(latitudes),
                            'center': sum(latitudes) / len(latitudes),
                            'range': max(latitudes) - min(latitudes),
                            'count': len(latitudes)
                        },
                        'longitude': {
                            'min': min(longitudes),
                            'max': max(longitudes),
                            'center': sum(longitudes) / len(longitudes),
                            'range': max(longitudes) - min(longitudes),
                            'count': len(longitudes)
                        }
                    }

                if timestamps:
                    result['timing'] = {
                        'first_timestamp': min(timestamps),
                        'last_timestamp': max(timestamps),
                        'duration_seconds': max(timestamps) - min(timestamps),
                        'timestamp_count': len(timestamps)
                    }

            # Extract ping information
            ping_packets = [packet for packet in p if hasattr(packet, 'ping_chan_headers')]

            result['ping_summary'] = {
                'total_packets': len(p),
                'navigation_packets': len(nav_packets),
                'ping_packets': len(ping_packets),
                'packet_types': list(set([type(packet).__name__ for packet in p]))
            }

            self.logger.info(f"pyxtf extraction successful - found {len(nav_packets)} nav packets")
            return result

        except Exception as e:
            self.logger.error(f"pyxtf extraction failed: {e}")
            return {'success': False, 'error': str(e)}

    def _convert_xtf_coordinate(self, coord_value):
        """XTF 좌표 형식을 십진도로 변환"""
        try:
            # XTF coordinates are often in minutes * 10000 format
            if abs(coord_value) > 1000000:  # Large number, likely in minutes format
                # Convert from minutes to decimal degrees
                degrees = int(coord_value / 600000)  # 60 minutes * 10000
                minutes = (coord_value % 600000) / 10000.0
                decimal_degrees = degrees + (minutes / 60.0)
            else:
                # Already in decimal degrees
                decimal_degrees = coord_value

            return decimal_degrees

        except Exception as e:
            self.logger.warning(f"Coordinate conversion failed: {e}")
            return 0.0

    def _is_valid_coordinate(self, lat, lon):
        """좌표 유효성 검사"""
        # Basic sanity checks for coordinates around Korea
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            # More specific check for Korea region
            if 33 <= lat <= 43 and 124 <= lon <= 132:
                return True
        return False

    def _extract_binary_header(self, xtf_path: Path) -> Dict[str, Any]:
        """바이너리 헤더 분석"""
        try:
            with open(xtf_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB for header analysis

            result = {
                'success': True,
                'file_signature': header[:4].hex(),
                'header_size': len(header)
            }

            # Look for coordinate-like patterns in binary data
            # This is a simplified approach - real XTF parsing is much more complex
            self.logger.info("Binary header analysis completed (basic)")
            return result

        except Exception as e:
            self.logger.error(f"Binary header analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_from_filename(self, xtf_path: Path) -> Dict[str, Any]:
        """파일명에서 정보 추출"""
        filename = xtf_path.name

        result = {
            'success': True,
            'inferred_info': {}
        }

        # Parse filename pattern: Location_System_Frequency_Date_etc
        parts = filename.replace('.xtf', '').split('_')

        if len(parts) >= 4:
            result['inferred_info'] = {
                'location': parts[0] + '_' + parts[1] if len(parts) > 1 else parts[0],
                'system': parts[2] if len(parts) > 2 else 'Unknown',
                'frequency': parts[3] if len(parts) > 3 else 'Unknown',
                'date_string': parts[4] if len(parts) > 4 else 'Unknown'
            }

            # Try to extract system type
            if 'Edgetech' in filename:
                result['inferred_info']['system_type'] = 'EdgeTech'
                if '4205' in filename:
                    result['inferred_info']['model'] = '4205'
            elif 'Klein' in filename:
                result['inferred_info']['system_type'] = 'Klein'
                if '3900' in filename:
                    result['inferred_info']['model'] = '3900'

        return result

    def analyze_coordinate_correspondence(self):
        """좌표 대응 관계 종합 분석"""
        self.logger.info("Starting comprehensive coordinate correspondence analysis")

        if not self.gps_data or not self.original_xtf_files:
            self.logger.error("GPS data or XTF files not loaded")
            return None

        analysis_results = {
            'gps_data': self.gps_data,
            'xtf_analyses': [],
            'coordinate_comparisons': [],
            'overlap_analysis': {},
            'distance_analysis': {},
            'conclusion': {}
        }

        # Analyze each XTF file
        for xtf_path in self.original_xtf_files:
            self.logger.info(f"Analyzing {xtf_path.name}")

            xtf_metadata = self.extract_xtf_metadata_robust(xtf_path)
            analysis_results['xtf_analyses'].append(xtf_metadata)

            # Compare coordinates if available
            if xtf_metadata.get('coordinates'):
                comparison = self._compare_coordinate_ranges(
                    self.gps_data['bounds'],
                    xtf_metadata['coordinates']
                )
                comparison['xtf_file'] = xtf_path.name
                analysis_results['coordinate_comparisons'].append(comparison)

        # Overall overlap analysis
        analysis_results['overlap_analysis'] = self._analyze_overall_overlap(
            analysis_results['coordinate_comparisons']
        )

        # Distance analysis
        analysis_results['distance_analysis'] = self._analyze_coordinate_distances(
            analysis_results['coordinate_comparisons']
        )

        # Final conclusion
        analysis_results['conclusion'] = self._generate_conclusion(analysis_results)

        self.analysis_results = analysis_results
        return analysis_results

    def _compare_coordinate_ranges(self, gps_bounds, xtf_coords):
        """GPS와 XTF 좌표 범위 비교"""
        comparison = {
            'overlap': {},
            'distance': {},
            'similarity_score': 0.0
        }

        try:
            gps_lat = gps_bounds['latitude']
            gps_lon = gps_bounds['longitude']
            xtf_lat = xtf_coords['latitude']
            xtf_lon = xtf_coords['longitude']

            # Overlap analysis
            lat_overlap = max(0, min(gps_lat['max'], xtf_lat['max']) - max(gps_lat['min'], xtf_lat['min']))
            lon_overlap = max(0, min(gps_lon['max'], xtf_lon['max']) - max(gps_lon['min'], xtf_lon['min']))

            gps_lat_range = gps_lat['max'] - gps_lat['min']
            gps_lon_range = gps_lon['max'] - gps_lon['min']
            xtf_lat_range = xtf_lat['max'] - xtf_lat['min']
            xtf_lon_range = xtf_lon['max'] - xtf_lon['min']

            comparison['overlap'] = {
                'latitude_overlap': lat_overlap,
                'longitude_overlap': lon_overlap,
                'latitude_overlap_percent_gps': (lat_overlap / gps_lat_range * 100) if gps_lat_range > 0 else 0,
                'longitude_overlap_percent_gps': (lon_overlap / gps_lon_range * 100) if gps_lon_range > 0 else 0,
                'latitude_overlap_percent_xtf': (lat_overlap / xtf_lat_range * 100) if xtf_lat_range > 0 else 0,
                'longitude_overlap_percent_xtf': (lon_overlap / xtf_lon_range * 100) if xtf_lon_range > 0 else 0
            }

            # Distance analysis between centers
            gps_center_lat = gps_lat['center']
            gps_center_lon = gps_lon['center']
            xtf_center_lat = xtf_lat['center']
            xtf_center_lon = xtf_lon['center']

            # Haversine distance approximation for short distances
            lat_diff = abs(gps_center_lat - xtf_center_lat)
            lon_diff = abs(gps_center_lon - xtf_center_lon)

            # Convert to approximate meters (rough approximation for Korea region)
            lat_diff_m = lat_diff * 111320  # ~111,320 meters per degree latitude
            lon_diff_m = lon_diff * 111320 * np.cos(np.radians(gps_center_lat))  # Adjust for longitude

            total_distance_m = np.sqrt(lat_diff_m**2 + lon_diff_m**2)

            comparison['distance'] = {
                'center_lat_diff_degrees': lat_diff,
                'center_lon_diff_degrees': lon_diff,
                'center_lat_diff_meters': lat_diff_m,
                'center_lon_diff_meters': lon_diff_m,
                'total_distance_meters': total_distance_m,
                'total_distance_km': total_distance_m / 1000
            }

            # Calculate similarity score
            lat_overlap_score = comparison['overlap']['latitude_overlap_percent_gps'] / 100
            lon_overlap_score = comparison['overlap']['longitude_overlap_percent_gps'] / 100
            distance_score = max(0, 1 - (total_distance_m / 10000))  # Penalize distances > 10km

            comparison['similarity_score'] = (lat_overlap_score + lon_overlap_score + distance_score) / 3

        except Exception as e:
            self.logger.error(f"Coordinate comparison failed: {e}")
            comparison['error'] = str(e)

        return comparison

    def _analyze_overall_overlap(self, comparisons):
        """전체 좌표 겹침 분석"""
        if not comparisons:
            return {}

        valid_comparisons = [c for c in comparisons if 'overlap' in c and 'error' not in c]

        if not valid_comparisons:
            return {'error': 'No valid coordinate comparisons'}

        lat_overlaps = [c['overlap']['latitude_overlap_percent_gps'] for c in valid_comparisons]
        lon_overlaps = [c['overlap']['longitude_overlap_percent_gps'] for c in valid_comparisons]
        distances = [c['distance']['total_distance_km'] for c in valid_comparisons]
        similarities = [c['similarity_score'] for c in valid_comparisons]

        return {
            'latitude_overlap_stats': {
                'max': max(lat_overlaps),
                'min': min(lat_overlaps),
                'mean': sum(lat_overlaps) / len(lat_overlaps),
                'std': np.std(lat_overlaps)
            },
            'longitude_overlap_stats': {
                'max': max(lon_overlaps),
                'min': min(lon_overlaps),
                'mean': sum(lon_overlaps) / len(lon_overlaps),
                'std': np.std(lon_overlaps)
            },
            'distance_stats': {
                'max_km': max(distances),
                'min_km': min(distances),
                'mean_km': sum(distances) / len(distances),
                'std_km': np.std(distances)
            },
            'similarity_stats': {
                'max': max(similarities),
                'min': min(similarities),
                'mean': sum(similarities) / len(similarities),
                'std': np.std(similarities)
            },
            'valid_comparisons': len(valid_comparisons),
            'total_xtf_files': len(comparisons)
        }

    def _analyze_coordinate_distances(self, comparisons):
        """좌표 거리 상세 분석"""
        if not comparisons:
            return {}

        valid_comparisons = [c for c in comparisons if 'distance' in c and 'error' not in c]

        if not valid_comparisons:
            return {'error': 'No valid distance comparisons'}

        distances = [c['distance']['total_distance_km'] for c in valid_comparisons]

        # Classify distances
        classifications = []
        for dist in distances:
            if dist < 0.1:  # < 100m
                classifications.append('very_close')
            elif dist < 1.0:  # < 1km
                classifications.append('close')
            elif dist < 10.0:  # < 10km
                classifications.append('nearby')
            elif dist < 100.0:  # < 100km
                classifications.append('distant')
            else:
                classifications.append('very_distant')

        classification_counts = {cls: classifications.count(cls) for cls in set(classifications)}

        return {
            'classification_counts': classification_counts,
            'closest_distance_km': min(distances),
            'furthest_distance_km': max(distances),
            'mean_distance_km': sum(distances) / len(distances),
            'median_distance_km': np.median(distances),
            'assessment': self._assess_distances(distances)
        }

    def _assess_distances(self, distances):
        """거리 기반 평가"""
        mean_dist = sum(distances) / len(distances)
        max_dist = max(distances)

        if mean_dist < 0.5:
            return "same_location"
        elif mean_dist < 5.0:
            return "same_area"
        elif mean_dist < 50.0:
            return "same_region"
        else:
            return "different_locations"

    def _generate_conclusion(self, analysis_results):
        """종합 결론 생성"""
        overlap_analysis = analysis_results.get('overlap_analysis', {})
        distance_analysis = analysis_results.get('distance_analysis', {})

        conclusion = {
            'coordinate_match_assessment': 'unknown',
            'confidence_level': 'unknown',
            'evidence': [],
            'recommendations': []
        }

        if overlap_analysis and distance_analysis:
            mean_overlap_lat = overlap_analysis.get('latitude_overlap_stats', {}).get('mean', 0)
            mean_overlap_lon = overlap_analysis.get('longitude_overlap_stats', {}).get('mean', 0)
            mean_distance = distance_analysis.get('mean_distance_km', float('inf'))
            assessment = distance_analysis.get('assessment', 'unknown')

            # Decision logic
            if mean_overlap_lat > 50 and mean_overlap_lon > 50 and mean_distance < 1.0:
                conclusion['coordinate_match_assessment'] = 'same_location'
                conclusion['confidence_level'] = 'high'
                conclusion['evidence'].append(f"High coordinate overlap (lat: {mean_overlap_lat:.1f}%, lon: {mean_overlap_lon:.1f}%)")
                conclusion['evidence'].append(f"Small distance between centers ({mean_distance:.2f} km)")
            elif mean_overlap_lat > 20 and mean_overlap_lon > 20 and mean_distance < 5.0:
                conclusion['coordinate_match_assessment'] = 'same_area'
                conclusion['confidence_level'] = 'medium'
                conclusion['evidence'].append(f"Moderate coordinate overlap (lat: {mean_overlap_lat:.1f}%, lon: {mean_overlap_lon:.1f}%)")
                conclusion['evidence'].append(f"Moderate distance between centers ({mean_distance:.2f} km)")
            elif mean_distance < 50.0:
                conclusion['coordinate_match_assessment'] = 'same_region'
                conclusion['confidence_level'] = 'medium'
                conclusion['evidence'].append(f"Low coordinate overlap but same region ({mean_distance:.2f} km)")
            else:
                conclusion['coordinate_match_assessment'] = 'different_locations'
                conclusion['confidence_level'] = 'high'
                conclusion['evidence'].append(f"Large distance between locations ({mean_distance:.2f} km)")

            # Add recommendations
            if conclusion['coordinate_match_assessment'] == 'same_location':
                conclusion['recommendations'].append("데이터가 동일한 위치를 나타냅니다. 정밀 분석을 진행하세요.")
            elif conclusion['coordinate_match_assessment'] == 'different_locations':
                conclusion['recommendations'].append("데이터가 서로 다른 위치를 나타냅니다. 데이터 출처를 재확인하세요.")
            else:
                conclusion['recommendations'].append("추가 검증이 필요합니다. 더 정밀한 좌표 분석을 수행하세요.")

        return conclusion

    def create_visualization(self):
        """분석 결과 시각화"""
        if not self.analysis_results:
            self.logger.error("No analysis results to visualize")
            return

        self.logger.info("Creating coordinate verification visualization")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # 1. GPS 및 XTF 좌표 범위 비교
        self._plot_coordinate_ranges(ax1)

        # 2. 거리 분석
        self._plot_distance_analysis(ax2)

        # 3. 겹침 분석
        self._plot_overlap_analysis(ax3)

        # 4. 결론 요약
        self._plot_conclusion_summary(ax4)

        plt.tight_layout()

        # Save visualization
        output_file = Path("analysis_results/coordinate_verification/coordinate_verification_analysis.png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization to: {output_file}")

    def _plot_coordinate_ranges(self, ax):
        """좌표 범위 플롯"""
        if not self.gps_data:
            ax.text(0.5, 0.5, 'No GPS data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Coordinate Ranges Comparison')
            return

        gps_bounds = self.gps_data['bounds']

        # GPS bounds
        gps_lat_range = [gps_bounds['latitude']['min'], gps_bounds['latitude']['max']]
        gps_lon_range = [gps_bounds['longitude']['min'], gps_bounds['longitude']['max']]

        # Plot GPS area
        rect_gps = patches.Rectangle(
            (gps_lon_range[0], gps_lat_range[0]),
            gps_lon_range[1] - gps_lon_range[0],
            gps_lat_range[1] - gps_lat_range[0],
            linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='GPS Area'
        )
        ax.add_patch(rect_gps)

        # Plot XTF areas
        comparisons = self.analysis_results.get('coordinate_comparisons', [])
        colors = ['red', 'green', 'orange', 'purple']

        for i, comp in enumerate(comparisons):
            if 'error' not in comp and i < len(colors):
                xtf_file = comp.get('xtf_file', f'XTF_{i}')
                # Note: We would need to extract XTF coordinates from analysis_results
                # This is a placeholder for the actual XTF coordinate plotting

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('GPS vs XTF Coordinate Ranges')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_distance_analysis(self, ax):
        """거리 분석 플롯"""
        distance_analysis = self.analysis_results.get('distance_analysis', {})

        if 'classification_counts' not in distance_analysis:
            ax.text(0.5, 0.5, 'No distance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distance Analysis')
            return

        classifications = distance_analysis['classification_counts']
        labels = list(classifications.keys())
        values = list(classifications.values())

        bars = ax.bar(labels, values, color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(labels)])

        ax.set_xlabel('Distance Classification')
        ax.set_ylabel('Number of XTF Files')
        ax.set_title('Distance Between GPS and XTF Centers')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value}', ha='center', va='bottom')

    def _plot_overlap_analysis(self, ax):
        """겹침 분석 플롯"""
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})

        if not overlap_analysis or 'latitude_overlap_stats' not in overlap_analysis:
            ax.text(0.5, 0.5, 'No overlap data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overlap Analysis')
            return

        lat_stats = overlap_analysis['latitude_overlap_stats']
        lon_stats = overlap_analysis['longitude_overlap_stats']

        categories = ['Latitude Overlap', 'Longitude Overlap']
        means = [lat_stats['mean'], lon_stats['mean']]
        stds = [lat_stats['std'], lon_stats['std']]

        bars = ax.bar(categories, means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral'], alpha=0.7)

        ax.set_ylabel('Overlap Percentage (%)')
        ax.set_title('Coordinate Overlap Analysis')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean:.1f}%', ha='center', va='bottom')

    def _plot_conclusion_summary(self, ax):
        """결론 요약 플롯"""
        ax.axis('off')

        conclusion = self.analysis_results.get('conclusion', {})

        if not conclusion:
            ax.text(0.5, 0.5, 'No conclusion available', ha='center', va='center', transform=ax.transAxes)
            return

        summary_text = []
        summary_text.append("📊 좌표 검증 분석 결과")
        summary_text.append("")
        summary_text.append(f"🎯 평가 결과: {conclusion.get('coordinate_match_assessment', 'Unknown')}")
        summary_text.append(f"🔍 신뢰도: {conclusion.get('confidence_level', 'Unknown')}")
        summary_text.append("")
        summary_text.append("📋 주요 증거:")

        for evidence in conclusion.get('evidence', []):
            summary_text.append(f"  • {evidence}")

        summary_text.append("")
        summary_text.append("💡 권장사항:")

        for rec in conclusion.get('recommendations', []):
            summary_text.append(f"  • {rec}")

        ax.text(0.1, 0.9, '\n'.join(summary_text), transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    def generate_report(self):
        """분석 보고서 생성"""
        if not self.analysis_results:
            self.logger.error("No analysis results to report")
            return

        self.logger.info("Generating coordinate verification report")

        report_lines = []
        report_lines.append("# 좌표 검증 분석 보고서")
        report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**분석자**: YMARX")
        report_lines.append("")

        # 분석 목적
        report_lines.append("## 🎯 **분석 목적**")
        report_lines.append("original XTF 파일들과 Location_MDGPS.xlsx의 좌표 범위를 정밀 비교하여")
        report_lines.append("기존 분석에서 '다른 위치'로 나온 원인을 규명하고 정확한 일치성을 판단")
        report_lines.append("")

        # GPS 데이터 개요
        if self.gps_data:
            report_lines.append("## 📍 **GPS 데이터 개요**")
            report_lines.append(f"- **총 기뢰 위치**: {self.gps_data['count']}개")
            report_lines.append(f"- **유효 좌표**: {self.gps_data['validation']['valid_count']}/{self.gps_data['validation']['total_count']}")

            bounds = self.gps_data['bounds']
            report_lines.append(f"- **위도 범위**: [{bounds['latitude']['min']:.6f}°, {bounds['latitude']['max']:.6f}°]")
            report_lines.append(f"- **경도 범위**: [{bounds['longitude']['min']:.6f}°, {bounds['longitude']['max']:.6f}°]")
            report_lines.append(f"- **중심 좌표**: ({bounds['latitude']['center']:.6f}°, {bounds['longitude']['center']:.6f}°)")
            report_lines.append("")

        # XTF 파일 분석
        report_lines.append("## 🔍 **Original XTF 파일 분석**")
        xtf_analyses = self.analysis_results.get('xtf_analyses', [])

        if xtf_analyses:
            report_lines.append(f"- **분석된 XTF 파일**: {len(xtf_analyses)}개")

            for i, xtf_analysis in enumerate(xtf_analyses):
                report_lines.append(f"### {i+1}. {xtf_analysis['file_name']}")
                report_lines.append(f"- **파일 크기**: {xtf_analysis['file_size_mb']:.1f} MB")
                report_lines.append(f"- **추출 방법**: {xtf_analysis.get('extraction_method', 'Unknown')}")

                if xtf_analysis.get('coordinates'):
                    coords = xtf_analysis['coordinates']
                    report_lines.append(f"- **위도 범위**: [{coords['latitude']['min']:.6f}°, {coords['latitude']['max']:.6f}°]")
                    report_lines.append(f"- **경도 범위**: [{coords['longitude']['min']:.6f}°, {coords['longitude']['max']:.6f}°]")
                    report_lines.append(f"- **내비게이션 포인트**: {coords['latitude']['count']}개")

                if xtf_analysis.get('inferred_info'):
                    info = xtf_analysis['inferred_info']
                    report_lines.append(f"- **시스템**: {info.get('system_type', 'Unknown')} {info.get('model', '')}")

                report_lines.append("")

        # 좌표 비교 분석
        report_lines.append("## 📊 **좌표 비교 분석**")

        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        if overlap_analysis:
            report_lines.append("### 겹침 분석")
            lat_stats = overlap_analysis.get('latitude_overlap_stats', {})
            lon_stats = overlap_analysis.get('longitude_overlap_stats', {})

            report_lines.append(f"- **위도 겹침**: 평균 {lat_stats.get('mean', 0):.1f}% (최대 {lat_stats.get('max', 0):.1f}%)")
            report_lines.append(f"- **경도 겹침**: 평균 {lon_stats.get('mean', 0):.1f}% (최대 {lon_stats.get('max', 0):.1f}%)")
            report_lines.append("")

        distance_analysis = self.analysis_results.get('distance_analysis', {})
        if distance_analysis:
            report_lines.append("### 거리 분석")
            report_lines.append(f"- **평균 거리**: {distance_analysis.get('mean_distance_km', 0):.2f} km")
            report_lines.append(f"- **최소 거리**: {distance_analysis.get('closest_distance_km', 0):.2f} km")
            report_lines.append(f"- **최대 거리**: {distance_analysis.get('furthest_distance_km', 0):.2f} km")
            report_lines.append(f"- **거리 평가**: {distance_analysis.get('assessment', 'Unknown')}")

            if 'classification_counts' in distance_analysis:
                report_lines.append("- **거리 분류**:")
                for cls, count in distance_analysis['classification_counts'].items():
                    report_lines.append(f"  - {cls}: {count}개 파일")
            report_lines.append("")

        # 최종 결론
        conclusion = self.analysis_results.get('conclusion', {})
        if conclusion:
            report_lines.append("## 🎯 **최종 결론**")

            assessment = conclusion.get('coordinate_match_assessment', 'unknown')
            confidence = conclusion.get('confidence_level', 'unknown')

            if assessment == 'same_location':
                report_lines.append("### ✅ **동일한 위치를 나타냅니다**")
            elif assessment == 'same_area':
                report_lines.append("### ⚠️ **동일한 지역을 나타냅니다**")
            elif assessment == 'same_region':
                report_lines.append("### 🔍 **동일한 지역권을 나타냅니다**")
            else:
                report_lines.append("### ❌ **서로 다른 위치를 나타냅니다**")

            report_lines.append(f"**신뢰도**: {confidence}")
            report_lines.append("")

            if conclusion.get('evidence'):
                report_lines.append("**주요 증거**:")
                for evidence in conclusion['evidence']:
                    report_lines.append(f"- {evidence}")
                report_lines.append("")

            if conclusion.get('recommendations'):
                report_lines.append("**권장사항**:")
                for rec in conclusion['recommendations']:
                    report_lines.append(f"- {rec}")
                report_lines.append("")

        # 기술적 세부사항
        report_lines.append("## 🛠️ **기술적 세부사항**")
        report_lines.append("")
        report_lines.append("**좌표 추출 방법**:")
        report_lines.append("- pyxtf 라이브러리를 통한 XTF 패킷 분석")
        report_lines.append("- 바이너리 헤더 분석 (fallback)")
        report_lines.append("- 파일명 기반 정보 추출")
        report_lines.append("")
        report_lines.append("**좌표 변환**:")
        report_lines.append("- XTF 좌표 형식에서 십진도 변환")
        report_lines.append("- 한국 지역 유효성 검증 (33-43°N, 124-132°E)")
        report_lines.append("")
        report_lines.append("**비교 분석**:")
        report_lines.append("- 좌표 범위 겹침 계산")
        report_lines.append("- 중심점 간 거리 계산 (Haversine 근사)")
        report_lines.append("- 종합 유사도 점수 산출")

        # 보고서 저장
        output_file = Path("analysis_results/coordinate_verification/COORDINATE_VERIFICATION_REPORT.md")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        self.logger.info(f"Saved coordinate verification report to: {output_file}")

        # JSON 데이터 저장
        json_file = Path("analysis_results/coordinate_verification/coordinate_verification_data.json")
        with open(json_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        self.logger.info(f"Saved coordinate verification data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Coordinate Verification Analysis")

    try:
        # Initialize analyzer
        analyzer = CoordinateVerificationAnalyzer()

        # Find original XTF files
        xtf_files = analyzer.find_original_xtf_files()
        if not xtf_files:
            logger.error("No original XTF files found")
            return 1

        # Load GPS data
        gps_data = analyzer.load_gps_data()
        if not gps_data:
            logger.error("Failed to load GPS data")
            return 1

        # Perform coordinate analysis
        results = analyzer.analyze_coordinate_correspondence()
        if not results:
            logger.error("Failed to analyze coordinate correspondence")
            return 1

        # Create visualization
        analyzer.create_visualization()

        # Generate report
        analyzer.generate_report()

        # Print summary
        print("\n" + "="*80)
        print("좌표 검증 분석 결과")
        print("="*80)

        conclusion = results.get('conclusion', {})
        print(f"📍 GPS 데이터: {gps_data['count']}개 기뢰 위치")
        print(f"📁 Original XTF 파일: {len(xtf_files)}개")

        if conclusion:
            print(f"\n🎯 **최종 평가**: {conclusion.get('coordinate_match_assessment', 'Unknown')}")
            print(f"🔍 **신뢰도**: {conclusion.get('confidence_level', 'Unknown')}")

            if conclusion.get('evidence'):
                print(f"\n📋 **주요 증거**:")
                for evidence in conclusion['evidence']:
                    print(f"   • {evidence}")

        overlap_analysis = results.get('overlap_analysis', {})
        if overlap_analysis and 'latitude_overlap_stats' in overlap_analysis:
            lat_overlap = overlap_analysis['latitude_overlap_stats']['mean']
            lon_overlap = overlap_analysis['longitude_overlap_stats']['mean']
            print(f"\n📊 **좌표 겹침**: 위도 {lat_overlap:.1f}%, 경도 {lon_overlap:.1f}%")

        distance_analysis = results.get('distance_analysis', {})
        if distance_analysis:
            mean_dist = distance_analysis.get('mean_distance_km', 0)
            assessment = distance_analysis.get('assessment', 'unknown')
            print(f"📏 **평균 거리**: {mean_dist:.2f} km ({assessment})")

        return 0

    except Exception as e:
        logger.error(f"Coordinate verification analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())