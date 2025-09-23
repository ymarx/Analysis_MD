#!/usr/bin/env python3
"""
Simple Coordinate Check using Working Pipeline
==============================================
기존 작동하는 파이프라인을 사용하여 XTF 좌표 추출 및 분석

Author: YMARX
Date: 2024-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.xtf_reader import XTFReader
from pipeline.modules.gps_parser import GPSParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_coordinates_with_working_reader(xtf_file_path):
    """작동하는 XTF 리더로 좌표 추출"""
    try:
        logger.info(f"Reading XTF file: {xtf_file_path}")

        reader = XTFReader(str(xtf_file_path))
        reader.load_file()
        reader.parse_pings(max_pings=500)  # 500개 ping으로 제한

        # Get ping data
        if not hasattr(reader, 'ping_data') or not reader.ping_data:
            logger.warning("No ping data available")
            return []

        coordinates = []
        for i, ping in enumerate(reader.ping_data):
            # Try to extract coordinates from ping data
            lat, lon = None, None

            # Method 1: Direct attributes
            if hasattr(ping, 'SensorPrimaryLatitude') and hasattr(ping, 'SensorPrimaryLongitude'):
                lat = ping.SensorPrimaryLatitude
                lon = ping.SensorPrimaryLongitude
            elif hasattr(ping, 'NavLatitude') and hasattr(ping, 'NavLongitude'):
                lat = ping.NavLatitude
                lon = ping.NavLongitude
            elif hasattr(ping, 'Latitude') and hasattr(ping, 'Longitude'):
                lat = ping.Latitude
                lon = ping.Longitude

            # Method 2: Check SensorCoordinate if exists
            if (not lat or not lon) and hasattr(ping, 'SensorCoordinate'):
                coord = ping.SensorCoordinate
                if hasattr(coord, 'Latitude') and hasattr(coord, 'Longitude'):
                    lat = coord.Latitude
                    lon = coord.Longitude

            # Add valid coordinates
            if lat and lon and lat != 0 and lon != 0:
                coordinates.append({
                    'ping': i,
                    'latitude': float(lat),
                    'longitude': float(lon)
                })

        logger.info(f"Extracted {len(coordinates)} coordinate points from {len(reader.ping_data)} pings")
        return coordinates

    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        return []


def simple_coordinate_analysis():
    """간단한 좌표 분석"""
    logger.info("="*60)
    logger.info("SIMPLE COORDINATE ANALYSIS")
    logger.info("="*60)

    # File paths
    xtf_files = [
        Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf"),
        Path("datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf")
    ]

    mine_locations_file = Path("datasets/Location_MDGPS.xlsx")

    # Parse mine locations
    gps_parser = GPSParser()
    mine_locations = gps_parser.parse_gps_file(mine_locations_file)

    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    logger.info(f"Mine locations: {len(mine_locations)} points")
    logger.info(f"Mine bounds: Lat[{mine_lat_min:.6f}, {mine_lat_max:.6f}], Lon[{mine_lon_min:.6f}, {mine_lon_max:.6f}]")

    # Analyze each XTF file
    results = {}

    for xtf_file in xtf_files:
        if not xtf_file.exists():
            logger.warning(f"File not found: {xtf_file}")
            continue

        logger.info(f"\nAnalyzing: {xtf_file.name}")

        coordinates = extract_coordinates_with_working_reader(xtf_file)

        if not coordinates:
            logger.warning(f"No coordinates extracted from {xtf_file.name}")
            results[xtf_file.name] = {
                'success': False,
                'message': 'No coordinates extracted'
            }
            continue

        # Calculate bounds
        lats = [coord['latitude'] for coord in coordinates]
        lons = [coord['longitude'] for coord in coordinates]

        xtf_lat_min, xtf_lat_max = min(lats), max(lats)
        xtf_lon_min, xtf_lon_max = min(lons), max(lons)

        # Calculate distances and coverage
        xtf_center_lat = (xtf_lat_min + xtf_lat_max) / 2
        xtf_center_lon = (xtf_lon_min + xtf_lon_max) / 2
        mine_center_lat = (mine_lat_min + mine_lat_max) / 2
        mine_center_lon = (mine_lon_min + mine_lon_max) / 2

        # Haversine distance
        lat1, lon1 = np.radians(xtf_center_lat), np.radians(xtf_center_lon)
        lat2, lon2 = np.radians(mine_center_lat), np.radians(mine_center_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))

        # Check overlap
        overlap_lat = max(0, min(xtf_lat_max, mine_lat_max) - max(xtf_lat_min, mine_lat_min))
        overlap_lon = max(0, min(xtf_lon_max, mine_lon_max) - max(xtf_lon_min, mine_lon_min))
        has_overlap = overlap_lat > 0 and overlap_lon > 0

        # Coverage area
        lat_coverage_km = (xtf_lat_max - xtf_lat_min) * 111
        lon_coverage_km = (xtf_lon_max - xtf_lon_min) * 111 * np.cos(np.radians(xtf_lat_min))

        results[xtf_file.name] = {
            'success': True,
            'coordinate_count': len(coordinates),
            'xtf_bounds': {
                'lat_min': xtf_lat_min,
                'lat_max': xtf_lat_max,
                'lon_min': xtf_lon_min,
                'lon_max': xtf_lon_max
            },
            'center': [xtf_center_lat, xtf_center_lon],
            'coverage_km': [lat_coverage_km, lon_coverage_km],
            'distance_to_mines_km': distance_km,
            'has_overlap': has_overlap,
            'sample_coordinates': coordinates[:3]
        }

        logger.info(f"  Coordinates: {len(coordinates)} points")
        logger.info(f"  Bounds: Lat[{xtf_lat_min:.6f}, {xtf_lat_max:.6f}], Lon[{xtf_lon_min:.6f}, {xtf_lon_max:.6f}]")
        logger.info(f"  Coverage: {lat_coverage_km:.1f} x {lon_coverage_km:.1f} km")
        logger.info(f"  Distance to mines: {distance_km:.1f} km")
        logger.info(f"  Overlap: {'YES' if has_overlap else 'NO'}")

    # Create visualization
    create_simple_visualization(results, mine_locations)

    # Generate report
    generate_simple_report(results, mine_locations)

    return results


def create_simple_visualization(results, mine_locations):
    """간단한 시각화 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Mine locations
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]

    ax.scatter(mine_lons, mine_lats, c='red', s=100, marker='x',
               label=f'Mine Locations ({len(mine_locations)})', alpha=0.8, linewidth=3)

    # XTF areas
    colors = ['blue', 'green', 'purple', 'orange']

    for i, (filename, data) in enumerate(results.items()):
        if not data.get('success', False):
            continue

        bounds = data['xtf_bounds']

        # Draw rectangle
        rect = patches.Rectangle(
            (bounds['lon_min'], bounds['lat_min']),
            bounds['lon_max'] - bounds['lon_min'],
            bounds['lat_max'] - bounds['lat_min'],
            linewidth=3,
            edgecolor=colors[i % len(colors)],
            facecolor=colors[i % len(colors)],
            alpha=0.3,
            label=f'XTF: {filename[:25]}...'
        )
        ax.add_patch(rect)

        # Mark center with distance
        center_lat, center_lon = data['center']
        ax.plot(center_lon, center_lat, 'o', color=colors[i % len(colors)],
                markersize=10, markeredgewidth=2, markeredgecolor='white')

        # Distance annotation
        ax.annotate(f'{data["distance_to_mines_km"]:.1f} km',
                   (center_lon, center_lat),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=10)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('XTF Coverage vs Mine Locations - Geographic Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path("analysis_results/simple_coordinate_check/coordinate_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to: {output_file}")


def generate_simple_report(results, mine_locations):
    """간단한 보고서 생성"""

    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    report_lines = []
    report_lines.append("# XTF 파일 좌표 분석 보고서")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Mine summary
    report_lines.append("## 기뢰 위치 데이터")
    report_lines.append(f"- 총 {len(mine_locations)}개 기뢰 위치")
    report_lines.append(f"- 위도: [{mine_lat_min:.6f}°, {mine_lat_max:.6f}°]")
    report_lines.append(f"- 경도: [{mine_lon_min:.6f}°, {mine_lon_max:.6f}°]")
    report_lines.append("")

    # XTF analysis
    report_lines.append("## XTF 파일 분석 결과")
    report_lines.append("")

    successful_analyses = 0
    overlapping_files = []

    for filename, data in results.items():
        if not data.get('success', False):
            report_lines.append(f"### ❌ {filename}")
            report_lines.append(f"- 상태: {data.get('message', '분석 실패')}")
            report_lines.append("")
            continue

        successful_analyses += 1

        bounds = data['xtf_bounds']
        coverage = data['coverage_km']
        distance = data['distance_to_mines_km']
        coord_count = data['coordinate_count']
        has_overlap = data['has_overlap']

        if has_overlap:
            overlapping_files.append(filename)

        status = "✅" if has_overlap else "❌"
        report_lines.append(f"### {status} {filename}")
        report_lines.append(f"- 좌표 포인트: {coord_count:,}개")
        report_lines.append(f"- 위도 범위: [{bounds['lat_min']:.6f}°, {bounds['lat_max']:.6f}°]")
        report_lines.append(f"- 경도 범위: [{bounds['lon_min']:.6f}°, {bounds['lon_max']:.6f}°]")
        report_lines.append(f"- 조사 영역: {coverage[0]:.1f} × {coverage[1]:.1f} km")
        report_lines.append(f"- 기뢰와의 거리: {distance:.1f} km")
        report_lines.append(f"- 지리적 중복: {'있음' if has_overlap else '없음'}")

        if 'sample_coordinates' in data:
            report_lines.append("- 샘플 좌표:")
            for coord in data['sample_coordinates']:
                report_lines.append(f"  - ({coord['latitude']:.6f}°, {coord['longitude']:.6f}°)")

        report_lines.append("")

    # Conclusion
    report_lines.append("## 분석 결론")
    report_lines.append("")

    if overlapping_files:
        report_lines.append(f"### ✅ 매핑 가능 ({len(overlapping_files)}개 파일)")
        for filename in overlapping_files:
            report_lines.append(f"- {filename}")
        report_lines.append("")
        report_lines.append("**결론**: 지리적 중복이 있는 파일에서 좌표 매핑이 가능합니다!")
        report_lines.append("이제 180도 회전/좌우 반전 변환을 적용하여 정확한 매핑을 수행할 수 있습니다.")
    else:
        report_lines.append("### ❌ 지리적 분리 확인")
        report_lines.append(f"분석된 {successful_analyses}개 파일 모두 기뢰 위치와 지리적으로 분리되어 있습니다.")
        report_lines.append("")
        report_lines.append("**결론**: 이전의 \"180도 회전/좌우 반전으로 문제 해결\" 주장은 잘못된 분석이었습니다.")
        report_lines.append("좌표 변환으로는 해결할 수 없는 근본적인 데이터 불일치 문제입니다.")
        report_lines.append("")
        report_lines.append("**해결 방안**:")
        report_lines.append("1. 기뢰 위치와 동일한 지역의 XTF 파일 확보")
        report_lines.append("2. 동일한 해양 조사에서 수집된 데이터 확인")
        report_lines.append("3. 좌표계 정보 재검토")

    # Save report
    output_file = Path("analysis_results/simple_coordinate_check/analysis_report.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved report to: {output_file}")


def main():
    """메인 실행"""
    logger.info("Starting Simple Coordinate Analysis")

    try:
        results = simple_coordinate_analysis()

        # Summary
        print("\n" + "="*60)
        print("XTF 좌표 분석 결과 요약")
        print("="*60)

        successful = 0
        overlapping = 0

        for filename, data in results.items():
            if data.get('success', False):
                successful += 1
                distance = data['distance_to_mines_km']
                has_overlap = data['has_overlap']
                coord_count = data['coordinate_count']

                if has_overlap:
                    overlapping += 1

                status = "✅ 매핑가능" if has_overlap else "❌ 분리됨"
                print(f"{status} {filename[:40]:40s}: {distance:6.1f}km, {coord_count:,}개 좌표")
            else:
                print(f"❌ 실패      {filename[:40]:40s}: {data.get('message', '분석 실패')}")

        print(f"\n📊 분석 요약:")
        print(f"   성공적 분석: {successful}개 파일")
        print(f"   매핑 가능: {overlapping}개 파일")
        print(f"   지리적 분리: {successful - overlapping}개 파일")

        if overlapping > 0:
            print(f"\n✅ 결론: 매핑 가능한 XTF 파일 발견!")
            print("   이제 좌표 변환을 적용하여 정확한 매핑을 수행할 수 있습니다.")
        else:
            print(f"\n❌ 결론: 모든 XTF 파일이 기뢰 위치와 지리적으로 분리됨")
            print("   이전 매핑 해결 주장은 잘못된 분석이었습니다.")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())