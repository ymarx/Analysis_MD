#!/usr/bin/env python3
"""
Direct XTF Coordinate Check
===========================
XTF 파일에서 직접 좌표 데이터를 추출하여 기뢰 위치와 비교

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
import json

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


def extract_coordinates_from_xtf(xtf_file_path):
    """XTF 파일에서 직접 좌표 추출"""
    try:
        # Import pyxtf and use basic reader
        import pyxtf

        logger.info(f"Opening XTF file: {xtf_file_path}")

        # Open and read XTF file
        with pyxtf.xtf_open(str(xtf_file_path)) as xtf_file:
            # Get basic navigation data
            pings = []
            coordinates = []

            ping_count = 0
            for packet in xtf_file:
                if hasattr(packet, 'data') and hasattr(packet.data, 'SensorCoordinate'):
                    coord = packet.data.SensorCoordinate
                    if coord.Latitude != 0 and coord.Longitude != 0:
                        coordinates.append({
                            'ping': ping_count,
                            'latitude': coord.Latitude,
                            'longitude': coord.Longitude
                        })
                    ping_count += 1

                    # Limit for analysis
                    if ping_count >= 1000:
                        break

            if coordinates:
                logger.info(f"Extracted {len(coordinates)} coordinate points")
                return coordinates
            else:
                logger.warning("No coordinates found with SensorCoordinate method")

        # Alternative method - try direct packet parsing
        logger.info("Trying alternative coordinate extraction method...")

        with pyxtf.xtf_open(str(xtf_file_path)) as xtf_file:
            coordinates = []
            ping_count = 0

            for packet in xtf_file:
                # Try to get coordinates from different packet types
                if hasattr(packet, 'data'):
                    data = packet.data

                    # Check for navigation fields
                    lat, lon = None, None

                    if hasattr(data, 'SensorPrimaryNavigationLatitude'):
                        lat = data.SensorPrimaryNavigationLatitude
                        lon = data.SensorPrimaryNavigationLongitude
                    elif hasattr(data, 'Latitude'):
                        lat = data.Latitude
                        lon = data.Longitude
                    elif hasattr(data, 'NavLatitude'):
                        lat = data.NavLatitude
                        lon = data.NavLongitude

                    if lat and lon and lat != 0 and lon != 0:
                        coordinates.append({
                            'ping': ping_count,
                            'latitude': lat,
                            'longitude': lon
                        })

                ping_count += 1
                if ping_count >= 1000:
                    break

            logger.info(f"Alternative method extracted {len(coordinates)} coordinate points")
            return coordinates

    except Exception as e:
        logger.error(f"Failed to extract coordinates from {xtf_file_path}: {e}")
        return []


def check_all_xtf_coordinates():
    """모든 XTF 파일의 좌표 범위 확인"""
    logger.info("="*60)
    logger.info("DIRECT XTF COORDINATE EXTRACTION")
    logger.info("="*60)

    # XTF files to check
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

    logger.info(f"Mine location bounds:")
    logger.info(f"  Latitude: [{mine_lat_min:.6f}, {mine_lat_max:.6f}]")
    logger.info(f"  Longitude: [{mine_lon_min:.6f}, {mine_lon_max:.6f}]")

    # Analyze each XTF file
    analysis_results = {}

    for xtf_file in xtf_files:
        if not xtf_file.exists():
            logger.warning(f"XTF file not found: {xtf_file}")
            continue

        logger.info(f"\nAnalyzing: {xtf_file.name}")

        coordinates = extract_coordinates_from_xtf(xtf_file)

        if not coordinates:
            logger.warning(f"No coordinates extracted from {xtf_file.name}")
            continue

        # Calculate bounds
        lats = [coord['latitude'] for coord in coordinates]
        lons = [coord['longitude'] for coord in coordinates]

        xtf_lat_min, xtf_lat_max = min(lats), max(lats)
        xtf_lon_min, xtf_lon_max = min(lons), max(lons)

        # Calculate center
        xtf_center_lat = (xtf_lat_min + xtf_lat_max) / 2
        xtf_center_lon = (xtf_lon_min + xtf_lon_max) / 2
        mine_center_lat = (mine_lat_min + mine_lat_max) / 2
        mine_center_lon = (mine_lon_min + mine_lon_max) / 2

        # Distance calculation
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

        analysis_results[xtf_file.name] = {
            'file_path': str(xtf_file),
            'coordinate_count': len(coordinates),
            'lat_range': [xtf_lat_min, xtf_lat_max],
            'lon_range': [xtf_lon_min, xtf_lon_max],
            'center': [xtf_center_lat, xtf_center_lon],
            'distance_to_mines_km': distance_km,
            'has_overlap': has_overlap,
            'sample_coordinates': coordinates[:5]  # First 5 for verification
        }

        logger.info(f"  Coordinates extracted: {len(coordinates)}")
        logger.info(f"  Latitude range: [{xtf_lat_min:.6f}, {xtf_lat_max:.6f}]")
        logger.info(f"  Longitude range: [{xtf_lon_min:.6f}, {xtf_lon_max:.6f}]")
        logger.info(f"  Distance to mines: {distance_km:.1f} km")
        logger.info(f"  Geographic overlap: {'YES' if has_overlap else 'NO'}")

    # Create visualization
    create_coordinate_comparison_plot(analysis_results, mine_locations)

    # Generate report
    generate_coordinate_analysis_report(analysis_results, mine_locations)

    return analysis_results


def create_coordinate_comparison_plot(analysis_results, mine_locations):
    """좌표 비교 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Extract mine coordinates
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]

    # Plot 1: Overall geographic view
    ax1.scatter(mine_lons, mine_lats, c='red', s=100, marker='x',
               label='Mine Locations (25)', alpha=0.8, linewidth=3)

    colors = ['blue', 'green', 'purple', 'orange']
    for i, (filename, data) in enumerate(analysis_results.items()):
        if 'lat_range' not in data:
            continue

        lat_min, lat_max = data['lat_range']
        lon_min, lon_max = data['lon_range']

        # Draw rectangle for XTF coverage
        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=3,
            edgecolor=colors[i % len(colors)],
            facecolor=colors[i % len(colors)],
            alpha=0.3,
            label=f'XTF: {filename[:30]}...'
        )
        ax1.add_patch(rect)

        # Mark center
        center_lat, center_lon = data['center']
        ax1.plot(center_lon, center_lat, 'o', color=colors[i % len(colors)],
                markersize=10, markeredgewidth=2, markeredgecolor='white')

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('XTF vs Mine Locations - Geographic Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add distance annotations
    for i, (filename, data) in enumerate(analysis_results.items()):
        if 'center' not in data:
            continue
        center_lat, center_lon = data['center']
        ax1.annotate(f'{data["distance_to_mines_km"]:.1f} km',
                    (center_lon, center_lat),
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=9)

    # Plot 2: Zoomed view around mine locations
    mine_lat_center = np.mean(mine_lats)
    mine_lon_center = np.mean(mine_lons)

    # Calculate appropriate buffer
    lat_range = max(mine_lats) - min(mine_lats)
    lon_range = max(mine_lons) - min(mine_lons)
    buffer = max(lat_range, lon_range, 0.001) * 2  # At least 200m buffer

    ax2.scatter(mine_lons, mine_lats, c='red', s=150, marker='x',
               label='Mine Locations', alpha=0.8, linewidth=3)

    # Add mine location numbers
    for i, (lat, lon) in enumerate(zip(mine_lats, mine_lons)):
        ax2.annotate(f'{i+1}', (lon, lat), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # Show XTF areas if they're reasonably close
    for i, (filename, data) in enumerate(analysis_results.items()):
        if 'lat_range' not in data:
            continue

        # Only show if within 10km for visibility
        if data['distance_to_mines_km'] < 10:
            lat_min, lat_max = data['lat_range']
            lon_min, lon_max = data['lon_range']

            rect = patches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=3,
                edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)],
                alpha=0.4,
                label=f'XTF: {filename[:20]}...'
            )
            ax2.add_patch(rect)

    ax2.set_xlim(mine_lon_center - buffer, mine_lon_center + buffer)
    ax2.set_ylim(mine_lat_center - buffer, mine_lat_center + buffer)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Mine Location Area - Detailed View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path("analysis_results/direct_coordinate_check/coordinate_comparison.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved coordinate comparison plot to: {output_file}")


def generate_coordinate_analysis_report(analysis_results, mine_locations):
    """좌표 분석 보고서 생성"""

    report_lines = []
    report_lines.append("# XTF 좌표 추출 및 매핑 가능성 분석 보고서")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Mine location summary
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    report_lines.append("## 🎯 기뢰 위치 데이터 (Location_MDGPS.xlsx)")
    report_lines.append(f"- **총 기뢰 개수**: {len(mine_locations)}개")
    report_lines.append(f"- **위도 범위**: [{mine_lat_min:.6f}°, {mine_lat_max:.6f}°] (범위: {(mine_lat_max-mine_lat_min)*111:.1f}m)")
    report_lines.append(f"- **경도 범위**: [{mine_lon_min:.6f}°, {mine_lon_max:.6f}°] (범위: {(mine_lon_max-mine_lon_min)*111*np.cos(np.radians(mine_lat_min)):.1f}m)")
    report_lines.append("")

    # XTF analysis
    report_lines.append("## 📡 XTF 파일별 좌표 범위 분석")
    report_lines.append("")

    has_overlap = False
    closest_file = None
    closest_distance = float('inf')

    for filename, data in analysis_results.items():
        if 'lat_range' not in data:
            report_lines.append(f"### ❌ {filename}")
            report_lines.append("- **상태**: 좌표 추출 실패")
            report_lines.append("")
            continue

        lat_min, lat_max = data['lat_range']
        lon_min, lon_max = data['lon_range']
        distance_km = data['distance_to_mines_km']
        coordinate_count = data['coordinate_count']
        overlap = data['has_overlap']

        if overlap:
            has_overlap = True

        if distance_km < closest_distance:
            closest_distance = distance_km
            closest_file = filename

        # Coverage calculations
        lat_coverage_km = (lat_max - lat_min) * 111
        lon_coverage_km = (lon_max - lon_min) * 111 * np.cos(np.radians(lat_min))

        status_icon = "✅" if overlap else "❌"
        report_lines.append(f"### {status_icon} {filename}")
        report_lines.append(f"- **좌표 포인트**: {coordinate_count:,}개")
        report_lines.append(f"- **위도 범위**: [{lat_min:.6f}°, {lat_max:.6f}°]")
        report_lines.append(f"- **경도 범위**: [{lon_min:.6f}°, {lon_max:.6f}°]")
        report_lines.append(f"- **조사 영역**: {lat_coverage_km:.1f} × {lon_coverage_km:.1f} km")
        report_lines.append(f"- **기뢰 위치와 거리**: {distance_km:.1f} km")
        report_lines.append(f"- **지리적 중복**: {'✅ 있음' if overlap else '❌ 없음'}")

        # Sample coordinates for verification
        if 'sample_coordinates' in data and data['sample_coordinates']:
            report_lines.append("- **샘플 좌표** (처음 3개):")
            for i, coord in enumerate(data['sample_coordinates'][:3]):
                report_lines.append(f"  - Ping {coord['ping']}: ({coord['latitude']:.6f}°, {coord['longitude']:.6f}°)")

        report_lines.append("")

    # Analysis conclusion
    report_lines.append("## 🔍 분석 결과 및 결론")
    report_lines.append("")

    if has_overlap:
        overlapping_files = [name for name, data in analysis_results.items()
                           if data.get('has_overlap', False)]
        report_lines.append("### ✅ **매핑 가능한 파일 발견!**")
        report_lines.append(f"**지리적 중복이 있는 파일**: {', '.join(overlapping_files)}")
        report_lines.append("")
        report_lines.append("**다음 단계**:")
        report_lines.append("1. 🔄 중복 영역이 있는 XTF 파일로 좌표 매핑 재시도")
        report_lines.append("2. 🎯 180도 회전/좌우 반전 변환 적용")
        report_lines.append("3. ✅ 기뢰 위치 매핑 정확도 검증")
        report_lines.append("4. 🚀 성공적인 매핑으로 파이프라인 진행")

    else:
        report_lines.append("### ❌ **지리적 분리 확인됨**")
        report_lines.append(f"**가장 가까운 파일**: {closest_file} (거리: {closest_distance:.1f} km)")
        report_lines.append("")
        report_lines.append("**문제점**:")
        report_lines.append("- 모든 XTF 파일이 기뢰 위치와 지리적으로 분리됨")
        report_lines.append("- 이전 \"180도 회전/좌우 반전 해결\" 주장은 잘못된 분석이었음")
        report_lines.append("- 좌표 변환으로는 해결할 수 없는 근본적인 데이터 불일치")
        report_lines.append("")
        report_lines.append("**권장사항**:")
        report_lines.append("1. 🔍 기뢰 위치와 동일한 지역의 XTF 데이터 확보")
        report_lines.append("2. 📍 기뢰 위치 좌표의 정확성 재검증")
        report_lines.append("3. 🗺️ 동일한 해양 조사에서 수집된 데이터인지 확인")
        report_lines.append("4. 🎯 다른 Pohang 지역 XTF 파일 탐색")

    # Technical details
    report_lines.append("")
    report_lines.append("## 📋 기술적 세부사항")
    report_lines.append("")
    report_lines.append("**좌표 추출 방법**:")
    report_lines.append("- pyxtf 라이브러리를 사용한 직접 패킷 파싱")
    report_lines.append("- SensorCoordinate, NavLatitude 등 다중 필드 확인")
    report_lines.append("- 0이 아닌 유효 좌표만 추출")
    report_lines.append("")
    report_lines.append("**거리 계산**:")
    report_lines.append("- Haversine 공식을 사용한 구면 거리 계산")
    report_lines.append("- 각 파일의 중심점과 기뢰 위치 중심점 간 거리")
    report_lines.append("")
    report_lines.append("**지리적 중복 판정**:")
    report_lines.append("- XTF 영역과 기뢰 위치 영역의 경계 상자 중복 검사")
    report_lines.append("- 위도/경도 모두에서 중복이 있어야 매핑 가능으로 판정")

    # Save report
    output_file = Path("analysis_results/direct_coordinate_check/coordinate_analysis_report.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved coordinate analysis report to: {output_file}")

    # Save detailed data
    json_file = Path("analysis_results/direct_coordinate_check/coordinate_data.json")
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    logger.info(f"Saved coordinate data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Direct XTF Coordinate Check")

    try:
        results = check_all_xtf_coordinates()

        # Print summary
        print("\n" + "="*60)
        print("XTF 좌표 추출 및 매핑 가능성 분석 결과")
        print("="*60)

        overlapping_files = 0
        total_files = 0

        for filename, data in results.items():
            if 'distance_to_mines_km' not in data:
                print(f"❌ {filename[:40]:40s}: 좌표 추출 실패")
                continue

            total_files += 1
            distance = data['distance_to_mines_km']
            has_overlap = data['has_overlap']
            coord_count = data['coordinate_count']

            if has_overlap:
                overlapping_files += 1

            status = "✅ 매핑가능" if has_overlap else "❌ 분리됨"
            print(f"{status} {filename[:35]:35s}: {distance:6.1f}km, {coord_count:,}개 좌표")

        print(f"\n📊 분석 요약:")
        print(f"   총 XTF 파일: {total_files}개")
        print(f"   매핑 가능: {overlapping_files}개")
        print(f"   지리적 분리: {total_files - overlapping_files}개")

        if overlapping_files > 0:
            print(f"\n✅ 결론: {overlapping_files}개 파일에서 기뢰 위치 매핑 가능!")
            print("   → 이제 180도 회전/좌우 반전 변환을 적용하여 정확한 매핑 수행 가능")
        else:
            print(f"\n❌ 결론: 모든 XTF 파일이 기뢰 위치와 지리적으로 분리됨")
            print("   → 이전 \"매핑 문제 해결\" 주장은 잘못된 분석이었음")
            print("   → 좌표 변환으로는 해결할 수 없는 근본적인 데이터 불일치")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())