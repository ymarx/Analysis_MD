#!/usr/bin/env python3
"""
Final Coordinate Analysis
=========================
XTF 파일의 실제 좌표를 추출하여 기뢰 위치와 비교 분석
올바른 XTF 리더 API를 사용한 최종 분석

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

from src.data_processing.xtf_reader import XTFReader
from pipeline.modules.gps_parser import GPSParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_coordinates_final(xtf_file_path):
    """최종 좌표 추출 - 올바른 API 사용"""
    try:
        logger.info(f"Analyzing XTF file: {xtf_file_path}")

        # Create reader with max_pings limit for efficiency
        reader = XTFReader(str(xtf_file_path), max_pings=500)

        # Load file
        if not reader.load_file():
            logger.error("Failed to load XTF file")
            return []

        # Parse pings
        ping_data_list = reader.parse_pings()

        if not ping_data_list:
            logger.warning("No ping data extracted")
            return []

        # Extract coordinates from PingData objects
        coordinates = []
        for i, ping_data in enumerate(ping_data_list):
            # ping_data is a PingData object with latitude/longitude attributes
            lat = ping_data.latitude
            lon = ping_data.longitude

            # Filter valid coordinates (non-zero)
            if lat != 0 and lon != 0:
                coordinates.append({
                    'ping': i,
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'timestamp': ping_data.timestamp,
                    'frequency': ping_data.frequency
                })

        logger.info(f"Extracted {len(coordinates)} valid coordinate points from {len(ping_data_list)} pings")
        return coordinates

    except Exception as e:
        logger.error(f"Failed to extract coordinates from {xtf_file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def final_coordinate_analysis():
    """최종 좌표 분석"""
    logger.info("="*60)
    logger.info("FINAL COORDINATE ANALYSIS")
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

        logger.info(f"\nProcessing: {xtf_file.name}")

        coordinates = extract_coordinates_final(xtf_file)

        if not coordinates:
            logger.warning(f"No coordinates extracted from {xtf_file.name}")
            results[xtf_file.name] = {
                'success': False,
                'message': 'No coordinates extracted',
                'file_path': str(xtf_file)
            }
            continue

        # Calculate coordinate statistics
        lats = [coord['latitude'] for coord in coordinates]
        lons = [coord['longitude'] for coord in coordinates]

        xtf_lat_min, xtf_lat_max = min(lats), max(lats)
        xtf_lon_min, xtf_lon_max = min(lons), max(lons)

        # Calculate centers
        xtf_center_lat = (xtf_lat_min + xtf_lat_max) / 2
        xtf_center_lon = (xtf_lon_min + xtf_lon_max) / 2
        mine_center_lat = (mine_lat_min + mine_lat_max) / 2
        mine_center_lon = (mine_lon_min + mine_lon_max) / 2

        # Haversine distance calculation
        lat1, lon1 = np.radians(xtf_center_lat), np.radians(xtf_center_lon)
        lat2, lon2 = np.radians(mine_center_lat), np.radians(mine_center_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))

        # Check geographic overlap
        overlap_lat = max(0, min(xtf_lat_max, mine_lat_max) - max(xtf_lat_min, mine_lat_min))
        overlap_lon = max(0, min(xtf_lon_max, mine_lon_max) - max(xtf_lon_min, mine_lon_min))
        has_overlap = overlap_lat > 0 and overlap_lon > 0

        # Coverage area calculations
        lat_coverage_km = (xtf_lat_max - xtf_lat_min) * 111
        lon_coverage_km = (xtf_lon_max - xtf_lon_min) * 111 * np.cos(np.radians(xtf_lat_min))

        # Store results
        results[xtf_file.name] = {
            'success': True,
            'file_path': str(xtf_file),
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
            'sample_coordinates': coordinates[:5],  # First 5 for verification
            'coordinate_stats': {
                'lat_mean': np.mean(lats),
                'lat_std': np.std(lats),
                'lon_mean': np.mean(lons),
                'lon_std': np.std(lons)
            }
        }

        logger.info(f"  Extracted coordinates: {len(coordinates)} points")
        logger.info(f"  Bounds: Lat[{xtf_lat_min:.6f}, {xtf_lat_max:.6f}], Lon[{xtf_lon_min:.6f}, {xtf_lon_max:.6f}]")
        logger.info(f"  Coverage: {lat_coverage_km:.1f} x {lon_coverage_km:.1f} km")
        logger.info(f"  Distance to mines: {distance_km:.1f} km")
        logger.info(f"  Geographic overlap: {'YES' if has_overlap else 'NO'}")

    # Create comprehensive visualization
    create_final_visualization(results, mine_locations)

    # Generate comprehensive report
    generate_final_report(results, mine_locations)

    return results


def create_final_visualization(results, mine_locations):
    """최종 종합 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Extract mine data
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]

    colors = ['blue', 'green', 'purple', 'orange']

    # Plot 1: Overall geographic comparison
    ax1.scatter(mine_lons, mine_lats, c='red', s=100, marker='x',
               label=f'Mine Locations ({len(mine_locations)})', alpha=0.8, linewidth=3)

    for i, (filename, data) in enumerate(results.items()):
        if not data.get('success', False):
            continue

        bounds = data['xtf_bounds']

        # Draw coverage rectangle
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
        ax1.add_patch(rect)

        # Mark center
        center_lat, center_lon = data['center']
        ax1.plot(center_lon, center_lat, 'o', color=colors[i % len(colors)],
                markersize=10, markeredgewidth=2, markeredgecolor='white')

        # Distance annotation
        ax1.annotate(f'{data["distance_to_mines_km"]:.1f} km',
                   (center_lon, center_lat),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=10)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('XTF Coverage vs Mine Locations - Overall View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mine location detail view
    mine_lat_center = np.mean(mine_lats)
    mine_lon_center = np.mean(mine_lons)
    buffer = max(max(mine_lats) - min(mine_lats), max(mine_lons) - min(mine_lons), 0.001) * 2

    ax2.scatter(mine_lons, mine_lats, c='red', s=150, marker='x',
               label='Mine Locations', alpha=0.8, linewidth=3)

    # Number mine locations
    for i, (lat, lon) in enumerate(zip(mine_lats, mine_lons)):
        ax2.annotate(f'{i+1}', (lon, lat), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # Show XTF areas if close enough
    for i, (filename, data) in enumerate(results.items()):
        if not data.get('success', False) or data['distance_to_mines_km'] > 10:
            continue

        bounds = data['xtf_bounds']
        rect = patches.Rectangle(
            (bounds['lon_min'], bounds['lat_min']),
            bounds['lon_max'] - bounds['lon_min'],
            bounds['lat_max'] - bounds['lat_min'],
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

    # Plot 3: Distance comparison
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    if successful_results:
        filenames = [name[:20] + '...' for name in successful_results.keys()]
        distances = [data['distance_to_mines_km'] for data in successful_results.values()]

        bars = ax3.bar(range(len(filenames)), distances, color=colors[:len(filenames)])
        ax3.set_xlabel('XTF Files')
        ax3.set_ylabel('Distance to Mines (km)')
        ax3.set_title('Distance from XTF Center to Mine Locations')
        ax3.set_xticks(range(len(filenames)))
        ax3.set_xticklabels(filenames, rotation=45, ha='right')

        # Add value labels on bars
        for bar, distance in zip(bars, distances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.01,
                    f'{distance:.1f} km', ha='center', va='bottom')

    # Plot 4: Coverage area comparison
    if successful_results:
        coverage_areas = [data['coverage_km'][0] * data['coverage_km'][1] for data in successful_results.values()]
        coord_counts = [data['coordinate_count'] for data in successful_results.values()]

        ax4.scatter(coverage_areas, coord_counts, s=100, c=colors[:len(successful_results)], alpha=0.7)

        for i, (filename, data) in enumerate(successful_results.items()):
            ax4.annotate(filename[:15] + '...',
                        (coverage_areas[i], coord_counts[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.set_xlabel('Coverage Area (km²)')
        ax4.set_ylabel('Number of Coordinate Points')
        ax4.set_title('XTF Coverage Area vs Coordinate Density')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path("analysis_results/final_coordinate_analysis/comprehensive_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comprehensive visualization to: {output_file}")


def generate_final_report(results, mine_locations):
    """최종 종합 보고서 생성"""

    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    report_lines = []
    report_lines.append("# XTF 파일 좌표 추출 및 기뢰 위치 매핑 가능성 최종 분석 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # Executive Summary
    successful_files = sum(1 for data in results.values() if data.get('success', False))
    overlapping_files = sum(1 for data in results.values() if data.get('has_overlap', False))

    report_lines.append("## 🎯 **핵심 결론**")
    report_lines.append("")

    if overlapping_files > 0:
        report_lines.append("### ✅ **지리적 중복 발견 - 매핑 가능**")
        report_lines.append(f"**매핑 가능한 파일**: {overlapping_files}개")
        report_lines.append("**이전 주장 검증**: ✅ 180도 회전/좌우 반전 변환 해결책이 실제로 존재")
        report_lines.append("**다음 단계**: 중복 영역이 있는 XTF 파일로 좌표 변환 재시도")
    else:
        report_lines.append("### ❌ **지리적 분리 확인됨**")
        report_lines.append(f"**분석된 파일**: {successful_files}개 (모두 기뢰 위치와 분리)")
        report_lines.append("**이전 주장 검증**: ❌ \"180도 회전/좌우 반전으로 문제 해결\" 주장은 **잘못된 분석**")
        report_lines.append("**근본 문제**: 좌표 변환으로는 해결할 수 없는 데이터 불일치")

    report_lines.append("")

    # Mine location reference
    report_lines.append("## 📍 **기뢰 위치 데이터 (Location_MDGPS.xlsx)**")
    report_lines.append(f"- **총 기뢰 개수**: {len(mine_locations)}개")
    report_lines.append(f"- **위도 범위**: [{mine_lat_min:.6f}°, {mine_lat_max:.6f}°] (범위: {(mine_lat_max-mine_lat_min)*111:.1f}m)")
    report_lines.append(f"- **경도 범위**: [{mine_lon_min:.6f}°, {mine_lon_max:.6f}°] (범위: {(mine_lon_max-mine_lon_min)*111*np.cos(np.radians(mine_lat_min)):.1f}m)")
    report_lines.append(f"- **중심점**: ({np.mean(mine_lats):.6f}°, {np.mean(mine_lons):.6f}°)")
    report_lines.append("")

    # XTF file analysis
    report_lines.append("## 📡 **XTF 파일별 상세 분석**")
    report_lines.append("")

    closest_file = None
    closest_distance = float('inf')

    for filename, data in results.items():
        if not data.get('success', False):
            report_lines.append(f"### ❌ **{filename}**")
            report_lines.append(f"- **상태**: 분석 실패")
            report_lines.append(f"- **사유**: {data.get('message', '알 수 없는 오류')}")
            report_lines.append("")
            continue

        bounds = data['xtf_bounds']
        coverage = data['coverage_km']
        distance = data['distance_to_mines_km']
        coord_count = data['coordinate_count']
        has_overlap = data['has_overlap']
        stats = data['coordinate_stats']

        if distance < closest_distance:
            closest_distance = distance
            closest_file = filename

        status_icon = "✅" if has_overlap else "❌"
        report_lines.append(f"### {status_icon} **{filename}**")
        report_lines.append(f"- **파일 경로**: `{data['file_path']}`")
        report_lines.append(f"- **추출된 좌표**: {coord_count:,}개")
        report_lines.append("")
        report_lines.append("**지리적 범위**:")
        report_lines.append(f"- 위도: [{bounds['lat_min']:.6f}°, {bounds['lat_max']:.6f}°]")
        report_lines.append(f"- 경도: [{bounds['lon_min']:.6f}°, {bounds['lon_max']:.6f}°]")
        report_lines.append(f"- 조사 영역: {coverage[0]:.1f} × {coverage[1]:.1f} km ({coverage[0]*coverage[1]:.2f} km²)")
        report_lines.append(f"- 중심점: ({data['center'][0]:.6f}°, {data['center'][1]:.6f}°)")
        report_lines.append("")
        report_lines.append("**기뢰 위치와의 관계**:")
        report_lines.append(f"- 중심점 간 거리: **{distance:.1f} km**")
        report_lines.append(f"- 지리적 중복: {'✅ 있음' if has_overlap else '❌ 없음'}")
        report_lines.append("")
        report_lines.append("**좌표 통계**:")
        report_lines.append(f"- 위도 평균: {stats['lat_mean']:.6f}° (표준편차: {stats['lat_std']:.6f}°)")
        report_lines.append(f"- 경도 평균: {stats['lon_mean']:.6f}° (표준편차: {stats['lon_std']:.6f}°)")
        report_lines.append("")

        # Sample coordinates
        if 'sample_coordinates' in data and data['sample_coordinates']:
            report_lines.append("**샘플 좌표** (처음 3개):")
            for coord in data['sample_coordinates'][:3]:
                report_lines.append(f"- Ping {coord['ping']}: ({coord['latitude']:.6f}°, {coord['longitude']:.6f}°) @{coord.get('frequency', 0):.0f}Hz")

        report_lines.append("")

    # Technical analysis
    report_lines.append("## 🔬 **기술적 분석**")
    report_lines.append("")

    if overlapping_files > 0:
        overlapping_file_names = [name for name, data in results.items() if data.get('has_overlap', False)]
        report_lines.append("### ✅ **매핑 가능 파일 확인됨**")
        for filename in overlapping_file_names:
            report_lines.append(f"- **{filename}**: 기뢰 위치와 지리적 중복 영역 존재")
        report_lines.append("")
        report_lines.append("**이는 다음을 의미합니다**:")
        report_lines.append("1. **이전 분석이 올바름**: 180도 회전/좌우 반전 변환으로 매핑 가능")
        report_lines.append("2. **파이프라인 복구 가능**: 중복 파일로 정확한 좌표 매핑 수행 가능")
        report_lines.append("3. **기뢰 탐지 모델 훈련 가능**: 실제 레이블 데이터로 모델 학습 진행 가능")
        report_lines.append("")
        report_lines.append("**권장 다음 단계**:")
        report_lines.append("1. 🔄 중복 영역이 있는 XTF 파일로 `correct_coordinate_analysis.py` 재실행")
        report_lines.append("2. ✅ 180도 회전/좌우 반전 변환의 정확성 검증")
        report_lines.append("3. 🎯 정확한 픽셀-GPS 매핑으로 레이블 데이터 생성")
        report_lines.append("4. 🚀 실제 기뢰 위치 데이터로 모델 훈련 시작")

    else:
        report_lines.append("### ❌ **지리적 분리 문제 확인됨**")
        report_lines.append(f"**가장 가까운 파일**: {closest_file} (거리: {closest_distance:.1f} km)")
        report_lines.append("")
        report_lines.append("**문제 분석**:")
        report_lines.append("1. **데이터 불일치**: XTF 파일과 기뢰 위치 데이터가 서로 다른 지역")
        report_lines.append("2. **이전 주장 오류**: \"180도 회전/좌우 반전으로 문제 해결\" 주장은 **잘못된 분석**")
        report_lines.append("3. **근본 원인**: 좌표 변환으로는 해결할 수 없는 지리적 분리")
        report_lines.append("")
        report_lines.append("**해결 방안**:")
        report_lines.append("1. 🔍 **데이터 재확보**: 기뢰 위치와 동일한 지역의 XTF 파일 확보")
        report_lines.append("2. 📋 **데이터 검증**: 기뢰 위치 좌표의 정확성 재검토")
        report_lines.append("3. 🗺️ **조사 일치 확인**: 동일한 해양 조사에서 수집된 데이터인지 확인")
        report_lines.append("4. 🎯 **추가 XTF 탐색**: 다른 Pohang 지역 XTF 파일 조사")

    # Summary statistics
    report_lines.append("")
    report_lines.append("## 📊 **분석 통계 요약**")
    report_lines.append("")
    report_lines.append("| 항목 | 값 |")
    report_lines.append("|------|-----|")
    report_lines.append(f"| 분석된 XTF 파일 | {len(results)}개 |")
    report_lines.append(f"| 성공적 분석 | {successful_files}개 |")
    report_lines.append(f"| 지리적 중복 파일 | {overlapping_files}개 |")
    report_lines.append(f"| 기뢰 위치 | {len(mine_locations)}개 |")

    if successful_files > 0:
        total_coordinates = sum(data['coordinate_count'] for data in results.values() if data.get('success', False))
        avg_distance = np.mean([data['distance_to_mines_km'] for data in results.values() if data.get('success', False)])
        report_lines.append(f"| 총 추출 좌표 | {total_coordinates:,}개 |")
        report_lines.append(f"| 평균 거리 | {avg_distance:.1f} km |")

    # Technical details
    report_lines.append("")
    report_lines.append("## 🛠️ **기술적 세부사항**")
    report_lines.append("")
    report_lines.append("**사용된 도구**:")
    report_lines.append("- `XTFReader`: 사이드스캔 소나 XTF 파일 파싱")
    report_lines.append("- `GPSParser`: GPS 좌표 데이터 파싱 및 검증")
    report_lines.append("- `pyxtf`: XTF 파일 저수준 처리")
    report_lines.append("")
    report_lines.append("**좌표 추출 방법**:")
    report_lines.append("- PingData 객체에서 latitude/longitude 속성 직접 추출")
    report_lines.append("- 0이 아닌 유효 좌표만 필터링")
    report_lines.append("- 최대 500개 ping으로 제한하여 효율성 확보")
    report_lines.append("")
    report_lines.append("**거리 계산**:")
    report_lines.append("- Haversine 공식을 사용한 구면 거리 계산")
    report_lines.append("- 각 XTF 파일의 중심점과 기뢰 위치 중심점 간 거리")
    report_lines.append("")
    report_lines.append("**지리적 중복 판정**:")
    report_lines.append("- XTF 영역과 기뢰 위치 영역의 경계 상자 중복 검사")
    report_lines.append("- 위도와 경도 모두에서 중복이 있어야 매핑 가능으로 판정")

    # Save report
    output_file = Path("analysis_results/final_coordinate_analysis/FINAL_COORDINATE_ANALYSIS_REPORT.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved final report to: {output_file}")

    # Save detailed data as JSON
    json_file = Path("analysis_results/final_coordinate_analysis/coordinate_analysis_data.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved analysis data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Final Coordinate Analysis")

    try:
        results = final_coordinate_analysis()

        # Print comprehensive summary
        print("\n" + "="*80)
        print("XTF 파일 좌표 추출 및 기뢰 위치 매핑 가능성 최종 분석 결과")
        print("="*80)

        successful = 0
        overlapping = 0
        failed = 0

        for filename, data in results.items():
            if data.get('success', False):
                successful += 1
                distance = data['distance_to_mines_km']
                has_overlap = data['has_overlap']
                coord_count = data['coordinate_count']

                if has_overlap:
                    overlapping += 1

                status = "✅ 매핑가능" if has_overlap else "❌ 분리됨"
                print(f"{status} {filename[:45]:45s}: {distance:6.1f}km, {coord_count:,}개 좌표")
            else:
                failed += 1
                print(f"❌ 실패      {filename[:45]:45s}: {data.get('message', '분석 실패')}")

        print(f"\n📊 **최종 분석 요약**:")
        print(f"   총 XTF 파일: {len(results)}개")
        print(f"   성공적 분석: {successful}개")
        print(f"   분석 실패: {failed}개")
        print(f"   매핑 가능: {overlapping}개")
        print(f"   지리적 분리: {successful - overlapping}개")

        print(f"\n🎯 **핵심 결론**:")
        if overlapping > 0:
            print(f"   ✅ {overlapping}개 파일에서 기뢰 위치와 지리적 중복 발견!")
            print("   → 이전 \"180도 회전/좌우 반전 해결\" 주장이 **실제로 정확함**")
            print("   → 중복 영역이 있는 XTF 파일로 좌표 변환 재시도 가능")
            print("   → 실제 기뢰 위치 데이터로 모델 훈련 진행 가능")
        else:
            print(f"   ❌ 모든 XTF 파일이 기뢰 위치와 지리적으로 분리됨")
            print("   → 이전 \"180도 회전/좌우 반전으로 문제 해결\" 주장은 **잘못된 분석**")
            print("   → 좌표 변환으로는 해결할 수 없는 근본적인 데이터 불일치")
            print("   → 동일 지역의 XTF 파일과 기뢰 위치 데이터 재확보 필요")

        return 0

    except Exception as e:
        logger.error(f"Final analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())