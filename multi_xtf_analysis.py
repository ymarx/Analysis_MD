#!/usr/bin/env python3
"""
Multi-XTF Analysis for Geographic Coverage
==========================================
여러 Pohang XTF 파일의 지리적 범위를 분석하여 기뢰 위치와 매핑되는 파일 찾기

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

from pipeline.unified_pipeline import UnifiedPipeline, PipelineConfig
from pipeline.modules.gps_parser import GPSParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_xtf_geographic_coverage():
    """여러 XTF 파일의 지리적 범위 분석"""
    logger.info("="*60)
    logger.info("MULTI-XTF GEOGRAPHIC COVERAGE ANALYSIS")
    logger.info("="*60)

    # Initialize components
    config = PipelineConfig(
        data_dir=Path("datasets"),
        output_dir=Path("analysis_results/multi_xtf_coverage"),
        save_intermediate=True,
        verbose=True
    )
    pipeline = UnifiedPipeline(config)
    gps_parser = GPSParser()

    # XTF files to analyze
    xtf_files = [
        Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf"),
        Path("datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf")
    ]

    mine_locations_file = Path("datasets/Location_MDGPS.xlsx")

    # Parse mine locations first
    logger.info("Parsing mine locations")
    mine_locations = gps_parser.parse_gps_file(mine_locations_file)
    validation = gps_parser.validate_coordinates(mine_locations)

    logger.info(f"Parsed {len(mine_locations)} mine locations")
    logger.info(f"Valid coordinates: {validation['valid_count']}/{validation['total_count']}")

    # Calculate mine location bounds
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    logger.info(f"Mine location bounds:")
    logger.info(f"  Latitude: [{mine_lat_min:.6f}, {mine_lat_max:.6f}]")
    logger.info(f"  Longitude: [{mine_lon_min:.6f}, {mine_lon_max:.6f}]")

    # Analyze each XTF file
    xtf_analysis = {}

    for xtf_file in xtf_files:
        if not xtf_file.exists():
            logger.warning(f"XTF file not found: {xtf_file}")
            continue

        logger.info(f"\nAnalyzing XTF file: {xtf_file.name}")

        try:
            # Read XTF data
            pipeline.read_xtf(xtf_file)

            xtf_data = pipeline.results.get('xtf_data', {})
            intensity_data = xtf_data.get('intensity_data', {})
            navigation = intensity_data.get('navigation', {})

            latitudes = navigation.get('latitudes', [])
            longitudes = navigation.get('longitudes', [])

            if not latitudes or not longitudes:
                logger.warning(f"No navigation data found in {xtf_file.name}")
                continue

            # Calculate XTF bounds
            xtf_lat_min, xtf_lat_max = min(latitudes), max(latitudes)
            xtf_lon_min, xtf_lon_max = min(longitudes), max(longitudes)

            # Calculate coverage area
            lat_range_km = (xtf_lat_max - xtf_lat_min) * 111  # ~111 km per degree
            lon_range_km = (xtf_lon_max - xtf_lon_min) * 111 * np.cos(np.radians(xtf_lat_min))

            # Check overlap with mine locations
            overlap_lat = max(0, min(xtf_lat_max, mine_lat_max) - max(xtf_lat_min, mine_lat_min))
            overlap_lon = max(0, min(xtf_lon_max, mine_lon_max) - max(xtf_lon_min, mine_lon_min))
            has_overlap = overlap_lat > 0 and overlap_lon > 0

            # Calculate distance between centers
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

            xtf_analysis[xtf_file.name] = {
                'file_path': str(xtf_file),
                'lat_range': [xtf_lat_min, xtf_lat_max],
                'lon_range': [xtf_lon_min, xtf_lon_max],
                'coverage_km': [lat_range_km, lon_range_km],
                'center': [xtf_center_lat, xtf_center_lon],
                'has_overlap': has_overlap,
                'distance_to_mines_km': distance_km,
                'ping_count': len(latitudes)
            }

            logger.info(f"  XTF bounds: Lat[{xtf_lat_min:.6f}, {xtf_lat_max:.6f}], Lon[{xtf_lon_min:.6f}, {xtf_lon_max:.6f}]")
            logger.info(f"  Coverage: {lat_range_km:.1f} × {lon_range_km:.1f} km")
            logger.info(f"  Center distance to mines: {distance_km:.1f} km")
            logger.info(f"  Overlap with mines: {'YES' if has_overlap else 'NO'}")

        except Exception as e:
            logger.error(f"Error analyzing {xtf_file.name}: {e}")
            continue

    # Generate comprehensive visualization
    create_geographic_coverage_map(xtf_analysis, mine_locations)

    # Generate detailed report
    generate_coverage_analysis_report(xtf_analysis, mine_locations)

    return xtf_analysis


def create_geographic_coverage_map(xtf_analysis, mine_locations):
    """지리적 범위 비교 지도 생성"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Extract mine coordinates
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]

    # Plot 1: Overall geographic view
    ax1.scatter(mine_lons, mine_lats, c='red', s=50, marker='x', label='Mine Locations', alpha=0.8)

    colors = ['blue', 'green', 'purple', 'orange']
    for i, (filename, data) in enumerate(xtf_analysis.items()):
        lat_min, lat_max = data['lat_range']
        lon_min, lon_max = data['lon_range']

        # Draw rectangle for XTF coverage
        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            label=f'XTF: {filename[:20]}...'
        )
        ax1.add_patch(rect)

        # Mark center
        center_lat, center_lon = data['center']
        ax1.plot(center_lon, center_lat, 'o', color=colors[i % len(colors)], markersize=8)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Geographic Coverage Comparison - Overall View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed view on mine locations
    mine_lat_center = np.mean(mine_lats)
    mine_lon_center = np.mean(mine_lons)
    buffer = 0.001  # ~100m buffer

    ax2.scatter(mine_lons, mine_lats, c='red', s=100, marker='x', label='Mine Locations', alpha=0.8)

    for i, (filename, data) in enumerate(xtf_analysis.items()):
        lat_min, lat_max = data['lat_range']
        lon_min, lon_max = data['lon_range']

        # Only show if within reasonable distance
        if data['distance_to_mines_km'] < 100:  # 100km threshold
            rect = patches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=3,
                edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)],
                alpha=0.3,
                label=f'XTF: {filename[:20]}...'
            )
            ax2.add_patch(rect)

    ax2.set_xlim(mine_lon_center - buffer, mine_lon_center + buffer)
    ax2.set_ylim(mine_lat_center - buffer, mine_lat_center + buffer)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Geographic Coverage - Mine Location Area (Zoomed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = Path("analysis_results/multi_xtf_coverage/geographic_coverage_map.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved geographic coverage map to: {output_file}")


def generate_coverage_analysis_report(xtf_analysis, mine_locations):
    """지리적 범위 분석 보고서 생성"""

    report_lines = []
    report_lines.append("# Multi-XTF Geographic Coverage Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Mine location summary
    mine_lats = [loc['latitude'] for loc in mine_locations]
    mine_lons = [loc['longitude'] for loc in mine_locations]
    mine_lat_min, mine_lat_max = min(mine_lats), max(mine_lats)
    mine_lon_min, mine_lon_max = min(mine_lons), max(mine_lons)

    report_lines.append("## Mine Location Reference")
    report_lines.append(f"- Total mine locations: {len(mine_locations)}")
    report_lines.append(f"- Latitude range: [{mine_lat_min:.6f}, {mine_lat_max:.6f}] (범위: {(mine_lat_max-mine_lat_min)*111:.1f}m)")
    report_lines.append(f"- Longitude range: [{mine_lon_min:.6f}, {mine_lon_max:.6f}] (범위: {(mine_lon_max-mine_lon_min)*111*np.cos(np.radians(mine_lat_min)):.1f}m)")
    report_lines.append("")

    # XTF file analysis
    report_lines.append("## XTF File Geographic Coverage")
    report_lines.append("")

    best_match = None
    best_distance = float('inf')

    for filename, data in xtf_analysis.items():
        lat_min, lat_max = data['lat_range']
        lon_min, lon_max = data['lon_range']
        coverage_km = data['coverage_km']
        distance_km = data['distance_to_mines_km']
        has_overlap = data['has_overlap']

        if distance_km < best_distance:
            best_distance = distance_km
            best_match = filename

        report_lines.append(f"### {filename}")
        report_lines.append(f"- **Coverage**: {coverage_km[0]:.1f} × {coverage_km[1]:.1f} km")
        report_lines.append(f"- **Coordinates**: Lat[{lat_min:.6f}, {lat_max:.6f}], Lon[{lon_min:.6f}, {lon_max:.6f}]")
        report_lines.append(f"- **Distance to mines**: {distance_km:.1f} km")
        report_lines.append(f"- **Overlap with mines**: {'✅ YES' if has_overlap else '❌ NO'}")
        report_lines.append(f"- **Ping count**: {data['ping_count']:,}")
        report_lines.append("")

    # Analysis conclusion
    report_lines.append("## Analysis Results")

    overlapping_files = [name for name, data in xtf_analysis.items() if data['has_overlap']]

    if overlapping_files:
        report_lines.append("### ✅ **Geographic Overlap Found**")
        report_lines.append(f"Files with mine location overlap: {', '.join(overlapping_files)}")
        report_lines.append("")
        report_lines.append("**Next Steps**:")
        report_lines.append("1. Test coordinate mapping with overlapping XTF files")
        report_lines.append("2. Apply 180-degree rotation/flip transformations")
        report_lines.append("3. Verify mine location mapping accuracy")
    else:
        report_lines.append("### ❌ **No Geographic Overlap**")
        report_lines.append(f"Best match: **{best_match}** (distance: {best_distance:.1f} km)")
        report_lines.append("")
        report_lines.append("**Problem Analysis**:")
        report_lines.append("- All XTF files are geographically separated from mine locations")
        report_lines.append("- Previous coordinate transformation solutions were not verified with correct geographic data")
        report_lines.append("- Need XTF data from the same geographic area as mine locations")
        report_lines.append("")
        report_lines.append("**Recommendations**:")
        report_lines.append("1. 🔍 Search for additional Pohang XTF files covering mine location area")
        report_lines.append("2. 📍 Verify mine location coordinates are correct for this survey area")
        report_lines.append("3. 🗺️ Confirm XTF and mine data are from the same maritime survey")

    # Save report
    output_file = Path("analysis_results/multi_xtf_coverage/coverage_analysis_report.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved coverage analysis report to: {output_file}")

    # Also save detailed data as JSON
    json_file = Path("analysis_results/multi_xtf_coverage/coverage_data.json")
    with open(json_file, 'w') as f:
        json.dump(xtf_analysis, f, indent=2, default=str)

    logger.info(f"Saved coverage data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Multi-XTF Geographic Coverage Analysis")

    try:
        results = analyze_xtf_geographic_coverage()

        # Print summary
        print("\n" + "="*60)
        print("MULTI-XTF GEOGRAPHIC COVERAGE SUMMARY")
        print("="*60)

        overlapping_files = []
        closest_file = None
        closest_distance = float('inf')

        for filename, data in results.items():
            distance = data['distance_to_mines_km']
            has_overlap = data['has_overlap']

            if has_overlap:
                overlapping_files.append(filename)

            if distance < closest_distance:
                closest_distance = distance
                closest_file = filename

            print(f"{filename[:30]:30s}: {distance:6.1f} km from mines, Overlap: {'YES' if has_overlap else 'NO'}")

        print(f"\nClosest XTF: {closest_file} ({closest_distance:.1f} km)")

        if overlapping_files:
            print(f"✅ Files with geographic overlap: {len(overlapping_files)}")
            print("  Coordinate mapping should work with these files!")
        else:
            print("❌ No geographic overlap found")
            print("  This explains why coordinate transformations are not working")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())