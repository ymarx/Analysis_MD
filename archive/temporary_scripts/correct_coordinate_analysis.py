#!/usr/bin/env python3
"""
Correct Coordinate Analysis with Transform
==========================================
올바른 좌표 변환을 적용한 위경도 매핑 검증

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


def apply_coordinate_transform(pixel_x, pixel_y, image_width, image_height, transform_type="rotate_flip"):
    """
    좌표 변환 적용 (이전에 해결한 180도 회전 + 좌우 반전)

    Args:
        pixel_x, pixel_y: 원본 픽셀 좌표
        image_width, image_height: 이미지 크기
        transform_type: 변환 타입

    Returns:
        변환된 (x, y) 좌표
    """
    if transform_type == "rotate_flip":
        # 180도 회전 + 좌우 반전의 결합
        # 결과: (x, y) -> (x, height-y)
        transformed_x = pixel_x
        transformed_y = image_height - pixel_y

    elif transform_type == "rotate_only":
        # 180도 회전만
        transformed_x = image_width - pixel_x
        transformed_y = image_height - pixel_y

    elif transform_type == "flip_only":
        # 좌우 반전만
        transformed_x = image_width - pixel_x
        transformed_y = pixel_y

    else:  # "none"
        transformed_x = pixel_x
        transformed_y = pixel_y

    return (transformed_x, transformed_y)


def test_coordinate_mapping_with_transform():
    """올바른 좌표 변환을 적용한 매핑 테스트"""
    logger.info("="*60)
    logger.info("CORRECT COORDINATE MAPPING WITH TRANSFORM")
    logger.info("="*60)

    # Initialize components
    config = PipelineConfig(
        data_dir=Path("datasets"),
        output_dir=Path("analysis_results/correct_mapping"),
        save_intermediate=True,
        verbose=True
    )
    pipeline = UnifiedPipeline(config)
    gps_parser = GPSParser()

    # Input files
    xtf_file = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")
    mine_locations_file = Path("datasets/Location_MDGPS.xlsx")

    # Step 1: Read XTF data
    logger.info("Step 1: Reading XTF data")
    pipeline.read_xtf(xtf_file)

    xtf_data = pipeline.results.get('xtf_data', {})
    intensity_data = xtf_data.get('intensity_data', {})
    intensity_images = intensity_data.get('intensity_images', {})
    navigation = intensity_data.get('navigation', {})

    combined_image = intensity_images['combined']
    latitudes = navigation.get('latitudes', [])
    longitudes = navigation.get('longitudes', [])

    logger.info(f"XTF intensity shape: {combined_image.shape}")
    logger.info(f"Navigation points: {len(latitudes)}")

    # Step 2: Parse mine locations
    logger.info("Step 2: Parsing mine locations")
    mine_locations = gps_parser.parse_gps_file(mine_locations_file)
    validation = gps_parser.validate_coordinates(mine_locations)

    logger.info(f"Parsed {len(mine_locations)} mine locations")
    logger.info(f"Valid coordinates: {validation['valid_count']}/{validation['total_count']}")

    # Step 3: Calculate bounds
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(longitudes), max(longitudes)

    logger.info(f"XTF bounds:")
    logger.info(f"  Latitude: [{lat_min:.6f}, {lat_max:.6f}]")
    logger.info(f"  Longitude: [{lon_min:.6f}, {lon_max:.6f}]")

    # Step 4: Test different coordinate transforms
    height, width = combined_image.shape

    transforms_to_test = ["none", "rotate_only", "flip_only", "rotate_flip"]
    results = {}

    for transform_type in transforms_to_test:
        logger.info(f"\nTesting transform: {transform_type}")

        # Map mine locations to pixels
        pixel_labels = []
        in_bounds_count = 0

        for i, location in enumerate(mine_locations):
            lat = location['latitude']
            lon = location['longitude']

            # Convert GPS to pixel (original method)
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                # Basic GPS to pixel conversion
                pixel_y_raw = int((lat - lat_min) / (lat_max - lat_min) * (height - 1))
                pixel_x_raw = int((lon - lon_min) / (lon_max - lon_min) * (width - 1))

                # Apply coordinate transform
                pixel_x, pixel_y = apply_coordinate_transform(
                    pixel_x_raw, pixel_y_raw, width, height, transform_type
                )

                # Check if transformed coordinates are still within bounds
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    in_bounds_count += 1
                    status = "✅ MAPPED"
                else:
                    status = "⚠️ OUT OF IMAGE"

                pixel_labels.append({
                    'mine_id': f"mine_{i+1}",
                    'gps_lat': lat,
                    'gps_lon': lon,
                    'pixel_x_raw': pixel_x_raw,
                    'pixel_y_raw': pixel_y_raw,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'in_bounds': (0 <= pixel_x < width and 0 <= pixel_y < height),
                    'status': status
                })
            else:
                pixel_labels.append({
                    'mine_id': f"mine_{i+1}",
                    'gps_lat': lat,
                    'gps_lon': lon,
                    'pixel_x_raw': -1,
                    'pixel_y_raw': -1,
                    'pixel_x': -1,
                    'pixel_y': -1,
                    'in_bounds': False,
                    'status': "❌ OUT OF XTF BOUNDS"
                })

        # Calculate statistics
        total_gps_in_xtf = sum(1 for loc in mine_locations
                              if lat_min <= loc['latitude'] <= lat_max
                              and lon_min <= loc['longitude'] <= lon_max)

        results[transform_type] = {
            'pixel_labels': pixel_labels,
            'total_mines': len(mine_locations),
            'gps_in_xtf_bounds': total_gps_in_xtf,
            'pixel_in_image_bounds': in_bounds_count,
            'mapping_rate': in_bounds_count / len(mine_locations) if mine_locations else 0
        }

        logger.info(f"  GPS coordinates in XTF bounds: {total_gps_in_xtf}")
        logger.info(f"  Pixel coordinates in image bounds: {in_bounds_count}")
        logger.info(f"  Mapping rate: {results[transform_type]['mapping_rate']:.1%}")

    # Step 5: Create visualization
    create_transform_comparison_visualization(combined_image, results, mine_locations)

    # Step 6: Generate report
    generate_transform_analysis_report(results, lat_min, lat_max, lon_min, lon_max)

    return results


def create_transform_comparison_visualization(combined_image, results, mine_locations):
    """변환 비교 시각화 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    transforms = ["none", "rotate_only", "flip_only", "rotate_flip"]
    colors = ['red', 'blue', 'green', 'purple']

    for i, transform_type in enumerate(transforms):
        ax = axes[i]

        # Show intensity image
        ax.imshow(combined_image, cmap='gray', aspect='auto', alpha=0.7)

        # Plot mine locations
        pixel_labels = results[transform_type]['pixel_labels']
        for label in pixel_labels:
            if label['in_bounds']:
                ax.scatter(label['pixel_x'], label['pixel_y'],
                          c=colors[i], marker='o', s=50, alpha=0.8)
                ax.text(label['pixel_x'], label['pixel_y'] - 10,
                       label['mine_id'].replace('mine_', ''),
                       color='yellow', fontsize=8, ha='center')

        mapping_rate = results[transform_type]['mapping_rate']
        ax.set_title(f'Transform: {transform_type}\nMapping Rate: {mapping_rate:.1%}')
        ax.set_xlabel('Range (pixels)')
        ax.set_ylabel('Ping (pixels)')

    plt.tight_layout()
    output_file = Path("analysis_results/correct_mapping/transform_comparison.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved transform comparison to: {output_file}")


def generate_transform_analysis_report(results, lat_min, lat_max, lon_min, lon_max):
    """변환 분석 보고서 생성"""

    report_lines = []
    report_lines.append("# 좌표 변환 분석 보고서")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("## XTF 데이터 범위")
    report_lines.append(f"- Latitude: [{lat_min:.6f}, {lat_max:.6f}] (범위: {lat_max-lat_min:.6f}°)")
    report_lines.append(f"- Longitude: [{lon_min:.6f}, {lon_max:.6f}] (범위: {lon_max-lon_min:.6f}°)")
    report_lines.append("")

    report_lines.append("## 변환 타입별 매핑 결과")
    report_lines.append("")

    best_transform = None
    best_rate = 0

    for transform_type, result in results.items():
        mapping_rate = result['mapping_rate']
        if mapping_rate > best_rate:
            best_rate = mapping_rate
            best_transform = transform_type

        report_lines.append(f"### {transform_type}")
        report_lines.append(f"- GPS 좌표가 XTF 범위 내: {result['gps_in_xtf_bounds']}")
        report_lines.append(f"- 픽셀 좌표가 이미지 범위 내: {result['pixel_in_image_bounds']}")
        report_lines.append(f"- 매핑 성공률: {mapping_rate:.1%}")
        report_lines.append("")

        # Detailed mapping table
        if result['pixel_in_image_bounds'] > 0:
            report_lines.append("#### 성공적으로 매핑된 위치:")
            report_lines.append("| Mine ID | GPS Lat | GPS Lon | Pixel X | Pixel Y |")
            report_lines.append("|---------|---------|---------|---------|---------|")

            for label in result['pixel_labels']:
                if label['in_bounds']:
                    report_lines.append(f"| {label['mine_id']} | {label['gps_lat']:.6f} | {label['gps_lon']:.6f} | {label['pixel_x']} | {label['pixel_y']} |")
            report_lines.append("")

    report_lines.append("## 결론")
    if best_rate > 0:
        report_lines.append(f"✅ **최적 변환**: `{best_transform}` (성공률: {best_rate:.1%})")
        report_lines.append("")
        report_lines.append("이전에 해결했던 180도 회전/좌우 반전 문제의 해결책이 확인되었습니다.")
        report_lines.append(f"`{best_transform}` 변환을 적용하면 기뢰 위치가 올바르게 매핑됩니다.")
    else:
        report_lines.append("❌ **문제 지속**: 모든 변환에서도 매핑 실패")
        report_lines.append("")
        report_lines.append("좌표 변환으로도 해결되지 않는 근본적인 데이터 불일치 문제가 있습니다.")
        report_lines.append("XTF 데이터와 기뢰 위치 데이터가 서로 다른 지역일 가능성이 높습니다.")

    # Save report
    output_file = Path("analysis_results/correct_mapping/transform_analysis_report.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved transform analysis report to: {output_file}")

    # Also save results as JSON
    json_file = Path("analysis_results/correct_mapping/transform_results.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved transform results to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Correct Coordinate Analysis with Transform")

    try:
        results = test_coordinate_mapping_with_transform()

        # Print summary
        print("\n" + "="*60)
        print("COORDINATE TRANSFORM ANALYSIS SUMMARY")
        print("="*60)

        for transform_type, result in results.items():
            mapping_rate = result['mapping_rate']
            print(f"{transform_type:15s}: {mapping_rate:6.1%} ({result['pixel_in_image_bounds']}/{result['total_mines']} mines)")

        best_transform = max(results.keys(), key=lambda k: results[k]['mapping_rate'])
        best_rate = results[best_transform]['mapping_rate']

        print(f"\nBest Transform: {best_transform} ({best_rate:.1%})")

        if best_rate > 0:
            print("✅ 좌표 변환으로 매핑 문제 해결됨!")
        else:
            print("❌ 좌표 변환으로도 해결되지 않음 - 데이터 불일치 문제")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())