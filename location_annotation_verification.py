#!/usr/bin/env python3
"""
Location_MDGPS vs PH_annotation Verification
============================================
Location_MDGPS.xlsx의 기뢰 위치 데이터와 PH_annotation.bmp가
같은 위치를 나타내는지 검증 분석

Author: YMARX
Date: 2025-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
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


def load_and_analyze_annotation_image():
    """PH_annotation.bmp 이미지 로드 및 기뢰 위치 마커 분석"""
    logger.info("Loading and analyzing PH_annotation.bmp")

    annotation_path = Path("datasets/PH_annotation.bmp")

    if not annotation_path.exists():
        logger.error(f"Annotation file not found: {annotation_path}")
        return None

    try:
        # 이미지 로드
        image = cv2.imread(str(annotation_path))
        if image is None:
            logger.error("Failed to load annotation image")
            return None

        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        logger.info(f"Loaded annotation image: {width}x{height}")

        # 이미지에서 기뢰 위치 마커 찾기
        mine_markers = detect_mine_markers(image_rgb)

        return {
            'image': image_rgb,
            'shape': (height, width),
            'path': str(annotation_path),
            'mine_markers': mine_markers,
            'marker_count': len(mine_markers)
        }

    except Exception as e:
        logger.error(f"Error loading annotation image: {e}")
        return None


def detect_mine_markers(image):
    """이미지에서 기뢰 위치 마커 감지"""
    logger.info("Detecting mine markers in annotation image")

    # 다양한 색상 범위로 마커 감지
    markers = []

    # HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 빨간색 마커 감지 (일반적인 기뢰 표시)
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 노란색 마커 감지
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # 초록색 마커 감지
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 파란색 마커 감지
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # 모든 마스크 결합
    combined_mask = cv2.bitwise_or(red_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50:  # 최소 면적 임계값
            # 중심점 계산
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 경계 상자
                x, y, w, h = cv2.boundingRect(contour)

                markers.append({
                    'id': i + 1,
                    'center': (cx, cy),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })

    logger.info(f"Detected {len(markers)} potential mine markers")
    return markers


def load_gps_data():
    """Location_MDGPS.xlsx에서 GPS 데이터 로드"""
    logger.info("Loading GPS data from Location_MDGPS.xlsx")

    gps_file = Path("datasets/Location_MDGPS.xlsx")

    if not gps_file.exists():
        logger.error(f"GPS file not found: {gps_file}")
        return None

    try:
        gps_parser = GPSParser()
        mine_locations = gps_parser.parse_gps_file(gps_file)
        validation = gps_parser.validate_coordinates(mine_locations)

        logger.info(f"Loaded {len(mine_locations)} mine locations")
        logger.info(f"Valid coordinates: {validation['valid_count']}/{validation['total_count']}")

        # GPS 범위 계산
        lats = [loc['latitude'] for loc in mine_locations]
        lons = [loc['longitude'] for loc in mine_locations]

        gps_bounds = {
            'lat_min': min(lats),
            'lat_max': max(lats),
            'lon_min': min(lons),
            'lon_max': max(lons),
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons)
        }

        return {
            'locations': mine_locations,
            'bounds': gps_bounds,
            'count': len(mine_locations),
            'validation': validation
        }

    except Exception as e:
        logger.error(f"Error loading GPS data: {e}")
        return None


def analyze_spatial_correspondence(annotation_data, gps_data):
    """Annotation 이미지와 GPS 데이터의 공간적 대응 관계 분석"""
    logger.info("Analyzing spatial correspondence between annotation and GPS data")

    if not annotation_data or not gps_data:
        logger.error("Missing annotation or GPS data")
        return None

    analysis = {
        'annotation_markers': annotation_data['marker_count'],
        'gps_locations': gps_data['count'],
        'count_match': False,
        'spatial_analysis': {},
        'correspondence_score': 0.0
    }

    # 1. 개수 비교
    marker_count = annotation_data['marker_count']
    gps_count = gps_data['count']

    logger.info(f"Marker count comparison: Annotation({marker_count}) vs GPS({gps_count})")

    if marker_count == gps_count:
        analysis['count_match'] = True
        logger.info("✅ Marker count matches GPS location count")
    else:
        logger.warning(f"❌ Count mismatch: {abs(marker_count - gps_count)} difference")

    # 2. 공간 분포 분석
    if annotation_data['mine_markers']:
        # Annotation 마커들의 분포 분석
        marker_positions = [marker['center'] for marker in annotation_data['mine_markers']]
        marker_x = [pos[0] for pos in marker_positions]
        marker_y = [pos[1] for pos in marker_positions]

        annotation_distribution = {
            'x_range': (min(marker_x), max(marker_x)),
            'y_range': (min(marker_y), max(marker_y)),
            'x_mean': np.mean(marker_x),
            'y_mean': np.mean(marker_y),
            'x_std': np.std(marker_x),
            'y_std': np.std(marker_y)
        }

        # GPS 위치들의 분포 분석
        gps_lats = [loc['latitude'] for loc in gps_data['locations']]
        gps_lons = [loc['longitude'] for loc in gps_data['locations']]

        gps_distribution = {
            'lat_range': (min(gps_lats), max(gps_lats)),
            'lon_range': (min(gps_lons), max(gps_lons)),
            'lat_mean': np.mean(gps_lats),
            'lon_mean': np.mean(gps_lons),
            'lat_std': np.std(gps_lats),
            'lon_std': np.std(gps_lons)
        }

        analysis['spatial_analysis'] = {
            'annotation_distribution': annotation_distribution,
            'gps_distribution': gps_distribution
        }

        # 3. 상대적 위치 패턴 비교
        if marker_count > 1 and gps_count > 1:
            # 가장 가까운 마커 쌍들 간의 거리 비교
            annotation_distances = calculate_pairwise_distances(marker_positions)
            gps_distances = calculate_gps_pairwise_distances(gps_data['locations'])

            # 거리 분포의 유사성 평가
            distance_correlation = compare_distance_distributions(annotation_distances, gps_distances)
            analysis['distance_correlation'] = distance_correlation

    # 4. 종합 대응 점수 계산
    correspondence_score = 0.0

    # 개수 일치 점수 (30%)
    if analysis['count_match']:
        correspondence_score += 0.3

    # 거리 분포 유사성 점수 (70%)
    if 'distance_correlation' in analysis:
        correspondence_score += 0.7 * analysis['distance_correlation']

    analysis['correspondence_score'] = correspondence_score

    logger.info(f"Spatial correspondence score: {correspondence_score:.3f}")

    return analysis


def calculate_pairwise_distances(positions):
    """위치들 간의 쌍별 거리 계산 (픽셀 단위)"""
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
    return distances


def calculate_gps_pairwise_distances(locations):
    """GPS 위치들 간의 쌍별 거리 계산 (미터 단위)"""
    distances = []
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            lat1, lon1 = locations[i]['latitude'], locations[i]['longitude']
            lat2, lon2 = locations[j]['latitude'], locations[j]['longitude']

            # Haversine 공식으로 거리 계산
            lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
            lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
            distance_m = distance_km * 1000

            distances.append(distance_m)

    return distances


def compare_distance_distributions(pixel_distances, gps_distances):
    """픽셀 거리와 GPS 거리 분포의 유사성 비교"""
    if not pixel_distances or not gps_distances:
        return 0.0

    # 정규화
    pixel_distances = np.array(pixel_distances)
    gps_distances = np.array(gps_distances)

    # Z-score 정규화
    if len(pixel_distances) > 1:
        pixel_norm = (pixel_distances - np.mean(pixel_distances)) / np.std(pixel_distances)
    else:
        pixel_norm = pixel_distances

    if len(gps_distances) > 1:
        gps_norm = (gps_distances - np.mean(gps_distances)) / np.std(gps_distances)
    else:
        gps_norm = gps_distances

    # 분포가 다른 경우 길이 맞추기
    min_len = min(len(pixel_norm), len(gps_norm))
    if min_len > 0:
        pixel_norm = pixel_norm[:min_len]
        gps_norm = gps_norm[:min_len]

        # 피어슨 상관계수 계산
        correlation = np.corrcoef(pixel_norm, gps_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        return max(0.0, correlation)  # 음수 상관관계는 0으로 처리

    return 0.0


def create_correspondence_visualization(annotation_data, gps_data, analysis_result):
    """대응 관계 시각화"""
    logger.info("Creating correspondence visualization")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # 1. Annotation 이미지와 마커
    if annotation_data:
        ax1.imshow(annotation_data['image'])

        # 감지된 마커 표시
        for i, marker in enumerate(annotation_data['mine_markers']):
            cx, cy = marker['center']
            ax1.plot(cx, cy, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax1.text(cx + 10, cy - 10, f'M{i+1}', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

        ax1.set_title(f'PH_annotation.bmp\n감지된 마커: {annotation_data["marker_count"]}개')
        ax1.axis('off')

    # 2. GPS 위치 분포
    if gps_data:
        lats = [loc['latitude'] for loc in gps_data['locations']]
        lons = [loc['longitude'] for loc in gps_data['locations']]

        ax2.scatter(lons, lats, c='blue', s=100, marker='x', alpha=0.8, linewidth=3)

        # 위치 번호 표시
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            ax2.text(lon + 0.000005, lat + 0.000005, f'G{i+1}', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7, color='white'))

        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'GPS 기뢰 위치\n총 {gps_data["count"]}개 위치')
        ax2.grid(True, alpha=0.3)

    # 3. 대응 관계 분석 결과
    if analysis_result:
        labels = ['개수 일치', '공간 분포\n유사성', '종합 점수']

        count_score = 1.0 if analysis_result['count_match'] else 0.0
        spatial_score = analysis_result.get('distance_correlation', 0.0)
        total_score = analysis_result['correspondence_score']

        scores = [count_score, spatial_score, total_score]
        colors = ['green' if s > 0.5 else 'orange' if s > 0.3 else 'red' for s in scores]

        bars = ax3.bar(labels, scores, color=colors, alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('점수')
        ax3.set_title('대응 관계 분석 점수')

        # 점수 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. 결론 요약
    ax4.axis('off')

    if analysis_result:
        conclusion_text = []
        conclusion_text.append("📊 Location_MDGPS vs PH_annotation 분석 결과")
        conclusion_text.append("")
        conclusion_text.append(f"마커 개수: Annotation({analysis_result['annotation_markers']}) vs GPS({analysis_result['gps_locations']})")
        conclusion_text.append(f"개수 일치: {'✅ 일치' if analysis_result['count_match'] else '❌ 불일치'}")
        conclusion_text.append("")
        conclusion_text.append(f"종합 대응 점수: {analysis_result['correspondence_score']:.3f}")
        conclusion_text.append("")

        if analysis_result['correspondence_score'] > 0.7:
            conclusion_text.append("✅ 결론: 같은 위치를 나타내는 것으로 판단")
        elif analysis_result['correspondence_score'] > 0.4:
            conclusion_text.append("⚠️ 결론: 부분적 대응 관계 존재")
        else:
            conclusion_text.append("❌ 결론: 다른 위치를 나타내는 것으로 판단")

        ax4.text(0.1, 0.9, '\n'.join(conclusion_text), transform=ax4.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    output_file = Path("analysis_results/location_annotation_verification/correspondence_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correspondence visualization to: {output_file}")


def generate_verification_report(annotation_data, gps_data, analysis_result):
    """검증 분석 보고서 생성"""
    logger.info("Generating verification analysis report")

    report_lines = []
    report_lines.append("# Location_MDGPS vs PH_annotation 대응 관계 검증 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # 분석 목적
    report_lines.append("## 🎯 **분석 목적**")
    report_lines.append("Location_MDGPS.xlsx의 기뢰 위치 데이터와 PH_annotation.bmp가 같은 위치를 나타내는지 검증")
    report_lines.append("GPS 좌표와 annotation 이미지의 마커들 간의 공간적 대응 관계 분석")
    report_lines.append("")

    # 데이터 개요
    report_lines.append("## 📊 **분석 데이터 개요**")
    report_lines.append("")

    if gps_data:
        report_lines.append("### GPS 데이터 (Location_MDGPS.xlsx)")
        report_lines.append(f"- **총 기뢰 위치**: {gps_data['count']}개")
        report_lines.append(f"- **유효 좌표**: {gps_data['validation']['valid_count']}/{gps_data['validation']['total_count']}")

        bounds = gps_data['bounds']
        report_lines.append(f"- **위도 범위**: [{bounds['lat_min']:.6f}°, {bounds['lat_max']:.6f}°]")
        report_lines.append(f"- **경도 범위**: [{bounds['lon_min']:.6f}°, {bounds['lon_max']:.6f}°]")
        report_lines.append(f"- **중심점**: ({bounds['center_lat']:.6f}°, {bounds['center_lon']:.6f}°)")
        report_lines.append("")

    if annotation_data:
        report_lines.append("### Annotation 이미지 (PH_annotation.bmp)")
        report_lines.append(f"- **이미지 크기**: {annotation_data['shape'][1]}×{annotation_data['shape'][0]} 픽셀")
        report_lines.append(f"- **감지된 마커**: {annotation_data['marker_count']}개")
        report_lines.append(f"- **파일 경로**: `{annotation_data['path']}`")

        if annotation_data['mine_markers']:
            report_lines.append("- **마커 세부 정보**:")
            for i, marker in enumerate(annotation_data['mine_markers'][:10]):  # 최대 10개만 표시
                cx, cy = marker['center']
                area = marker['area']
                report_lines.append(f"  - 마커 {i+1}: 위치({cx}, {cy}), 면적({area:.0f} 픽셀)")

            if len(annotation_data['mine_markers']) > 10:
                report_lines.append(f"  - ... 외 {len(annotation_data['mine_markers']) - 10}개 마커")

        report_lines.append("")

    # 분석 결과
    if analysis_result:
        report_lines.append("## 📈 **분석 결과**")
        report_lines.append("")

        # 개수 비교
        report_lines.append("### 1. 개수 비교")
        report_lines.append(f"- **Annotation 마커**: {analysis_result['annotation_markers']}개")
        report_lines.append(f"- **GPS 위치**: {analysis_result['gps_locations']}개")
        report_lines.append(f"- **개수 일치**: {'✅ 일치' if analysis_result['count_match'] else '❌ 불일치'}")

        if not analysis_result['count_match']:
            diff = abs(analysis_result['annotation_markers'] - analysis_result['gps_locations'])
            report_lines.append(f"- **차이**: {diff}개")

        report_lines.append("")

        # 공간 분포 분석
        if 'spatial_analysis' in analysis_result:
            spatial = analysis_result['spatial_analysis']

            report_lines.append("### 2. 공간 분포 분석")

            if 'annotation_distribution' in spatial:
                ann_dist = spatial['annotation_distribution']
                report_lines.append("**Annotation 마커 분포**:")
                report_lines.append(f"- X 범위: [{ann_dist['x_range'][0]:.0f}, {ann_dist['x_range'][1]:.0f}] 픽셀")
                report_lines.append(f"- Y 범위: [{ann_dist['y_range'][0]:.0f}, {ann_dist['y_range'][1]:.0f}] 픽셀")
                report_lines.append(f"- 중심점: ({ann_dist['x_mean']:.0f}, {ann_dist['y_mean']:.0f}) 픽셀")
                report_lines.append("")

            if 'gps_distribution' in spatial:
                gps_dist = spatial['gps_distribution']
                report_lines.append("**GPS 위치 분포**:")
                report_lines.append(f"- 위도 범위: [{gps_dist['lat_range'][0]:.6f}°, {gps_dist['lat_range'][1]:.6f}°]")
                report_lines.append(f"- 경도 범위: [{gps_dist['lon_range'][0]:.6f}°, {gps_dist['lon_range'][1]:.6f}°]")
                report_lines.append(f"- 중심점: ({gps_dist['lat_mean']:.6f}°, {gps_dist['lon_mean']:.6f}°)")
                report_lines.append("")

        # 거리 분포 유사성
        if 'distance_correlation' in analysis_result:
            corr = analysis_result['distance_correlation']
            report_lines.append("### 3. 거리 분포 유사성")
            report_lines.append(f"- **상관계수**: {corr:.3f}")

            if corr > 0.7:
                report_lines.append("- **평가**: ✅ 높은 유사성 (동일 패턴)")
            elif corr > 0.4:
                report_lines.append("- **평가**: ⚠️ 중간 유사성 (부분적 일치)")
            else:
                report_lines.append("- **평가**: ❌ 낮은 유사성 (다른 패턴)")

            report_lines.append("")

        # 종합 점수
        score = analysis_result['correspondence_score']
        report_lines.append("### 4. 종합 대응 점수")
        report_lines.append(f"- **점수**: {score:.3f} / 1.000")
        report_lines.append("")
        report_lines.append("**점수 구성**:")
        report_lines.append("- 개수 일치 (30%)")
        report_lines.append("- 공간 분포 유사성 (70%)")
        report_lines.append("")

    # 최종 결론
    report_lines.append("## 🎯 **최종 결론**")
    report_lines.append("")

    if analysis_result:
        score = analysis_result['correspondence_score']

        if score > 0.7:
            report_lines.append("### ✅ **같은 위치를 나타내는 것으로 판단됩니다**")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 종합 대응 점수: {score:.3f} (임계값 0.7 초과)")
            if analysis_result['count_match']:
                report_lines.append("- 마커 개수 완전 일치")
            if 'distance_correlation' in analysis_result and analysis_result['distance_correlation'] > 0.4:
                report_lines.append("- 공간 분포 패턴 유사성 확인")
            report_lines.append("")
            report_lines.append("**의미**:")
            report_lines.append("- Location_MDGPS.xlsx와 PH_annotation.bmp는 동일한 지역의 기뢰 위치 정보")
            report_lines.append("- GPS 좌표를 이용한 정확한 위치 매핑 가능")
            report_lines.append("- PH_annotation.bmp를 기준으로 한 기뢰 탐지 모델 훈련 타당성 확보")

        elif score > 0.4:
            report_lines.append("### ⚠️ **부분적 대응 관계가 존재합니다**")
            report_lines.append("")
            report_lines.append("**분석**:")
            report_lines.append(f"- 종합 대응 점수: {score:.3f} (중간 수준)")
            report_lines.append("- 완전한 일치는 아니지만 상당한 연관성 존재")
            report_lines.append("")
            report_lines.append("**가능한 원인**:")
            report_lines.append("- 일부 기뢰 위치의 annotation 누락 또는 추가")
            report_lines.append("- 마커 감지 알고리즘의 한계")
            report_lines.append("- 서로 다른 시점의 데이터일 가능성")

        else:
            report_lines.append("### ❌ **다른 위치를 나타내는 것으로 판단됩니다**")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 종합 대응 점수: {score:.3f} (임계값 0.4 미달)")
            if not analysis_result['count_match']:
                report_lines.append("- 마커 개수 불일치")
            if 'distance_correlation' in analysis_result and analysis_result['distance_correlation'] <= 0.4:
                report_lines.append("- 공간 분포 패턴 상이")
            report_lines.append("")
            report_lines.append("**의미**:")
            report_lines.append("- Location_MDGPS.xlsx와 PH_annotation.bmp는 서로 다른 지역 또는 다른 데이터")
            report_lines.append("- 이전 XTF-GPS 좌표 분석과 일치하는 결과")
            report_lines.append("- 전체적으로 데이터 간 불일치 문제 확인")

    # 기술적 세부사항
    report_lines.append("")
    report_lines.append("## 🛠️ **기술적 세부사항**")
    report_lines.append("")
    report_lines.append("**마커 감지 방법**:")
    report_lines.append("- HSV 색상 공간에서 빨강, 노랑, 초록, 파랑 색상 범위 기반 감지")
    report_lines.append("- 형태학적 연산을 통한 노이즈 제거")
    report_lines.append("- 최소 면적 임계값(50 픽셀) 적용")
    report_lines.append("")
    report_lines.append("**대응 관계 분석**:")
    report_lines.append("- 개수 일치성: 마커 수와 GPS 위치 수 비교")
    report_lines.append("- 공간 분포 유사성: 정규화된 거리 분포의 피어슨 상관계수")
    report_lines.append("- 종합 점수: 개수 일치(30%) + 공간 분포 유사성(70%)")

    # 보고서 저장
    output_file = Path("analysis_results/location_annotation_verification/LOCATION_ANNOTATION_VERIFICATION_REPORT.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved verification report to: {output_file}")

    # JSON 데이터 저장
    json_data = {
        'annotation_data': {
            'marker_count': annotation_data['marker_count'] if annotation_data else 0,
            'shape': annotation_data['shape'] if annotation_data else None,
            'path': annotation_data['path'] if annotation_data else None
        },
        'gps_data': {
            'count': gps_data['count'] if gps_data else 0,
            'bounds': gps_data['bounds'] if gps_data else None,
            'validation': gps_data['validation'] if gps_data else None
        },
        'analysis_result': analysis_result,
        'analysis_timestamp': datetime.now().isoformat()
    }

    json_file = Path("analysis_results/location_annotation_verification/verification_data.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info(f"Saved verification data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Location_MDGPS vs PH_annotation Verification")

    try:
        # Annotation 이미지 분석
        annotation_data = load_and_analyze_annotation_image()

        # GPS 데이터 로드
        gps_data = load_gps_data()

        if not annotation_data and not gps_data:
            logger.error("Failed to load both annotation and GPS data")
            return 1

        # 대응 관계 분석
        analysis_result = analyze_spatial_correspondence(annotation_data, gps_data)

        # 시각화 생성
        create_correspondence_visualization(annotation_data, gps_data, analysis_result)

        # 보고서 생성
        generate_verification_report(annotation_data, gps_data, analysis_result)

        # 결과 요약 출력
        print("\n" + "="*80)
        print("Location_MDGPS vs PH_annotation 대응 관계 검증 결과")
        print("="*80)

        if annotation_data:
            print(f"📁 PH_annotation.bmp: {annotation_data['shape'][1]}×{annotation_data['shape'][0]}")
            print(f"   감지된 마커: {annotation_data['marker_count']}개")

        if gps_data:
            print(f"🗺️ Location_MDGPS.xlsx: {gps_data['count']}개 GPS 위치")
            print(f"   유효 좌표: {gps_data['validation']['valid_count']}/{gps_data['validation']['total_count']}")

        if analysis_result:
            print(f"\n📊 분석 결과:")
            print(f"   개수 일치: {'✅ 일치' if analysis_result['count_match'] else '❌ 불일치'}")
            if 'distance_correlation' in analysis_result:
                print(f"   공간 분포 유사성: {analysis_result['distance_correlation']:.3f}")
            print(f"   종합 대응 점수: {analysis_result['correspondence_score']:.3f}")

            score = analysis_result['correspondence_score']
            print(f"\n🎯 **최종 결론**:")
            if score > 0.7:
                print("   ✅ Location_MDGPS와 PH_annotation은 **같은 위치**를 나타냅니다!")
                print("   → GPS 좌표와 annotation 이미지가 동일한 지역의 기뢰 위치 정보")
                print("   → 정확한 위치 매핑 및 모델 훈련 가능")
            elif score > 0.4:
                print("   ⚠️ **부분적 대응 관계**가 존재합니다")
                print("   → 완전한 일치는 아니지만 상당한 연관성 확인")
                print("   → 추가 검증 필요")
            else:
                print("   ❌ Location_MDGPS와 PH_annotation은 **다른 위치**를 나타냅니다")
                print("   → 이전 XTF-GPS 분석과 일치하는 데이터 불일치 문제")
                print("   → 전체적인 데이터 정합성 재검토 필요")

        return 0

    except Exception as e:
        logger.error(f"Verification analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())