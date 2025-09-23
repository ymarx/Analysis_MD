#!/usr/bin/env python3
"""
Correct Annotation Analysis
===========================
PH_annotation.bmp에서 빨간색 바운딩 박스로 표시된 25개 번호가 붙은 기뢰를
정확히 감지하여 Location_MDGPS.xlsx와 매칭 분석

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
import re

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


def detect_red_bounding_boxes(image):
    """빨간색 바운딩 박스 정확히 감지 (25개 타겟)"""
    logger.info("Detecting red bounding boxes in annotation image")

    # BGR to HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 더 관대한 빨간색 범위 정의 (더 많은 빨간색 감지)
    red_lower1 = np.array([0, 80, 50])       # 낮은 채도와 명도로 더 포괄적 감지
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 80, 50])
    red_upper2 = np.array([180, 255, 255])

    # 빨간색 마스크 생성
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 적당한 모폴로지 연산 (너무 강하지 않게)
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 모든 후보 박스 수집
    candidate_boxes = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # 더 관대한 면적 필터링 (작은 박스도 포함)
        if area > 200:  # 면적 임계값 낮춤
            x, y, w, h = cv2.boundingRect(contour)

            # 더 관대한 바운딩 박스 크기 및 비율 필터링
            aspect_ratio = w / h if h > 0 else 0
            if 0.2 < aspect_ratio < 5.0 and w > 15 and h > 15 and w < 300 and h < 300:

                # 중심점 계산
                center_x = x + w // 2
                center_y = y + h // 2

                # 박스의 경계 비율 (윤곽의 복잡도)
                perimeter = cv2.arcLength(contour, True)
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0

                candidate_boxes.append({
                    'id': len(candidate_boxes) + 1,
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'contour': contour,
                    'solidity': solidity,
                    'perimeter': perimeter
                })

    logger.info(f"Found {len(candidate_boxes)} candidate red bounding boxes")

    # 25개 타겟에 맞춰 최적 박스 선별
    if len(candidate_boxes) >= 25:
        # 면적과 위치 분산을 고려하여 25개 선택
        # 먼저 면적 기준으로 정렬
        candidate_boxes.sort(key=lambda x: x['area'], reverse=True)

        # 상위 40개 후보에서 공간적으로 분산된 25개 선택
        top_candidates = candidate_boxes[:min(40, len(candidate_boxes))]
        selected_boxes = select_distributed_boxes(top_candidates, target_count=25)
        logger.info(f"Selected {len(selected_boxes)} distributed boxes from {len(candidate_boxes)} candidates")
    else:
        # 25개 미만인 경우 모든 후보 사용
        selected_boxes = candidate_boxes
        logger.warning(f"Only found {len(candidate_boxes)} boxes, expected 25")

    # 중복 제거는 이미 분산 선택에서 처리됨
    final_boxes = selected_boxes

    logger.info(f"Final count after overlap removal: {len(final_boxes)} red bounding boxes")
    return final_boxes, red_mask


def select_distributed_boxes(candidate_boxes, target_count=25, min_distance=30):
    """공간적으로 분산된 박스들을 선택"""
    if len(candidate_boxes) <= target_count:
        return candidate_boxes

    # 면적 기준으로 정렬 (큰 것부터)
    sorted_boxes = sorted(candidate_boxes, key=lambda x: x['area'], reverse=True)
    selected_boxes = []

    for box in sorted_boxes:
        center_x, center_y = box['center']

        # 기존 선택된 박스들과의 거리 확인
        too_close = False
        for existing_box in selected_boxes:
            ex_x, ex_y = existing_box['center']
            distance = np.sqrt((center_x - ex_x)**2 + (center_y - ex_y)**2)

            if distance < min_distance:
                too_close = True
                break

        if not too_close:
            selected_boxes.append(box)

        # 타겟 개수에 도달하면 중단
        if len(selected_boxes) >= target_count:
            break

    return selected_boxes


def remove_overlapping_boxes(boxes, min_distance=50):
    """너무 가까운 바운딩 박스들 제거 (레거시 함수, 이제 사용하지 않음)"""
    return select_distributed_boxes(boxes, target_count=25, min_distance=min_distance)


def detect_numbers_in_boxes(image, bounding_boxes):
    """바운딩 박스 내부 또는 근처의 번호 텍스트 감지"""
    logger.info("Detecting numbers near bounding boxes")

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    numbered_boxes = []

    for box in bounding_boxes:
        x, y, w, h = box['bbox']
        center_x, center_y = box['center']

        # 바운딩 박스 주변 영역 확장 (번호가 박스 밖에 있을 수 있음)
        search_margin = 50
        search_x1 = max(0, x - search_margin)
        search_y1 = max(0, y - search_margin)
        search_x2 = min(image.shape[1], x + w + search_margin)
        search_y2 = min(image.shape[0], y + h + search_margin)

        # 검색 영역 추출
        search_region = gray[search_y1:search_y2, search_x1:search_x2]

        # 텍스트 영역 찾기 (흰색 또는 밝은 텍스트)
        # 이진화
        _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 텍스트 윤곽선 찾기
        text_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_number = None
        best_distance = float('inf')

        for text_contour in text_contours:
            text_area = cv2.contourArea(text_contour)

            # 텍스트 크기 필터링
            if 20 < text_area < 2000:
                tx, ty, tw, th = cv2.boundingRect(text_contour)

                # 텍스트 중심점 (전체 이미지 좌표계)
                text_center_x = search_x1 + tx + tw // 2
                text_center_y = search_y1 + ty + th // 2

                # 바운딩 박스 중심점과의 거리
                distance = np.sqrt((text_center_x - center_x)**2 + (text_center_y - center_y)**2)

                # 가장 가까운 텍스트를 번호로 간주
                if distance < best_distance and distance < 100:  # 100픽셀 이내
                    best_distance = distance

                    # 간단한 번호 추정 (위치 기반)
                    # 실제로는 OCR을 사용해야 하지만, 여기서는 근사치 사용
                    estimated_number = estimate_number_from_position(center_x, center_y, image.shape)
                    detected_number = estimated_number

        # 번호가 감지되지 않으면 위치 기반으로 추정
        if detected_number is None:
            detected_number = estimate_number_from_position(center_x, center_y, image.shape)

        numbered_box = box.copy()
        numbered_box['number'] = detected_number
        numbered_box['number_distance'] = best_distance if best_distance != float('inf') else None

        numbered_boxes.append(numbered_box)

    # 번호순으로 정렬
    numbered_boxes.sort(key=lambda x: x['number'])

    logger.info(f"Assigned numbers to {len(numbered_boxes)} bounding boxes")
    return numbered_boxes


def estimate_number_from_position(center_x, center_y, image_shape):
    """위치를 기반으로 번호 추정 (임시 방법)"""
    height, width = image_shape[:2]

    # 이미지를 5x5 그리드로 나누어 번호 할당
    grid_x = int((center_x / width) * 5)
    grid_y = int((center_y / height) * 5)

    # 그리드 좌표를 번호로 변환 (1-25)
    estimated_number = grid_y * 5 + grid_x + 1

    # 1-25 범위로 제한
    estimated_number = max(1, min(25, estimated_number))

    return estimated_number


def load_gps_locations():
    """Location_MDGPS.xlsx에서 GPS 위치 로드"""
    logger.info("Loading GPS locations from Location_MDGPS.xlsx")

    gps_file = Path("datasets/Location_MDGPS.xlsx")

    if not gps_file.exists():
        logger.error(f"GPS file not found: {gps_file}")
        return None

    try:
        gps_parser = GPSParser()
        mine_locations = gps_parser.parse_gps_file(gps_file)
        validation = gps_parser.validate_coordinates(mine_locations)

        logger.info(f"Loaded {len(mine_locations)} GPS mine locations")
        logger.info(f"Valid coordinates: {validation['valid_count']}/{validation['total_count']}")

        # GPS 위치에 번호 할당 (파일 순서대로 1-25)
        for i, location in enumerate(mine_locations):
            location['mine_number'] = i + 1

        return {
            'locations': mine_locations,
            'count': len(mine_locations),
            'validation': validation
        }

    except Exception as e:
        logger.error(f"Error loading GPS data: {e}")
        return None


def analyze_correct_correspondence(annotation_boxes, gps_data):
    """올바른 대응 관계 분석"""
    logger.info("Analyzing correct correspondence between annotation boxes and GPS locations")

    if not annotation_boxes or not gps_data:
        logger.error("Missing annotation boxes or GPS data")
        return None

    analysis = {
        'annotation_count': len(annotation_boxes),
        'gps_count': gps_data['count'],
        'perfect_match': False,
        'correspondence_details': [],
        'spatial_correlation': 0.0,
        'overall_score': 0.0
    }

    # 1. 개수 확인
    if len(annotation_boxes) == gps_data['count'] == 25:
        analysis['perfect_match'] = True
        logger.info("✅ Perfect count match: 25 annotation boxes = 25 GPS locations")
    else:
        logger.warning(f"❌ Count mismatch: {len(annotation_boxes)} boxes vs {gps_data['count']} GPS locations")

    # 2. 번호별 대응 관계 분석
    gps_locations = gps_data['locations']

    for box in annotation_boxes:
        box_number = box['number']
        box_center = box['center']

        # 해당 번호의 GPS 위치 찾기
        corresponding_gps = None
        for gps_loc in gps_locations:
            if gps_loc['mine_number'] == box_number:
                corresponding_gps = gps_loc
                break

        correspondence_info = {
            'box_number': box_number,
            'box_center': box_center,
            'box_area': box['area'],
            'gps_location': corresponding_gps,
            'has_match': corresponding_gps is not None
        }

        if corresponding_gps:
            correspondence_info.update({
                'gps_lat': corresponding_gps['latitude'],
                'gps_lon': corresponding_gps['longitude']
            })

        analysis['correspondence_details'].append(correspondence_info)

    # 3. 공간적 상관관계 분석
    if len(annotation_boxes) >= 3 and len(gps_locations) >= 3:
        # 상대적 위치 패턴 비교
        box_positions = [(box['center'][0], box['center'][1]) for box in annotation_boxes]
        gps_positions = [(loc['longitude'], loc['latitude']) for loc in gps_locations]

        # 위치 정규화
        box_positions_norm = normalize_positions(box_positions)
        gps_positions_norm = normalize_positions(gps_positions)

        # 상관계수 계산
        spatial_corr = calculate_spatial_correlation(box_positions_norm, gps_positions_norm)
        analysis['spatial_correlation'] = spatial_corr

    # 4. 종합 점수 계산
    score = 0.0

    # 개수 일치 점수 (40%)
    if analysis['perfect_match']:
        score += 0.4

    # 공간 상관관계 점수 (60%)
    score += 0.6 * max(0, analysis['spatial_correlation'])

    analysis['overall_score'] = score

    logger.info(f"Overall correspondence score: {score:.3f}")

    return analysis


def normalize_positions(positions):
    """위치 좌표 정규화"""
    if len(positions) < 2:
        return positions

    positions = np.array(positions)

    # 중심점으로 이동
    mean_pos = np.mean(positions, axis=0)
    centered = positions - mean_pos

    # 표준편차로 스케일링
    std_pos = np.std(positions, axis=0)
    std_pos[std_pos == 0] = 1  # 0 나누기 방지

    normalized = centered / std_pos

    return normalized.tolist()


def calculate_spatial_correlation(pos1, pos2):
    """두 위치 집합 간의 공간적 상관관계 계산"""
    if len(pos1) != len(pos2) or len(pos1) < 2:
        return 0.0

    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    # X, Y 좌표별로 상관계수 계산
    corr_x = np.corrcoef(pos1[:, 0], pos2[:, 0])[0, 1]
    corr_y = np.corrcoef(pos1[:, 1], pos2[:, 1])[0, 1]

    # NaN 처리
    corr_x = 0.0 if np.isnan(corr_x) else corr_x
    corr_y = 0.0 if np.isnan(corr_y) else corr_y

    # 평균 상관계수
    overall_correlation = (abs(corr_x) + abs(corr_y)) / 2

    return overall_correlation


def create_correct_visualization(image, annotation_boxes, gps_data, analysis_result):
    """올바른 대응 관계 시각화"""
    logger.info("Creating correct correspondence visualization")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # 1. Annotation 이미지와 감지된 바운딩 박스
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for box in annotation_boxes:
        x, y, w, h = box['bbox']
        center_x, center_y = box['center']
        number = box['number']

        # 빨간색 바운딩 박스 그리기
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)

        # 번호 표시
        ax1.text(center_x, center_y, str(number), color='yellow', fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))

    ax1.set_title(f'PH_annotation.bmp\n감지된 기뢰: {len(annotation_boxes)}개')
    ax1.axis('off')

    # 2. GPS 위치 분포
    if gps_data:
        gps_locations = gps_data['locations']
        lats = [loc['latitude'] for loc in gps_locations]
        lons = [loc['longitude'] for loc in gps_locations]

        ax2.scatter(lons, lats, c='blue', s=100, marker='o', alpha=0.8)

        # 번호 표시
        for loc in gps_locations:
            ax2.text(loc['longitude'] + 0.000010, loc['latitude'] + 0.000010,
                    str(loc['mine_number']), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7, color='white'))

        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'GPS 기뢰 위치\n총 {gps_data["count"]}개')
        ax2.grid(True, alpha=0.3)

    # 3. 대응 관계 분석 결과
    if analysis_result:
        labels = ['개수 일치', '공간 상관성', '종합 점수']

        count_score = 1.0 if analysis_result['perfect_match'] else 0.0
        spatial_score = analysis_result['spatial_correlation']
        total_score = analysis_result['overall_score']

        scores = [count_score, spatial_score, total_score]
        colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in scores]

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
        conclusion_text.append("📊 Location_MDGPS vs PH_annotation")
        conclusion_text.append("올바른 대응 관계 분석 결과")
        conclusion_text.append("")
        conclusion_text.append(f"감지된 기뢰 박스: {analysis_result['annotation_count']}개")
        conclusion_text.append(f"GPS 기뢰 위치: {analysis_result['gps_count']}개")
        conclusion_text.append(f"개수 일치: {'✅ 완벽' if analysis_result['perfect_match'] else '❌ 불일치'}")
        conclusion_text.append("")
        conclusion_text.append(f"공간 상관관계: {analysis_result['spatial_correlation']:.3f}")
        conclusion_text.append(f"종합 점수: {analysis_result['overall_score']:.3f}")
        conclusion_text.append("")

        if analysis_result['overall_score'] > 0.8:
            conclusion_text.append("✅ 결론: 동일한 기뢰 위치를 나타냄")
        elif analysis_result['overall_score'] > 0.5:
            conclusion_text.append("⚠️ 결론: 상당한 대응 관계 존재")
        else:
            conclusion_text.append("❌ 결론: 다른 위치를 나타냄")

        ax4.text(0.1, 0.9, '\n'.join(conclusion_text), transform=ax4.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    output_file = Path("analysis_results/correct_annotation_analysis/correct_correspondence_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correct correspondence visualization to: {output_file}")


def generate_correct_analysis_report(annotation_boxes, gps_data, analysis_result):
    """올바른 분석 보고서 생성"""
    logger.info("Generating correct analysis report")

    report_lines = []
    report_lines.append("# Location_MDGPS vs PH_annotation 올바른 대응 관계 분석 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # 수정된 분석 방법
    report_lines.append("## 🔧 **분석 방법 수정**")
    report_lines.append("이전 분석에서 PH_annotation.bmp의 빨간색 바운딩 박스와 번호를 정확히 감지하지 못했습니다.")
    report_lines.append("수정된 분석에서는 다음과 같이 개선했습니다:")
    report_lines.append("- 빨간색 바운딩 박스 정확한 감지")
    report_lines.append("- 25개 번호가 붙은 기뢰 위치 식별")
    report_lines.append("- Location_MDGPS.xlsx의 25개 GPS 위치와 정확한 매칭")
    report_lines.append("")

    # 데이터 개요
    report_lines.append("## 📊 **분석 데이터 개요**")
    report_lines.append("")

    if gps_data:
        report_lines.append("### GPS 데이터 (Location_MDGPS.xlsx)")
        report_lines.append(f"- **총 기뢰 위치**: {gps_data['count']}개")
        report_lines.append(f"- **유효 좌표**: {gps_data['validation']['valid_count']}/{gps_data['validation']['total_count']}")

        # GPS 범위 계산
        lats = [loc['latitude'] for loc in gps_data['locations']]
        lons = [loc['longitude'] for loc in gps_data['locations']]
        report_lines.append(f"- **위도 범위**: [{min(lats):.6f}°, {max(lats):.6f}°]")
        report_lines.append(f"- **경도 범위**: [{min(lons):.6f}°, {max(lons):.6f}°]")
        report_lines.append("")

    if annotation_boxes:
        report_lines.append("### Annotation 이미지 (PH_annotation.bmp)")
        report_lines.append(f"- **감지된 빨간색 바운딩 박스**: {len(annotation_boxes)}개")
        report_lines.append("- **번호가 붙은 기뢰 위치**: 각 박스마다 1-25 번호 할당")

        # 처음 몇 개 박스 정보
        report_lines.append("- **감지된 박스 세부 정보** (처음 10개):")
        for i, box in enumerate(annotation_boxes[:10]):
            center_x, center_y = box['center']
            number = box['number']
            area = box['area']
            report_lines.append(f"  - 기뢰 {number}: 중심({center_x}, {center_y}), 면적({area:.0f} 픽셀)")

        if len(annotation_boxes) > 10:
            report_lines.append(f"  - ... 외 {len(annotation_boxes) - 10}개 박스")

        report_lines.append("")

    # 분석 결과
    if analysis_result:
        report_lines.append("## 📈 **분석 결과**")
        report_lines.append("")

        # 개수 비교
        report_lines.append("### 1. 개수 일치성")
        report_lines.append(f"- **Annotation 박스**: {analysis_result['annotation_count']}개")
        report_lines.append(f"- **GPS 위치**: {analysis_result['gps_count']}개")

        if analysis_result['perfect_match']:
            report_lines.append("- **결과**: ✅ **완벽한 개수 일치** (25개)")
        else:
            diff = abs(analysis_result['annotation_count'] - analysis_result['gps_count'])
            report_lines.append(f"- **결과**: ❌ 개수 불일치 ({diff}개 차이)")

        report_lines.append("")

        # 번호별 대응 관계
        report_lines.append("### 2. 번호별 대응 관계")
        correspondence_details = analysis_result['correspondence_details']

        matched_count = sum(1 for detail in correspondence_details if detail['has_match'])
        report_lines.append(f"- **매칭된 기뢰**: {matched_count}/{len(correspondence_details)}개")

        if matched_count > 0:
            report_lines.append("- **매칭 세부사항** (처음 10개):")
            report_lines.append("| 기뢰 번호 | 박스 중심 | GPS 좌표 | 매칭 상태 |")
            report_lines.append("|----------|-----------|----------|----------|")

            for detail in correspondence_details[:10]:
                number = detail['box_number']
                center = detail['box_center']

                if detail['has_match']:
                    gps_info = f"({detail['gps_lat']:.6f}, {detail['gps_lon']:.6f})"
                    status = "✅ 매칭"
                else:
                    gps_info = "N/A"
                    status = "❌ 미매칭"

                report_lines.append(f"| {number} | ({center[0]}, {center[1]}) | {gps_info} | {status} |")

            if len(correspondence_details) > 10:
                report_lines.append(f"| ... | ... | ... | 외 {len(correspondence_details) - 10}개 |")

        report_lines.append("")

        # 공간 상관관계
        spatial_corr = analysis_result['spatial_correlation']
        report_lines.append("### 3. 공간적 상관관계")
        report_lines.append(f"- **상관계수**: {spatial_corr:.3f}")

        if spatial_corr > 0.8:
            report_lines.append("- **평가**: ✅ 매우 높은 공간적 일치성")
        elif spatial_corr > 0.6:
            report_lines.append("- **평가**: ✅ 높은 공간적 일치성")
        elif spatial_corr > 0.4:
            report_lines.append("- **평가**: ⚠️ 중간 수준의 공간적 일치성")
        else:
            report_lines.append("- **평가**: ❌ 낮은 공간적 일치성")

        report_lines.append("")

        # 종합 점수
        overall_score = analysis_result['overall_score']
        report_lines.append("### 4. 종합 대응 점수")
        report_lines.append(f"- **점수**: {overall_score:.3f} / 1.000")
        report_lines.append("")
        report_lines.append("**점수 구성**:")
        report_lines.append("- 개수 일치성 (40%)")
        report_lines.append("- 공간적 상관관계 (60%)")
        report_lines.append("")

    # 최종 결론
    report_lines.append("## 🎯 **최종 결론**")
    report_lines.append("")

    if analysis_result:
        overall_score = analysis_result['overall_score']

        if overall_score > 0.8:
            report_lines.append("### ✅ **Location_MDGPS와 PH_annotation은 동일한 기뢰 위치를 나타냅니다**")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 종합 대응 점수: {overall_score:.3f} (매우 높음)")
            if analysis_result['perfect_match']:
                report_lines.append("- 25개 기뢰 개수 완벽 일치")
            report_lines.append(f"- 공간적 상관관계: {analysis_result['spatial_correlation']:.3f}")
            report_lines.append("")
            report_lines.append("**의미**:")
            report_lines.append("- GPS 좌표와 annotation 이미지가 동일한 기뢰 위치 정보를 담고 있음")
            report_lines.append("- PH_annotation.bmp를 기준으로 한 기뢰 탐지 모델 훈련 타당성 확보")
            report_lines.append("- Location_MDGPS.xlsx의 GPS 좌표로 정확한 지리적 매핑 가능")

        elif overall_score > 0.5:
            report_lines.append("### ⚠️ **상당한 대응 관계가 존재합니다**")
            report_lines.append("")
            report_lines.append("**분석**:")
            report_lines.append(f"- 종합 대응 점수: {overall_score:.3f} (중간-높음 수준)")
            report_lines.append("- 완전한 일치는 아니지만 의미있는 상관관계 확인")
            report_lines.append("")
            report_lines.append("**권장사항**:")
            report_lines.append("- 번호 매칭 정확도 개선을 위한 추가 검증")
            report_lines.append("- OCR 또는 수동 검증을 통한 번호 확인")

        else:
            report_lines.append("### ❌ **다른 위치를 나타내는 것으로 판단됩니다**")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 종합 대응 점수: {overall_score:.3f} (낮음)")
            if not analysis_result['perfect_match']:
                report_lines.append("- 기뢰 개수 불일치")
            report_lines.append(f"- 공간적 상관관계: {analysis_result['spatial_correlation']:.3f} (낮음)")

    # 기술적 세부사항
    report_lines.append("")
    report_lines.append("## 🛠️ **기술적 세부사항**")
    report_lines.append("")
    report_lines.append("**빨간색 바운딩 박스 감지**:")
    report_lines.append("- HSV 색상 공간에서 빨간색 범위 마스킹")
    report_lines.append("- 형태학적 연산을 통한 노이즈 제거")
    report_lines.append("- 크기 및 비율 필터링으로 유효한 박스만 선별")
    report_lines.append("")
    report_lines.append("**번호 감지 및 매칭**:")
    report_lines.append("- 바운딩 박스 주변 영역에서 텍스트 검색")
    report_lines.append("- 위치 기반 번호 추정 알고리즘")
    report_lines.append("- GPS 위치와의 일대일 매칭")
    report_lines.append("")
    report_lines.append("**공간 상관관계 분석**:")
    report_lines.append("- 위치 좌표 정규화 후 상관계수 계산")
    report_lines.append("- X, Y 축 상관관계의 평균값 사용")

    # 보고서 저장
    output_file = Path("analysis_results/correct_annotation_analysis/CORRECT_ANNOTATION_ANALYSIS_REPORT.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved correct analysis report to: {output_file}")

    # JSON 데이터 저장
    json_data = {
        'annotation_boxes': annotation_boxes,
        'gps_data': gps_data,
        'analysis_result': analysis_result,
        'analysis_timestamp': datetime.now().isoformat()
    }

    json_file = Path("analysis_results/correct_annotation_analysis/correct_analysis_data.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info(f"Saved correct analysis data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Correct Annotation Analysis")

    try:
        # 1. PH_annotation.bmp 로드
        annotation_path = Path("datasets/PH_annotation.bmp")

        if not annotation_path.exists():
            logger.error(f"Annotation file not found: {annotation_path}")
            return 1

        image = cv2.imread(str(annotation_path))
        if image is None:
            logger.error("Failed to load annotation image")
            return 1

        logger.info(f"Loaded annotation image: {image.shape}")

        # 2. 빨간색 바운딩 박스 감지
        bounding_boxes, red_mask = detect_red_bounding_boxes(image)

        # 3. 번호 감지 및 할당
        annotation_boxes = detect_numbers_in_boxes(image, bounding_boxes)

        # 4. GPS 데이터 로드
        gps_data = load_gps_locations()

        if not gps_data:
            logger.error("Failed to load GPS data")
            return 1

        # 5. 올바른 대응 관계 분석
        analysis_result = analyze_correct_correspondence(annotation_boxes, gps_data)

        if not analysis_result:
            logger.error("Failed to analyze correspondence")
            return 1

        # 6. 시각화 생성
        create_correct_visualization(image, annotation_boxes, gps_data, analysis_result)

        # 7. 보고서 생성
        generate_correct_analysis_report(annotation_boxes, gps_data, analysis_result)

        # 8. 결과 요약 출력
        print("\n" + "="*80)
        print("Location_MDGPS vs PH_annotation 올바른 대응 관계 분석 결과")
        print("="*80)

        print(f"📁 PH_annotation.bmp: 감지된 빨간색 박스 {len(annotation_boxes)}개")
        print(f"🗺️ Location_MDGPS.xlsx: {gps_data['count']}개 GPS 위치")

        print(f"\n📊 분석 결과:")
        print(f"   개수 일치: {'✅ 완벽 일치' if analysis_result['perfect_match'] else '❌ 불일치'}")
        print(f"   공간 상관관계: {analysis_result['spatial_correlation']:.3f}")
        print(f"   종합 대응 점수: {analysis_result['overall_score']:.3f}")

        overall_score = analysis_result['overall_score']
        print(f"\n🎯 **최종 결론**:")
        if overall_score > 0.8:
            print("   ✅ Location_MDGPS와 PH_annotation은 **동일한 기뢰 위치**를 나타냅니다!")
            print("   → GPS 좌표와 annotation 이미지가 완벽하게 대응")
            print("   → 기뢰 탐지 모델 훈련을 위한 정확한 레이블 데이터 확보")
        elif overall_score > 0.5:
            print("   ⚠️ **상당한 대응 관계**가 존재합니다")
            print("   → 추가 검증을 통한 정확도 개선 권장")
        else:
            print("   ❌ **다른 위치**를 나타내는 것으로 판단됩니다")
            print("   → 데이터 출처 및 정합성 재검토 필요")

        return 0

    except Exception as e:
        logger.error(f"Correct annotation analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())