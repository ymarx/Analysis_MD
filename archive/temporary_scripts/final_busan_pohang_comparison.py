#!/usr/bin/env python3
"""
부산-포항 위치 데이터 최종 비교 분석

목적: 올바른 컬럼으로 부산과 포항 데이터 비교
"""

import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_coordinate(coord_str):
    """도분초 좌표를 십진도로 변환"""

    if pd.isna(coord_str) or coord_str == '':
        return None

    coord_str = str(coord_str).strip()

    # 패턴 1: 이미 십진도 형태 (36.5933983 N)
    pattern1 = r'^(\d+\.?\d*)\s*([NSEW])$'
    match1 = re.match(pattern1, coord_str)
    if match1:
        value = float(match1.group(1))
        direction = match1.group(2)
        if direction in ['S', 'W']:
            value = -value
        return value

    # 패턴 2: 도 분 형태 (129 30.557773 E)
    pattern2 = r'^(\d+)\s+(\d+\.?\d*)\s*([NSEW])$'
    match2 = re.match(pattern2, coord_str)
    if match2:
        degrees = float(match2.group(1))
        minutes = float(match2.group(2))
        direction = match2.group(3)

        value = degrees + minutes / 60.0
        if direction in ['S', 'W']:
            value = -value
        return value

    # 패턴 3: 도 분 초 형태 (129 30 33.4641 E)
    pattern3 = r'^(\d+)\s+(\d+)\s+(\d+\.?\d*)\s*([NSEW])$'
    match3 = re.match(pattern3, coord_str)
    if match3:
        degrees = float(match3.group(1))
        minutes = float(match3.group(2))
        seconds = float(match3.group(3))
        direction = match3.group(4)

        value = degrees + minutes / 60.0 + seconds / 3600.0
        if direction in ['S', 'W']:
            value = -value
        return value

    return None

def load_pohang_data():
    """포항 Location_MDGPS 데이터 로드"""

    print("📍 포항 Location_MDGPS 데이터 로드 중...")

    try:
        df = pd.read_excel("datasets/Location_MDGPS.xlsx")

        # 좌표 변환
        df['위도_십진도'] = df['위도'].apply(parse_coordinate)
        df['경도_십진도'] = df['경도'].apply(parse_coordinate)

        valid_coords = df.dropna(subset=['위도_십진도', '경도_십진도'])

        print(f"   포항 데이터: {len(valid_coords)}개 좌표")
        print(f"   위도 범위: {valid_coords['위도_십진도'].min():.6f} ~ {valid_coords['위도_십진도'].max():.6f}")
        print(f"   경도 범위: {valid_coords['경도_십진도'].min():.6f} ~ {valid_coords['경도_십진도'].max():.6f}")

        # 중심점
        center_lat = valid_coords['위도_십진도'].mean()
        center_lon = valid_coords['경도_십진도'].mean()
        print(f"   중심점: {center_lat:.6f}, {center_lon:.6f}")

        return valid_coords[['정점', '위도_십진도', '경도_십진도']].copy()

    except Exception as e:
        print(f"❌ 포항 데이터 로드 실패: {e}")
        return None

def load_busan_data():
    """부산 위치자료 데이터 로드 (올바른 컬럼 사용)"""

    print("\n📍 부산 위치자료 데이터 로드 중...")

    busan_file = "[샘플]데이터/[위치]부산위치자료-도분초-위경도변환.xlsx"

    try:
        # 전체 데이터 읽기
        df = pd.read_excel(busan_file)

        # Unnamed: 7이 위도, Unnamed: 8이 경도
        lat_col = 'Unnamed: 7'
        lon_col = 'Unnamed: 8'

        # 숫자 데이터로 변환
        df['위도_십진도'] = pd.to_numeric(df[lat_col], errors='coerce')
        df['경도_십진도'] = pd.to_numeric(df[lon_col], errors='coerce')

        # 유효한 좌표만 필터링 (위도 30-40, 경도 120-140 범위)
        valid_coords = df[
            (df['위도_십진도'] >= 30) & (df['위도_십진도'] <= 40) &
            (df['경도_십진도'] >= 120) & (df['경도_십진도'] <= 140)
        ].dropna(subset=['위도_십진도', '경도_십진도'])

        print(f"   부산 데이터: {len(valid_coords)}개 좌표")
        print(f"   위도 범위: {valid_coords['위도_십진도'].min():.6f} ~ {valid_coords['위도_십진도'].max():.6f}")
        print(f"   경도 범위: {valid_coords['경도_십진도'].min():.6f} ~ {valid_coords['경도_십진도'].max():.6f}")

        # 중심점
        center_lat = valid_coords['위도_십진도'].mean()
        center_lon = valid_coords['경도_십진도'].mean()
        print(f"   중심점: {center_lat:.6f}, {center_lon:.6f}")

        # 정점명 생성 (BS_01, BS_02, ...)
        valid_coords = valid_coords.reset_index(drop=True)
        valid_coords['정점'] = 'BS_' + (valid_coords.index + 1).astype(str).str.zfill(2)

        return valid_coords[['정점', '위도_십진도', '경도_십진도']].copy()

    except Exception as e:
        print(f"❌ 부산 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_overlap_and_distance(pohang_data, busan_data):
    """두 데이터셋 간 중복도 및 거리 분석"""

    print(f"\n🔍 거리 분석 및 중복도 계산 중...")

    # 중심점 간 거리
    ph_center = (pohang_data['위도_십진도'].mean(), pohang_data['경도_십진도'].mean())
    bs_center = (busan_data['위도_십진도'].mean(), busan_data['경도_십진도'].mean())
    center_distance = geodesic(ph_center, bs_center).kilometers

    print(f"   데이터셋 중심점 간 거리: {center_distance:.1f} km")

    # 최단 거리 계산 (각 포항 점에 대해 가장 가까운 부산 점)
    min_distances = []

    for _, ph_row in pohang_data.iterrows():
        ph_point = (ph_row['위도_십진도'], ph_row['경도_십진도'])

        distances_to_busan = []
        for _, bs_row in busan_data.iterrows():
            bs_point = (bs_row['위도_십진도'], bs_row['경도_십진도'])
            distance = geodesic(ph_point, bs_point).kilometers
            distances_to_busan.append(distance)

        min_distance = min(distances_to_busan)
        min_distances.append(min_distance)

    # 거리 통계
    distance_stats = {
        'center_distance': center_distance,
        'min_distance': min(min_distances),
        'max_distance': max(min_distances),
        'mean_distance': np.mean(min_distances),
        'median_distance': np.median(min_distances),
        'std_distance': np.std(min_distances)
    }

    print(f"\n📊 거리 통계:")
    print(f"   중심점 간 거리: {distance_stats['center_distance']:.1f} km")
    print(f"   최단 거리: {distance_stats['min_distance']:.1f} km")
    print(f"   최장 거리: {distance_stats['max_distance']:.1f} km")
    print(f"   평균 거리: {distance_stats['mean_distance']:.1f} km")
    print(f"   중간값: {distance_stats['median_distance']:.1f} km")

    # 중복도 분석 (거리 기준)
    overlap_thresholds = [5, 10, 20, 50, 100]
    overlap_analysis = {}

    print(f"\n🎯 중복도 분석:")
    for threshold in overlap_thresholds:
        overlap_count = len([d for d in min_distances if d <= threshold])
        overlap_percentage = (overlap_count / len(min_distances)) * 100
        overlap_analysis[threshold] = {
            'count': overlap_count,
            'percentage': overlap_percentage
        }
        print(f"   {threshold}km 이내: {overlap_count}개 ({overlap_percentage:.1f}%)")

    return distance_stats, overlap_analysis, min_distances

def determine_regional_relationship(pohang_data, busan_data, distance_stats):
    """지역적 관계 판단"""

    print(f"\n🗺️ 지역적 관계 분석:")

    # 포항 데이터 위치 분석
    ph_center_lat = pohang_data['위도_십진도'].mean()
    ph_center_lon = pohang_data['경도_십진도'].mean()

    # 부산 데이터 위치 분석
    bs_center_lat = busan_data['위도_십진도'].mean()
    bs_center_lon = busan_data['경도_십진도'].mean()

    # 지역 판단
    def get_region_name(lat, lon):
        if lat >= 36.0 and lon >= 129.0:
            return "포항 해역 (경북 동해안)"
        elif 35.0 <= lat < 36.0 and lon >= 129.0:
            return "부산/울산 해역 (경남 동해안)"
        elif 35.0 <= lat < 36.0 and 128.0 <= lon < 129.0:
            return "부산 근해 (남해 동부)"
        elif lat < 35.0 and 127.0 <= lon <= 129.0:
            return "남해 중부"
        else:
            return "기타 해역"

    ph_region = get_region_name(ph_center_lat, ph_center_lon)
    bs_region = get_region_name(bs_center_lat, bs_center_lon)

    print(f"   포항 데이터 지역: {ph_region}")
    print(f"   부산 데이터 지역: {bs_region}")

    # 관계 판단
    center_distance = distance_stats['center_distance']
    mean_distance = distance_stats['mean_distance']

    if center_distance < 50:
        relationship = "인접 해역"
        explanation = "두 데이터는 인접한 해역에 위치"
    elif center_distance < 150:
        relationship = "같은 해역권"
        explanation = "동일한 해역권 내 서로 다른 지점들"
    else:
        relationship = "별개 해역"
        explanation = "지리적으로 분리된 서로 다른 해역"

    print(f"   관계: {relationship} ({center_distance:.1f}km 거리)")
    print(f"   설명: {explanation}")

    return {
        'pohang_region': ph_region,
        'busan_region': bs_region,
        'relationship': relationship,
        'explanation': explanation,
        'center_distance': center_distance
    }

def generate_final_report(pohang_data, busan_data, distance_stats, overlap_analysis, regional_info):
    """최종 비교 보고서 생성"""

    from pathlib import Path
    import json

    # 출력 디렉토리 생성
    output_dir = Path("analysis_results/final_busan_pohang_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 보고서 생성
    report_file = output_dir / "FINAL_BUSAN_POHANG_COMPARISON_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 부산-포항 위치 데이터 최종 비교 분석 보고서
**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석자**: YMARX

## 🎯 **분석 목적**
Location_MDGPS(포항)와 부산위치자료의 지리적 관계 및 중복도 최종 분석

## 📊 **데이터 개요**

### 포항 데이터 (Location_MDGPS)
- **좌표 개수**: {len(pohang_data)}개
- **위치**: {regional_info['pohang_region']}
- **위도 범위**: {pohang_data['위도_십진도'].min():.6f}° ~ {pohang_data['위도_십진도'].max():.6f}°N
- **경도 범위**: {pohang_data['경도_십진도'].min():.6f}° ~ {pohang_data['경도_십진도'].max():.6f}°E

### 부산 데이터
- **좌표 개수**: {len(busan_data)}개
- **위치**: {regional_info['busan_region']}
- **위도 범위**: {busan_data['위도_십진도'].min():.6f}° ~ {busan_data['위도_십진도'].max():.6f}°N
- **경도 범위**: {busan_data['경도_십진도'].min():.6f}° ~ {busan_data['경도_십진도'].max():.6f}°E

## 📏 **거리 분석 결과**

### 기본 거리 통계
- **중심점 간 거리**: {distance_stats['center_distance']:.1f} km
- **최단 거리**: {distance_stats['min_distance']:.1f} km
- **최장 거리**: {distance_stats['max_distance']:.1f} km
- **평균 거리**: {distance_stats['mean_distance']:.1f} km
- **중간값 거리**: {distance_stats['median_distance']:.1f} km

### 중복도 분석
""")

        for threshold, data in overlap_analysis.items():
            f.write(f"- **{threshold}km 이내**: {data['count']}개 ({data['percentage']:.1f}%)\n")

        f.write(f"""

## 🗺️ **지역적 관계**

### 위치 분석
- **포항 데이터**: {regional_info['pohang_region']}
- **부산 데이터**: {regional_info['busan_region']}
- **관계**: {regional_info['relationship']}

### 설명
{regional_info['explanation']}

## 🎯 **중복도 평가**

""")

        # 중복도 평가
        high_overlap = overlap_analysis.get(10, {'percentage': 0})['percentage']
        medium_overlap = overlap_analysis.get(50, {'percentage': 0})['percentage']

        if high_overlap > 50:
            overlap_level = "매우 높음"
            overlap_desc = "두 데이터가 거의 같은 지역을 대상으로 함"
        elif high_overlap > 20:
            overlap_level = "높음"
            overlap_desc = "상당 부분 중복되는 지역이 존재"
        elif medium_overlap > 50:
            overlap_level = "중간"
            overlap_desc = "부분적으로 겹치는 영역이 있음"
        elif medium_overlap > 20:
            overlap_level = "낮음"
            overlap_desc = "일부 인접한 지역이 있으나 대부분 별개"
        else:
            overlap_level = "거의 없음"
            overlap_desc = "완전히 서로 다른 지역의 데이터"

        f.write(f"""### 중복도 수준: {overlap_level}
**평가**: {overlap_desc}

**근거**:
- 10km 이내 중복: {high_overlap:.1f}%
- 50km 이내 중복: {medium_overlap:.1f}%
- 평균 거리: {distance_stats['mean_distance']:.1f}km

## 💡 **최종 결론**

### 지리적 관계
""")

        if distance_stats['center_distance'] < 100:
            f.write(f"""**인접 해역**: 두 데이터는 {distance_stats['center_distance']:.1f}km 떨어진 인접 해역에 위치합니다.

""")
        else:
            f.write(f"""**별개 해역**: 두 데이터는 {distance_stats['center_distance']:.1f}km 떨어진 서로 다른 해역에 위치합니다.

""")

        f.write(f"""### 데이터 특성
- **포항 데이터**: 포항 북동쪽 해상의 기뢰 관련 위치 정보
- **부산 데이터**: 부산 연안의 위치 기준점 정보
- **목적**: 서로 다른 조사/작업 목적으로 수집된 독립적 데이터

### 결론
Location_MDGPS(포항)와 부산위치자료는 **{regional_info['relationship'].lower()}**의 관계에 있으며,
**{overlap_level.lower()}의 중복도**를 보입니다. 이는 **서로 다른 목적으로 수집된 독립적인 해양 위치 데이터**임을 의미합니다.

""")

        if high_overlap < 10:
            f.write("두 데이터셋은 **지리적으로 분리된 별개의 조사 지역**을 다루고 있습니다.\n")
        elif high_overlap < 30:
            f.write("두 데이터셋은 **일부 인접한 지역을 포함**하지만 주로 다른 해역을 다룹니다.\n")
        else:
            f.write("두 데이터셋은 **상당 부분 중복되는 해역**을 다루고 있습니다.\n")

    print(f"\n📁 최종 보고서 저장: {report_file}")
    return report_file

def main():
    """메인 실행 함수"""

    print("="*70)
    print("부산-포항 위치 데이터 최종 비교 분석")
    print("="*70)

    try:
        # 1. 포항 데이터 로드
        pohang_data = load_pohang_data()
        if pohang_data is None:
            return False

        # 2. 부산 데이터 로드
        busan_data = load_busan_data()
        if busan_data is None:
            return False

        # 3. 거리 및 중복도 분석
        distance_stats, overlap_analysis, min_distances = analyze_overlap_and_distance(pohang_data, busan_data)

        # 4. 지역적 관계 분석
        regional_info = determine_regional_relationship(pohang_data, busan_data, distance_stats)

        # 5. 최종 보고서 생성
        report_file = generate_final_report(pohang_data, busan_data, distance_stats, overlap_analysis, regional_info)

        # 결과 요약 출력
        print(f"\n{'='*70}")
        print("🎯 최종 분석 결과")
        print(f"{'='*70}")

        print(f"\n📊 기본 정보:")
        print(f"   포항 데이터: {len(pohang_data)}개 좌표")
        print(f"   부산 데이터: {len(busan_data)}개 좌표")

        print(f"\n📏 거리 분석:")
        print(f"   중심점 간 거리: {distance_stats['center_distance']:.1f} km")
        print(f"   평균 거리: {distance_stats['mean_distance']:.1f} km")

        print(f"\n🎯 중복도 분석:")
        for threshold in [10, 50, 100]:
            if threshold in overlap_analysis:
                data = overlap_analysis[threshold]
                print(f"   {threshold}km 이내: {data['count']}개 ({data['percentage']:.1f}%)")

        print(f"\n🗺️ 지역적 관계:")
        print(f"   관계: {regional_info['relationship']}")
        print(f"   설명: {regional_info['explanation']}")

        return True

    except Exception as e:
        logger.error(f"최종 비교 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 부산-포항 위치 데이터 최종 비교 분석이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 최종 비교 분석 중 오류가 발생했습니다.")