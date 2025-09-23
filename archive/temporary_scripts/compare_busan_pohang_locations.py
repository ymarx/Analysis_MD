#!/usr/bin/env python3
"""
Location_MDGPS와 부산위치자료 비교 분석

목적: 포항 Location_MDGPS 위치와 부산 위치 자료의 중복도 및 거리 분석
"""

import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic
from datetime import datetime
import logging
import matplotlib.pyplot as plt

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

        return valid_coords[['정점', '위도_십진도', '경도_십진도']].copy()

    except Exception as e:
        print(f"❌ 포항 데이터 로드 실패: {e}")
        return None

def load_busan_data():
    """부산 위치자료 데이터 로드"""

    print("\n📍 부산 위치자료 데이터 로드 중...")

    busan_file = "[샘플]데이터/[위치]부산위치자료-도분초-위경도변환.xlsx"

    try:
        # Excel 파일 읽기 (시트가 여러 개 있을 수 있음)
        try:
            df = pd.read_excel(busan_file)
        except:
            # 첫 번째 시트만 읽기
            df = pd.read_excel(busan_file, sheet_name=0)

        print(f"   부산 데이터 기본 정보:")
        print(f"   행 수: {len(df)}")
        print(f"   컬럼: {list(df.columns)}")

        # 상위 몇 행 출력
        print(f"\n   데이터 샘플:")
        print(df.head().to_string())

        # 좌표 컬럼 찾기
        lat_columns = []
        lon_columns = []

        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['lat', '위도', 'latitude']):
                lat_columns.append(col)
            elif any(term in col_lower for term in ['lon', 'lng', '경도', 'longitude']):
                lon_columns.append(col)

        print(f"\n   좌표 컬럼:")
        print(f"   위도: {lat_columns}")
        print(f"   경도: {lon_columns}")

        # 좌표 변환
        if lat_columns and lon_columns:
            lat_col = lat_columns[0]
            lon_col = lon_columns[0]

            df['위도_십진도'] = df[lat_col].apply(parse_coordinate)
            df['경도_십진도'] = df[lon_col].apply(parse_coordinate)

            valid_coords = df.dropna(subset=['위도_십진도', '경도_십진도'])

            print(f"\n   변환 결과:")
            print(f"   성공적으로 변환된 좌표: {len(valid_coords)}/{len(df)}")

            if len(valid_coords) > 0:
                print(f"   위도 범위: {valid_coords['위도_십진도'].min():.6f} ~ {valid_coords['위도_십진도'].max():.6f}")
                print(f"   경도 범위: {valid_coords['경도_십진도'].min():.6f} ~ {valid_coords['경도_십진도'].max():.6f}")

                # 정점명 컬럼 찾기
                point_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(term in col_lower for term in ['정점', 'point', 'station', '번호', 'id']):
                        point_col = col
                        break

                if point_col:
                    return valid_coords[[point_col, '위도_십진도', '경도_십진도']].copy().rename(columns={point_col: '정점'})
                else:
                    # 인덱스를 정점명으로 사용
                    valid_coords['정점'] = 'BS_' + (valid_coords.index + 1).astype(str)
                    return valid_coords[['정점', '위도_십진도', '경도_십진도']].copy()

        return None

    except Exception as e:
        print(f"❌ 부산 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_distances_and_overlaps(pohang_data, busan_data):
    """두 데이터셋 간 거리 계산 및 중복 분석"""

    print(f"\n🔍 거리 분석 및 중복도 계산 중...")

    results = {
        'pohang_count': len(pohang_data),
        'busan_count': len(busan_data),
        'distance_matrix': [],
        'closest_pairs': [],
        'overlap_analysis': {},
        'regional_analysis': {}
    }

    # 각 포항 좌표에 대해 부산 좌표와의 최단 거리 계산
    for _, ph_row in pohang_data.iterrows():
        ph_point = (ph_row['위도_십진도'], ph_row['경도_십진도'])
        ph_name = ph_row['정점']

        min_distance = float('inf')
        closest_busan = None

        distances_to_busan = []

        for _, bs_row in busan_data.iterrows():
            bs_point = (bs_row['위도_십진도'], bs_row['경도_십진도'])
            bs_name = bs_row['정점']

            distance = geodesic(ph_point, bs_point).kilometers
            distances_to_busan.append({
                'pohang_point': ph_name,
                'busan_point': bs_name,
                'distance_km': distance,
                'pohang_coords': ph_point,
                'busan_coords': bs_point
            })

            if distance < min_distance:
                min_distance = distance
                closest_busan = bs_name

        results['distance_matrix'].extend(distances_to_busan)
        results['closest_pairs'].append({
            'pohang_point': ph_name,
            'closest_busan_point': closest_busan,
            'distance_km': min_distance,
            'pohang_coords': ph_point
        })

    # 거리 통계
    all_distances = [pair['distance_km'] for pair in results['closest_pairs']]

    results['distance_stats'] = {
        'min_distance': min(all_distances),
        'max_distance': max(all_distances),
        'mean_distance': np.mean(all_distances),
        'median_distance': np.median(all_distances),
        'std_distance': np.std(all_distances)
    }

    # 중복도 분석 (거리 기준)
    overlap_thresholds = [1, 5, 10, 20, 50, 100]
    for threshold in overlap_thresholds:
        overlap_count = len([d for d in all_distances if d <= threshold])
        overlap_percentage = (overlap_count / len(all_distances)) * 100
        results['overlap_analysis'][f'{threshold}km'] = {
            'count': overlap_count,
            'percentage': overlap_percentage
        }

    print(f"   거리 통계:")
    print(f"   최단 거리: {results['distance_stats']['min_distance']:.1f} km")
    print(f"   최장 거리: {results['distance_stats']['max_distance']:.1f} km")
    print(f"   평균 거리: {results['distance_stats']['mean_distance']:.1f} km")
    print(f"   중간 거리: {results['distance_stats']['median_distance']:.1f} km")

    return results

def analyze_regional_distribution(pohang_data, busan_data, results):
    """지역별 분포 분석"""

    print(f"\n🗺️ 지역별 분포 분석...")

    # 포항 데이터 중심점
    ph_center_lat = pohang_data['위도_십진도'].mean()
    ph_center_lon = pohang_data['경도_십진도'].mean()

    # 부산 데이터 중심점
    bs_center_lat = busan_data['위도_십진도'].mean()
    bs_center_lon = busan_data['경도_십진도'].mean()

    # 중심점 간 거리
    center_distance = geodesic((ph_center_lat, ph_center_lon), (bs_center_lat, bs_center_lon)).kilometers

    print(f"   포항 데이터 중심: {ph_center_lat:.6f}, {ph_center_lon:.6f}")
    print(f"   부산 데이터 중심: {bs_center_lat:.6f}, {bs_center_lon:.6f}")
    print(f"   중심점 간 거리: {center_distance:.1f} km")

    # 지역 판단
    def get_region(lat, lon):
        if lat >= 36.0 and lon >= 129.0:
            if lat >= 36.5:
                return "포항 북부 해역"
            else:
                return "포항 남부 해역"
        elif lat >= 35.0 and lat < 36.0 and lon >= 128.5:
            return "부산/울산 해역"
        elif lat >= 35.0 and lat < 36.0 and lon < 128.5:
            return "남해 서부"
        else:
            return "기타 해역"

    ph_region = get_region(ph_center_lat, ph_center_lon)
    bs_region = get_region(bs_center_lat, bs_center_lon)

    results['regional_analysis'] = {
        'pohang_center': (ph_center_lat, ph_center_lon),
        'busan_center': (bs_center_lat, bs_center_lon),
        'center_distance': center_distance,
        'pohang_region': ph_region,
        'busan_region': bs_region,
        'same_region': ph_region == bs_region
    }

    print(f"   포항 데이터 지역: {ph_region}")
    print(f"   부산 데이터 지역: {bs_region}")
    print(f"   동일 지역 여부: {'예' if results['regional_analysis']['same_region'] else '아니오'}")

    return results

def generate_comparison_report(results):
    """비교 분석 보고서 생성"""

    from pathlib import Path
    import json

    # 출력 디렉토리 생성
    output_dir = Path("analysis_results/busan_pohang_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 결과 저장 (거리 매트릭스는 너무 크므로 요약만)
    summary_results = results.copy()
    summary_results['distance_matrix'] = f"총 {len(results['distance_matrix'])}개 거리 계산"

    detail_file = output_dir / "busan_pohang_comparison_detail.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False, default=str)

    # 마크다운 보고서 생성
    report_file = output_dir / "BUSAN_POHANG_LOCATION_COMPARISON_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 부산-포항 위치 데이터 비교 분석 보고서
**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석자**: YMARX

## 🎯 **분석 목적**
Location_MDGPS(포항)와 부산위치자료의 지리적 중복도 및 관계 분석

## 📊 **데이터 개요**
- **포항 데이터**: {results['pohang_count']}개 좌표
- **부산 데이터**: {results['busan_count']}개 좌표
- **총 거리 계산**: {len(results['distance_matrix'])}회

## 📏 **거리 분석 결과**

### 통계 요약
- **최단 거리**: {results['distance_stats']['min_distance']:.1f} km
- **최장 거리**: {results['distance_stats']['max_distance']:.1f} km
- **평균 거리**: {results['distance_stats']['mean_distance']:.1f} km
- **중간값 거리**: {results['distance_stats']['median_distance']:.1f} km
- **표준편차**: {results['distance_stats']['std_distance']:.1f} km

### 거리별 중복도 분석
""")

        for threshold, data in results['overlap_analysis'].items():
            f.write(f"- **{threshold} 이내**: {data['count']}개 ({data['percentage']:.1f}%)\n")

        f.write(f"""

## 🗺️ **지역 분석**

### 중심점 위치
- **포항 데이터 중심**: {results['regional_analysis']['pohang_center'][0]:.6f}°N, {results['regional_analysis']['pohang_center'][1]:.6f}°E
- **부산 데이터 중심**: {results['regional_analysis']['busan_center'][0]:.6f}°N, {results['regional_analysis']['busan_center'][1]:.6f}°E
- **중심점 간 거리**: {results['regional_analysis']['center_distance']:.1f} km

### 지역 분류
- **포항 데이터**: {results['regional_analysis']['pohang_region']}
- **부산 데이터**: {results['regional_analysis']['busan_region']}
- **동일 지역**: {'예' if results['regional_analysis']['same_region'] else '아니오'}

## 🎯 **중복도 평가**

### 높은 중복도 (< 10km)
""")

        high_overlap = results['overlap_analysis']['10km']
        f.write(f"- **중복 포인트**: {high_overlap['count']}개 ({high_overlap['percentage']:.1f}%)\n")

        if high_overlap['percentage'] > 50:
            overlap_level = "매우 높음"
            explanation = "두 데이터셋이 거의 같은 지역을 대상으로 함"
        elif high_overlap['percentage'] > 20:
            overlap_level = "높음"
            explanation = "상당 부분 중복되는 지역이 존재"
        elif high_overlap['percentage'] > 5:
            overlap_level = "보통"
            explanation = "부분적으로 중복되는 영역이 있음"
        else:
            overlap_level = "낮음"
            explanation = "서로 다른 지역의 데이터"

        f.write(f"- **중복도 평가**: {overlap_level}\n")
        f.write(f"- **해석**: {explanation}\n")

        f.write(f"""

## 🔍 **가장 가까운 포인트들**

""")

        # 상위 5개 가장 가까운 쌍들
        closest_pairs = sorted(results['closest_pairs'], key=lambda x: x['distance_km'])[:5]
        for i, pair in enumerate(closest_pairs, 1):
            f.write(f"### {i}. {pair['pohang_point']} ↔ {pair['closest_busan_point']}\n")
            f.write(f"- **거리**: {pair['distance_km']:.1f} km\n")
            f.write(f"- **포항 좌표**: {pair['pohang_coords'][0]:.6f}, {pair['pohang_coords'][1]:.6f}\n\n")

        f.write(f"""

## 💡 **결론**

### 지리적 관계
""")

        if results['distance_stats']['mean_distance'] < 50:
            f.write("- **근접한 해역**: 두 데이터셋 모두 인접한 해역에 위치\n")
        elif results['distance_stats']['mean_distance'] < 200:
            f.write("- **같은 권역**: 동일한 해역 권역 내 서로 다른 지점\n")
        else:
            f.write("- **서로 다른 해역**: 지리적으로 분리된 별개 해역\n")

        f.write(f"""
### 데이터 특성
- **평균 거리 {results['distance_stats']['mean_distance']:.1f}km**: """)

        if results['distance_stats']['mean_distance'] < 20:
            f.write("거의 동일한 조사 구역\n")
        elif results['distance_stats']['mean_distance'] < 100:
            f.write("인접한 조사 구역들\n")
        else:
            f.write("서로 다른 조사 목적의 별개 구역\n")

        f.write(f"""
### 최종 판단
""")

        if high_overlap['percentage'] > 30:
            f.write("**높은 중복도**: 두 데이터는 상당 부분 중복되는 지역을 다룸\n")
        elif high_overlap['percentage'] > 10:
            f.write("**부분 중복**: 일부 지역에서 중복되는 조사 지점 존재\n")
        else:
            f.write("**별개 지역**: 서로 다른 목적의 독립적인 조사 지역\n")

    print(f"\n📁 보고서 저장: {report_file}")
    return report_file

def main():
    """메인 실행 함수"""

    print("="*70)
    print("부산-포항 위치 데이터 비교 분석")
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

        # 3. 거리 및 중복도 계산
        results = calculate_distances_and_overlaps(pohang_data, busan_data)

        # 4. 지역 분석
        results = analyze_regional_distribution(pohang_data, busan_data, results)

        # 5. 보고서 생성
        report_file = generate_comparison_report(results)

        # 결과 요약 출력
        print(f"\n{'='*70}")
        print("🎯 비교 분석 결과")
        print(f"{'='*70}")

        print(f"\n📊 기본 정보:")
        print(f"   포항 좌표: {results['pohang_count']}개")
        print(f"   부산 좌표: {results['busan_count']}개")

        print(f"\n📏 거리 분석:")
        print(f"   평균 거리: {results['distance_stats']['mean_distance']:.1f} km")
        print(f"   최단 거리: {results['distance_stats']['min_distance']:.1f} km")
        print(f"   최장 거리: {results['distance_stats']['max_distance']:.1f} km")

        print(f"\n🎯 중복도 분석:")
        for threshold in ['10km', '50km', '100km']:
            data = results['overlap_analysis'][threshold]
            print(f"   {threshold} 이내: {data['count']}개 ({data['percentage']:.1f}%)")

        print(f"\n🗺️ 지역 분석:")
        print(f"   중심점 간 거리: {results['regional_analysis']['center_distance']:.1f} km")
        print(f"   포항 지역: {results['regional_analysis']['pohang_region']}")
        print(f"   부산 지역: {results['regional_analysis']['busan_region']}")

        return True

    except Exception as e:
        logger.error(f"비교 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 부산-포항 위치 데이터 비교 분석이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 비교 분석 중 오류가 발생했습니다.")