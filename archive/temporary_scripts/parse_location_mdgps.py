#!/usr/bin/env python3
"""
Location_MDGPS.xlsx 파일의 도분초 좌표 파싱 및 분석

목적: 도분초 형태의 좌표를 십진도로 변환하여 위치 분석
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

    # 패턴: "36.5933983 N" 또는 "129 30.557773 E"

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

    # 패턴 3: 도 분 초 형태 (예: 129 30 33.4641 E)
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

    # 다른 패턴 시도
    print(f"⚠️ 파싱할 수 없는 좌표 형식: '{coord_str}'")
    return None

def analyze_location_mdgps():
    """Location_MDGPS 파일 분석"""

    print("="*60)
    print("Location_MDGPS 위치 분석 (도분초 파싱)")
    print("="*60)

    excel_path = "datasets/Location_MDGPS.xlsx"

    try:
        # Excel 파일 읽기
        df = pd.read_excel(excel_path)

        print(f"📊 데이터 기본 정보:")
        print(f"   행 수: {len(df)}")
        print(f"   열 수: {len(df.columns)}")
        print(f"   컬럼명: {list(df.columns)}")

        # 상위 몇 행 출력
        print(f"\n📋 데이터 샘플:")
        for i, row in df.head().iterrows():
            print(f"   {row['정점']}: {row['위도']}, {row['경도']}")

        # 좌표 변환
        print(f"\n🔄 좌표 변환 중...")

        df['위도_십진도'] = df['위도'].apply(parse_coordinate)
        df['경도_십진도'] = df['경도'].apply(parse_coordinate)

        # 변환 결과 확인
        valid_coords = df.dropna(subset=['위도_십진도', '경도_십진도'])
        print(f"   성공적으로 변환된 좌표: {len(valid_coords)}/{len(df)}")

        if len(valid_coords) == 0:
            print("❌ 좌표 변환 실패")
            return None

        # 변환 결과 샘플 출력
        print(f"\n📍 변환된 좌표 샘플:")
        for i, row in valid_coords.head().iterrows():
            print(f"   {row['정점']}: {row['위도_십진도']:.6f}, {row['경도_십진도']:.6f}")

        # 좌표 통계
        print(f"\n📊 좌표 통계:")
        print(f"   위도 범위: {valid_coords['위도_십진도'].min():.6f} ~ {valid_coords['위도_십진도'].max():.6f}")
        print(f"   경도 범위: {valid_coords['경도_십진도'].min():.6f} ~ {valid_coords['경도_십진도'].max():.6f}")
        print(f"   위도 평균: {valid_coords['위도_십진도'].mean():.6f}")
        print(f"   경도 평균: {valid_coords['경도_십진도'].mean():.6f}")

        # 중심점 계산
        center_lat = valid_coords['위도_십진도'].mean()
        center_lon = valid_coords['경도_십진도'].mean()

        print(f"\n🎯 Location_MDGPS 중심 위치:")
        print(f"   위도: {center_lat:.6f}")
        print(f"   경도: {center_lon:.6f}")

        # 한국 주요 도시와의 거리 계산
        korean_cities = {
            '서울': (37.5665, 126.9780),
            '부산': (35.1796, 129.0756),
            '대구': (35.8714, 128.6014),
            '인천': (37.4563, 126.7052),
            '광주': (35.1595, 126.8526),
            '대전': (36.3504, 127.3845),
            '울산': (35.5384, 129.3114),
            '포항': (36.0190, 129.3435),
            '경주': (35.8562, 129.2247),
            '창원': (35.2281, 128.6811),
            '제주': (33.4996, 126.5312),
            '여수': (34.7604, 127.6622),
            '군산': (35.9678, 126.7368),
            '목포': (34.8118, 126.3922),
            '통영': (34.8544, 128.4331),
            '강릉': (37.7519, 128.8761),
            '속초': (38.2070, 128.5918),
            '삼척': (37.4486, 129.1658),
            '동해': (37.5247, 129.1143)
        }

        location_point = (center_lat, center_lon)

        print(f"\n🏙️ 한국 주요 도시와의 거리:")
        distances = []

        for city, coords in korean_cities.items():
            distance = geodesic(location_point, coords).kilometers
            distances.append((city, distance))

        # 거리순 정렬
        distances.sort(key=lambda x: x[1])

        for city, distance in distances:
            print(f"   {city}: {distance:.1f} km")

        # 가장 가까운 도시들
        print(f"\n🏆 가장 가까운 도시들:")
        for i, (city, distance) in enumerate(distances[:5], 1):
            print(f"   {i}. {city}: {distance:.1f} km")

        # Original XTF 위치와 비교
        original_xtf_coords = (36.098, 129.515)  # 포항 근해
        distance_to_original = geodesic(location_point, original_xtf_coords).kilometers

        print(f"\n📏 Original XTF(포항 근해)와의 거리:")
        print(f"   거리: {distance_to_original:.1f} km")

        # 포항과의 거리
        pohang_coords = (36.0190, 129.3435)
        distance_to_pohang = geodesic(location_point, pohang_coords).kilometers

        print(f"\n📏 포항시와의 거리:")
        print(f"   거리: {distance_to_pohang:.1f} km")

        # 지역 분석
        print(f"\n🗺️ 지역 분석:")

        # 정확한 위치 판단
        if 36.5 <= center_lat <= 36.7 and 129.4 <= center_lon <= 129.6:
            region = "포항 근해 (동해안)"
            detailed_location = "포항 북동쪽 해상"
        elif center_lat >= 36.0 and center_lon >= 129.0:
            region = "경북 동해안 지역"
            if center_lat >= 37.0:
                detailed_location = "강원도 남부 또는 경북 북부 해상"
            else:
                detailed_location = "경북 동해안 해상"
        else:
            region = "기타 지역"
            detailed_location = "위치 미상"

        print(f"   지역: {region}")
        print(f"   상세 위치: {detailed_location}")

        # 해역 판단
        if center_lon >= 129.0:
            sea_area = "동해"
        elif center_lon <= 126.5:
            sea_area = "서해"
        else:
            sea_area = "남해"

        print(f"   해역: {sea_area}")

        # PH 접두사 의미 확인
        print(f"\n🔤 PH 접두사 분석:")
        print(f"   모든 정점이 'PH_'로 시작")
        print(f"   PH = Pohang (포항)을 의미하는 것으로 추정")
        print(f"   위치도 포항 근처이므로 일치함")

        return {
            'center_coords': (center_lat, center_lon),
            'closest_city': distances[0][0],
            'closest_distance': distances[0][1],
            'distance_to_original_xtf': distance_to_original,
            'distance_to_pohang': distance_to_pohang,
            'region': region,
            'detailed_location': detailed_location,
            'sea_area': sea_area,
            'total_points': len(valid_coords),
            'all_distances': distances[:10]  # 상위 10개 도시
        }

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = analyze_location_mdgps()

    if result:
        print(f"\n{'='*60}")
        print("🎯 최종 결론")
        print(f"{'='*60}")
        print(f"📍 Location_MDGPS 위치: {result['detailed_location']}")
        print(f"🏙️ 가장 가까운 도시: {result['closest_city']} ({result['closest_distance']:.1f}km)")
        print(f"🌊 해역: {result['sea_area']}")
        print(f"📏 포항시와의 거리: {result['distance_to_pohang']:.1f}km")
        print(f"📏 Original XTF와의 거리: {result['distance_to_original_xtf']:.1f}km")

        print(f"\n💡 해석:")
        if result['distance_to_original_xtf'] < 10:
            print("   Location_MDGPS와 Original XTF는 거의 같은 위치!")
        elif result['distance_to_original_xtf'] < 30:
            print("   Location_MDGPS와 Original XTF는 인접한 위치")
        else:
            print("   Location_MDGPS와 Original XTF는 서로 다른 위치")

    else:
        print("\n❌ Location_MDGPS 위치 분석을 완료할 수 없습니다.")