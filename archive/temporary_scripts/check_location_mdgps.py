#!/usr/bin/env python3
"""
Location_MDGPS.xlsx 파일의 실제 위치 정보 분석

목적: Location_MDGPS의 좌표를 확인하고 어느 지역에 가까운지 분석
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_location_mdgps():
    """Location_MDGPS 파일 분석"""

    print("="*60)
    print("Location_MDGPS 위치 분석")
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
        print(f"\n📋 데이터 샘플 (상위 5행):")
        print(df.head().to_string())

        # 좌표 컬럼 찾기
        coordinate_columns = []
        lat_columns = []
        lon_columns = []

        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['lat', '위도']):
                lat_columns.append(col)
                coordinate_columns.append(col)
            elif any(term in col_lower for term in ['lon', 'lng', '경도']):
                lon_columns.append(col)
                coordinate_columns.append(col)
            elif any(term in col_lower for term in ['x', 'y', 'coordinate']):
                coordinate_columns.append(col)

        print(f"\n🗺️ 좌표 관련 컬럼:")
        print(f"   위도 컬럼: {lat_columns}")
        print(f"   경도 컬럼: {lon_columns}")
        print(f"   기타 좌표 컬럼: {coordinate_columns}")

        # 좌표 데이터 분석
        if lat_columns and lon_columns:
            lat_col = lat_columns[0]
            lon_col = lon_columns[0]

            # 숫자형으로 변환
            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

            # 유효한 좌표만 필터링
            valid_coords = df.dropna(subset=[lat_col, lon_col])

            if len(valid_coords) > 0:
                print(f"\n📍 좌표 통계:")
                print(f"   유효한 좌표 수: {len(valid_coords)}")
                print(f"   위도 범위: {valid_coords[lat_col].min():.6f} ~ {valid_coords[lat_col].max():.6f}")
                print(f"   경도 범위: {valid_coords[lon_col].min():.6f} ~ {valid_coords[lon_col].max():.6f}")
                print(f"   위도 평균: {valid_coords[lat_col].mean():.6f}")
                print(f"   경도 평균: {valid_coords[lon_col].mean():.6f}")

                # 중심점 계산
                center_lat = valid_coords[lat_col].mean()
                center_lon = valid_coords[lon_col].mean()

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
                    '통영': (34.8544, 128.4331)
                }

                location_point = (center_lat, center_lon)

                print(f"\n🏙️ 한국 주요 도시와의 거리:")
                distances = []

                for city, coords in korean_cities.items():
                    distance = geodesic(location_point, coords).kilometers
                    distances.append((city, distance))
                    print(f"   {city}: {distance:.1f} km")

                # 가장 가까운 도시들
                distances.sort(key=lambda x: x[1])
                print(f"\n🏆 가장 가까운 도시들:")
                for i, (city, distance) in enumerate(distances[:5], 1):
                    print(f"   {i}. {city}: {distance:.1f} km")

                # Original XTF 위치와 비교
                original_xtf_coords = (36.098, 129.515)  # 포항 근해
                distance_to_original = geodesic(location_point, original_xtf_coords).kilometers

                print(f"\n📏 Original XTF(포항 근해)와의 거리:")
                print(f"   거리: {distance_to_original:.1f} km")

                # 지역 판단
                print(f"\n🗺️ 지역 분석:")
                if center_lat >= 35.0 and center_lat <= 38.0 and center_lon >= 126.0 and center_lon <= 130.0:
                    if center_lat >= 36.5:
                        region = "한국 중부 지역 (경기/강원 일대)"
                    elif center_lat >= 35.5:
                        region = "한국 중남부 지역 (충청/경북 일대)"
                    else:
                        region = "한국 남부 지역 (경남/전라 일대)"

                    if center_lon >= 129.0:
                        region += " - 동해안 쪽"
                    elif center_lon <= 127.0:
                        region += " - 서해안 쪽"
                    else:
                        region += " - 내륙 지역"
                else:
                    region = "한국 밖 지역"

                print(f"   예상 지역: {region}")

                # 좌표 형식 판단
                if 30 <= center_lat <= 40 and 120 <= center_lon <= 135:
                    coord_format = "십진도 (WGS84)"
                elif center_lat > 100000:
                    coord_format = "UTM 좌표계"
                else:
                    coord_format = "불명"

                print(f"   좌표 형식: {coord_format}")

                return {
                    'center_coords': (center_lat, center_lon),
                    'closest_city': distances[0][0],
                    'closest_distance': distances[0][1],
                    'distance_to_original_xtf': distance_to_original,
                    'region': region,
                    'coordinate_format': coord_format,
                    'total_points': len(valid_coords)
                }

        # 다른 형태의 좌표 컬럼이 있는지 확인
        print(f"\n🔍 다른 좌표 형태 탐색:")
        for col in df.columns:
            sample_values = df[col].dropna().head(5).tolist()
            print(f"   {col}: {sample_values}")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

if __name__ == "__main__":
    result = analyze_location_mdgps()

    if result:
        print(f"\n{'='*60}")
        print("🎯 결론")
        print(f"{'='*60}")
        print(f"Location_MDGPS는 {result['closest_city']}에서 {result['closest_distance']:.1f}km 떨어진 곳에 위치")
        print(f"Original XTF(포항 근해)와는 {result['distance_to_original_xtf']:.1f}km 거리 차이")
        print(f"지역: {result['region']}")
    else:
        print("\n❌ Location_MDGPS 위치 분석을 완료할 수 없습니다.")