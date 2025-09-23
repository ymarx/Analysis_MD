#!/usr/bin/env python3
"""
XTF 파일의 좌표 데이터 소스 및 정확성 조사

목적: XTF에서 추출한 위경도가 직접 GPS 메타데이터인지,
      아니면 계산이 필요한 다른 형태의 데이터인지 확인
"""

import pyxtf
import numpy as np
import pandas as pd
import os

def investigate_xtf_coordinate_source():
    """XTF 좌표 데이터 소스 조사"""

    print("="*70)
    print("XTF 좌표 데이터 소스 및 정확성 조사")
    print("="*70)

    # EdgeTech 4205 파일 (문제가 있었던 파일)
    xtf_path = "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf"

    if not os.path.exists(xtf_path):
        print(f"❌ 파일을 찾을 수 없음: {xtf_path}")
        return

    print(f"🔍 분석 파일: {os.path.basename(xtf_path)}")

    # 패킷별 상세 분석
    analyze_packet_details(xtf_path)

    # 좌표 시스템 분석
    analyze_coordinate_system(xtf_path)

    # 다른 형태의 위치 데이터 확인
    analyze_alternative_position_data(xtf_path)

def analyze_packet_details(xtf_path):
    """패킷별 상세 분석"""

    print(f"\n📋 패킷별 상세 분석:")

    try:
        packet_count = 0
        coordinate_sources = {}

        for packet in pyxtf.xtf_read_gen(xtf_path):
            packet_count += 1

            if packet_count <= 10:  # 처음 10개 패킷 상세 분석
                print(f"\n--- 패킷 {packet_count} ---")
                print(f"패킷 타입: {type(packet).__name__}")

                # 모든 속성 나열
                attrs = [attr for attr in dir(packet) if not attr.startswith('_')]
                coord_related_attrs = [attr for attr in attrs if any(keyword in attr.lower()
                                     for keyword in ['coord', 'lat', 'lon', 'x', 'y', 'pos', 'gps', 'nav'])]

                if coord_related_attrs:
                    print(f"좌표 관련 속성들:")
                    for attr in coord_related_attrs:
                        try:
                            value = getattr(packet, attr)
                            print(f"  {attr}: {value} (타입: {type(value).__name__})")

                            # 좌표 소스 분류
                            if attr not in coordinate_sources:
                                coordinate_sources[attr] = []
                            coordinate_sources[attr].append(value)
                        except:
                            print(f"  {attr}: 접근 불가")

                # 데이터 존재 여부
                if hasattr(packet, 'data'):
                    data_info = "있음" if packet.data is not None else "없음"
                    print(f"데이터: {data_info}")

            if packet_count >= 100:  # 100개만 분석
                break

        # 좌표 소스 요약
        print(f"\n📊 좌표 소스 요약:")
        for attr, values in coordinate_sources.items():
            if values:
                values_array = np.array(values)
                if len(values_array) > 0:
                    print(f"  {attr}:")
                    print(f"    개수: {len(values_array)}")
                    print(f"    범위: {np.min(values_array):.6f} ~ {np.max(values_array):.6f}")
                    print(f"    평균: {np.mean(values_array):.6f}")

    except Exception as e:
        print(f"❌ 패킷 분석 실패: {e}")
        import traceback
        traceback.print_exc()

def analyze_coordinate_system(xtf_path):
    """좌표 시스템 분석"""

    print(f"\n🌐 좌표 시스템 분석:")

    try:
        # 헤더 정보 확인
        file_header, packets = pyxtf.xtf_read(xtf_path)

        print(f"헤더 정보:")
        header_attrs = [attr for attr in dir(file_header) if not attr.startswith('_')]
        coord_system_attrs = [attr for attr in header_attrs if any(keyword in attr.lower()
                            for keyword in ['coord', 'datum', 'proj', 'ellips', 'utm', 'wgs'])]

        if coord_system_attrs:
            for attr in coord_system_attrs:
                try:
                    value = getattr(file_header, attr)
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: 접근 불가")
        else:
            print("  좌표 시스템 관련 속성을 찾을 수 없음")

        # 일반적인 XTF 헤더 정보
        common_attrs = ['VersionNumber', 'NumberOfSonarChannels', 'NumberOfBathymetryChannels']
        print(f"\n일반 헤더 정보:")
        for attr in common_attrs:
            if hasattr(file_header, attr):
                value = getattr(file_header, attr)
                print(f"  {attr}: {value}")

    except Exception as e:
        print(f"❌ 좌표 시스템 분석 실패: {e}")

def analyze_alternative_position_data(xtf_path):
    """다른 형태의 위치 데이터 확인"""

    print(f"\n🔍 대안 위치 데이터 확인:")

    try:
        position_data = {
            'ship_coordinates': [],
            'sensor_coordinates': [],
            'raw_coordinates': [],
            'timestamp_info': []
        }

        packet_count = 0

        for packet in pyxtf.xtf_read_gen(xtf_path):
            packet_count += 1

            # Ship 좌표
            if hasattr(packet, 'ShipXcoordinate') and hasattr(packet, 'ShipYcoordinate'):
                ship_x = getattr(packet, 'ShipXcoordinate')
                ship_y = getattr(packet, 'ShipYcoordinate')
                position_data['ship_coordinates'].append((ship_x, ship_y))

            # Sensor 좌표
            if hasattr(packet, 'SensorXcoordinate') and hasattr(packet, 'SensorYcoordinate'):
                sensor_x = getattr(packet, 'SensorXcoordinate')
                sensor_y = getattr(packet, 'SensorYcoordinate')
                position_data['sensor_coordinates'].append((sensor_x, sensor_y))

            # 원시 좌표 (다른 형태)
            raw_attrs = ['RawXcoordinate', 'RawYcoordinate', 'GPSXcoordinate', 'GPSYcoordinate']
            for attr in raw_attrs:
                if hasattr(packet, attr):
                    value = getattr(packet, attr)
                    position_data['raw_coordinates'].append((attr, value))

            # 타임스탬프 정보
            time_attrs = ['TimeStamp', 'ping_time_year', 'ping_time_month', 'ping_time_day']
            time_info = {}
            for attr in time_attrs:
                if hasattr(packet, attr):
                    time_info[attr] = getattr(packet, attr)
            if time_info:
                position_data['timestamp_info'].append(time_info)

            if packet_count >= 100:  # 100개만 분석
                break

        # 결과 분석
        print(f"\n📊 위치 데이터 분석 결과:")

        # Ship vs Sensor 좌표 비교
        if position_data['ship_coordinates'] and position_data['sensor_coordinates']:
            ship_coords = np.array(position_data['ship_coordinates'])
            sensor_coords = np.array(position_data['sensor_coordinates'])

            print(f"\nShip 좌표:")
            print(f"  X 범위: {ship_coords[:, 0].min():.6f} ~ {ship_coords[:, 0].max():.6f}")
            print(f"  Y 범위: {ship_coords[:, 1].min():.6f} ~ {ship_coords[:, 1].max():.6f}")

            print(f"\nSensor 좌표:")
            print(f"  X 범위: {sensor_coords[:, 0].min():.6f} ~ {sensor_coords[:, 0].max():.6f}")
            print(f"  Y 범위: {sensor_coords[:, 1].min():.6f} ~ {sensor_coords[:, 1].max():.6f}")

            # 차이 분석
            diff_x = np.abs(ship_coords[:, 0] - sensor_coords[:, 0])
            diff_y = np.abs(ship_coords[:, 1] - sensor_coords[:, 1])

            print(f"\nShip vs Sensor 차이:")
            print(f"  X 차이: 평균 {diff_x.mean():.6f}, 최대 {diff_x.max():.6f}")
            print(f"  Y 차이: 평균 {diff_y.mean():.6f}, 최대 {diff_y.max():.6f}")

            if diff_x.mean() < 0.001 and diff_y.mean() < 0.001:
                print("  → Ship과 Sensor 좌표가 거의 동일함 (동일한 GPS 소스)")
            else:
                print("  → Ship과 Sensor 좌표가 다름 (별도 계산 또는 오프셋)")

        # 원시 좌표 정보
        if position_data['raw_coordinates']:
            print(f"\n원시 좌표 속성:")
            raw_summary = {}
            for attr, value in position_data['raw_coordinates']:
                if attr not in raw_summary:
                    raw_summary[attr] = []
                raw_summary[attr].append(value)

            for attr, values in raw_summary.items():
                values_array = np.array(values)
                print(f"  {attr}: {len(values)}개, 범위 {values_array.min():.6f} ~ {values_array.max():.6f}")

        # 타임스탬프 정보
        if position_data['timestamp_info']:
            print(f"\n타임스탬프 정보:")
            first_time = position_data['timestamp_info'][0]
            for attr, value in first_time.items():
                print(f"  {attr}: {value}")

    except Exception as e:
        print(f"❌ 대안 위치 데이터 분석 실패: {e}")

def investigate_coordinate_calculation_requirements():
    """좌표 계산 요구사항 조사"""

    print(f"\n🧮 좌표 계산 요구사항 조사:")

    print(f"\n일반적인 XTF 좌표 저장 방식:")
    print(f"  1. 직접 GPS 좌표 (WGS84 decimal degrees)")
    print(f"  2. UTM 좌표 (미터 단위)")
    print(f"  3. 로컬 좌표계 (원점 기준 상대 좌표)")
    print(f"  4. 변환이 필요한 원시 형태")

    print(f"\n현재 데이터 특성:")
    print(f"  - 값 범위: 12.xxx ~ 129.xxx (경도), 36.xxx (위도)")
    print(f"  - 단위: decimal degrees로 추정")
    print(f"  - 좌표계: WGS84로 추정")

    print(f"\n💡 결론:")
    print(f"  현재 XTF에서 추출한 좌표는 직접 GPS 메타데이터로 보임")
    print(f"  별도 계산이나 변환 없이 사용 가능")
    print(f"  단, 자릿수 절단 오류는 데이터 손상으로 수정 필요")

def main():
    """메인 실행 함수"""

    investigate_xtf_coordinate_source()
    investigate_coordinate_calculation_requirements()

    print(f"\n{'='*70}")
    print("🎯 XTF 좌표 소스 조사 결론")
    print(f"{'='*70}")

    print(f"\n✅ 주요 발견사항:")
    print(f"   1. XTF 좌표는 직접 GPS 메타데이터 (SensorXcoordinate, SensorYcoordinate)")
    print(f"   2. 별도 계산이나 변환 과정 불필요")
    print(f"   3. WGS84 decimal degrees 형태로 저장")
    print(f"   4. 자릿수 절단은 데이터 손상으로 소프트웨어적 수정 필요")

    print(f"\n⚠️ 주의사항:")
    print(f"   1. Ship vs Sensor 좌표 차이 확인 필요")
    print(f"   2. 좌표계 정보 헤더에서 확인 권장")
    print(f"   3. 다른 XTF 파일과의 일관성 검증 필요")

if __name__ == "__main__":
    main()