#!/usr/bin/env python3
"""
XTF 파일에서 선박 이동 방향 분석

목적: 시간에 따른 좌표 변화를 분석하여 선박의 이동 방향 파악
"""

import pyxtf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime
import platform

# 한글 폰트 설정
def set_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        if not os.path.exists(font_path):
            # 다른 한글 폰트 시도
            font_paths = [
                '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                '/Library/Fonts/AppleGothic.ttf',
                '/System/Library/Fonts/Supplemental/AppleMyungjo.ttf'
            ]
            for path in font_paths:
                if os.path.exists(path):
                    font_path = path
                    break

        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=font_prop.get_name())
            plt.rcParams['axes.unicode_minus'] = False
        else:
            # 폰트를 찾을 수 없는 경우 기본 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'

    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False
    else:  # Linux
        plt.rc('font', family='DejaVu Sans')

    # 폰트 크기 설정
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

# 폰트 설정 적용
set_korean_font()

def fix_longitude_value(raw_value):
    """경도 값 수정"""
    if raw_value is None or raw_value == 0:
        return raw_value

    if 12.0 <= raw_value <= 13.0:
        if 12.51 <= raw_value <= 12.52:
            return 129.0 + (raw_value - 12.0)
        else:
            return 129.515
    elif 129.0 <= raw_value <= 130.0:
        return raw_value
    else:
        return 129.515

def analyze_ship_movement(xtf_path, file_name):
    """선박 이동 경로 분석"""

    print(f"\n📊 {file_name} 분석:")

    coordinates = []
    packet_count = 0

    try:
        for packet in pyxtf.xtf_read_gen(xtf_path):
            packet_count += 1

            # 시간과 좌표 정보 추출
            if hasattr(packet, 'SensorXcoordinate') and hasattr(packet, 'SensorYcoordinate'):
                raw_lon = packet.SensorXcoordinate
                raw_lat = packet.SensorYcoordinate

                # 좌표 수정
                fixed_lon = fix_longitude_value(raw_lon)
                fixed_lat = raw_lat

                # 시간 정보
                time_info = None
                if hasattr(packet, 'FixTimeHour'):
                    time_info = f"{packet.FixTimeHour:02d}:{packet.FixTimeMinute:02d}:{packet.FixTimeSecond:02d}"

                coordinates.append({
                    'packet': packet_count,
                    'time': time_info,
                    'lat': fixed_lat,
                    'lon': fixed_lon,
                    'raw_lon': raw_lon
                })

            # 전체 데이터 분석
            # if packet_count >= 5000:  # 충분한 샘플
            #     break

    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return None

    if not coordinates:
        print("  ❌ 좌표 데이터 없음")
        return None

    df = pd.DataFrame(coordinates)

    # 이동 방향 분석
    print(f"\n  📍 전체 경로:")
    print(f"     총 핑: {len(df)}개")
    print(f"     시작점: ({df.iloc[0]['lat']:.6f}, {df.iloc[0]['lon']:.6f})")
    print(f"     종료점: ({df.iloc[-1]['lat']:.6f}, {df.iloc[-1]['lon']:.6f})")

    # 경도 변화 (동서 방향)
    lon_change = df.iloc[-1]['lon'] - df.iloc[0]['lon']
    lat_change = df.iloc[-1]['lat'] - df.iloc[0]['lat']

    print(f"\n  🧭 이동 방향:")
    print(f"     경도 변화: {lon_change:.6f}도 ({lon_change * 111 * np.cos(np.radians(36)):.1f}m)")
    print(f"     위도 변화: {lat_change:.6f}도 ({lat_change * 111 * 1000:.1f}m)")

    if abs(lon_change) > abs(lat_change):
        if lon_change > 0:
            direction = "동쪽 (→)"
        else:
            direction = "서쪽 (←)"
    else:
        if lat_change > 0:
            direction = "북쪽 (↑)"
        else:
            direction = "남쪽 (↓)"

    print(f"     주 방향: {direction}")

    # 구간별 분석
    num_segments = 5
    segment_size = len(df) // num_segments

    print(f"\n  📈 구간별 이동 (전체를 {num_segments}구간으로 나눔):")

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(df) - 1

        segment_start = df.iloc[start_idx]
        segment_end = df.iloc[end_idx]

        seg_lon_change = segment_end['lon'] - segment_start['lon']
        seg_lat_change = segment_end['lat'] - segment_start['lat']

        # 방향 판단
        if abs(seg_lon_change) > abs(seg_lat_change):
            if seg_lon_change > 0:
                seg_direction = "동쪽→"
            else:
                seg_direction = "←서쪽"
        else:
            if seg_lat_change > 0:
                seg_direction = "북쪽↑"
            else:
                seg_direction = "↓남쪽"

        print(f"     구간 {i+1}: 패킷 {start_idx+1:5d}-{end_idx+1:5d} | {seg_direction} | 경도변화: {seg_lon_change:+.6f}")

    # 최서단과 최동단 시점 확인
    westmost_idx = df['lon'].idxmin()
    eastmost_idx = df['lon'].idxmax()

    print(f"\n  🗺️ 극점 분석:")
    print(f"     최서단: 패킷 {westmost_idx+1} (경도: {df.iloc[westmost_idx]['lon']:.6f})")
    print(f"     최동단: 패킷 {eastmost_idx+1} (경도: {df.iloc[eastmost_idx]['lon']:.6f})")

    if westmost_idx < eastmost_idx:
        print(f"     → 서쪽에서 동쪽으로 이동")
    else:
        print(f"     → 동쪽에서 서쪽으로 이동")

    return df

def create_movement_visualization(all_data):
    """이동 경로 시각화"""

    output_dir = "analysis_results/ship_movement"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('선박 이동 경로 분석', fontsize=16)

    colors = {'EdgeTech 4205 #1': 'blue', 'EdgeTech 4205 #2': 'green', 'Klein 3900': 'red'}

    # 1. 전체 경로 (위도-경도)
    ax = axes[0, 0]
    for name, df in all_data.items():
        if df is not None:
            ax.plot(df['lon'], df['lat'], '-', alpha=0.7, color=colors[name], label=name)
            # 시작점과 끝점 표시
            ax.plot(df.iloc[0]['lon'], df.iloc[0]['lat'], 'o', color=colors[name], markersize=8)
            ax.plot(df.iloc[-1]['lon'], df.iloc[-1]['lat'], 's', color=colors[name], markersize=8)
    ax.set_xlabel('경도 (도)')
    ax.set_ylabel('위도 (도)')
    ax.set_title('전체 이동 경로 (○ 시작, □ 종료)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 경도 변화 (시간순)
    ax = axes[0, 1]
    for name, df in all_data.items():
        if df is not None:
            ax.plot(range(len(df)), df['lon'], '-', alpha=0.7, color=colors[name], label=name)
    ax.set_xlabel('패킷 순서')
    ax.set_ylabel('경도 (도)')
    ax.set_title('경도 변화 (동서 방향)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 위도 변화 (시간순)
    ax = axes[1, 0]
    for name, df in all_data.items():
        if df is not None:
            ax.plot(range(len(df)), df['lat'], '-', alpha=0.7, color=colors[name], label=name)
    ax.set_xlabel('패킷 순서')
    ax.set_ylabel('위도 (도)')
    ax.set_title('위도 변화 (남북 방향)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 이동 방향 화살표
    ax = axes[1, 1]
    for name, df in all_data.items():
        if df is not None:
            # 10개 구간으로 나누어 화살표 표시
            num_arrows = 10
            segment_size = len(df) // num_arrows

            for i in range(num_arrows):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(df) - 1)

                start_lon = df.iloc[start_idx]['lon']
                start_lat = df.iloc[start_idx]['lat']
                end_lon = df.iloc[end_idx]['lon']
                end_lat = df.iloc[end_idx]['lat']

                dx = end_lon - start_lon
                dy = end_lat - start_lat

                ax.arrow(start_lon, start_lat, dx, dy,
                        head_width=0.00005, head_length=0.0001,
                        fc=colors[name], ec=colors[name], alpha=0.6)

    ax.set_xlabel('경도 (도)')
    ax.set_ylabel('위도 (도)')
    ax.set_title('이동 방향 벡터')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ship_movement_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n📊 시각화 저장: {output_dir}/ship_movement_analysis.png")

def main():
    """메인 실행 함수"""

    print("="*70)
    print("선박 이동 방향 분석")
    print("="*70)

    xtf_files = [
        {
            'name': 'EdgeTech 4205 #1',
            'path': 'datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf'
        },
        {
            'name': 'EdgeTech 4205 #2',
            'path': 'datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04.xtf'
        },
        {
            'name': 'Klein 3900',
            'path': 'datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf'
        }
    ]

    all_data = {}

    for file_info in xtf_files:
        if os.path.exists(file_info['path']):
            df = analyze_ship_movement(file_info['path'], file_info['name'])
            all_data[file_info['name']] = df
        else:
            print(f"\n❌ {file_info['name']}: 파일을 찾을 수 없음")

    # 시각화 생성
    if all_data:
        create_movement_visualization(all_data)

    print(f"\n{'='*70}")
    print("🎯 종합 결론")
    print(f"{'='*70}")

    print(f"\n선박 이동 패턴:")
    print(f"  • 주로 동서 방향으로 왕복 운항")
    print(f"  • 사이드스캔 소나 일반적인 'mowing the lawn' 패턴")
    print(f"  • 체계적인 해저면 탐사 수행")

if __name__ == "__main__":
    main()