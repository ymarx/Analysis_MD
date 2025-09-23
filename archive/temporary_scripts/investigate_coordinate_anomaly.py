#!/usr/bin/env python3
"""
EdgeTech 4205 좌표 이상치 조사

목적: 12.51 ~ 129.51 범위가 나오는 원인 규명
"""

import os
import numpy as np
import pandas as pd
import pyxtf
import matplotlib.pyplot as plt
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate_coordinate_anomaly():
    """좌표 이상치 상세 조사"""

    print("="*70)
    print("EdgeTech 4205 좌표 이상치 조사")
    print("="*70)

    # 문제가 있는 EdgeTech 파일
    xtf_path = "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf"

    if not os.path.exists(xtf_path):
        print(f"❌ 파일을 찾을 수 없음: {xtf_path}")
        return

    print(f"🔍 분석 파일: {os.path.basename(xtf_path)}")

    try:
        # 모든 좌표 데이터 수집
        coordinates = []
        packet_info = []
        packet_count = 0

        for packet in pyxtf.xtf_read_gen(xtf_path):
            packet_count += 1

            if hasattr(packet, 'data') and packet.data is not None:
                coord_data = {}

                # 좌표 속성들
                coord_attrs = ['SensorXcoordinate', 'SensorYcoordinate', 'ShipXcoordinate', 'ShipYcoordinate']

                for attr in coord_attrs:
                    if hasattr(packet, attr):
                        value = getattr(packet, attr)
                        coord_data[attr] = value

                if coord_data:
                    coord_data['packet_number'] = packet_count
                    coordinates.append(coord_data)

            # 처음 2000개 패킷만 분석
            if packet_count >= 2000:
                break

        print(f"📊 수집된 좌표 데이터: {len(coordinates)}개")

        if not coordinates:
            print("❌ 좌표 데이터를 찾을 수 없습니다.")
            return

        # DataFrame으로 변환
        df = pd.DataFrame(coordinates)

        print(f"\n📋 데이터 기본 정보:")
        print(f"   총 레코드: {len(df)}")
        print(f"   컬럼: {list(df.columns)}")

        # 각 좌표 필드 분석
        for coord_field in ['SensorXcoordinate', 'SensorYcoordinate']:
            if coord_field in df.columns:
                analyze_coordinate_field(df, coord_field)

        # 이상치 패턴 분석
        analyze_anomaly_patterns(df)

        # 시간순 변화 분석
        analyze_temporal_changes(df)

        # 데이터 시각화
        create_coordinate_plots(df)

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()

def analyze_coordinate_field(df, field_name):
    """특정 좌표 필드 상세 분석"""

    print(f"\n🔍 {field_name} 분석:")

    values = df[field_name].dropna()
    if len(values) == 0:
        print(f"   데이터 없음")
        return

    print(f"   데이터 개수: {len(values)}")
    print(f"   최솟값: {values.min():.6f}")
    print(f"   최댓값: {values.max():.6f}")
    print(f"   평균: {values.mean():.6f}")
    print(f"   표준편차: {values.std():.6f}")

    # 값 분포 분석
    print(f"\n   값 분포:")

    # 10도 간격으로 분포 확인
    ranges = [
        (0, 50, "0-50도 (이상치)"),
        (50, 100, "50-100도 (중간값)"),
        (100, 140, "100-140도 (정상 한국 경도)"),
        (140, 200, "140-200도 (이상치)")
    ]

    for min_val, max_val, description in ranges:
        count = len(values[(values >= min_val) & (values < max_val)])
        percentage = (count / len(values)) * 100
        print(f"   {description}: {count}개 ({percentage:.1f}%)")

    # 구체적인 값들 확인
    unique_values = values.value_counts().head(10)
    print(f"\n   상위 10개 값:")
    for val, count in unique_values.items():
        print(f"   {val:.6f}: {count}회")

    # 이상치 탐지 (IQR 방법)
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = values[(values < lower_bound) | (values > upper_bound)]
    print(f"\n   이상치 (IQR 방법):")
    print(f"   정상 범위: {lower_bound:.6f} ~ {upper_bound:.6f}")
    print(f"   이상치 개수: {len(outliers)} ({len(outliers)/len(values)*100:.1f}%)")

    if len(outliers) > 0:
        print(f"   이상치 예시: {outliers.head().tolist()}")

def analyze_anomaly_patterns(df):
    """이상치 패턴 분석"""

    print(f"\n🔍 이상치 패턴 분석:")

    # SensorXcoordinate 분석 (경도)
    if 'SensorXcoordinate' in df.columns:
        x_coords = df['SensorXcoordinate'].dropna()

        # 12도대와 129도대 분리
        low_coords = x_coords[x_coords < 50]  # 12도대
        high_coords = x_coords[x_coords > 100]  # 129도대

        print(f"   경도 분포:")
        print(f"   낮은 값 (< 50도): {len(low_coords)}개")
        print(f"   높은 값 (> 100도): {len(high_coords)}개")

        if len(low_coords) > 0:
            print(f"   낮은 값 범위: {low_coords.min():.6f} ~ {low_coords.max():.6f}")
            print(f"   낮은 값 예시: {low_coords.head().tolist()}")

        if len(high_coords) > 0:
            print(f"   높은 값 범위: {high_coords.min():.6f} ~ {high_coords.max():.6f}")
            print(f"   높은 값 예시: {high_coords.head().tolist()}")

        # 패킷 번호와 좌표값 관계 분석
        if len(low_coords) > 0 and len(high_coords) > 0:
            print(f"\n   패킷 번호별 분포:")

            # 낮은 값들이 어느 패킷에서 나오는지
            low_packets = df[df['SensorXcoordinate'] < 50]['packet_number'].tolist()
            high_packets = df[df['SensorXcoordinate'] > 100]['packet_number'].tolist()

            print(f"   낮은 값 패킷 번호: {low_packets[:10]} ...")
            print(f"   높은 값 패킷 번호: {high_packets[:10]} ...")

            # 연속성 확인
            low_continuous = is_continuous_sequence(low_packets)
            high_continuous = is_continuous_sequence(high_packets)

            print(f"   낮은 값 연속성: {'연속적' if low_continuous else '산발적'}")
            print(f"   높은 값 연속성: {'연속적' if high_continuous else '산발적'}")

def is_continuous_sequence(packet_numbers):
    """패킷 번호가 연속적인지 확인"""

    if len(packet_numbers) < 2:
        return False

    sorted_nums = sorted(packet_numbers)
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i-1] > 5:  # 5개 이상 간격이면 불연속
            return False
    return True

def analyze_temporal_changes(df):
    """시간순 좌표 변화 분석"""

    print(f"\n🔍 시간순 변화 분석:")

    if 'SensorXcoordinate' not in df.columns:
        print("   SensorXcoordinate 데이터 없음")
        return

    # 처음 100개와 마지막 100개 비교
    if len(df) >= 200:
        first_100 = df.head(100)['SensorXcoordinate'].dropna()
        last_100 = df.tail(100)['SensorXcoordinate'].dropna()

        print(f"   처음 100개:")
        print(f"   범위: {first_100.min():.6f} ~ {first_100.max():.6f}")
        print(f"   평균: {first_100.mean():.6f}")

        print(f"   마지막 100개:")
        print(f"   범위: {last_100.min():.6f} ~ {last_100.max():.6f}")
        print(f"   평균: {last_100.mean():.6f}")

        # 변화 패턴 확인
        if first_100.mean() < 50 and last_100.mean() > 100:
            print(f"   ⚠️ 패턴: 낮은 값에서 높은 값으로 변화")
        elif first_100.mean() > 100 and last_100.mean() < 50:
            print(f"   ⚠️ 패턴: 높은 값에서 낮은 값으로 변화")
        else:
            print(f"   패턴: 일관된 값 유지")

    # 10구간으로 나누어 변화 추적
    num_sections = 10
    section_size = len(df) // num_sections

    print(f"\n   구간별 평균 경도:")
    for i in range(num_sections):
        start_idx = i * section_size
        end_idx = (i + 1) * section_size if i < num_sections - 1 else len(df)

        section_data = df.iloc[start_idx:end_idx]['SensorXcoordinate'].dropna()
        if len(section_data) > 0:
            print(f"   구간 {i+1:2d}: {section_data.mean():8.3f}도 (패킷 {start_idx+1}-{end_idx})")

def create_coordinate_plots(df):
    """좌표 데이터 시각화"""

    print(f"\n📊 데이터 시각화 생성 중...")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('EdgeTech 4205 좌표 이상치 분석', fontsize=16)

        # 1. 경도 시계열
        if 'SensorXcoordinate' in df.columns:
            x_coords = df['SensorXcoordinate'].dropna()
            axes[0, 0].plot(x_coords.index, x_coords.values, 'b-', alpha=0.7)
            axes[0, 0].set_title('경도 시계열 변화')
            axes[0, 0].set_xlabel('패킷 순서')
            axes[0, 0].set_ylabel('경도 (도)')
            axes[0, 0].grid(True)

        # 2. 경도 히스토그램
        if 'SensorXcoordinate' in df.columns:
            x_coords = df['SensorXcoordinate'].dropna()
            axes[0, 1].hist(x_coords.values, bins=50, alpha=0.7, color='blue')
            axes[0, 1].set_title('경도 분포')
            axes[0, 1].set_xlabel('경도 (도)')
            axes[0, 1].set_ylabel('빈도')
            axes[0, 1].grid(True)

        # 3. 위도 시계열
        if 'SensorYcoordinate' in df.columns:
            y_coords = df['SensorYcoordinate'].dropna()
            axes[1, 0].plot(y_coords.index, y_coords.values, 'r-', alpha=0.7)
            axes[1, 0].set_title('위도 시계열 변화')
            axes[1, 0].set_xlabel('패킷 순서')
            axes[1, 0].set_ylabel('위도 (도)')
            axes[1, 0].grid(True)

        # 4. 좌표 산점도
        if 'SensorXcoordinate' in df.columns and 'SensorYcoordinate' in df.columns:
            x_coords = df['SensorXcoordinate'].dropna()
            y_coords = df['SensorYcoordinate'].dropna()

            # 길이 맞추기
            min_len = min(len(x_coords), len(y_coords))
            x_coords = x_coords.iloc[:min_len]
            y_coords = y_coords.iloc[:min_len]

            axes[1, 1].scatter(x_coords.values, y_coords.values, alpha=0.5, s=1)
            axes[1, 1].set_title('위도-경도 산점도')
            axes[1, 1].set_xlabel('경도 (도)')
            axes[1, 1].set_ylabel('위도 (도)')
            axes[1, 1].grid(True)

        plt.tight_layout()

        # 저장
        output_dir = "analysis_results/coordinate_anomaly_investigation"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/coordinate_anomaly_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   시각화 저장: {output_dir}/coordinate_anomaly_analysis.png")

    except Exception as e:
        print(f"   시각화 실패: {e}")

def investigate_data_corruption():
    """데이터 손상 가능성 조사"""

    print(f"\n🔍 데이터 손상 가능성 조사:")

    # 가능한 원인들
    possible_causes = [
        "1. 자릿수 절단 (121.xxx → 12.xxx)",
        "2. 소수점 위치 이동 (1295.xxx → 12.95xxx)",
        "3. 데이터 타입 변환 오류",
        "4. 패킷 헤더 손상",
        "5. 다른 좌표계 데이터 혼입",
        "6. 파일 손상 또는 일부 복원"
    ]

    print("   가능한 원인들:")
    for cause in possible_causes:
        print(f"   {cause}")

    # 121 → 12 패턴 확인
    print(f"\n   121→12 패턴 검증:")
    print(f"   예상: 121.xxx가 12.xxx로 변환되었을 가능성")
    print(f"   확인 방법: 12.xxx 값에 10을 곱하면 129.xxx 근처가 되는지")

def main():
    """메인 실행 함수"""

    investigate_coordinate_anomaly()
    investigate_data_corruption()

    print(f"\n{'='*70}")
    print("🎯 조사 결론")
    print(f"{'='*70}")

    print(f"\n💡 주요 발견사항:")
    print(f"   1. EdgeTech 4205 파일에서 12.xx ~ 129.xx 범위 확인")
    print(f"   2. Klein 3900은 정상적으로 129.xx 범위만 보임")
    print(f"   3. 같은 지역에서 이런 차이가 나는 것은 명백한 오류")

    print(f"\n⚠️ 문제점:")
    print(f"   - 지리적으로 불가능한 좌표 범위")
    print(f"   - 데이터 일관성 부족")
    print(f"   - 메타데이터 추출 과정에서 오류 가능성")

    print(f"\n🔧 권장 조치:")
    print(f"   1. EdgeTech 4205 파일의 원시 데이터 재검토")
    print(f"   2. 자릿수 절단 오류 여부 확인")
    print(f"   3. 올바른 좌표 추출 방법 적용")
    print(f"   4. Location_MDGPS와의 거리 재계산")

if __name__ == "__main__":
    main()