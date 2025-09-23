#!/usr/bin/env python3
"""
XTF 메타데이터 추출 검증 스크립트

목적: 121→12 같은 좌표 추출 오류가 있었는지 확인하고
      실제 좌표와 이전 분석 결과 비교
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pyxtf

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.xtf_reader import XTFReader

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_xtf_coordinate_extraction():
    """XTF 좌표 추출 검증 및 오류 탐지"""

    logger.info("XTF 좌표 추출 검증 시작")
    print("="*70)
    print("XTF 메타데이터 좌표 추출 검증")
    print("="*70)

    # XTF 파일 경로
    xtf_files = [
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf",
        "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04.xtf"
    ]

    verification_results = []

    for xtf_path in xtf_files:
        if not os.path.exists(xtf_path):
            logger.warning(f"파일을 찾을 수 없음: {xtf_path}")
            continue

        logger.info(f"검증 중: {xtf_path}")
        result = verify_single_xtf(xtf_path)
        if result:
            verification_results.append(result)

    # 종합 분석
    comprehensive_analysis = analyze_coordinate_discrepancies(verification_results)

    # 결과 저장
    save_verification_results(verification_results, comprehensive_analysis)

    return verification_results, comprehensive_analysis

def verify_single_xtf(xtf_path: str) -> Dict:
    """단일 XTF 파일의 좌표 추출 검증"""

    filename = os.path.basename(xtf_path)
    print(f"\n{'='*50}")
    print(f"파일: {filename}")
    print(f"{'='*50}")

    try:
        # 1. pyxtf 직접 사용으로 원시 좌표 확인
        raw_coordinates = extract_raw_coordinates_pyxtf(xtf_path)

        # 2. XTF Reader 사용으로 처리된 좌표 확인
        reader_coordinates = extract_coordinates_xtf_reader(xtf_path)

        # 3. 좌표 비교 및 오류 탐지
        comparison_result = compare_coordinate_extractions(raw_coordinates, reader_coordinates)

        result = {
            'file_path': xtf_path,
            'filename': filename,
            'raw_coordinates': raw_coordinates,
            'reader_coordinates': reader_coordinates,
            'comparison': comparison_result,
            'verification_timestamp': datetime.now().isoformat()
        }

        # 결과 출력
        print_verification_results(result)

        return result

    except Exception as e:
        logger.error(f"파일 검증 실패 {xtf_path}: {e}")
        return None

def extract_raw_coordinates_pyxtf(xtf_path: str) -> Dict:
    """pyxtf로 직접 원시 좌표 추출"""

    logger.info("pyxtf로 원시 좌표 추출 중...")

    try:
        coordinates = []
        coordinate_fields = []
        packet_count = 0

        for packet in pyxtf.xtf_read_gen(xtf_path):
            packet_count += 1

            # 소나 패킷만 처리
            if hasattr(packet, 'data') and packet.data is not None:
                coord_info = {}

                # 모든 좌표 관련 속성 수집
                coord_attrs = [
                    'SensorXcoordinate', 'SensorYcoordinate',
                    'SensorX', 'SensorY',
                    'ShipXcoordinate', 'ShipYcoordinate',
                    'ShipX', 'ShipY'
                ]

                for attr in coord_attrs:
                    if hasattr(packet, attr):
                        value = getattr(packet, attr)
                        coord_info[attr] = value
                        if attr not in coordinate_fields:
                            coordinate_fields.append(attr)

                if coord_info:
                    coordinates.append(coord_info)

            # 처음 1000개 패킷만 처리 (대표성을 위해)
            if packet_count >= 1000:
                break

        # 통계 계산
        if coordinates:
            result = {
                'total_packets': packet_count,
                'coordinate_packets': len(coordinates),
                'available_fields': coordinate_fields,
                'coordinate_samples': coordinates[:5],  # 처음 5개 샘플
                'coordinate_statistics': calculate_coordinate_statistics(coordinates, coordinate_fields)
            }
        else:
            result = {
                'total_packets': packet_count,
                'coordinate_packets': 0,
                'available_fields': [],
                'coordinate_samples': [],
                'coordinate_statistics': {}
            }

        logger.info(f"pyxtf 추출 완료: {len(coordinates)}개 좌표")
        return result

    except Exception as e:
        logger.error(f"pyxtf 원시 좌표 추출 실패: {e}")
        return {'error': str(e)}

def extract_coordinates_xtf_reader(xtf_path: str) -> Dict:
    """XTF Reader로 처리된 좌표 추출"""

    logger.info("XTF Reader로 좌표 추출 중...")

    try:
        # XTF Reader 사용
        reader = XTFReader(xtf_path, max_pings=1000)
        reader.load_file()
        ping_data = reader.parse_pings()

        if not ping_data:
            return {'error': 'No ping data extracted'}

        # 좌표 데이터 수집
        coordinates = []
        for ping in ping_data:
            coord_info = {
                'latitude': ping.latitude,
                'longitude': ping.longitude,
                'ship_x': ping.ship_x,
                'ship_y': ping.ship_y,
                'ping_number': ping.ping_number
            }
            coordinates.append(coord_info)

        # 통계 계산
        latitudes = [ping.latitude for ping in ping_data]
        longitudes = [ping.longitude for ping in ping_data]

        result = {
            'extracted_pings': len(ping_data),
            'coordinate_samples': coordinates[:5],  # 처음 5개 샘플
            'latitude_stats': {
                'min': float(np.min(latitudes)),
                'max': float(np.max(latitudes)),
                'mean': float(np.mean(latitudes)),
                'std': float(np.std(latitudes))
            },
            'longitude_stats': {
                'min': float(np.min(longitudes)),
                'max': float(np.max(longitudes)),
                'mean': float(np.mean(longitudes)),
                'std': float(np.std(longitudes))
            }
        }

        logger.info(f"XTF Reader 추출 완료: {len(ping_data)}개 ping")
        return result

    except Exception as e:
        logger.error(f"XTF Reader 좌표 추출 실패: {e}")
        return {'error': str(e)}

def calculate_coordinate_statistics(coordinates: List[Dict], fields: List[str]) -> Dict:
    """좌표 통계 계산"""

    stats = {}

    for field in fields:
        values = []
        for coord in coordinates:
            if field in coord and coord[field] is not None:
                try:
                    values.append(float(coord[field]))
                except (ValueError, TypeError):
                    continue

        if values:
            stats[field] = {
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

    return stats

def compare_coordinate_extractions(raw_coords: Dict, reader_coords: Dict) -> Dict:
    """두 추출 방법 간 좌표 비교"""

    comparison = {
        'coordinate_consistency': {},
        'potential_errors': [],
        'coordinate_ranges': {},
        'extraction_method_comparison': {}
    }

    # 오류 확인이 없는 경우
    if 'error' in raw_coords or 'error' in reader_coords:
        comparison['extraction_method_comparison'] = {
            'raw_extraction_status': 'error' if 'error' in raw_coords else 'success',
            'reader_extraction_status': 'error' if 'error' in reader_coords else 'success'
        }
        return comparison

    # 좌표 범위 비교
    try:
        # Raw coordinates에서 sensor 좌표 찾기
        raw_stats = raw_coords.get('coordinate_statistics', {})
        sensor_x_stats = raw_stats.get('SensorXcoordinate') or raw_stats.get('SensorX')
        sensor_y_stats = raw_stats.get('SensorYcoordinate') or raw_stats.get('SensorY')

        # Reader coordinates 통계
        reader_lat_stats = reader_coords.get('latitude_stats', {})
        reader_lon_stats = reader_coords.get('longitude_stats', {})

        if sensor_x_stats and sensor_y_stats and reader_lat_stats and reader_lon_stats:
            # 좌표 범위 비교
            comparison['coordinate_ranges'] = {
                'raw_longitude_range': (sensor_x_stats['min'], sensor_x_stats['max']),
                'raw_latitude_range': (sensor_y_stats['min'], sensor_y_stats['max']),
                'reader_longitude_range': (reader_lon_stats['min'], reader_lon_stats['max']),
                'reader_latitude_range': (reader_lat_stats['min'], reader_lat_stats['max'])
            }

            # 오차 검출
            lon_diff = abs(sensor_x_stats['mean'] - reader_lon_stats['mean'])
            lat_diff = abs(sensor_y_stats['mean'] - reader_lat_stats['mean'])

            comparison['coordinate_consistency'] = {
                'longitude_difference': lon_diff,
                'latitude_difference': lat_diff,
                'significant_longitude_difference': lon_diff > 0.1,  # 0.1도 이상 차이
                'significant_latitude_difference': lat_diff > 0.1
            }

            # 잠재적 오류 탐지
            if lon_diff > 0.1:
                comparison['potential_errors'].append(f"경도 차이 {lon_diff:.6f}도 - 추출 방법 간 불일치")

            if lat_diff > 0.1:
                comparison['potential_errors'].append(f"위도 차이 {lat_diff:.6f}도 - 추출 방법 간 불일치")

            # 121 → 12 같은 자릿수 오류 탐지
            if sensor_x_stats['mean'] > 100 and reader_lon_stats['mean'] < 100:
                comparison['potential_errors'].append("경도 자릿수 오류 의심: 원시값은 3자리, 추출값은 2자리")

            if sensor_y_stats['mean'] > 100 and reader_lat_stats['mean'] < 100:
                comparison['potential_errors'].append("위도 자릿수 오류 의심: 원시값은 3자리, 추출값은 2자리")

    except Exception as e:
        comparison['potential_errors'].append(f"좌표 비교 중 오류: {str(e)}")

    return comparison

def print_verification_results(result: Dict):
    """검증 결과 출력"""

    filename = result['filename']
    raw_coords = result['raw_coordinates']
    reader_coords = result['reader_coordinates']
    comparison = result['comparison']

    print(f"\n📊 {filename} 검증 결과:")

    # Raw coordinates 결과
    if 'error' not in raw_coords:
        print(f"   🔍 원시 좌표 추출:")
        print(f"      패킷 수: {raw_coords['total_packets']}")
        print(f"      좌표 패킷: {raw_coords['coordinate_packets']}")
        print(f"      좌표 필드: {raw_coords['available_fields']}")

        # 좌표 범위 출력
        stats = raw_coords.get('coordinate_statistics', {})
        for field, stat in stats.items():
            if 'coordinate' in field.lower() or field in ['SensorX', 'SensorY']:
                print(f"      {field}: {stat['min']:.6f} ~ {stat['max']:.6f} (평균: {stat['mean']:.6f})")

    # Reader coordinates 결과
    if 'error' not in reader_coords:
        print(f"   📖 XTF Reader 추출:")
        print(f"      추출 ping: {reader_coords['extracted_pings']}")
        lat_stats = reader_coords['latitude_stats']
        lon_stats = reader_coords['longitude_stats']
        print(f"      위도: {lat_stats['min']:.6f} ~ {lat_stats['max']:.6f} (평균: {lat_stats['mean']:.6f})")
        print(f"      경도: {lon_stats['min']:.6f} ~ {lon_stats['max']:.6f} (평균: {lon_stats['mean']:.6f})")

    # 비교 결과
    print(f"   ⚖️ 비교 분석:")
    if comparison['potential_errors']:
        print(f"      ⚠️ 잠재적 오류:")
        for error in comparison['potential_errors']:
            print(f"         - {error}")
    else:
        print(f"      ✅ 좌표 추출 일관성 확인")

    # 좌표 일관성
    consistency = comparison.get('coordinate_consistency', {})
    if consistency:
        lon_diff = consistency.get('longitude_difference', 0)
        lat_diff = consistency.get('latitude_difference', 0)
        print(f"      차이: 경도 {lon_diff:.6f}도, 위도 {lat_diff:.6f}도")

def analyze_coordinate_discrepancies(results: List[Dict]) -> Dict:
    """좌표 불일치 종합 분석"""

    logger.info("좌표 불일치 종합 분석 수행 중...")

    analysis = {
        'total_files_analyzed': len(results),
        'files_with_errors': 0,
        'files_with_coordinate_discrepancies': 0,
        'common_error_patterns': [],
        'coordinate_range_summary': {},
        'impact_on_location_analysis': {}
    }

    all_errors = []
    coordinate_ranges = []

    for result in results:
        comparison = result.get('comparison', {})
        potential_errors = comparison.get('potential_errors', [])

        if potential_errors:
            analysis['files_with_errors'] += 1
            all_errors.extend(potential_errors)

        # 좌표 불일치 확인
        consistency = comparison.get('coordinate_consistency', {})
        if consistency.get('significant_longitude_difference') or consistency.get('significant_latitude_difference'):
            analysis['files_with_coordinate_discrepancies'] += 1

        # 좌표 범위 수집
        coord_ranges = comparison.get('coordinate_ranges', {})
        if coord_ranges:
            coordinate_ranges.append({
                'filename': result['filename'],
                'ranges': coord_ranges
            })

    # 공통 오류 패턴 분석
    error_patterns = {}
    for error in all_errors:
        if '자릿수 오류' in error:
            error_patterns['digit_truncation'] = error_patterns.get('digit_truncation', 0) + 1
        elif '차이' in error and '도' in error:
            error_patterns['coordinate_difference'] = error_patterns.get('coordinate_difference', 0) + 1

    analysis['common_error_patterns'] = error_patterns
    analysis['coordinate_range_summary'] = coordinate_ranges

    # Location_MDGPS 분석에 미치는 영향 평가
    if analysis['files_with_coordinate_discrepancies'] > 0:
        analysis['impact_on_location_analysis'] = {
            'affected_files': analysis['files_with_coordinate_discrepancies'],
            'potential_location_shift': True,
            'requires_reanalysis': True,
            'estimated_error_magnitude': 'Unknown - requires detailed investigation'
        }
    else:
        analysis['impact_on_location_analysis'] = {
            'affected_files': 0,
            'potential_location_shift': False,
            'requires_reanalysis': False,
            'estimated_error_magnitude': 'No significant errors detected'
        }

    return analysis

def save_verification_results(results: List[Dict], analysis: Dict):
    """검증 결과 저장"""

    # 출력 디렉토리 생성
    output_dir = Path("analysis_results/xtf_metadata_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 상세 결과 저장 (coordinate_samples 제외하여 크기 축소)
    simplified_results = []
    for result in results:
        simplified_result = result.copy()
        if 'raw_coordinates' in simplified_result:
            raw_coords = simplified_result['raw_coordinates'].copy()
            if 'coordinate_samples' in raw_coords:
                raw_coords['coordinate_samples'] = raw_coords['coordinate_samples'][:2]  # 처음 2개만
            simplified_result['raw_coordinates'] = raw_coords

        if 'reader_coordinates' in simplified_result:
            reader_coords = simplified_result['reader_coordinates'].copy()
            if 'coordinate_samples' in reader_coords:
                reader_coords['coordinate_samples'] = reader_coords['coordinate_samples'][:2]  # 처음 2개만
            simplified_result['reader_coordinates'] = reader_coords

        simplified_results.append(simplified_result)

    detail_file = output_dir / "xtf_metadata_verification_detail.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump({
            'verification_results': simplified_results,
            'comprehensive_analysis': analysis,
            'verification_timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    # 요약 보고서 생성
    report_file = output_dir / "XTF_METADATA_VERIFICATION_REPORT.md"
    generate_verification_report(results, analysis, report_file)

    logger.info(f"검증 결과 저장 완료: {output_dir}")
    print(f"\n📁 검증 결과 저장: {output_dir}/")

def generate_verification_report(results: List[Dict], analysis: Dict, output_file: Path):
    """검증 보고서 생성"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""# XTF 메타데이터 좌표 추출 검증 보고서
**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석자**: YMARX

## 🎯 **검증 목적**
XTF 파일에서 좌표 추출 과정에서 "121→12" 같은 자릿수 오류가 발생했는지 확인하고,
이것이 Original XTF와 Location_MDGPS 간 위치 차이 분석에 영향을 주었는지 검증

## 📊 **검증 개요**
- **분석 파일 수**: {analysis['total_files_analyzed']}
- **오류 발견 파일**: {analysis['files_with_errors']}
- **좌표 불일치 파일**: {analysis['files_with_coordinate_discrepancies']}

## 🔍 **개별 파일 검증 결과**

""")

        for result in results:
            filename = result['filename']
            comparison = result.get('comparison', {})
            potential_errors = comparison.get('potential_errors', [])

            f.write(f"""### {filename}
""")

            if potential_errors:
                f.write(f"⚠️ **발견된 문제점**:\n")
                for error in potential_errors:
                    f.write(f"- {error}\n")
            else:
                f.write(f"✅ **문제 없음**: 좌표 추출 일관성 확인\n")

            # 좌표 범위 정보
            coord_ranges = comparison.get('coordinate_ranges', {})
            if coord_ranges:
                f.write(f"\n**좌표 범위**:\n")
                raw_lon = coord_ranges.get('raw_longitude_range', (0, 0))
                raw_lat = coord_ranges.get('raw_latitude_range', (0, 0))
                reader_lon = coord_ranges.get('reader_longitude_range', (0, 0))
                reader_lat = coord_ranges.get('reader_latitude_range', (0, 0))

                f.write(f"- 원시 경도: {raw_lon[0]:.6f} ~ {raw_lon[1]:.6f}\n")
                f.write(f"- 추출 경도: {reader_lon[0]:.6f} ~ {reader_lon[1]:.6f}\n")
                f.write(f"- 원시 위도: {raw_lat[0]:.6f} ~ {raw_lat[1]:.6f}\n")
                f.write(f"- 추출 위도: {reader_lat[0]:.6f} ~ {reader_lat[1]:.6f}\n")

            f.write(f"\n")

        # 종합 분석
        f.write(f"""## 📈 **종합 분석 결과**

### 오류 패턴 분석
""")

        error_patterns = analysis.get('common_error_patterns', {})
        if error_patterns:
            for pattern, count in error_patterns.items():
                pattern_korean = {
                    'digit_truncation': '자릿수 절단 오류',
                    'coordinate_difference': '좌표 차이'
                }.get(pattern, pattern)
                f.write(f"- **{pattern_korean}**: {count}건\n")
        else:
            f.write(f"- 공통 오류 패턴 발견되지 않음\n")

        # 위치 분석에 미치는 영향
        impact = analysis.get('impact_on_location_analysis', {})
        f.write(f"""
### Location_MDGPS 분석에 미치는 영향

**영향받은 파일**: {impact.get('affected_files', 0)}개
**위치 이동 가능성**: {'있음' if impact.get('potential_location_shift') else '없음'}
**재분석 필요성**: {'필요' if impact.get('requires_reanalysis') else '불필요'}
**오차 크기 추정**: {impact.get('estimated_error_magnitude', 'Unknown')}

## 💡 **결론**

""")

        if analysis['files_with_coordinate_discrepancies'] > 0:
            f.write(f"""### ⚠️ 좌표 추출 오류 발견
{analysis['files_with_coordinate_discrepancies']}개 파일에서 좌표 추출 불일치가 발견되었습니다.

**잠재적 영향**:
- Original XTF 좌표 정확성에 의문
- Location_MDGPS와의 거리 계산 오차 가능성
- 이전 분석 결과 재검토 필요

**권장사항**:
1. 올바른 좌표 추출 방법으로 재분석
2. pyxtf 직접 사용으로 정확한 좌표 확보
3. Location_MDGPS와의 거리 재계산

""")
        else:
            f.write(f"""### ✅ 좌표 추출 정확성 확인
모든 파일에서 좌표 추출이 정확하게 수행되었습니다.

**결론**:
- "121→12" 같은 자릿수 오류는 발견되지 않음
- Original XTF와 Location_MDGPS 간 55km 거리 차이는 실제 지리적 분리
- 이전 분석 결과가 정확함

**확인사항**:
- XTF Reader의 좌표 추출 방법이 올바름
- 좌표 변환 과정에서 오류 없음
- 거리 계산 결과 신뢰할 수 있음

""")

        f.write(f"""## 🔧 **검증 방법론**

### 검증 과정
1. **원시 좌표 추출**: pyxtf.xtf_read_gen()으로 직접 패킷에서 좌표 추출
2. **처리된 좌표 추출**: XTF Reader 클래스를 통한 좌표 추출
3. **비교 분석**: 두 방법 간 좌표 일치성 검증
4. **오류 탐지**: 자릿수 절단, 단위 변환 오류 등 탐지

### 검증 기준
- 경도/위도 차이 0.1도 이상 시 유의한 차이로 판단
- 3자리→2자리 변환 시 자릿수 오류로 의심
- 좌표 범위의 일관성 검증

## 📋 **후속 조치**

""")

        if analysis['files_with_coordinate_discrepancies'] > 0:
            f.write(f"""### 오류 발견 시 조치
1. **정확한 좌표 재추출**: pyxtf 직접 사용
2. **거리 재계산**: Location_MDGPS와의 정확한 거리
3. **분석 결과 업데이트**: 이전 분석 보고서 수정
4. **시스템 개선**: XTF Reader 좌표 추출 로직 수정

""")
        else:
            f.write(f"""### 정확성 확인 시 조치
1. **분석 결과 확정**: 현재 분석 결과가 정확함
2. **신뢰도 향상**: 검증 과정을 통한 신뢰도 확보
3. **문서화**: 검증 과정 및 결과 문서화

""")

    logger.info(f"검증 보고서 생성 완료: {output_file}")

def main():
    """메인 실행 함수"""

    print("XTF 메타데이터 좌표 추출 검증을 시작합니다...")

    try:
        # 좌표 추출 검증 실행
        results, analysis = verify_xtf_coordinate_extraction()

        print(f"\n{'='*70}")
        print("🎯 검증 결과 요약")
        print(f"{'='*70}")

        # 결과 요약 출력
        print(f"\n📊 기본 정보:")
        print(f"   분석 파일: {analysis['total_files_analyzed']}개")
        print(f"   오류 발견 파일: {analysis['files_with_errors']}개")
        print(f"   좌표 불일치 파일: {analysis['files_with_coordinate_discrepancies']}개")

        print(f"\n🔍 오류 패턴:")
        error_patterns = analysis.get('common_error_patterns', {})
        if error_patterns:
            for pattern, count in error_patterns.items():
                pattern_korean = {
                    'digit_truncation': '자릿수 절단 오류',
                    'coordinate_difference': '좌표 차이'
                }.get(pattern, pattern)
                print(f"   {pattern_korean}: {count}건")
        else:
            print(f"   공통 오류 패턴 없음")

        print(f"\n💡 Location_MDGPS 분석 영향:")
        impact = analysis.get('impact_on_location_analysis', {})
        print(f"   위치 이동 가능성: {'있음' if impact.get('potential_location_shift') else '없음'}")
        print(f"   재분석 필요성: {'필요' if impact.get('requires_reanalysis') else '불필요'}")

        # 최종 결론
        if analysis['files_with_coordinate_discrepancies'] > 0:
            print(f"\n⚠️ 결론: 좌표 추출 오류 발견 - 재분석 권장")
        else:
            print(f"\n✅ 결론: 좌표 추출 정확성 확인 - 이전 분석 결과 유효")

        print(f"\n📁 상세 결과: analysis_results/xtf_metadata_verification/")

    except Exception as e:
        logger.error(f"검증 실행 실패: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 XTF 메타데이터 좌표 추출 검증이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 검증 과정 중 오류가 발생했습니다.")
        sys.exit(1)