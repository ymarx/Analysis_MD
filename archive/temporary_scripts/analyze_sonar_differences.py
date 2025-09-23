#!/usr/bin/env python3
"""
사이드 스캔 소나 기종별 차이점 분석
=====================================
EdgeTech 4205 vs Klein 3900의 패킷 구조, 메타데이터, 강도 데이터 차이점을 분석합니다.

Author: YMARX
Date: 2025-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.xtf_reader import XTFReader
from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_sonar_system_differences():
    """사이드 스캔 소나 기종별 차이점 상세 분석"""

    logger.info("="*70)
    logger.info("사이드 스캔 소나 기종별 차이점 분석")
    logger.info("="*70)

    # 분석할 파일들
    test_files = {
        'EdgeTech_4205_1': {
            'path': "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf",
            'manufacturer': 'EdgeTech',
            'model': '4205',
            'frequency': '800 kHz',
            'expected_features': ['dual_frequency', 'chirp_capability', 'high_resolution']
        },
        'EdgeTech_4205_2': {
            'path': "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04.xtf",
            'manufacturer': 'EdgeTech',
            'model': '4205',
            'frequency': '800 kHz',
            'expected_features': ['dual_frequency', 'chirp_capability', 'high_resolution']
        },
        'Klein_3900': {
            'path': "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
            'manufacturer': 'Klein',
            'model': '3900',
            'frequency': '900 kHz',
            'expected_features': ['dual_frequency', 'backscatter_analysis', 'bathymetry']
        }
    }

    analysis_results = {}

    for system_id, file_info in test_files.items():
        file_path = Path(file_info['path'])

        if not file_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"분석 중: {file_info['manufacturer']} {file_info['model']}")
        logger.info(f"파일: {file_path.name}")

        try:
            # 1. XTF Reader로 기본 분석
            reader = XTFReader(file_path, max_pings=1000)
            reader.load_file()
            ping_data = reader.parse_pings()

            # 2. Intensity Extractor로 상세 분석
            extractor = XTFIntensityExtractor(max_memory_mb=256)
            extracted_data = extractor.extract_intensity_data(
                str(file_path),
                ping_range=(0, 500)
            )

            # 3. 기종별 특성 분석
            system_analysis = analyze_system_characteristics(
                reader, extracted_data, file_info
            )

            analysis_results[system_id] = system_analysis

        except Exception as e:
            logger.error(f"분석 실패 ({system_id}): {e}")
            continue

    # 4. 기종별 비교 분석
    comparison_results = compare_sonar_systems(analysis_results)

    # 5. 결과 저장
    save_analysis_results(analysis_results, comparison_results)

    return analysis_results, comparison_results


def analyze_system_characteristics(reader, extracted_data, file_info):
    """개별 소나 시스템 특성 분석"""

    logger.info(f"시스템 특성 분석: {file_info['manufacturer']} {file_info['model']}")

    analysis = {
        'manufacturer': file_info['manufacturer'],
        'model': file_info['model'],
        'frequency': file_info['frequency'],
        'file_characteristics': {},
        'packet_structure': {},
        'data_format': {},
        'performance_metrics': {},
        'unique_features': []
    }

    try:
        # 1. 파일 특성 분석
        metadata = reader.metadata
        if metadata:
            analysis['file_characteristics'] = {
                'total_pings': metadata.total_pings,
                'sonar_channels': metadata.num_sonar_channels,
                'bathymetry_channels': metadata.num_bathymetry_channels,
                'frequency_info': metadata.frequency_info,
                'coordinate_bounds': metadata.coordinate_bounds,
                'time_range': metadata.time_range
            }

        # 2. 패킷 구조 분석
        if reader.ping_data:
            sample_ping = reader.ping_data[0]

            analysis['packet_structure'] = {
                'ping_number_range': {
                    'min': min(p.ping_number for p in reader.ping_data),
                    'max': max(p.ping_number for p in reader.ping_data)
                },
                'samples_per_ping': {
                    'min': min(p.range_samples for p in reader.ping_data),
                    'max': max(p.range_samples for p in reader.ping_data),
                    'typical': sample_ping.range_samples
                },
                'coordinate_precision': {
                    'lat_precision': len(str(sample_ping.latitude).split('.')[-1]) if '.' in str(sample_ping.latitude) else 0,
                    'lon_precision': len(str(sample_ping.longitude).split('.')[-1]) if '.' in str(sample_ping.longitude) else 0
                },
                'timestamp_format': type(sample_ping.timestamp).__name__
            }

        # 3. 데이터 포맷 분석
        intensity_images = extracted_data.get('intensity_images', {})
        if intensity_images:
            combined_img = intensity_images.get('combined', np.array([]))
            port_img = intensity_images.get('port', np.array([]))
            starboard_img = intensity_images.get('starboard', np.array([]))

            analysis['data_format'] = {
                'combined_shape': combined_img.shape if combined_img.size > 0 else None,
                'port_shape': port_img.shape if port_img.size > 0 else None,
                'starboard_shape': starboard_img.shape if starboard_img.size > 0 else None,
                'data_type': str(combined_img.dtype) if combined_img.size > 0 else None,
                'value_range': {
                    'combined': [float(combined_img.min()), float(combined_img.max())] if combined_img.size > 0 else None,
                    'port': [float(port_img.min()), float(port_img.max())] if port_img.size > 0 else None,
                    'starboard': [float(starboard_img.min()), float(starboard_img.max())] if starboard_img.size > 0 else None
                }
            }

        # 4. 성능 지표 분석
        if reader.ping_data and intensity_images:
            # 데이터 품질 지표
            intensity_matrix = reader.extract_intensity_matrix()

            analysis['performance_metrics'] = {
                'data_completeness': {
                    'pings_with_coordinates': sum(1 for p in reader.ping_data if p.latitude != 0 and p.longitude != 0) / len(reader.ping_data),
                    'pings_with_intensity': sum(1 for p in reader.ping_data if p.data.size > 0) / len(reader.ping_data)
                },
                'intensity_statistics': {
                    'mean': float(intensity_matrix.mean()) if intensity_matrix.size > 0 else 0,
                    'std': float(intensity_matrix.std()) if intensity_matrix.size > 0 else 0,
                    'dynamic_range': float(intensity_matrix.max() - intensity_matrix.min()) if intensity_matrix.size > 0 else 0,
                    'signal_to_noise_estimate': estimate_snr(intensity_matrix) if intensity_matrix.size > 0 else 0
                },
                'spatial_coverage': calculate_spatial_coverage(reader.ping_data),
                'temporal_consistency': assess_temporal_consistency(reader.ping_data)
            }

        # 5. 기종별 고유 특성 식별
        analysis['unique_features'] = identify_unique_features(file_info, reader, extracted_data)

        logger.info(f"시스템 특성 분석 완료: {len(analysis)} 카테고리")

    except Exception as e:
        logger.error(f"시스템 특성 분석 중 오류: {e}")

    return analysis


def estimate_snr(intensity_matrix):
    """Signal-to-Noise Ratio 추정"""
    if intensity_matrix.size == 0:
        return 0

    try:
        # 간단한 SNR 추정: 신호의 평균 대비 노이즈의 표준편차
        signal = np.mean(intensity_matrix)
        noise = np.std(intensity_matrix[intensity_matrix < np.percentile(intensity_matrix, 20)])  # 하위 20%를 노이즈로 가정
        return float(signal / noise) if noise > 0 else 0
    except:
        return 0


def calculate_spatial_coverage(ping_data):
    """공간적 커버리지 계산"""
    if not ping_data:
        return {}

    try:
        lats = [p.latitude for p in ping_data if p.latitude != 0]
        lons = [p.longitude for p in ping_data if p.longitude != 0]

        if not lats or not lons:
            return {}

        return {
            'lat_range_km': (max(lats) - min(lats)) * 111,  # 대략적 변환
            'lon_range_km': (max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats))),
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons),
            'coverage_area_km2': ((max(lats) - min(lats)) * 111) * ((max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats))))
        }
    except:
        return {}


def assess_temporal_consistency(ping_data):
    """시간적 일관성 평가"""
    if not ping_data:
        return {}

    try:
        # Ping 번호의 일관성 확인
        ping_numbers = [p.ping_number for p in ping_data]

        return {
            'ping_sequence_consistent': all(ping_numbers[i] <= ping_numbers[i+1] for i in range(len(ping_numbers)-1)),
            'ping_number_gaps': len(set(range(min(ping_numbers), max(ping_numbers)+1))) - len(set(ping_numbers)),
            'average_ping_interval': (max(ping_numbers) - min(ping_numbers)) / len(ping_numbers) if len(ping_numbers) > 1 else 0
        }
    except:
        return {}


def identify_unique_features(file_info, reader, extracted_data):
    """기종별 고유 특성 식별"""
    features = []

    try:
        manufacturer = file_info['manufacturer']
        model = file_info['model']

        # EdgeTech 특성
        if manufacturer == 'EdgeTech':
            features.append('edgetech_format')

            # 4205 모델 특성
            if model == '4205':
                features.append('dual_frequency_capable')
                features.append('chirp_sonar')
                features.append('high_resolution_bathymetry')

                # 데이터 구조에서 특성 확인
                if reader.ping_data and reader.ping_data[0].data.size > 6000:
                    features.append('high_sample_rate')

        # Klein 특성
        elif manufacturer == 'Klein':
            features.append('klein_format')

            # 3900 모델 특성
            if model == '3900':
                features.append('dual_frequency_capable')
                features.append('sidescan_bathymetry')
                features.append('backscatter_analysis')

                # Klein 특유의 패킷 구조 확인
                if 'kleinv4_data_page' in str(extracted_data):
                    features.append('klein_v4_protocol')

        # 데이터 특성 기반 특성 식별
        intensity_images = extracted_data.get('intensity_images', {})
        if intensity_images:
            combined_img = intensity_images.get('combined', np.array([]))

            # 해상도 특성
            if combined_img.size > 0:
                if combined_img.shape[1] > 6500:  # 높은 range resolution
                    features.append('high_range_resolution')
                elif combined_img.shape[1] < 6500:
                    features.append('standard_range_resolution')

                # 동적 범위 특성
                dynamic_range = combined_img.max() - combined_img.min()
                if dynamic_range > 30000:
                    features.append('high_dynamic_range')
                elif dynamic_range < 5000:
                    features.append('standard_dynamic_range')

    except Exception as e:
        logger.warning(f"고유 특성 식별 중 오류: {e}")

    return features


def compare_sonar_systems(analysis_results):
    """소나 시스템 간 비교 분석"""

    logger.info("소나 시스템 간 비교 분석 시작")

    if len(analysis_results) < 2:
        logger.warning("비교할 시스템이 부족합니다")
        return {}

    comparison = {
        'manufacturer_comparison': {},
        'performance_comparison': {},
        'data_format_comparison': {},
        'feature_comparison': {},
        'recommendations': {}
    }

    try:
        # 제조사별 그룹화
        edgetech_systems = {k: v for k, v in analysis_results.items() if v['manufacturer'] == 'EdgeTech'}
        klein_systems = {k: v for k, v in analysis_results.items() if v['manufacturer'] == 'Klein'}

        # 1. 제조사별 비교
        comparison['manufacturer_comparison'] = {
            'EdgeTech': {
                'count': len(edgetech_systems),
                'models': list(set(s['model'] for s in edgetech_systems.values())),
                'frequencies': list(set(s['frequency'] for s in edgetech_systems.values())),
                'avg_pings': np.mean([s['file_characteristics'].get('total_pings', 0) for s in edgetech_systems.values()]) if edgetech_systems else 0,
                'unique_features': list(set().union(*[s['unique_features'] for s in edgetech_systems.values()])) if edgetech_systems else []
            },
            'Klein': {
                'count': len(klein_systems),
                'models': list(set(s['model'] for s in klein_systems.values())),
                'frequencies': list(set(s['frequency'] for s in klein_systems.values())),
                'avg_pings': np.mean([s['file_characteristics'].get('total_pings', 0) for s in klein_systems.values()]) if klein_systems else 0,
                'unique_features': list(set().union(*[s['unique_features'] for s in klein_systems.values()])) if klein_systems else []
            }
        }

        # 2. 성능 비교
        all_systems = list(analysis_results.values())
        performance_metrics = [s.get('performance_metrics', {}) for s in all_systems]

        comparison['performance_comparison'] = {
            'data_completeness': {
                'coordinate_completeness': [pm.get('data_completeness', {}).get('pings_with_coordinates', 0) for pm in performance_metrics],
                'intensity_completeness': [pm.get('data_completeness', {}).get('pings_with_intensity', 0) for pm in performance_metrics]
            },
            'intensity_quality': {
                'dynamic_range': [pm.get('intensity_statistics', {}).get('dynamic_range', 0) for pm in performance_metrics],
                'snr_estimate': [pm.get('intensity_statistics', {}).get('signal_to_noise_estimate', 0) for pm in performance_metrics]
            },
            'spatial_coverage': [pm.get('spatial_coverage', {}).get('coverage_area_km2', 0) for pm in performance_metrics]
        }

        # 3. 데이터 포맷 비교
        data_formats = [s.get('data_format', {}) for s in all_systems]
        comparison['data_format_comparison'] = {
            'sample_counts': {
                'combined_width': [df.get('combined_shape', [0, 0])[1] if df.get('combined_shape') else 0 for df in data_formats],
                'port_width': [df.get('port_shape', [0, 0])[1] if df.get('port_shape') else 0 for df in data_formats],
                'starboard_width': [df.get('starboard_shape', [0, 0])[1] if df.get('starboard_shape') else 0 for df in data_formats]
            },
            'value_ranges': {
                'max_values': [df.get('value_range', {}).get('combined', [0, 0])[1] if df.get('value_range', {}).get('combined') else 0 for df in data_formats]
            }
        }

        # 4. 특성 비교
        all_features = list(set().union(*[s['unique_features'] for s in all_systems]))
        feature_matrix = {}
        for feature in all_features:
            feature_matrix[feature] = {
                system_id: feature in system_data['unique_features']
                for system_id, system_data in analysis_results.items()
            }
        comparison['feature_comparison'] = feature_matrix

        # 5. 권장사항 생성
        comparison['recommendations'] = generate_recommendations(analysis_results, comparison)

        logger.info("소나 시스템 비교 분석 완료")

    except Exception as e:
        logger.error(f"비교 분석 중 오류: {e}")

    return comparison


def generate_recommendations(analysis_results, comparison):
    """분석 결과 기반 권장사항 생성"""

    recommendations = {
        'processing_optimization': [],
        'quality_improvement': [],
        'feature_utilization': [],
        'system_specific': {}
    }

    try:
        # 처리 최적화 권장사항
        max_samples = max([
            s.get('data_format', {}).get('combined_shape', [0, 0])[1]
            for s in analysis_results.values()
        ])

        if max_samples > 6500:
            recommendations['processing_optimization'].append(
                "고해상도 데이터(>6500 samples)에 대해 메모리 효율적 처리 필요"
            )

        # 품질 개선 권장사항
        snr_values = [
            s.get('performance_metrics', {}).get('intensity_statistics', {}).get('signal_to_noise_estimate', 0)
            for s in analysis_results.values()
        ]

        if any(snr < 10 for snr in snr_values):
            recommendations['quality_improvement'].append(
                "낮은 SNR 시스템에 대해 노이즈 필터링 적용 권장"
            )

        # 특성 활용 권장사항
        edgetech_count = len([s for s in analysis_results.values() if s['manufacturer'] == 'EdgeTech'])
        klein_count = len([s for s in analysis_results.values() if s['manufacturer'] == 'Klein'])

        if edgetech_count > 0 and klein_count > 0:
            recommendations['feature_utilization'].append(
                "다중 제조사 데이터 융합을 통한 성능 향상 가능"
            )

        # 시스템별 권장사항
        for system_id, system_data in analysis_results.items():
            system_rec = []

            if 'high_dynamic_range' in system_data['unique_features']:
                system_rec.append("높은 동적 범위 활용한 세밀한 객체 탐지 가능")

            if 'dual_frequency_capable' in system_data['unique_features']:
                system_rec.append("듀얼 주파수 데이터 융합 처리 권장")

            recommendations['system_specific'][system_id] = system_rec

    except Exception as e:
        logger.warning(f"권장사항 생성 중 오류: {e}")

    return recommendations


def save_analysis_results(analysis_results, comparison_results):
    """분석 결과 저장"""

    output_dir = Path("analysis_results/sonar_system_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # JSON 데이터 저장
        full_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'individual_analysis': analysis_results,
            'comparative_analysis': comparison_results,
            'summary': {
                'total_systems_analyzed': len(analysis_results),
                'manufacturers': list(set(s['manufacturer'] for s in analysis_results.values())),
                'models': list(set(s['model'] for s in analysis_results.values())),
                'frequencies': list(set(s['frequency'] for s in analysis_results.values()))
            }
        }

        json_file = output_dir / "sonar_system_analysis_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

        # 마크다운 보고서 생성
        generate_markdown_report(full_results, output_dir)

        logger.info(f"분석 결과 저장 완료: {output_dir}")

    except Exception as e:
        logger.error(f"결과 저장 중 오류: {e}")


def generate_markdown_report(full_results, output_dir):
    """마크다운 보고서 생성"""

    analysis_results = full_results['individual_analysis']
    comparison_results = full_results['comparative_analysis']

    report_lines = []
    report_lines.append("# 사이드 스캔 소나 기종별 차이점 분석 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # 요약
    summary = full_results['summary']
    report_lines.append("## 📊 **분석 요약**")
    report_lines.append(f"- **분석된 시스템**: {summary['total_systems_analyzed']}개")
    report_lines.append(f"- **제조사**: {', '.join(summary['manufacturers'])}")
    report_lines.append(f"- **모델**: {', '.join(summary['models'])}")
    report_lines.append(f"- **주파수**: {', '.join(summary['frequencies'])}")
    report_lines.append("")

    # 개별 시스템 분석
    report_lines.append("## 🔍 **개별 시스템 분석**")
    report_lines.append("")

    for system_id, system_data in analysis_results.items():
        report_lines.append(f"### {system_data['manufacturer']} {system_data['model']} ({system_data['frequency']})")

        # 파일 특성
        file_char = system_data.get('file_characteristics', {})
        report_lines.append(f"- **총 Ping 수**: {file_char.get('total_pings', 'N/A'):,}")
        report_lines.append(f"- **소나 채널**: {file_char.get('sonar_channels', 'N/A')}")

        # 데이터 포맷
        data_format = system_data.get('data_format', {})
        if data_format.get('combined_shape'):
            report_lines.append(f"- **데이터 크기**: {data_format['combined_shape'][0]} × {data_format['combined_shape'][1]}")

        # 성능 지표
        perf = system_data.get('performance_metrics', {})
        if perf:
            intensity_stats = perf.get('intensity_statistics', {})
            report_lines.append(f"- **평균 강도**: {intensity_stats.get('mean', 0):.1f}")
            report_lines.append(f"- **동적 범위**: {intensity_stats.get('dynamic_range', 0):.1f}")
            report_lines.append(f"- **SNR 추정**: {intensity_stats.get('signal_to_noise_estimate', 0):.2f}")

        # 고유 특성
        features = system_data.get('unique_features', [])
        if features:
            report_lines.append(f"- **고유 특성**: {', '.join(features)}")

        report_lines.append("")

    # 비교 분석
    if comparison_results:
        report_lines.append("## ⚖️ **비교 분석**")
        report_lines.append("")

        # 제조사별 비교
        mfg_comp = comparison_results.get('manufacturer_comparison', {})
        if mfg_comp:
            report_lines.append("### 제조사별 비교")

            for mfg, data in mfg_comp.items():
                if data['count'] > 0:
                    report_lines.append(f"**{mfg}**:")
                    report_lines.append(f"- 분석된 시스템: {data['count']}개")
                    report_lines.append(f"- 평균 Ping 수: {data['avg_pings']:,.0f}")
                    report_lines.append(f"- 고유 특성: {', '.join(data['unique_features'])}")
                    report_lines.append("")

        # 성능 비교
        perf_comp = comparison_results.get('performance_comparison', {})
        if perf_comp:
            report_lines.append("### 성능 비교")

            dynamic_ranges = perf_comp.get('intensity_quality', {}).get('dynamic_range', [])
            if dynamic_ranges:
                report_lines.append(f"- **동적 범위**: 최소 {min(dynamic_ranges):.0f}, 최대 {max(dynamic_ranges):.0f}")

            coverage_areas = perf_comp.get('spatial_coverage', [])
            if coverage_areas and any(area > 0 for area in coverage_areas):
                valid_areas = [area for area in coverage_areas if area > 0]
                report_lines.append(f"- **공간 커버리지**: 평균 {np.mean(valid_areas):.2f} km²")

            report_lines.append("")

    # 권장사항
    recommendations = comparison_results.get('recommendations', {})
    if recommendations:
        report_lines.append("## 💡 **권장사항**")
        report_lines.append("")

        for category, recs in recommendations.items():
            if recs and category != 'system_specific':
                category_name = {
                    'processing_optimization': '처리 최적화',
                    'quality_improvement': '품질 개선',
                    'feature_utilization': '특성 활용'
                }.get(category, category)

                report_lines.append(f"### {category_name}")
                for rec in recs:
                    report_lines.append(f"- {rec}")
                report_lines.append("")

    # 보고서 저장
    report_file = output_dir / "SONAR_SYSTEM_ANALYSIS_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"마크다운 보고서 저장: {report_file}")


def main():
    """메인 실행 함수"""
    logger.info("사이드 스캔 소나 기종별 차이점 분석 시작")

    try:
        analysis_results, comparison_results = analyze_sonar_system_differences()

        # 요약 출력
        print("\n" + "="*70)
        print("사이드 스캔 소나 기종별 차이점 분석 완료")
        print("="*70)

        if analysis_results:
            print(f"📊 분석된 시스템: {len(analysis_results)}개")

            manufacturers = list(set(s['manufacturer'] for s in analysis_results.values()))
            models = list(set(s['model'] for s in analysis_results.values()))

            print(f"🏭 제조사: {', '.join(manufacturers)}")
            print(f"📱 모델: {', '.join(models)}")

            # 주요 차이점 요약
            if len(manufacturers) > 1:
                print(f"\n🔍 주요 차이점:")

                # 데이터 크기 비교
                data_sizes = []
                for s in analysis_results.values():
                    shape = s.get('data_format', {}).get('combined_shape')
                    if shape:
                        data_sizes.append(shape[1])

                if data_sizes:
                    print(f"   - 샘플 수 범위: {min(data_sizes)} ~ {max(data_sizes)}")

                # 동적 범위 비교
                dynamic_ranges = []
                for s in analysis_results.values():
                    dr = s.get('performance_metrics', {}).get('intensity_statistics', {}).get('dynamic_range', 0)
                    if dr > 0:
                        dynamic_ranges.append(dr)

                if dynamic_ranges:
                    print(f"   - 동적 범위: {min(dynamic_ranges):.0f} ~ {max(dynamic_ranges):.0f}")

        print(f"\n📁 상세 결과: analysis_results/sonar_system_analysis/")
        return 0

    except Exception as e:
        logger.error(f"분석 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())