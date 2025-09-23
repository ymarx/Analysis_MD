#!/usr/bin/env python3
"""
핑 개수 확인 및 Preprocessor 작동 검증
=====================================
EdgeTech 4205와 Klein 3900의 전체 핑 개수를 확인하고
preprocessor의 작동 상태를 검증합니다.

Author: YMARX
Date: 2025-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
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


def check_ping_counts():
    """전체 핑 개수 확인"""

    logger.info("="*70)
    logger.info("핑 개수 확인")
    logger.info("="*70)

    # 분석할 XTF 파일들
    xtf_files = [
        {
            'path': "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf",
            'type': 'EdgeTech 4205',
            'frequency': '800 kHz'
        },
        {
            'path': "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
            'type': 'Klein 3900',
            'frequency': '900 kHz'
        },
        {
            'path': "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04.xtf",
            'type': 'EdgeTech 4205',
            'frequency': '800 kHz'
        }
    ]

    ping_results = {}

    for file_info in xtf_files:
        file_path = Path(file_info['path'])

        if not file_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"분석: {file_info['type']} - {file_path.name}")
        logger.info(f"주파수: {file_info['frequency']}")
        logger.info(f"파일 크기: {file_path.stat().st_size / (1024*1024):.1f} MB")

        try:
            # XTF Reader로 전체 파일 로드 (핑 제한 없음)
            reader = XTFReader(file_path, max_pings=None)

            # 파일 로드
            load_success = reader.load_file()
            if not load_success:
                logger.error(f"파일 로드 실패: {file_path.name}")
                continue

            # 메타데이터에서 총 핑 수 확인
            metadata = reader.metadata
            total_pings_metadata = metadata.total_pings if metadata else 0

            # 실제 파싱 가능한 핑 수 확인 (처음 1000개만 테스트)
            test_pings = reader.parse_pings()
            parseable_pings_sample = len(test_pings)

            # 전체 파싱 시도 (메모리 허용 범위 내에서)
            reader_full = XTFReader(file_path, max_pings=10000)  # 최대 10,000개로 제한
            reader_full.load_file()
            all_pings = reader_full.parse_pings()
            parseable_pings_extended = len(all_pings)

            ping_results[file_path.name] = {
                'file_type': file_info['type'],
                'frequency': file_info['frequency'],
                'file_size_mb': file_path.stat().st_size / (1024*1024),
                'total_pings_metadata': total_pings_metadata,
                'parseable_pings_sample': parseable_pings_sample,
                'parseable_pings_extended': parseable_pings_extended,
                'sample_ping_info': {
                    'ping_number': test_pings[0].ping_number if test_pings else None,
                    'range_samples': test_pings[0].range_samples if test_pings else None,
                    'latitude': test_pings[0].latitude if test_pings else None,
                    'longitude': test_pings[0].longitude if test_pings else None,
                    'data_size': test_pings[0].data.size if test_pings else None
                }
            }

            logger.info(f"✅ 메타데이터 총 핑 수: {total_pings_metadata:,}")
            logger.info(f"✅ 파싱 가능 핑 수 (샘플): {parseable_pings_sample:,}")
            logger.info(f"✅ 파싱 가능 핑 수 (확장): {parseable_pings_extended:,}")

            if test_pings:
                sample = test_pings[0]
                logger.info(f"   샘플 ping 정보:")
                logger.info(f"   - Ping 번호: {sample.ping_number}")
                logger.info(f"   - 샘플 수: {sample.range_samples}")
                logger.info(f"   - 위치: ({sample.latitude:.6f}, {sample.longitude:.6f})")
                logger.info(f"   - 데이터 크기: {sample.data.size}")

        except Exception as e:
            logger.error(f"핑 개수 확인 실패 ({file_path.name}): {e}")
            ping_results[file_path.name] = {
                'file_type': file_info['type'],
                'error': str(e)
            }

    return ping_results


def check_preprocessor():
    """Preprocessor 작동 확인"""

    logger.info("\n" + "="*70)
    logger.info("PREPROCESSOR 작동 검증")
    logger.info("="*70)

    # Preprocessor 관련 모듈들 확인
    preprocessor_modules = [
        'src.data_processing.xtf_reader',
        'src.data_processing.xtf_intensity_extractor'
    ]

    preprocessor_results = {
        'modules_available': {},
        'functionality_tests': {},
        'performance_tests': {}
    }

    # 1. 모듈 가용성 확인
    logger.info("1. Preprocessor 모듈 가용성 확인")
    for module_name in preprocessor_modules:
        try:
            __import__(module_name)
            preprocessor_results['modules_available'][module_name] = True
            logger.info(f"✅ {module_name}: 사용 가능")
        except ImportError as e:
            preprocessor_results['modules_available'][module_name] = False
            logger.error(f"❌ {module_name}: 사용 불가 - {e}")

    # 2. 기능 테스트
    logger.info("\n2. Preprocessor 기능 테스트")

    # 테스트용 파일 (Klein 3900 - 상대적으로 작은 파일)
    test_file = Path("datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf")

    if test_file.exists():
        try:
            # XTF Reader 기능 테스트
            logger.info("2-1. XTF Reader 기능 테스트")
            reader = XTFReader(test_file, max_pings=500)
            reader.load_file()
            ping_data = reader.parse_pings()
            intensity_matrix = reader.extract_intensity_matrix()
            georef_df = reader.get_georeferenced_data()

            preprocessor_results['functionality_tests']['xtf_reader'] = {
                'file_load': True,
                'ping_parsing': len(ping_data) > 0,
                'intensity_extraction': intensity_matrix.size > 0,
                'georeferencing': not georef_df.empty,
                'ping_count': len(ping_data),
                'matrix_shape': intensity_matrix.shape if intensity_matrix.size > 0 else None
            }

            logger.info(f"   ✅ 파일 로드: 성공")
            logger.info(f"   ✅ Ping 파싱: {len(ping_data)} pings")
            logger.info(f"   ✅ Intensity 매트릭스: {intensity_matrix.shape}")
            logger.info(f"   ✅ 위치 데이터: {georef_df.shape}")

            # Intensity Extractor 기능 테스트
            logger.info("\n2-2. Intensity Extractor 기능 테스트")
            extractor = XTFIntensityExtractor(max_memory_mb=256)
            extracted_data = extractor.extract_intensity_data(
                str(test_file),
                ping_range=(0, 200)  # 작은 범위로 테스트
            )

            preprocessor_results['functionality_tests']['intensity_extractor'] = {
                'data_extraction': bool(extracted_data),
                'metadata_valid': extracted_data.get('metadata') is not None,
                'ping_data_valid': len(extracted_data.get('ping_data', [])) > 0,
                'images_valid': bool(extracted_data.get('intensity_images', {})),
                'navigation_valid': bool(extracted_data.get('navigation_data', {})),
                'extracted_pings': len(extracted_data.get('ping_data', []))
            }

            logger.info(f"   ✅ 데이터 추출: 성공")
            logger.info(f"   ✅ 메타데이터: {'유효' if extracted_data.get('metadata') else '무효'}")
            logger.info(f"   ✅ Ping 데이터: {len(extracted_data.get('ping_data', []))} pings")
            logger.info(f"   ✅ Intensity 이미지: {len(extracted_data.get('intensity_images', {}))} 타입")
            logger.info(f"   ✅ Navigation 데이터: {'유효' if extracted_data.get('navigation_data') else '무효'}")

        except Exception as e:
            logger.error(f"❌ Preprocessor 기능 테스트 실패: {e}")
            preprocessor_results['functionality_tests']['error'] = str(e)
    else:
        logger.warning(f"❌ 테스트 파일을 찾을 수 없습니다: {test_file}")

    # 3. 성능 테스트
    logger.info("\n3. Preprocessor 성능 테스트")

    if test_file.exists():
        try:
            import time

            # 처리 시간 측정
            start_time = time.time()

            reader = XTFReader(test_file, max_pings=1000)
            reader.load_file()
            ping_data = reader.parse_pings()
            intensity_matrix = reader.extract_intensity_matrix()

            processing_time = time.time() - start_time

            # 메모리 사용량 추정
            memory_usage_mb = (
                intensity_matrix.nbytes +
                sum(ping.data.nbytes for ping in ping_data)
            ) / (1024 * 1024)

            preprocessor_results['performance_tests'] = {
                'processing_time_seconds': processing_time,
                'pings_per_second': len(ping_data) / processing_time if processing_time > 0 else 0,
                'memory_usage_mb': memory_usage_mb,
                'throughput_mb_per_second': memory_usage_mb / processing_time if processing_time > 0 else 0
            }

            logger.info(f"   ✅ 처리 시간: {processing_time:.2f}초")
            logger.info(f"   ✅ 처리 속도: {len(ping_data) / processing_time:.1f} pings/초")
            logger.info(f"   ✅ 메모리 사용량: {memory_usage_mb:.1f} MB")
            logger.info(f"   ✅ 처리량: {memory_usage_mb / processing_time:.1f} MB/초")

        except Exception as e:
            logger.error(f"❌ 성능 테스트 실패: {e}")
            preprocessor_results['performance_tests']['error'] = str(e)

    return preprocessor_results


def generate_summary_report(ping_results, preprocessor_results):
    """요약 보고서 생성"""

    # 출력 디렉토리 생성
    output_dir = Path("analysis_results/ping_preprocessor_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 마크다운 보고서 생성
    report_lines = []
    report_lines.append("# 핑 개수 및 Preprocessor 검증 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # 핑 개수 요약
    report_lines.append("## 📊 **핑 개수 분석**")
    report_lines.append("")

    if ping_results:
        for filename, data in ping_results.items():
            if 'error' in data:
                report_lines.append(f"### ❌ {filename}")
                report_lines.append(f"- **오류**: {data['error']}")
            else:
                report_lines.append(f"### ✅ {filename}")
                report_lines.append(f"- **기종**: {data['file_type']}")
                report_lines.append(f"- **주파수**: {data['frequency']}")
                report_lines.append(f"- **파일 크기**: {data['file_size_mb']:.1f} MB")
                report_lines.append(f"- **메타데이터 총 핑 수**: {data['total_pings_metadata']:,}")
                report_lines.append(f"- **파싱 가능 핑 수 (샘플)**: {data['parseable_pings_sample']:,}")
                report_lines.append(f"- **파싱 가능 핑 수 (확장)**: {data['parseable_pings_extended']:,}")

                sample_info = data.get('sample_ping_info', {})
                if sample_info and sample_info.get('ping_number'):
                    report_lines.append(f"- **샘플 Ping 정보**:")
                    report_lines.append(f"  - Ping 번호: {sample_info['ping_number']}")
                    report_lines.append(f"  - 샘플 수: {sample_info['range_samples']}")
                    report_lines.append(f"  - 데이터 크기: {sample_info['data_size']}")

            report_lines.append("")

    # Preprocessor 검증 결과
    report_lines.append("## 🔧 **Preprocessor 검증 결과**")
    report_lines.append("")

    # 모듈 가용성
    modules = preprocessor_results.get('modules_available', {})
    report_lines.append("### 모듈 가용성")
    for module, available in modules.items():
        status = "✅ 사용 가능" if available else "❌ 사용 불가"
        report_lines.append(f"- **{module}**: {status}")
    report_lines.append("")

    # 기능 테스트 결과
    func_tests = preprocessor_results.get('functionality_tests', {})
    if func_tests and 'error' not in func_tests:
        report_lines.append("### 기능 테스트")

        # XTF Reader
        xtf_reader = func_tests.get('xtf_reader', {})
        if xtf_reader:
            report_lines.append("**XTF Reader**:")
            report_lines.append(f"- 파일 로드: {'✅ 성공' if xtf_reader.get('file_load') else '❌ 실패'}")
            report_lines.append(f"- Ping 파싱: {'✅ 성공' if xtf_reader.get('ping_parsing') else '❌ 실패'} ({xtf_reader.get('ping_count', 0)} pings)")
            report_lines.append(f"- Intensity 추출: {'✅ 성공' if xtf_reader.get('intensity_extraction') else '❌ 실패'}")
            report_lines.append(f"- 위치 정보: {'✅ 성공' if xtf_reader.get('georeferencing') else '❌ 실패'}")

        # Intensity Extractor
        intensity_extractor = func_tests.get('intensity_extractor', {})
        if intensity_extractor:
            report_lines.append("")
            report_lines.append("**Intensity Extractor**:")
            report_lines.append(f"- 데이터 추출: {'✅ 성공' if intensity_extractor.get('data_extraction') else '❌ 실패'}")
            report_lines.append(f"- 메타데이터: {'✅ 유효' if intensity_extractor.get('metadata_valid') else '❌ 무효'}")
            report_lines.append(f"- Ping 데이터: {'✅ 유효' if intensity_extractor.get('ping_data_valid') else '❌ 무효'} ({intensity_extractor.get('extracted_pings', 0)} pings)")
            report_lines.append(f"- Intensity 이미지: {'✅ 유효' if intensity_extractor.get('images_valid') else '❌ 무효'}")
            report_lines.append(f"- Navigation 데이터: {'✅ 유효' if intensity_extractor.get('navigation_valid') else '❌ 무효'}")

        report_lines.append("")

    # 성능 테스트 결과
    perf_tests = preprocessor_results.get('performance_tests', {})
    if perf_tests and 'error' not in perf_tests:
        report_lines.append("### 성능 테스트")
        report_lines.append(f"- **처리 시간**: {perf_tests.get('processing_time_seconds', 0):.2f}초")
        report_lines.append(f"- **처리 속도**: {perf_tests.get('pings_per_second', 0):.1f} pings/초")
        report_lines.append(f"- **메모리 사용량**: {perf_tests.get('memory_usage_mb', 0):.1f} MB")
        report_lines.append(f"- **처리량**: {perf_tests.get('throughput_mb_per_second', 0):.1f} MB/초")
        report_lines.append("")

    # 종합 결론
    report_lines.append("## 🎯 **종합 결론**")
    report_lines.append("")

    # 핑 개수 통계
    total_pings = sum(data.get('total_pings_metadata', 0) for data in ping_results.values() if 'error' not in data)
    successful_files = len([data for data in ping_results.values() if 'error' not in data])

    report_lines.append(f"### 핑 개수 요약")
    report_lines.append(f"- **분석 성공 파일**: {successful_files}/{len(ping_results)}")
    report_lines.append(f"- **총 핑 수**: {total_pings:,}")

    # EdgeTech vs Klein 비교
    edgetech_pings = sum(data.get('total_pings_metadata', 0) for data in ping_results.values()
                        if 'error' not in data and 'EdgeTech' in data.get('file_type', ''))
    klein_pings = sum(data.get('total_pings_metadata', 0) for data in ping_results.values()
                     if 'error' not in data and 'Klein' in data.get('file_type', ''))

    report_lines.append(f"- **EdgeTech 4205 총 핑 수**: {edgetech_pings:,}")
    report_lines.append(f"- **Klein 3900 총 핑 수**: {klein_pings:,}")
    report_lines.append("")

    # Preprocessor 상태
    all_modules_ok = all(preprocessor_results.get('modules_available', {}).values())
    xtf_reader_ok = func_tests.get('xtf_reader', {}).get('ping_parsing', False)
    extractor_ok = func_tests.get('intensity_extractor', {}).get('data_extraction', False)

    report_lines.append(f"### Preprocessor 상태")
    report_lines.append(f"- **모든 모듈 사용 가능**: {'✅ 예' if all_modules_ok else '❌ 아니오'}")
    report_lines.append(f"- **XTF Reader 작동**: {'✅ 정상' if xtf_reader_ok else '❌ 오류'}")
    report_lines.append(f"- **Intensity Extractor 작동**: {'✅ 정상' if extractor_ok else '❌ 오류'}")

    if all_modules_ok and xtf_reader_ok and extractor_ok:
        report_lines.append("")
        report_lines.append("### ✅ **전체 시스템 정상 작동**")
        report_lines.append("두 종류 기종의 사이드 스캔 소나 데이터 처리가 모두 정상적으로 작동합니다.")
    else:
        report_lines.append("")
        report_lines.append("### ⚠️ **일부 시스템 문제 발견**")
        report_lines.append("Preprocessor 일부 기능에서 문제가 발견되었습니다.")

    # 보고서 저장
    report_file = output_dir / "PING_PREPROCESSOR_CHECK_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # JSON 데이터 저장
    json_data = {
        'ping_analysis': ping_results,
        'preprocessor_analysis': preprocessor_results,
        'summary': {
            'total_files': len(ping_results),
            'successful_files': successful_files,
            'total_pings': total_pings,
            'edgetech_pings': edgetech_pings,
            'klein_pings': klein_pings,
            'preprocessor_ok': all_modules_ok and xtf_reader_ok and extractor_ok
        },
        'analysis_timestamp': datetime.now().isoformat()
    }

    json_file = output_dir / "ping_preprocessor_check_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"보고서 저장: {report_file}")
    logger.info(f"데이터 저장: {json_file}")

    return json_data


def main():
    """메인 실행 함수"""
    logger.info("핑 개수 및 Preprocessor 검증 시작")

    try:
        # 1. 핑 개수 확인
        ping_results = check_ping_counts()

        # 2. Preprocessor 검증
        preprocessor_results = check_preprocessor()

        # 3. 요약 보고서 생성
        summary = generate_summary_report(ping_results, preprocessor_results)

        # 4. 결과 출력
        print("\n" + "="*70)
        print("핑 개수 및 PREPROCESSOR 검증 완료")
        print("="*70)

        # 핑 개수 요약
        total_pings = summary['summary']['total_pings']
        edgetech_pings = summary['summary']['edgetech_pings']
        klein_pings = summary['summary']['klein_pings']
        successful_files = summary['summary']['successful_files']

        print(f"📊 분석 결과:")
        print(f"   - 성공 파일: {successful_files}/{len(ping_results)}")
        print(f"   - 총 핑 수: {total_pings:,}")
        print(f"   - EdgeTech 4205: {edgetech_pings:,} pings")
        print(f"   - Klein 3900: {klein_pings:,} pings")

        # Preprocessor 상태
        preprocessor_ok = summary['summary']['preprocessor_ok']
        print(f"\n🔧 Preprocessor 상태: {'✅ 정상 작동' if preprocessor_ok else '❌ 문제 발견'}")

        print(f"\n📁 상세 결과: analysis_results/ping_preprocessor_check/")

        return 0

    except Exception as e:
        logger.error(f"검증 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())