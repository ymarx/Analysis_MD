#!/usr/bin/env python3
"""
XTF Reader & Intensity Extractor 검증 테스트
=============================================
두 종류 기종(EdgeTech 4205, Klein 3900)의 사이드 스캔 소나에서
패킷 정보, 메타데이터, 넘파이 배열 형태의 강도 데이터 추출을 검증합니다.

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

# Import custom modules
from src.data_processing.xtf_reader import XTFReader, BatchXTFProcessor, PingData, XTFMetadata
from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor, IntensityMetadata, IntensityPing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_xtf_reader_extraction():
    """XTF Reader의 패킷 정보 및 메타데이터 추출 검증"""

    logger.info("="*70)
    logger.info("XTF READER 검증 테스트")
    logger.info("="*70)

    # 테스트할 XTF 파일들 (두 기종)
    test_files = [
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

    results = {}

    for file_info in test_files:
        file_path = Path(file_info['path'])

        if not file_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"테스트: {file_info['type']} - {file_path.name}")
        logger.info(f"주파수: {file_info['frequency']}")

        try:
            # XTF Reader 초기화 및 로드 (메모리 효율성을 위해 1000 ping으로 제한)
            reader = XTFReader(file_path, max_pings=1000)

            # 1. 파일 로드 테스트
            load_success = reader.load_file()
            logger.info(f"파일 로드: {'✅ 성공' if load_success else '❌ 실패'}")

            if not load_success:
                continue

            # 2. 메타데이터 검증
            metadata = reader.metadata
            if metadata:
                logger.info(f"메타데이터 생성: ✅ 성공")
                logger.info(f"  - 총 ping 수: {metadata.total_pings:,}")
                logger.info(f"  - 소나 채널 수: {metadata.num_sonar_channels}")
                logger.info(f"  - 주파수 정보: {metadata.frequency_info}")
                logger.info(f"  - 좌표 범위: {metadata.coordinate_bounds}")
                logger.info(f"  - 시간 범위: {metadata.time_range}")
            else:
                logger.warning("메타데이터 생성 실패")

            # 3. Ping 데이터 파싱 테스트
            ping_data = reader.parse_pings()
            logger.info(f"Ping 데이터 파싱: {'✅ 성공' if ping_data else '❌ 실패'}")
            logger.info(f"  - 파싱된 ping 수: {len(ping_data):,}")

            if ping_data:
                # 샘플 ping 데이터 검증
                sample_ping = ping_data[0]
                logger.info(f"  - 샘플 ping 정보:")
                logger.info(f"    * Ping 번호: {sample_ping.ping_number}")
                logger.info(f"    * 타임스탬프: {sample_ping.timestamp}")
                logger.info(f"    * 위도: {sample_ping.latitude:.6f}")
                logger.info(f"    * 경도: {sample_ping.longitude:.6f}")
                logger.info(f"    * 주파수: {sample_ping.frequency} Hz")
                logger.info(f"    * 데이터 크기: {sample_ping.data.shape if sample_ping.data.size > 0 else 'N/A'}")
                logger.info(f"    * 샘플 수: {sample_ping.range_samples}")

            # 4. Intensity 매트릭스 추출 테스트
            intensity_matrix = reader.extract_intensity_matrix()
            logger.info(f"Intensity 매트릭스 추출: {'✅ 성공' if intensity_matrix.size > 0 else '❌ 실패'}")
            if intensity_matrix.size > 0:
                logger.info(f"  - 매트릭스 크기: {intensity_matrix.shape}")
                logger.info(f"  - 데이터 타입: {intensity_matrix.dtype}")
                logger.info(f"  - 값 범위: [{intensity_matrix.min():.3f}, {intensity_matrix.max():.3f}]")
                logger.info(f"  - 평균 강도: {intensity_matrix.mean():.3f}")

            # 5. 위치 정보 데이터프레임 테스트
            georef_df = reader.get_georeferenced_data()
            logger.info(f"위치 정보 데이터프레임: {'✅ 성공' if not georef_df.empty else '❌ 실패'}")
            if not georef_df.empty:
                logger.info(f"  - 데이터프레임 크기: {georef_df.shape}")
                logger.info(f"  - 컬럼: {list(georef_df.columns)}")

                # 좌표 통계
                if 'latitude' in georef_df.columns and 'longitude' in georef_df.columns:
                    lat_stats = georef_df['latitude'].describe()
                    lon_stats = georef_df['longitude'].describe()
                    logger.info(f"  - 위도 범위: [{lat_stats['min']:.6f}, {lat_stats['max']:.6f}]")
                    logger.info(f"  - 경도 범위: [{lon_stats['min']:.6f}, {lon_stats['max']:.6f}]")

            # 6. 요약 정보 테스트
            summary = reader.get_summary()
            logger.info(f"요약 정보: {'✅ 성공' if summary else '❌ 실패'}")

            # 결과 저장
            results[file_path.name] = {
                'file_type': file_info['type'],
                'frequency': file_info['frequency'],
                'load_success': load_success,
                'metadata_valid': metadata is not None,
                'ping_count': len(ping_data) if ping_data else 0,
                'intensity_matrix_shape': intensity_matrix.shape if intensity_matrix.size > 0 else None,
                'georef_data_valid': not georef_df.empty,
                'summary_valid': bool(summary)
            }

        except Exception as e:
            logger.error(f"XTF Reader 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return results


def test_intensity_extractor():
    """Intensity Extractor의 강도 데이터 추출 검증"""

    logger.info("="*70)
    logger.info("INTENSITY EXTRACTOR 검증 테스트")
    logger.info("="*70)

    # 테스트할 XTF 파일들
    test_files = [
        {
            'path': "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf",
            'type': 'EdgeTech 4205'
        },
        {
            'path': "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
            'type': 'Klein 3900'
        }
    ]

    results = {}

    # Intensity Extractor 초기화
    extractor = XTFIntensityExtractor(max_memory_mb=512)  # 메모리 제한

    for file_info in test_files:
        file_path = Path(file_info['path'])

        if not file_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {file_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"테스트: {file_info['type']} - {file_path.name}")

        try:
            # 강도 데이터 추출 (ping 제한으로 메모리 효율성)
            output_dir = f"analysis_results/xtf_extraction_test/{file_path.stem}"
            extracted_data = extractor.extract_intensity_data(
                str(file_path),
                output_dir=output_dir,
                ping_range=(0, 500)  # 처음 500 ping만 테스트
            )

            # 1. 메타데이터 검증
            metadata = extracted_data.get('metadata')
            if metadata:
                logger.info(f"메타데이터 추출: ✅ 성공")
                logger.info(f"  - 파일 경로: {metadata.file_path}")
                logger.info(f"  - Ping 수: {metadata.ping_count}")
                logger.info(f"  - 채널 수: {metadata.channel_count}")
                logger.info(f"  - 주파수: {metadata.frequency} Hz")
                logger.info(f"  - 샘플링 주파수: {metadata.sample_rate} Hz")
                logger.info(f"  - 거리 해상도: {metadata.range_resolution} m")
                logger.info(f"  - 시간 범위: {metadata.timestamp_range}")
                logger.info(f"  - 좌표 범위: {metadata.coordinate_bounds}")
            else:
                logger.warning("메타데이터 추출 실패")

            # 2. Ping 데이터 검증
            ping_data = extracted_data.get('ping_data', [])
            logger.info(f"Ping 데이터 추출: {'✅ 성공' if ping_data else '❌ 실패'}")
            logger.info(f"  - 추출된 ping 수: {len(ping_data)}")

            if ping_data:
                sample_ping = ping_data[0]
                logger.info(f"  - 샘플 ping 정보:")
                logger.info(f"    * Ping 번호: {sample_ping.ping_number}")
                logger.info(f"    * 타임스탬프: {sample_ping.timestamp}")
                logger.info(f"    * 위치: ({sample_ping.latitude:.6f}, {sample_ping.longitude:.6f})")
                logger.info(f"    * 방향: {sample_ping.heading}°")
                logger.info(f"    * Port 데이터: {sample_ping.port_intensity.shape}")
                logger.info(f"    * Starboard 데이터: {sample_ping.starboard_intensity.shape}")

            # 3. Intensity 이미지 검증
            intensity_images = extracted_data.get('intensity_images', {})
            logger.info(f"Intensity 이미지 생성: {'✅ 성공' if intensity_images else '❌ 실패'}")

            for img_type, img_array in intensity_images.items():
                if img_array.size > 0:
                    logger.info(f"  - {img_type.upper()} 이미지:")
                    logger.info(f"    * 크기: {img_array.shape}")
                    logger.info(f"    * 데이터 타입: {img_array.dtype}")
                    logger.info(f"    * 값 범위: [{img_array.min():.3f}, {img_array.max():.3f}]")
                    logger.info(f"    * 평균: {img_array.mean():.3f}")
                    logger.info(f"    * 표준편차: {img_array.std():.3f}")
                else:
                    logger.warning(f"  - {img_type.upper()} 이미지: 데이터 없음")

            # 4. Navigation 데이터 검증
            nav_data = extracted_data.get('navigation_data', {})
            logger.info(f"Navigation 데이터 추출: {'✅ 성공' if nav_data else '❌ 실패'}")

            if nav_data:
                for data_type, data_array in nav_data.items():
                    if isinstance(data_array, np.ndarray) and data_array.size > 0:
                        logger.info(f"  - {data_type}: {data_array.shape}, 범위 [{data_array.min():.3f}, {data_array.max():.3f}]")

            # 5. 데이터 품질 검증
            quality_score = assess_data_quality(extracted_data)
            logger.info(f"데이터 품질 점수: {quality_score:.2f}/100")

            # 결과 저장
            results[file_path.name] = {
                'file_type': file_info['type'],
                'extraction_success': bool(extracted_data),
                'metadata_valid': metadata is not None,
                'ping_count': len(ping_data),
                'intensity_images': {k: v.shape if v.size > 0 else None for k, v in intensity_images.items()},
                'navigation_data_valid': bool(nav_data),
                'quality_score': quality_score,
                'output_directory': output_dir
            }

        except Exception as e:
            logger.error(f"Intensity Extractor 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return results


def assess_data_quality(extracted_data):
    """추출된 데이터의 품질 평가"""
    score = 0

    # 메타데이터 품질 (25점)
    metadata = extracted_data.get('metadata')
    if metadata:
        score += 10
        if metadata.ping_count > 0:
            score += 5
        if metadata.frequency > 0:
            score += 5
        if metadata.coordinate_bounds:
            score += 5

    # Ping 데이터 품질 (25점)
    ping_data = extracted_data.get('ping_data', [])
    if ping_data:
        score += 10
        if len(ping_data) > 100:
            score += 5
        # 좌표 유효성 검사
        valid_coords = sum(1 for ping in ping_data if ping.latitude != 0 and ping.longitude != 0)
        if valid_coords > len(ping_data) * 0.8:  # 80% 이상 유효한 좌표
            score += 5
        # 강도 데이터 유효성
        valid_intensity = sum(1 for ping in ping_data if ping.port_intensity.size > 0 or ping.starboard_intensity.size > 0)
        if valid_intensity > len(ping_data) * 0.9:  # 90% 이상 유효한 강도 데이터
            score += 5

    # 이미지 품질 (25점)
    intensity_images = extracted_data.get('intensity_images', {})
    if intensity_images:
        valid_images = sum(1 for img in intensity_images.values() if img.size > 0)
        score += min(valid_images * 8, 25)  # 최대 25점

    # Navigation 데이터 품질 (25점)
    nav_data = extracted_data.get('navigation_data', {})
    if nav_data:
        valid_nav = sum(1 for data in nav_data.values() if isinstance(data, np.ndarray) and data.size > 0)
        score += min(valid_nav * 5, 25)  # 최대 25점

    return score


def test_batch_processing():
    """배치 프로세싱 테스트"""

    logger.info("="*70)
    logger.info("BATCH PROCESSING 검증 테스트")
    logger.info("="*70)

    # 모든 XTF 파일
    file_paths = [
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf",
        "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xtf",
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04.xtf"
    ]

    # 존재하는 파일만 필터링
    existing_files = [fp for fp in file_paths if Path(fp).exists()]
    logger.info(f"배치 처리할 파일 수: {len(existing_files)}")

    try:
        # 배치 프로세서 초기화
        batch_processor = BatchXTFProcessor(existing_files, max_pings_per_file=500)

        # 모든 파일 처리
        readers = batch_processor.process_all()
        logger.info(f"배치 처리 결과: {len(readers)}/{len(existing_files)} 파일 성공")

        # 종합 요약 정보
        combined_summary = batch_processor.get_combined_summary()
        logger.info(f"종합 요약:")
        logger.info(f"  - 총 파일 수: {combined_summary.get('total_files', 0)}")
        logger.info(f"  - 총 ping 수: {combined_summary.get('total_pings', 0):,}")
        logger.info(f"  - 주파수 종류: {combined_summary.get('unique_frequencies', [])}")
        logger.info(f"  - 채널 종류: {combined_summary.get('channels', [])}")

        return True

    except Exception as e:
        logger.error(f"배치 처리 테스트 실패: {e}")
        return False


def generate_test_report(reader_results, extractor_results, batch_success):
    """테스트 결과 보고서 생성"""

    report_lines = []
    report_lines.append("# XTF Reader & Intensity Extractor 검증 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # XTF Reader 결과
    report_lines.append("## 🔍 **XTF Reader 검증 결과**")
    report_lines.append("")

    if reader_results:
        for filename, result in reader_results.items():
            status = "✅" if result['load_success'] else "❌"
            report_lines.append(f"### {status} {filename}")
            report_lines.append(f"- **기종**: {result['file_type']}")
            report_lines.append(f"- **주파수**: {result['frequency']}")
            report_lines.append(f"- **파일 로드**: {'성공' if result['load_success'] else '실패'}")
            report_lines.append(f"- **메타데이터**: {'유효' if result['metadata_valid'] else '무효'}")
            report_lines.append(f"- **Ping 수**: {result['ping_count']:,}")
            report_lines.append(f"- **Intensity 매트릭스**: {result['intensity_matrix_shape'] if result['intensity_matrix_shape'] else 'N/A'}")
            report_lines.append(f"- **위치 데이터**: {'유효' if result['georef_data_valid'] else '무효'}")
            report_lines.append("")
    else:
        report_lines.append("❌ XTF Reader 테스트 결과 없음")
        report_lines.append("")

    # Intensity Extractor 결과
    report_lines.append("## 🎨 **Intensity Extractor 검증 결과**")
    report_lines.append("")

    if extractor_results:
        for filename, result in extractor_results.items():
            status = "✅" if result['extraction_success'] else "❌"
            report_lines.append(f"### {status} {filename}")
            report_lines.append(f"- **기종**: {result['file_type']}")
            report_lines.append(f"- **추출 성공**: {'성공' if result['extraction_success'] else '실패'}")
            report_lines.append(f"- **메타데이터**: {'유효' if result['metadata_valid'] else '무효'}")
            report_lines.append(f"- **Ping 수**: {result['ping_count']:,}")
            report_lines.append(f"- **품질 점수**: {result['quality_score']:.1f}/100")

            if result['intensity_images']:
                report_lines.append(f"- **Intensity 이미지**:")
                for img_type, shape in result['intensity_images'].items():
                    report_lines.append(f"  - {img_type.upper()}: {shape if shape else 'N/A'}")

            report_lines.append(f"- **Navigation 데이터**: {'유효' if result['navigation_data_valid'] else '무효'}")
            report_lines.append(f"- **출력 디렉토리**: {result['output_directory']}")
            report_lines.append("")
    else:
        report_lines.append("❌ Intensity Extractor 테스트 결과 없음")
        report_lines.append("")

    # 배치 처리 결과
    report_lines.append("## 🔄 **배치 처리 검증 결과**")
    report_lines.append("")
    report_lines.append(f"**배치 처리**: {'✅ 성공' if batch_success else '❌ 실패'}")
    report_lines.append("")

    # 종합 결론
    report_lines.append("## 🎯 **종합 결론**")
    report_lines.append("")

    # XTF Reader 성공률
    reader_success_rate = 0
    if reader_results:
        successful_readers = sum(1 for r in reader_results.values() if r['load_success'])
        reader_success_rate = successful_readers / len(reader_results) * 100

    # Intensity Extractor 성공률
    extractor_success_rate = 0
    if extractor_results:
        successful_extractors = sum(1 for r in extractor_results.values() if r['extraction_success'])
        extractor_success_rate = successful_extractors / len(extractor_results) * 100

    # 평균 품질 점수
    avg_quality = 0
    if extractor_results:
        quality_scores = [r['quality_score'] for r in extractor_results.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    report_lines.append(f"- **XTF Reader 성공률**: {reader_success_rate:.1f}%")
    report_lines.append(f"- **Intensity Extractor 성공률**: {extractor_success_rate:.1f}%")
    report_lines.append(f"- **평균 데이터 품질**: {avg_quality:.1f}/100")
    report_lines.append(f"- **배치 처리**: {'성공' if batch_success else '실패'}")
    report_lines.append("")

    if reader_success_rate >= 80 and extractor_success_rate >= 80 and avg_quality >= 70:
        report_lines.append("### ✅ **전체 검증 성공**")
        report_lines.append("두 종류 기종의 사이드 스캔 소나에서 패킷 정보, 메타데이터, 넘파이 배열 형태의 강도 데이터가 성공적으로 추출됩니다.")
    else:
        report_lines.append("### ⚠️ **부분적 검증 성공**")
        report_lines.append("일부 기능에서 개선이 필요합니다.")

    # 보고서 저장
    output_dir = Path("analysis_results/xtf_extraction_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "XTF_EXTRACTION_VERIFICATION_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # JSON 결과도 저장
    json_data = {
        'xtf_reader_results': reader_results,
        'intensity_extractor_results': extractor_results,
        'batch_processing_success': batch_success,
        'summary': {
            'reader_success_rate': reader_success_rate,
            'extractor_success_rate': extractor_success_rate,
            'average_quality_score': avg_quality
        },
        'test_timestamp': datetime.now().isoformat()
    }

    json_file = output_dir / "xtf_extraction_verification_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"검증 보고서 저장: {report_file}")
    logger.info(f"검증 데이터 저장: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("XTF Reader & Intensity Extractor 검증 테스트 시작")

    try:
        # 1. XTF Reader 테스트
        logger.info("1단계: XTF Reader 검증")
        reader_results = test_xtf_reader_extraction()

        # 2. Intensity Extractor 테스트
        logger.info("\n2단계: Intensity Extractor 검증")
        extractor_results = test_intensity_extractor()

        # 3. 배치 처리 테스트
        logger.info("\n3단계: 배치 처리 검증")
        batch_success = test_batch_processing()

        # 4. 보고서 생성
        logger.info("\n4단계: 검증 보고서 생성")
        generate_test_report(reader_results, extractor_results, batch_success)

        # 요약 출력
        print("\n" + "="*70)
        print("XTF EXTRACTION 검증 테스트 완료")
        print("="*70)

        reader_success = sum(1 for r in reader_results.values() if r['load_success']) if reader_results else 0
        extractor_success = sum(1 for r in extractor_results.values() if r['extraction_success']) if extractor_results else 0

        print(f"📁 XTF Reader: {reader_success}/{len(reader_results) if reader_results else 0} 성공")
        print(f"🎨 Intensity Extractor: {extractor_success}/{len(extractor_results) if extractor_results else 0} 성공")
        print(f"🔄 배치 처리: {'성공' if batch_success else '실패'}")
        print(f"📊 결과 저장: analysis_results/xtf_extraction_test/")

        return 0

    except Exception as e:
        logger.error(f"검증 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())