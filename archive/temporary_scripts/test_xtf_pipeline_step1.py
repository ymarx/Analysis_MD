#!/usr/bin/env python3
"""
1단계: XTF 메타데이터/강도 데이터 추출 포괄적 테스트

목적: 모든 XTF 파일에 대해 데이터 추출 기능 검증
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_all_xtf_files():
    """모든 XTF 파일 찾기"""

    print("🔍 XTF 파일 탐색 중...")

    xtf_files = []
    datasets_path = Path("datasets")

    if datasets_path.exists():
        for xtf_file in datasets_path.rglob("*.xtf"):
            xtf_files.append(xtf_file)

    print(f"   발견된 XTF 파일: {len(xtf_files)}개")

    # 파일별 정보 출력
    for i, xtf_file in enumerate(xtf_files[:10], 1):  # 처음 10개만 출력
        size_mb = xtf_file.stat().st_size / (1024 * 1024)
        print(f"   {i:2d}. {xtf_file.name} ({size_mb:.1f}MB)")

    if len(xtf_files) > 10:
        print(f"   ... 및 {len(xtf_files)-10}개 더")

    return xtf_files

def test_xtf_reader_with_file(xtf_file_path):
    """개별 XTF 파일로 XTF Reader 테스트"""

    try:
        from src.data_processing.xtf_reader import XTFReader

        # XTF Reader 초기화 (파일 경로 필요)
        reader = XTFReader(str(xtf_file_path))

        # 메타데이터 가져오기
        metadata = reader.get_metadata()

        # 좌표 데이터 가져오기
        coordinates = reader.get_navigation_data()

        # Ping 데이터 가져오기 (처음 50개만)
        ping_data = reader.get_ping_data(max_pings=50)

        return True, {
            "metadata": {
                "total_pings": metadata.total_pings if metadata else 0,
                "frequency_info": metadata.frequency_info if metadata else {},
                "coordinate_bounds": metadata.coordinate_bounds if metadata else {}
            },
            "coordinates_count": len(coordinates),
            "ping_data_count": len(ping_data),
            "coordinate_sample": coordinates[:3] if len(coordinates) > 0 else [],
            "ping_sample": len(ping_data[0].data) if len(ping_data) > 0 else 0
        }

    except Exception as e:
        return False, str(e)

def test_pipeline_xtf_reader_with_file(xtf_file_path):
    """Pipeline XTF Reader로 테스트"""

    try:
        from pipeline.modules.xtf_reader import XTFReader

        # Pipeline XTF Reader 사용
        reader = XTFReader()
        result = reader.read(xtf_file_path)

        return True, {
            "ping_count": result.get("ping_count", 0),
            "intensity_shape": result.get("intensity_matrix_shape"),
            "coordinate_stats": result.get("coordinate_stats"),
            "summary": result.get("summary", {})
        }

    except Exception as e:
        return False, str(e)

def test_intensity_extractor_with_file(xtf_file_path):
    """XTF Intensity Extractor 직접 테스트"""

    try:
        from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor

        # Intensity Extractor 초기화
        extractor = XTFIntensityExtractor()

        # 강도 데이터 추출
        intensity_result = extractor.extract_intensity_data(str(xtf_file_path))

        return True, {
            "intensity_pings_count": len(intensity_result.get("intensity_pings", [])),
            "port_channel_data": intensity_result.get("port_channel_data") is not None,
            "starboard_channel_data": intensity_result.get("starboard_channel_data") is not None,
            "combined_intensity_shape": intensity_result.get("combined_intensity_data", np.array([])).shape
        }

    except Exception as e:
        return False, str(e)

def run_comprehensive_xtf_tests():
    """포괄적 XTF 테스트 실행"""

    print("=" * 70)
    print("1단계: XTF 데이터 추출 포괄적 테스트")
    print("=" * 70)

    # XTF 파일들 찾기
    xtf_files = find_all_xtf_files()

    if not xtf_files:
        print("❌ 테스트할 XTF 파일이 없습니다.")
        return False

    # 테스트할 파일 선정 (처음 3개)
    test_files = xtf_files[:3]
    print(f"\n📋 테스트 대상: {len(test_files)}개 파일")

    test_results = []

    for i, xtf_file in enumerate(test_files, 1):
        print(f"\n🔬 테스트 {i}/{len(test_files)}: {xtf_file.name}")
        print(f"   📁 경로: {xtf_file}")

        file_result = {
            "filename": xtf_file.name,
            "filepath": str(xtf_file),
            "size_mb": xtf_file.stat().st_size / (1024 * 1024)
        }

        # 1. Pipeline XTF Reader 테스트
        print(f"   🔧 Pipeline XTF Reader 테스트...")
        pipeline_success, pipeline_result = test_pipeline_xtf_reader_with_file(xtf_file)

        if pipeline_success:
            print(f"   ✅ Pipeline Reader 성공")
            print(f"      - Ping 수: {pipeline_result.get('ping_count', 0)}")
            print(f"      - Intensity Shape: {pipeline_result.get('intensity_shape')}")
            file_result["pipeline_success"] = True
            file_result["pipeline_result"] = pipeline_result
        else:
            print(f"   ❌ Pipeline Reader 실패: {pipeline_result}")
            file_result["pipeline_success"] = False
            file_result["pipeline_error"] = pipeline_result

        # 2. Direct XTF Reader 테스트
        print(f"   🔧 Direct XTF Reader 테스트...")
        direct_success, direct_result = test_xtf_reader_with_file(xtf_file)

        if direct_success:
            print(f"   ✅ Direct Reader 성공")
            print(f"      - 좌표 수: {direct_result.get('coordinates_count', 0)}")
            print(f"      - Ping 수: {direct_result.get('ping_data_count', 0)}")
            file_result["direct_success"] = True
            file_result["direct_result"] = direct_result
        else:
            print(f"   ❌ Direct Reader 실패: {direct_result}")
            file_result["direct_success"] = False
            file_result["direct_error"] = direct_result

        # 3. Intensity Extractor 테스트
        print(f"   🔧 Intensity Extractor 테스트...")
        intensity_success, intensity_result = test_intensity_extractor_with_file(xtf_file)

        if intensity_success:
            print(f"   ✅ Intensity Extractor 성공")
            print(f"      - Intensity Pings: {intensity_result.get('intensity_pings_count', 0)}")
            print(f"      - Combined Shape: {intensity_result.get('combined_intensity_shape')}")
            file_result["intensity_success"] = True
            file_result["intensity_result"] = intensity_result
        else:
            print(f"   ❌ Intensity Extractor 실패: {intensity_result}")
            file_result["intensity_success"] = False
            file_result["intensity_error"] = intensity_result

        test_results.append(file_result)

    return test_results

def generate_test_summary(test_results):
    """테스트 결과 요약 생성"""

    print(f"\n{'='*70}")
    print("📊 1단계 테스트 결과 요약")
    print(f"{'='*70}")

    if not test_results:
        print("❌ 테스트 결과 없음")
        return False

    total_files = len(test_results)
    pipeline_success_count = sum(1 for r in test_results if r.get("pipeline_success", False))
    direct_success_count = sum(1 for r in test_results if r.get("direct_success", False))
    intensity_success_count = sum(1 for r in test_results if r.get("intensity_success", False))

    print(f"📁 테스트 파일 수: {total_files}")
    print(f"✅ Pipeline Reader: {pipeline_success_count}/{total_files} ({pipeline_success_count/total_files*100:.1f}%)")
    print(f"✅ Direct Reader: {direct_success_count}/{total_files} ({direct_success_count/total_files*100:.1f}%)")
    print(f"✅ Intensity Extractor: {intensity_success_count}/{total_files} ({intensity_success_count/total_files*100:.1f}%)")

    # 성공한 파일들의 데이터 통계
    print(f"\n📊 추출된 데이터 통계:")

    for result in test_results:
        if result.get("pipeline_success"):
            pipeline_data = result["pipeline_result"]
            print(f"\n   📄 {result['filename']}:")
            print(f"      - Ping 수: {pipeline_data.get('ping_count', 0)}")
            print(f"      - Intensity 크기: {pipeline_data.get('intensity_shape')}")

    # 전체 성공률 계산
    overall_success_rate = (pipeline_success_count + direct_success_count + intensity_success_count) / (3 * total_files) * 100

    return overall_success_rate >= 80

def save_test_results(test_results):
    """테스트 결과 저장"""

    # JSON으로 상세 결과 저장
    output_file = f"analysis_results/data_validation/xtf_step1_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # numpy array는 JSON 직렬화 불가능하므로 문자열로 변환
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return f"numpy.array{obj.shape}"
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    # 결과 정리
    clean_results = []
    for result in test_results:
        clean_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                clean_result[key] = {k: convert_for_json(v) for k, v in value.items()}
            else:
                clean_result[key] = convert_for_json(value)
        clean_results.append(clean_result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "test_description": "1단계 XTF 데이터 추출 포괄적 테스트",
            "total_files_tested": len(test_results),
            "results": clean_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 테스트 결과 저장: {output_file}")

def main():
    """메인 실행 함수"""

    print("🔧 1단계: XTF 데이터 추출 포괄적 테스트 시작")

    # 포괄적 테스트 실행
    test_results = run_comprehensive_xtf_tests()

    if not test_results:
        print("\n❌ 테스트 실행 실패")
        return False

    # 결과 요약
    success = generate_test_summary(test_results)

    # 결과 저장
    save_test_results(test_results)

    print(f"\n{'='*70}")
    if success:
        print("✅ 1단계 XTF 데이터 추출 테스트 완료 - 성공")
        print("🎯 다음 단계: 3단계 데이터 증강 테스트 진행 가능")
    else:
        print("⚠️ 1단계 XTF 데이터 추출 테스트 완료 - 일부 개선 필요")
    print(f"{'='*70}")

    return success

if __name__ == "__main__":
    main()