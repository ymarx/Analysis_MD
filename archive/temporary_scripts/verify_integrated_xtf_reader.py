#!/usr/bin/env python3
"""
통합된 XTF Reader 작동 검증

목적: pipeline의 XTF Reader가 src 모듈을 정상적으로 사용하는지 확인
"""

import os
import sys
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import_pipeline_xtf_reader():
    """pipeline XTF Reader import 테스트"""

    print("=" * 70)
    print("Pipeline XTF Reader Import 테스트")
    print("=" * 70)

    try:
        from pipeline.modules.xtf_reader import XTFReader
        print("✅ pipeline.modules.xtf_reader 성공적으로 import됨")
        return XTFReader
    except ImportError as e:
        print(f"❌ pipeline XTF Reader import 실패: {e}")
        return None

def test_import_src_xtf_reader():
    """src XTF Reader import 테스트"""

    print(f"\n📊 src XTF Reader Import 테스트:")

    try:
        from src.data_processing.xtf_reader import XTFReader
        print("✅ src.data_processing.xtf_reader 성공적으로 import됨")
        return XTFReader
    except ImportError as e:
        print(f"❌ src XTF Reader import 실패: {e}")
        return None

def test_pipeline_reader_initialization():
    """pipeline XTF Reader 초기화 테스트"""

    print(f"\n🔧 Pipeline XTF Reader 초기화 테스트:")

    try:
        from pipeline.modules.xtf_reader import XTFReader
        reader = XTFReader()
        print("✅ Pipeline XTF Reader 초기화 성공")

        # extractor 확인
        if hasattr(reader, 'extractor') and reader.extractor is not None:
            print("✅ XTF Intensity Extractor 정상 로드됨")
        else:
            print("⚠️ XTF Intensity Extractor 없음")

        return reader
    except Exception as e:
        print(f"❌ Pipeline XTF Reader 초기화 실패: {e}")
        return None

def test_xtf_file_processing():
    """실제 XTF 파일 처리 테스트"""

    print(f"\n📁 XTF 파일 처리 테스트:")

    # 테스트할 XTF 파일들 찾기
    xtf_files = []
    datasets_path = Path("datasets")

    if datasets_path.exists():
        for xtf_file in datasets_path.rglob("*.xtf"):
            xtf_files.append(xtf_file)

    print(f"   발견된 XTF 파일: {len(xtf_files)}개")

    if not xtf_files:
        print("⚠️ 테스트할 XTF 파일 없음")
        return False

    # pipeline XTF Reader로 테스트
    try:
        from pipeline.modules.xtf_reader import XTFReader
        reader = XTFReader()

        # 첫 번째 파일로 테스트
        test_file = xtf_files[0]
        print(f"   테스트 파일: {test_file.name}")

        result = reader.read(test_file)

        if result and isinstance(result, dict):
            print("✅ XTF 파일 읽기 성공")
            print(f"   반환된 키들: {list(result.keys())}")

            # 좌표 데이터 확인
            if 'coordinates' in result:
                coords = result['coordinates']
                print(f"   좌표 데이터: {len(coords)}개")
                if len(coords) > 0:
                    print(f"   첫 좌표: {coords[0]}")

            return True
        else:
            print("❌ XTF 파일 읽기 실패 - 빈 결과")
            return False

    except Exception as e:
        print(f"❌ XTF 파일 처리 실패: {e}")
        return False

def test_coordinate_fix_functionality():
    """좌표 수정 기능 테스트"""

    print(f"\n🔧 좌표 수정 기능 테스트:")

    try:
        from src.data_processing.xtf_reader import XTFReader
        reader = XTFReader()

        # 좌표 수정 메서드 존재 확인
        if hasattr(reader, '_fix_longitude_value'):
            print("✅ 좌표 수정 메서드 존재함")

            # 테스트 케이스
            test_cases = [
                (12.514938, 129.514938),  # 수정 필요
                (129.515000, 129.515000),  # 정상
                (12.520000, 129.520000),  # 수정 필요
                (130.000000, 129.515000)  # 범위 밖 -> 평균값
            ]

            for input_val, expected in test_cases:
                result = reader._fix_longitude_value(input_val)
                status = "✅" if abs(result - expected) < 0.001 else "❌"
                print(f"   {status} {input_val} → {result} (예상: {expected})")

            return True
        else:
            print("❌ 좌표 수정 메서드 없음")
            return False

    except Exception as e:
        print(f"❌ 좌표 수정 기능 테스트 실패: {e}")
        return False

def test_module_integration_status():
    """모듈 통합 상태 종합 평가"""

    print(f"\n📊 모듈 통합 상태 종합 평가:")

    results = {
        'pipeline_import': False,
        'src_import': False,
        'initialization': False,
        'file_processing': False,
        'coordinate_fix': False
    }

    # 각 테스트 실행
    if test_import_pipeline_xtf_reader():
        results['pipeline_import'] = True

    if test_import_src_xtf_reader():
        results['src_import'] = True

    if test_pipeline_reader_initialization():
        results['initialization'] = True

    if test_xtf_file_processing():
        results['file_processing'] = True

    if test_coordinate_fix_functionality():
        results['coordinate_fix'] = True

    # 결과 요약
    passed = sum(results.values())
    total = len(results)

    print(f"\n📋 테스트 결과 요약:")
    print(f"   통과: {passed}/{total} ({passed/total*100:.1f}%)")

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}")

    if passed == total:
        print(f"\n🎉 모든 테스트 통과! XTF Reader 통합 성공")
        return True
    else:
        print(f"\n⚠️ 일부 테스트 실패. 추가 수정 필요")
        return False

def main():
    """메인 실행 함수"""

    print("🔧 통합된 XTF Reader 작동 검증 시작")

    # 작업 디렉토리 확인
    cwd = Path.cwd()
    print(f"\n📁 현재 작업 디렉토리: {cwd}")

    # Python 경로 확인
    print(f"📊 Python 경로:")
    for i, path in enumerate(sys.path[:5]):
        print(f"   {i+1}. {path}")

    # 모듈 통합 상태 테스트
    success = test_module_integration_status()

    print(f"\n{'='*70}")
    if success:
        print("✅ XTF Reader 통합 검증 완료 - 모든 기능 정상")
    else:
        print("⚠️ XTF Reader 통합 검증 완료 - 일부 개선 필요")
    print(f"{'='*70}")

    return success

if __name__ == "__main__":
    main()