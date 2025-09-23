#!/usr/bin/env python3
"""
전체 분석 파이프라인 모듈 상태 진단

목적: 9단계 분석 파이프라인의 모든 모듈 작동 상태 확인
"""

import os
import sys
from pathlib import Path
import logging
import importlib.util
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_module_import(module_path, module_name):
    """모듈 import 테스트"""

    try:
        if not os.path.exists(module_path):
            return False, f"파일 없음: {module_path}"

        # 모듈 spec 생성
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False, "모듈 spec 생성 실패"

        # 모듈 로드
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return True, f"성공적으로 import됨"

    except Exception as e:
        return False, f"Import 실패: {str(e)}"

def test_class_instantiation(module_path, module_name, class_name):
    """클래스 인스턴스화 테스트"""

    try:
        # 모듈 import
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 클래스 존재 확인
        if not hasattr(module, class_name):
            return False, f"클래스 {class_name} 없음"

        # 클래스 인스턴스화 시도
        class_obj = getattr(module, class_name)
        instance = class_obj()

        return True, f"클래스 {class_name} 인스턴스화 성공"

    except Exception as e:
        return False, f"인스턴스화 실패: {str(e)}"

def diagnose_pipeline_modules():
    """파이프라인 모듈들 진단"""

    print("=" * 70)
    print("분석 파이프라인 모듈 상태 진단")
    print("=" * 70)

    # 진단할 모듈들 정의
    modules_to_diagnose = {
        "1단계 - XTF 데이터 추출": [
            {
                "path": "src/data_processing/xtf_reader.py",
                "name": "xtf_reader",
                "class": "XTFReader",
                "description": "XTF 파일 읽기 및 메타데이터 추출"
            },
            {
                "path": "src/data_processing/xtf_intensity_extractor.py",
                "name": "xtf_intensity_extractor",
                "class": "XTFIntensityExtractor",
                "description": "강도 데이터 추출"
            }
        ],
        "2단계 - 좌표 매핑 및 레이블링": [
            {
                "path": "src/data_processing/coordinate_mapper.py",
                "name": "coordinate_mapper",
                "class": "CoordinateMapper",
                "description": "위경도-픽셀 좌표 매핑"
            },
            {
                "path": "pipeline/modules/gps_parser.py",
                "name": "gps_parser",
                "class": None,  # 함수 기반일 수 있음
                "description": "GPS 데이터 파싱"
            }
        ],
        "3단계 - 데이터 증강": [
            {
                "path": "src/data_augmentation/augmentation_engine.py",
                "name": "augmentation_engine",
                "class": "AugmentationEngine",
                "description": "데이터 증강 (회전, blur 등)"
            }
        ],
        "4단계 - 특징 추출": [
            {
                "path": "src/feature_extraction/hog_extractor.py",
                "name": "hog_extractor",
                "class": "HOGExtractor",
                "description": "HOG 특징 추출"
            },
            {
                "path": "src/feature_extraction/lbp_extractor.py",
                "name": "lbp_extractor",
                "class": "LBPExtractor",
                "description": "LBP 특징 추출"
            },
            {
                "path": "src/feature_extraction/gabor_extractor.py",
                "name": "gabor_extractor",
                "class": "GaborExtractor",
                "description": "Gabor 특징 추출"
            },
            {
                "path": "src/feature_extraction/sfs_extractor.py",
                "name": "sfs_extractor",
                "class": "SFSExtractor",
                "description": "SFS 특징 추출"
            },
            {
                "path": "src/feature_extraction/feature_ensemble.py",
                "name": "feature_ensemble",
                "class": "FeatureEnsemble",
                "description": "특징 앙상블"
            }
        ],
        "5단계 - 분류": [
            {
                "path": "src/models/cnn_detector.py",
                "name": "cnn_detector",
                "class": "CNNDetector",
                "description": "CNN 기반 탐지"
            },
            {
                "path": "pipeline/modules/mine_classifier.py",
                "name": "mine_classifier",
                "class": "MineClassifier",
                "description": "기뢰 분류"
            }
        ],
        "지원 모듈": [
            {
                "path": "src/data_processing/preprocessor.py",
                "name": "preprocessor",
                "class": "Preprocessor",
                "description": "데이터 전처리"
            },
            {
                "path": "pipeline/modules/terrain_analyzer.py",
                "name": "terrain_analyzer",
                "class": "TerrainAnalyzer",
                "description": "지형 분석"
            }
        ]
    }

    results = {}

    for stage, modules in modules_to_diagnose.items():
        print(f"\n📊 {stage}:")
        stage_results = []

        for module_info in modules:
            module_path = module_info["path"]
            module_name = module_info["name"]
            class_name = module_info["class"]
            description = module_info["description"]

            # Import 테스트
            import_success, import_msg = test_module_import(module_path, module_name)

            # 클래스 인스턴스화 테스트 (클래스가 정의된 경우)
            if import_success and class_name:
                class_success, class_msg = test_class_instantiation(module_path, module_name, class_name)
            else:
                class_success = import_success
                class_msg = import_msg if not class_name else "클래스 정보 없음"

            # 결과 출력
            status = "✅" if import_success else "❌"
            class_status = "✅" if class_success else "❌"

            print(f"   {status} {description}")
            print(f"      📁 파일: {module_path}")
            print(f"      📦 Import: {import_msg}")
            if class_name:
                print(f"      🏗️ 클래스: {class_msg}")

            # 결과 저장
            stage_results.append({
                "module": module_name,
                "path": module_path,
                "description": description,
                "import_success": import_success,
                "class_success": class_success,
                "import_msg": import_msg,
                "class_msg": class_msg
            })

        results[stage] = stage_results

    return results

def generate_diagnosis_summary(results):
    """진단 결과 요약 생성"""

    print(f"\n{'='*70}")
    print("📋 진단 결과 요약")
    print(f"{'='*70}")

    total_modules = 0
    working_modules = 0

    for stage, modules in results.items():
        stage_working = 0
        stage_total = len(modules)

        for module in modules:
            total_modules += 1
            if module["import_success"]:
                working_modules += 1
                stage_working += 1

        success_rate = (stage_working / stage_total * 100) if stage_total > 0 else 0
        status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"

        print(f"{status} {stage}: {stage_working}/{stage_total} ({success_rate:.1f}%)")

    overall_rate = (working_modules / total_modules * 100) if total_modules > 0 else 0
    overall_status = "✅" if overall_rate >= 80 else "⚠️" if overall_rate >= 50 else "❌"

    print(f"\n{overall_status} 전체: {working_modules}/{total_modules} ({overall_rate:.1f}%)")

    return overall_rate >= 80

def identify_critical_issues(results):
    """중요한 문제점들 식별"""

    print(f"\n🔧 중요 이슈 및 해결 방안:")

    critical_modules = [
        "xtf_reader", "xtf_intensity_extractor",  # 1단계 핵심
        "augmentation_engine",  # 3단계 핵심
        "feature_ensemble",  # 4단계 핵심
        "cnn_detector"  # 5단계 핵심
    ]

    issues_found = []

    for stage, modules in results.items():
        for module in modules:
            module_name = module["module"]

            if not module["import_success"]:
                is_critical = module_name in critical_modules
                priority = "🔴 긴급" if is_critical else "🟡 중요"

                print(f"\n{priority} {module['description']} ({module_name})")
                print(f"   📁 {module['path']}")
                print(f"   ❌ 문제: {module['import_msg']}")

                # 해결 방안 제시
                if "No module named" in module['import_msg']:
                    print(f"   💡 해결방안: 의존성 라이브러리 설치 필요")
                elif "파일 없음" in module['import_msg']:
                    print(f"   💡 해결방안: 모듈 파일 생성 또는 경로 확인 필요")
                else:
                    print(f"   💡 해결방안: 코드 구문 오류 수정 필요")

                issues_found.append({
                    "module": module_name,
                    "critical": is_critical,
                    "issue": module['import_msg']
                })

    if not issues_found:
        print("\n✅ 중요한 이슈 없음")

    return issues_found

def recommend_next_steps(results, issues):
    """다음 단계 권장사항"""

    print(f"\n🎯 다음 단계 권장사항:")

    # 즉시 수행 가능한 작업들
    print(f"\n📋 즉시 수행 가능:")

    # 1단계 확인
    xtf_working = True
    for stage, modules in results.items():
        if "1단계" in stage:
            for module in modules:
                if not module["import_success"]:
                    xtf_working = False

    if xtf_working:
        print("   ✅ 1단계: XTF 추가 파일 처리 테스트")
    else:
        print("   🔧 1단계: XTF 모듈 수정 필요")

    # 3단계 데이터 증강 확인
    aug_working = False
    for stage, modules in results.items():
        if "3단계" in stage:
            for module in modules:
                if module["import_success"]:
                    aug_working = True

    if aug_working:
        print("   ✅ 3단계: 데이터 증강 기능 테스트")
    else:
        print("   🔧 3단계: 데이터 증강 모듈 수정 필요")

    # GPS 데이터 대기 중 작업
    print(f"\n⏳ GPS 데이터 수령 대기 중:")
    print("   📋 2단계: Coordinate Mapper 구조 점검 및 준비")
    print("   📋 더미 데이터로 매핑 로직 검증")

    # 우선순위별 수정 작업
    critical_issues = [issue for issue in issues if issue["critical"]]
    if critical_issues:
        print(f"\n🔴 긴급 수정 필요:")
        for issue in critical_issues:
            print(f"   - {issue['module']}: {issue['issue']}")

    return xtf_working and aug_working

def main():
    """메인 실행 함수"""

    print("🔧 분석 파이프라인 모듈 상태 진단 시작")

    # 작업 디렉토리 확인
    cwd = Path.cwd()
    print(f"\n📁 현재 작업 디렉토리: {cwd}")

    # 모듈 진단 실행
    results = diagnose_pipeline_modules()

    # 결과 요약
    overall_healthy = generate_diagnosis_summary(results)

    # 중요 이슈 식별
    issues = identify_critical_issues(results)

    # 다음 단계 권장
    ready_for_testing = recommend_next_steps(results, issues)

    print(f"\n{'='*70}")
    if overall_healthy:
        print("✅ 분석 파이프라인 진단 완료 - 대부분 모듈 정상")
    else:
        print("⚠️ 분석 파이프라인 진단 완료 - 일부 모듈 수정 필요")
    print(f"{'='*70}")

    return overall_healthy

if __name__ == "__main__":
    main()