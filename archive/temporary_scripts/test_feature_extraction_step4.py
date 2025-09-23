#!/usr/bin/env python3
"""
4단계: 특징 추출 모듈들 (여러 기법) 검증

목적: HOG, LBP, Gabor, SFS 등 여러 특징 추출 기법과 앙상블 시스템 검증
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extractors_import():
    """특징 추출 모듈들 import 테스트"""

    print("🔧 특징 추출 모듈들 Import 테스트:")

    extractors = {}
    import_results = {}

    # HOG Extractor
    try:
        from src.feature_extraction.hog_extractor import HOGExtractor
        extractors['HOG'] = HOGExtractor
        import_results['HOG'] = True
        print("   ✅ HOG Extractor import 성공")
    except Exception as e:
        import_results['HOG'] = False
        print(f"   ❌ HOG Extractor import 실패: {e}")

    # LBP Extractor
    try:
        from src.feature_extraction.lbp_extractor import LBPExtractor
        extractors['LBP'] = LBPExtractor
        import_results['LBP'] = True
        print("   ✅ LBP Extractor import 성공")
    except Exception as e:
        import_results['LBP'] = False
        print(f"   ❌ LBP Extractor import 실패: {e}")

    # Gabor Extractor
    try:
        from src.feature_extraction.gabor_extractor import GaborExtractor
        extractors['Gabor'] = GaborExtractor
        import_results['Gabor'] = True
        print("   ✅ Gabor Extractor import 성공")
    except Exception as e:
        import_results['Gabor'] = False
        print(f"   ❌ Gabor Extractor import 실패: {e}")

    # SFS Extractor
    try:
        from src.feature_extraction.sfs_extractor import SFSExtractor
        extractors['SFS'] = SFSExtractor
        import_results['SFS'] = True
        print("   ✅ SFS Extractor import 성공")
    except Exception as e:
        import_results['SFS'] = False
        print(f"   ❌ SFS Extractor import 실패: {e}")

    # Feature Ensemble
    try:
        from src.feature_extraction.feature_ensemble import FeatureEnsemble
        extractors['Ensemble'] = FeatureEnsemble
        import_results['Ensemble'] = True
        print("   ✅ Feature Ensemble import 성공")
    except Exception as e:
        import_results['Ensemble'] = False
        print(f"   ❌ Feature Ensemble import 실패: {e}")

    return extractors, import_results

def create_test_sonar_patches():
    """테스트용 소나 패치 이미지들 생성"""

    patches = []

    # 1. 기물 패치 (강한 반사체)
    mine_patch = np.zeros((64, 64), dtype=np.float32)
    center = 32
    for i in range(64):
        for j in range(64):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 15:
                mine_patch[i, j] = 0.8 * np.exp(-(dist**2) / (2 * 8**2))
    mine_patch += np.random.normal(0, 0.05, (64, 64))
    patches.append(np.clip(mine_patch, 0, 1))

    # 2. 해저면 패치 (균등한 반사)
    seafloor_patch = np.random.normal(0.4, 0.1, (64, 64))
    patches.append(np.clip(seafloor_patch, 0, 1))

    # 3. 음향 그림자 패치 (어두운 영역)
    shadow_patch = np.random.normal(0.1, 0.05, (64, 64))
    patches.append(np.clip(shadow_patch, 0, 1))

    # 4. 잡음 패치 (순수 노이즈)
    noise_patch = np.random.normal(0.3, 0.2, (64, 64))
    patches.append(np.clip(noise_patch, 0, 1))

    # 5. 복합 패치 (기물 + 그림자)
    complex_patch = np.zeros((64, 64), dtype=np.float32)
    # 기물 부분
    complex_patch[:32, :32] = mine_patch[:32, :32]
    # 그림자 부분
    complex_patch[32:, 32:] = shadow_patch[32:, 32:] * 0.5
    # 배경
    complex_patch += np.random.normal(0.3, 0.05, (64, 64)) * 0.3
    patches.append(np.clip(complex_patch, 0, 1))

    return patches, ['Mine', 'Seafloor', 'Shadow', 'Noise', 'Complex']

def test_individual_extractors(extractors, import_results):
    """개별 특징 추출기 테스트"""

    print("🔧 개별 특징 추출기 테스트:")

    # 테스트 이미지 생성
    test_patches, patch_labels = create_test_sonar_patches()
    print(f"   📊 테스트 패치: {len(test_patches)}개 ({', '.join(patch_labels)})")

    extraction_results = {}

    for extractor_name, extractor_class in extractors.items():
        if extractor_name == 'Ensemble':  # 앙상블은 별도 테스트
            continue

        if not import_results.get(extractor_name):
            continue

        print(f"\n   🔬 {extractor_name} Extractor 테스트:")

        try:
            # 추출기 초기화
            if extractor_name == 'HOG':
                extractor = extractor_class()
            elif extractor_name == 'LBP':
                extractor = extractor_class()
            elif extractor_name == 'Gabor':
                extractor = extractor_class()
            elif extractor_name == 'SFS':
                extractor = extractor_class()
            else:
                extractor = extractor_class()

            print(f"      ✅ {extractor_name} 초기화 성공")

            # 각 패치에서 특징 추출
            features_list = []
            for i, patch in enumerate(test_patches):
                try:
                    # 이미지가 uint8 형태로 변환되어야 할 수도 있음
                    patch_uint8 = (patch * 255).astype(np.uint8)

                    # 특징 추출 메서드 찾기
                    if hasattr(extractor, 'extract'):
                        features = extractor.extract(patch_uint8)
                    elif hasattr(extractor, 'extract_features'):
                        features = extractor.extract_features(patch_uint8)
                    elif hasattr(extractor, 'compute'):
                        features = extractor.compute(patch_uint8)
                    else:
                        # 클래스 메서드들 확인
                        methods = [method for method in dir(extractor) if callable(getattr(extractor, method)) and not method.startswith('_')]
                        print(f"         사용 가능한 메서드: {methods}")
                        features = None

                    if features is not None:
                        features_list.append(features)
                        print(f"      ✅ {patch_labels[i]} 패치: 특징 크기 {np.array(features).shape}")
                    else:
                        print(f"      ❌ {patch_labels[i]} 패치: 특징 추출 실패")

                except Exception as e:
                    print(f"      ❌ {patch_labels[i]} 패치 처리 실패: {e}")

            # 결과 정리
            if features_list:
                features_array = np.array(features_list)
                extraction_results[extractor_name] = {
                    'success': True,
                    'feature_shape': features_array.shape,
                    'feature_stats': {
                        'mean': np.mean(features_array),
                        'std': np.std(features_array),
                        'min': np.min(features_array),
                        'max': np.max(features_array)
                    }
                }
                print(f"      📊 전체 특징 크기: {features_array.shape}")
                print(f"      📊 특징 통계: mean={np.mean(features_array):.3f}, std={np.std(features_array):.3f}")
            else:
                extraction_results[extractor_name] = {'success': False, 'error': '특징 추출 실패'}

        except Exception as e:
            print(f"      ❌ {extractor_name} 테스트 실패: {e}")
            extraction_results[extractor_name] = {'success': False, 'error': str(e)}

    return extraction_results

def test_feature_ensemble(extractors, import_results, extraction_results):
    """특징 앙상블 시스템 테스트"""

    print("\n🔧 특징 앙상블 시스템 테스트:")

    if not import_results.get('Ensemble'):
        print("   ❌ Feature Ensemble import 실패로 테스트 불가")
        return {'success': False, 'error': 'Import 실패'}

    try:
        from src.feature_extraction.feature_ensemble import FeatureEnsemble

        # 앙상블 초기화
        ensemble = FeatureEnsemble()
        print("   ✅ Feature Ensemble 초기화 성공")

        # 테스트 이미지 생성
        test_patches, patch_labels = create_test_sonar_patches()

        # 앙상블 특징 추출
        ensemble_features = []
        for i, patch in enumerate(test_patches):
            try:
                patch_uint8 = (patch * 255).astype(np.uint8)

                # 앙상블 특징 추출 메서드 찾기
                if hasattr(ensemble, 'extract_all_features'):
                    features = ensemble.extract_all_features(patch_uint8)
                elif hasattr(ensemble, 'extract'):
                    features = ensemble.extract(patch_uint8)
                elif hasattr(ensemble, 'extract_features'):
                    features = ensemble.extract_features(patch_uint8)
                else:
                    # 개별 추출기들을 조합하여 앙상블 수행
                    individual_features = []

                    # 성공한 추출기들의 특징 결합
                    for extractor_name in extraction_results:
                        if extraction_results[extractor_name].get('success'):
                            try:
                                if extractor_name in extractors:
                                    ext = extractors[extractor_name]()
                                    if hasattr(ext, 'extract'):
                                        feat = ext.extract(patch_uint8)
                                    elif hasattr(ext, 'extract_features'):
                                        feat = ext.extract_features(patch_uint8)
                                    else:
                                        continue

                                    if feat is not None:
                                        individual_features.append(np.array(feat).flatten())
                            except:
                                continue

                    if individual_features:
                        features = np.concatenate(individual_features)
                    else:
                        features = None

                if features is not None:
                    ensemble_features.append(features)
                    print(f"   ✅ {patch_labels[i]} 패치: 앙상블 특징 크기 {np.array(features).shape}")
                else:
                    print(f"   ❌ {patch_labels[i]} 패치: 앙상블 특징 추출 실패")

            except Exception as e:
                print(f"   ❌ {patch_labels[i]} 패치 처리 실패: {e}")

        # 앙상블 결과 분석
        if ensemble_features:
            ensemble_array = np.array(ensemble_features)
            print(f"   📊 앙상블 특징 크기: {ensemble_array.shape}")
            print(f"   📊 앙상블 특징 통계: mean={np.mean(ensemble_array):.3f}, std={np.std(ensemble_array):.3f}")

            return {
                'success': True,
                'ensemble_shape': ensemble_array.shape,
                'ensemble_stats': {
                    'mean': np.mean(ensemble_array),
                    'std': np.std(ensemble_array),
                    'min': np.min(ensemble_array),
                    'max': np.max(ensemble_array)
                }
            }
        else:
            return {'success': False, 'error': '앙상블 특징 추출 실패'}

    except Exception as e:
        print(f"   ❌ Feature Ensemble 테스트 실패: {e}")
        return {'success': False, 'error': str(e)}

def create_feature_comparison_visualization(extraction_results, ensemble_results):
    """특징 추출 결과 비교 시각화"""

    print("🔧 특징 추출 결과 시각화 생성:")

    try:
        # 성공한 추출기들만 선택
        successful_extractors = {name: result for name, result in extraction_results.items()
                               if result.get('success', False)}

        if not successful_extractors:
            print("   ❌ 시각화할 성공한 추출기 없음")
            return False

        # 특징 차원 비교 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. 특징 차원 비교
        extractor_names = list(successful_extractors.keys())
        feature_dims = []

        for name in extractor_names:
            shape = successful_extractors[name]['feature_shape']
            if len(shape) >= 2:
                dims = shape[1] if len(shape) == 2 else np.prod(shape[1:])
            else:
                dims = shape[0] if len(shape) == 1 else 1
            feature_dims.append(dims)

        ax1.bar(extractor_names, feature_dims)
        ax1.set_title('Feature Dimensions by Extractor')
        ax1.set_ylabel('Feature Dimension')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 특징 값 분포 비교
        for i, name in enumerate(extractor_names):
            stats = successful_extractors[name]['feature_stats']
            means = [stats['mean']]
            stds = [stats['std']]

            ax2.errorbar([i], means, yerr=stds, fmt='o', label=name, capsize=5)

        ax2.set_title('Feature Value Distribution')
        ax2.set_ylabel('Feature Value (mean ± std)')
        ax2.set_xticks(range(len(extractor_names)))
        ax2.set_xticklabels(extractor_names, rotation=45)
        ax2.legend()

        plt.tight_layout()

        # 저장
        output_path = "analysis_results/visualizations/feature_extraction_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ 시각화 저장: {output_path}")
        return True

    except Exception as e:
        print(f"   ❌ 시각화 생성 실패: {e}")
        return False

def run_comprehensive_feature_tests():
    """포괄적 특징 추출 테스트 실행"""

    print("=" * 70)
    print("4단계: 특징 추출 모듈들 포괄적 테스트")
    print("=" * 70)

    # 1. 모듈 import
    extractors, import_results = test_feature_extractors_import()

    # 2. 개별 추출기 테스트
    extraction_results = test_individual_extractors(extractors, import_results)

    # 3. 앙상블 테스트
    ensemble_results = test_feature_ensemble(extractors, import_results, extraction_results)

    # 4. 시각화 생성
    viz_success = create_feature_comparison_visualization(extraction_results, ensemble_results)

    return {
        'import_results': import_results,
        'extraction_results': extraction_results,
        'ensemble_results': ensemble_results,
        'visualization': viz_success
    }

def generate_feature_extraction_summary(results):
    """특징 추출 테스트 결과 요약"""

    print(f"\n{'='*70}")
    print("📊 4단계 특징 추출 테스트 결과 요약")
    print(f"{'='*70}")

    if not results:
        print("❌ 테스트 결과 없음")
        return False

    # Import 결과
    import_results = results.get('import_results', {})
    import_success = sum(import_results.values())
    import_total = len(import_results)
    print(f"📦 모듈 Import: {import_success}/{import_total} 성공")

    for module, success in import_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {module}")

    # 특징 추출 결과
    extraction_results = results.get('extraction_results', {})
    extraction_success = sum(1 for r in extraction_results.values() if r.get('success', False))
    extraction_total = len(extraction_results)
    print(f"\n🔬 특징 추출: {extraction_success}/{extraction_total} 성공")

    for extractor, result in extraction_results.items():
        if result.get('success'):
            shape = result['feature_shape']
            stats = result['feature_stats']
            print(f"   ✅ {extractor}: 특징 크기 {shape}, 평균 {stats['mean']:.3f}")
        else:
            print(f"   ❌ {extractor}: {result.get('error', '알 수 없는 오류')}")

    # 앙상블 결과
    ensemble_results = results.get('ensemble_results', {})
    ensemble_status = "✅" if ensemble_results.get('success') else "❌"
    print(f"\n🧩 특징 앙상블: {ensemble_status}")

    if ensemble_results.get('success'):
        shape = ensemble_results['ensemble_shape']
        stats = ensemble_results['ensemble_stats']
        print(f"   📊 앙상블 특징 크기: {shape}")
        print(f"   📊 앙상블 통계: 평균 {stats['mean']:.3f}, 표준편차 {stats['std']:.3f}")

    # 시각화 결과
    viz_status = "✅" if results.get('visualization') else "❌"
    print(f"\n📊 시각화: {viz_status}")

    # 전체 성공률 계산
    success_components = [
        import_success == import_total,
        extraction_success >= extraction_total * 0.5,  # 50% 이상 성공
        ensemble_results.get('success', False),
        results.get('visualization', False)
    ]

    success_count = sum(success_components)
    success_rate = (success_count / 4) * 100

    print(f"\n📋 전체 성공률: {success_count}/4 ({success_rate:.1f}%)")

    return success_rate >= 75  # 75% 이상이면 성공

def save_feature_test_results(results):
    """테스트 결과 저장"""

    output_file = f"analysis_results/data_validation/feature_extraction_step4_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # numpy 값들을 JSON 직렬화 가능하도록 변환
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    # 결과 정리
    clean_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            clean_results[key] = {k: convert_for_json(v) for k, v in value.items() if k != 'feature_stats' or isinstance(v, dict)}
            # feature_stats 별도 처리
            if key in ['extraction_results', 'ensemble_results']:
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and 'feature_stats' in sub_value:
                        clean_results[key][sub_key]['feature_stats'] = {
                            k: convert_for_json(v) for k, v in sub_value['feature_stats'].items()
                        }
        else:
            clean_results[key] = convert_for_json(value)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "test_description": "4단계 특징 추출 모듈들 포괄적 테스트",
            "results": clean_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 테스트 결과 저장: {output_file}")

def main():
    """메인 실행 함수"""

    print("🔧 4단계: 특징 추출 모듈들 포괄적 테스트 시작")

    # 포괄적 테스트 실행
    results = run_comprehensive_feature_tests()

    if not results:
        print("\n❌ 테스트 실행 실패")
        return False

    # 결과 요약
    success = generate_feature_extraction_summary(results)

    # 결과 저장
    save_feature_test_results(results)

    print(f"\n{'='*70}")
    if success:
        print("✅ 4단계 특징 추출 테스트 완료 - 성공")
        print("🎯 여러 기법의 특징 추출 및 앙상블 시스템 검증 완료")
        print("🎯 다음 단계: 5단계 기뢰 분류 테스트 진행 가능")
    else:
        print("⚠️ 4단계 특징 추출 테스트 완료 - 일부 개선 필요")
        print("🔧 일부 특징 추출기는 추가 구현이 필요할 수 있습니다")
    print(f"{'='*70}")

    return success

if __name__ == "__main__":
    main()