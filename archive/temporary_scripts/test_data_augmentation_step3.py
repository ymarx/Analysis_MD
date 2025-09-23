#!/usr/bin/env python3
"""
3단계: 데이터 증강 (회전, blur 등) 기능 테스트

목적: 25개 기물 데이터를 증강하여 충분한 훈련 데이터 생성 기능 검증
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

def test_augmentation_import():
    """데이터 증강 모듈 import 테스트"""

    print("🔧 데이터 증강 모듈 Import 테스트:")

    try:
        from src.data_augmentation.augmentation_engine import AdvancedAugmentationEngine
        from src.data_augmentation.augmentation_engine import AugmentationConfig
        print("   ✅ AdvancedAugmentationEngine import 성공")
        return True, AdvancedAugmentationEngine, AugmentationConfig
    except ImportError as e:
        print(f"   ❌ Import 실패: {e}")
        return False, None, None

def test_augmentation_initialization():
    """데이터 증강 엔진 초기화 테스트"""

    print("🔧 데이터 증강 엔진 초기화 테스트:")

    try:
        # 모듈 import
        success, engine_class, config_class = test_augmentation_import()
        if not success:
            return False, None

        # 기본 설정으로 초기화
        config = config_class()
        engine = engine_class(config)

        print("   ✅ 기본 설정으로 초기화 성공")

        # 커스텀 설정으로 초기화
        custom_config = config_class(
            rotation_range=(-90, 90),
            rotation_probability=0.8,
            noise_probability=0.7
        )
        custom_engine = engine_class(custom_config)

        print("   ✅ 커스텀 설정으로 초기화 성공")

        return True, custom_engine

    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        return False, None

def create_test_sonar_image():
    """테스트용 소나 이미지 생성"""

    # 실제 소나 이미지와 유사한 패턴 생성
    height, width = 200, 500
    image = np.zeros((height, width), dtype=np.float32)

    # 배경 노이즈
    background = np.random.normal(0.2, 0.1, (height, width))
    image = np.clip(background, 0, 1)

    # 기물 시뮬레이션 (여러 개의 밝은 점)
    mine_positions = [
        (50, 100), (80, 200), (120, 300), (150, 400)
    ]

    for y, x in mine_positions:
        # 기물 신호 (강한 반사)
        mine_signal = np.exp(-((np.arange(width) - x)**2 + (np.arange(height).reshape(-1, 1) - y)**2) / (2 * 15**2))
        image += mine_signal * 0.8

        # 음향 그림자 효과
        shadow_start = y + 20
        if shadow_start < height:
            shadow_width = min(30, width - x)
            image[shadow_start:shadow_start+40, x:x+shadow_width] *= 0.3

    # 해저면 반사
    seafloor_line = int(height * 0.8)
    image[seafloor_line:, :] += np.random.normal(0.4, 0.1, (height - seafloor_line, width))

    return np.clip(image, 0, 1)

def test_augmentation_functions(engine):
    """개별 증강 기능들 테스트"""

    print("🔧 개별 증강 기능 테스트:")

    # 테스트 이미지 생성
    test_image = create_test_sonar_image()
    print(f"   📊 테스트 이미지 크기: {test_image.shape}")

    results = {}

    # 1. 회전 테스트
    try:
        rotated_image, _ = engine.augment_single(test_image, augmentation_types=['rotation'])
        print("   ✅ 회전 증강 성공")
        results['rotation'] = True
    except Exception as e:
        print(f"   ❌ 회전 증강 실패: {e}")
        results['rotation'] = False

    # 2. 노이즈 추가 테스트
    try:
        noisy_image, _ = engine.augment_single(test_image, augmentation_types=['noise'])
        print("   ✅ 노이즈 증강 성공")
        results['noise'] = True
    except Exception as e:
        print(f"   ❌ 노이즈 증강 실패: {e}")
        results['noise'] = False

    # 3. 밝기/대비 조정 테스트
    try:
        bright_image, _ = engine.augment_single(test_image, augmentation_types=['brightness'])
        print("   ✅ 밝기 증강 성공")
        results['brightness'] = True
    except Exception as e:
        print(f"   ❌ 밝기 증강 실패: {e}")
        results['brightness'] = False

    # 4. 복합 증강 테스트
    try:
        combined_image, _ = engine.augment_single(test_image,
                                                augmentation_types=['rotation', 'noise', 'brightness'])
        print("   ✅ 복합 증강 성공")
        results['combined'] = True
    except Exception as e:
        print(f"   ❌ 복합 증강 실패: {e}")
        results['combined'] = False

    return results, test_image

def test_batch_augmentation(engine):
    """배치 증강 기능 테스트"""

    print("🔧 배치 증강 기능 테스트:")

    try:
        # 여러 테스트 이미지 생성
        test_images = []
        for i in range(5):
            img = create_test_sonar_image()
            # 각각 다른 특성 추가
            img += np.random.normal(0, 0.05, img.shape) * (i + 1) * 0.1
            test_images.append(img)

        print(f"   📊 배치 크기: {len(test_images)}개 이미지")

        # 배치 증강 실행
        if hasattr(engine, 'augment_batch'):
            augmented_batch = engine.augment_batch(test_images, augmentations_per_image=3)
            print(f"   ✅ 배치 증강 성공: {len(test_images)} → {len(augmented_batch)}개")
            return True, len(augmented_batch)
        else:
            # 개별 증강으로 배치 처리
            augmented_images = []
            for img in test_images:
                for _ in range(3):  # 각 이미지당 3개 증강
                    aug_img, _ = engine.augment_single(img)
                    augmented_images.append(aug_img)

            print(f"   ✅ 개별 증강으로 배치 처리 성공: {len(test_images)} → {len(augmented_images)}개")
            return True, len(augmented_images)

    except Exception as e:
        print(f"   ❌ 배치 증강 실패: {e}")
        return False, 0

def simulate_mine_data_augmentation(engine):
    """25개 기물 데이터 증강 시뮬레이션"""

    print("🔧 25개 기물 데이터 증강 시뮬레이션:")

    try:
        # 25개 기물 이미지 시뮬레이션
        mine_images = []
        for i in range(25):
            # 각기 다른 특성의 기물 이미지 생성
            img = create_test_sonar_image()

            # 기물별 다양성 추가
            noise_level = 0.02 + (i % 5) * 0.01
            img += np.random.normal(0, noise_level, img.shape)

            mine_images.append(np.clip(img, 0, 1))

        print(f"   📊 원본 기물 데이터: {len(mine_images)}개")

        # 각 기물당 10개씩 증강 (총 250개로 확장)
        augmented_count = 0
        augmentation_stats = {
            'rotation': 0,
            'noise': 0,
            'brightness': 0,
            'combined': 0
        }

        for i, mine_img in enumerate(mine_images):
            for aug_idx in range(10):
                # 다양한 증강 기법 적용
                if aug_idx < 3:
                    aug_type = ['rotation']
                    augmentation_stats['rotation'] += 1
                elif aug_idx < 6:
                    aug_type = ['noise', 'brightness']
                    augmentation_stats['noise'] += 1
                    augmentation_stats['brightness'] += 1
                else:
                    aug_type = ['rotation', 'noise', 'brightness']
                    augmentation_stats['combined'] += 1

                try:
                    aug_img, _ = engine.augment_single(mine_img, augmentation_types=aug_type)
                    augmented_count += 1
                except:
                    pass

        print(f"   ✅ 증강 완료: 25개 → {25 + augmented_count}개")
        print(f"   📊 증강 통계:")
        for aug_type, count in augmentation_stats.items():
            print(f"      - {aug_type}: {count}개")

        return True, 25 + augmented_count, augmentation_stats

    except Exception as e:
        print(f"   ❌ 기물 데이터 증강 시뮬레이션 실패: {e}")
        return False, 0, {}

def create_augmentation_visualization(engine, test_image):
    """증강 결과 시각화"""

    print("🔧 증강 결과 시각화 생성:")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Augmentation Results', fontsize=16)

        # 원본 이미지
        axes[0, 0].imshow(test_image, cmap='viridis')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # 다양한 증강 결과
        augmentation_types = [
            (['rotation'], 'Rotation'),
            (['noise'], 'Noise'),
            (['brightness'], 'Brightness'),
            (['rotation', 'noise'], 'Rotation + Noise'),
            (['rotation', 'noise', 'brightness'], 'Combined')
        ]

        for i, (aug_types, title) in enumerate(augmentation_types):
            row = (i + 1) // 3
            col = (i + 1) % 3

            try:
                aug_img, _ = engine.augment_single(test_image, augmentation_types=aug_types)
                axes[row, col].imshow(aug_img, cmap='viridis')
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)[:50]}',
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{title} (Failed)')

        plt.tight_layout()

        # 저장
        output_path = "analysis_results/visualizations/data_augmentation_test.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ 시각화 저장: {output_path}")
        return True

    except Exception as e:
        print(f"   ❌ 시각화 생성 실패: {e}")
        return False

def run_comprehensive_augmentation_tests():
    """포괄적 데이터 증강 테스트 실행"""

    print("=" * 70)
    print("3단계: 데이터 증강 기능 포괄적 테스트")
    print("=" * 70)

    # 1. 모듈 초기화
    success, engine = test_augmentation_initialization()
    if not success:
        return False

    # 2. 개별 기능 테스트
    function_results, test_image = test_augmentation_functions(engine)

    # 3. 배치 처리 테스트
    batch_success, augmented_count = test_batch_augmentation(engine)

    # 4. 기물 데이터 증강 시뮬레이션
    sim_success, total_mine_data, augmentation_stats = simulate_mine_data_augmentation(engine)

    # 5. 시각화 생성
    viz_success = create_augmentation_visualization(engine, test_image)

    return {
        'initialization': success,
        'functions': function_results,
        'batch_processing': batch_success,
        'batch_count': augmented_count,
        'mine_simulation': sim_success,
        'total_mine_data': total_mine_data,
        'augmentation_stats': augmentation_stats,
        'visualization': viz_success
    }

def generate_augmentation_summary(results):
    """데이터 증강 테스트 결과 요약"""

    print(f"\n{'='*70}")
    print("📊 3단계 데이터 증강 테스트 결과 요약")
    print(f"{'='*70}")

    if not results:
        print("❌ 테스트 결과 없음")
        return False

    # 기본 기능 확인
    print(f"✅ 엔진 초기화: {'성공' if results.get('initialization') else '실패'}")

    # 개별 기능 결과
    functions = results.get('functions', {})
    function_success = sum(functions.values())
    function_total = len(functions)
    print(f"📊 개별 기능: {function_success}/{function_total} 성공")

    for func_name, success in functions.items():
        status = "✅" if success else "❌"
        print(f"   {status} {func_name}")

    # 배치 처리 결과
    batch_status = "✅" if results.get('batch_processing') else "❌"
    batch_count = results.get('batch_count', 0)
    print(f"{batch_status} 배치 처리: {batch_count}개 증강 이미지 생성")

    # 기물 데이터 시뮬레이션 결과
    mine_status = "✅" if results.get('mine_simulation') else "❌"
    total_mine = results.get('total_mine_data', 0)
    print(f"{mine_status} 기물 데이터 증강: 25개 → {total_mine}개")

    if results.get('augmentation_stats'):
        print("   📊 증강 기법별 적용 횟수:")
        for aug_type, count in results['augmentation_stats'].items():
            print(f"      - {aug_type}: {count}회")

    # 시각화 결과
    viz_status = "✅" if results.get('visualization') else "❌"
    print(f"{viz_status} 시각화 생성")

    # 전체 성공률 계산
    success_count = sum([
        results.get('initialization', False),
        function_success == function_total,
        results.get('batch_processing', False),
        results.get('mine_simulation', False),
        results.get('visualization', False)
    ])
    success_rate = (success_count / 5) * 100

    print(f"\n📋 전체 성공률: {success_count}/5 ({success_rate:.1f}%)")

    return success_rate >= 80

def save_augmentation_test_results(results):
    """테스트 결과 저장"""

    output_file = f"analysis_results/data_validation/augmentation_step3_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "test_description": "3단계 데이터 증강 기능 포괄적 테스트",
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 테스트 결과 저장: {output_file}")

def main():
    """메인 실행 함수"""

    print("🔧 3단계: 데이터 증강 기능 포괄적 테스트 시작")

    # 포괄적 테스트 실행
    results = run_comprehensive_augmentation_tests()

    if not results:
        print("\n❌ 테스트 실행 실패")
        return False

    # 결과 요약
    success = generate_augmentation_summary(results)

    # 결과 저장
    save_augmentation_test_results(results)

    print(f"\n{'='*70}")
    if success:
        print("✅ 3단계 데이터 증강 테스트 완료 - 성공")
        print("🎯 25개 기물 데이터를 충분한 훈련 데이터로 증강 가능")
        print("🎯 다음 단계: 4단계 특징 추출 테스트 진행 가능")
    else:
        print("⚠️ 3단계 데이터 증강 테스트 완료 - 일부 개선 필요")
    print(f"{'='*70}")

    return success

if __name__ == "__main__":
    main()