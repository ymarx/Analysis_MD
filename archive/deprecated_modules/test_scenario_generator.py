#!/usr/bin/env python3
"""
시나리오별 모의데이터 생성기 테스트 (matplotlib 없는 버전)
"""

import numpy as np
import logging
from pathlib import Path
import json
import sys
import time
from typing import Dict

# 프로젝트 모듈 import
sys.path.append('src')

from data_simulation.scenario_generator import ScenarioDataGenerator, MarineEnvironment

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_image_as_text(image: np.ndarray, filepath: Path) -> None:
    """이미지를 텍스트 형태로 저장 (matplotlib 대체)"""
    h, w = image.shape
    
    with open(filepath, 'w') as f:
        f.write(f"# 이미지 크기: {h}x{w}\n")
        f.write(f"# 값 범위: {image.min():.3f} ~ {image.max():.3f}\n")
        f.write(f"# 평균: {image.mean():.3f}, 표준편차: {image.std():.3f}\n\n")
        
        # 간단한 ASCII 아트로 시각화
        f.write("ASCII 시각화 (32x32 다운샘플링):\n")
        step_h, step_w = max(1, h//32), max(1, w//32)
        
        for i in range(0, h, step_h):
            line = ""
            for j in range(0, w, step_w):
                value = image[i, j]
                if value < 0.2:
                    line += "."
                elif value < 0.4:
                    line += "-"
                elif value < 0.6:
                    line += "o"
                elif value < 0.8:
                    line += "O"
                else:
                    line += "#"
            f.write(line + "\n")


def test_single_scenario(generator: ScenarioDataGenerator, scenario_name: str) -> Dict:
    """단일 시나리오 테스트"""
    logger.info(f"\n=== 시나리오 {scenario_name} 테스트 ===")
    
    results = {}
    
    try:
        # 양성 샘플 생성
        start_time = time.time()
        positive_sample = generator.generate_scenario_sample(
            scenario_name, target_present=True, image_size=(96, 96)
        )
        positive_time = (time.time() - start_time) * 1000
        
        # 음성 샘플 생성
        start_time = time.time()
        negative_sample = generator.generate_scenario_sample(
            scenario_name, target_present=False, image_size=(96, 96)
        )
        negative_time = (time.time() - start_time) * 1000
        
        # 통계 계산
        pos_img = positive_sample['image']
        neg_img = negative_sample['image']
        
        results = {
            'scenario_name': scenario_name,
            'success': True,
            'positive_stats': {
                'mean': float(pos_img.mean()),
                'std': float(pos_img.std()),
                'min': float(pos_img.min()),
                'max': float(pos_img.max()),
                'generation_time_ms': positive_time
            },
            'negative_stats': {
                'mean': float(neg_img.mean()),
                'std': float(neg_img.std()),
                'min': float(neg_img.min()),
                'max': float(neg_img.max()),
                'generation_time_ms': negative_time
            },
            'metadata': {
                'positive': positive_sample['metadata'],
                'negative': negative_sample['metadata']
            }
        }
        
        logger.info(f"✅ {scenario_name} 성공:")
        logger.info(f"  양성 샘플: {pos_img.mean():.3f}±{pos_img.std():.3f}, {positive_time:.1f}ms")
        logger.info(f"  음성 샘플: {neg_img.mean():.3f}±{neg_img.std():.3f}, {negative_time:.1f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ {scenario_name} 실패: {e}")
        return {'scenario_name': scenario_name, 'success': False, 'error': str(e)}


def test_batch_generation(generator: ScenarioDataGenerator) -> Dict:
    """배치 생성 테스트"""
    logger.info("\n=== 배치 생성 테스트 ===")
    
    try:
        # 작은 배치로 테스트
        dataset = generator.generate_scenario_dataset(
            'C_medium_depth',  # 중간 복잡도 시나리오
            n_positive=5,
            n_negative=5,
            image_size=(64, 64)
        )
        
        images = np.array(dataset['images'])
        labels = np.array(dataset['labels'])
        
        # 클래스별 통계
        positive_images = images[labels == 1]
        negative_images = images[labels == 0]
        
        results = {
            'success': True,
            'total_samples': len(images),
            'positive_count': len(positive_images),
            'negative_count': len(negative_images),
            'positive_stats': {
                'mean': float(positive_images.mean()),
                'std': float(positive_images.std())
            },
            'negative_stats': {
                'mean': float(negative_images.mean()),
                'std': float(negative_images.std())
            }
        }
        
        logger.info(f"✅ 배치 생성 성공: {len(images)}개 샘플")
        logger.info(f"  양성: {len(positive_images)}개 ({positive_images.mean():.3f}±{positive_images.std():.3f})")
        logger.info(f"  음성: {len(negative_images)}개 ({negative_images.mean():.3f}±{negative_images.std():.3f})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 배치 생성 실패: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """메인 테스트 함수"""
    
    logger.info("해양환경 시나리오별 모의데이터 생성기 테스트 시작")
    
    # 출력 디렉토리 생성
    output_dir = Path("data/results/scenario_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 생성기 초기화
    generator = ScenarioDataGenerator()
    
    all_results = {}
    
    # 1. 각 시나리오별 개별 테스트
    logger.info("\n" + "="*50)
    logger.info("단일 시나리오 테스트")
    logger.info("="*50)
    
    for scenario_name in generator.scenarios.keys():
        result = test_single_scenario(generator, scenario_name)
        all_results[scenario_name] = result
        
        # 성공한 경우 샘플 이미지를 텍스트로 저장
        if result.get('success', False):
            # 한 개씩 더 생성해서 텍스트로 저장
            sample = generator.generate_scenario_sample(scenario_name, target_present=True, image_size=(64, 64))
            
            # 텍스트 형태로 저장
            save_image_as_text(
                sample['image'],
                output_dir / f'{scenario_name}_sample.txt'
            )
    
    # 2. 배치 생성 테스트
    logger.info("\n" + "="*50)
    logger.info("배치 생성 테스트")
    logger.info("="*50)
    
    batch_result = test_batch_generation(generator)
    all_results['batch_test'] = batch_result
    
    # 3. 결과 요약
    logger.info("\n" + "="*50)
    logger.info("테스트 결과 요약")
    logger.info("="*50)
    
    success_count = sum(1 for r in all_results.values() if r.get('success', False))
    total_scenarios = len(generator.scenarios) + 1  # +1 for batch test
    
    logger.info(f"전체 테스트: {success_count}/{total_scenarios} 성공")
    
    for name, result in all_results.items():
        if name == 'batch_test':
            continue
        status = "✅" if result.get('success', False) else "❌"
        logger.info(f"  {status} {name}")
    
    # 배치 테스트 결과
    batch_status = "✅" if batch_result.get('success', False) else "❌"
    logger.info(f"  {batch_status} batch_test")
    
    # 4. 결과 저장
    with open(output_dir / 'scenario_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n테스트 결과가 {output_dir}에 저장되었습니다.")
    
    # 5. 환경별 특성 분석
    logger.info("\n" + "="*50)
    logger.info("환경별 특성 분석")
    logger.info("="*50)
    
    for scenario_name, config in generator.scenarios.items():
        logger.info(f"\n{scenario_name}:")
        logger.info(f"  환경: {config.environment.value}")
        logger.info(f"  수심: {config.depth_range[0]}-{config.depth_range[1]}m")
        logger.info(f"  노이즈 레벨: {config.noise_level:.1f}")
        logger.info(f"  텍스처 복잡도: {config.texture_complexity:.1f}")
        logger.info(f"  해류 강도: {config.current_strength:.1f}")
        logger.info(f"  기뢰 가시성: {config.target_visibility:.1f}")
        logger.info(f"  그림자 강도: {config.shadow_strength:.1f}")
    
    return all_results


if __name__ == "__main__":
    main()