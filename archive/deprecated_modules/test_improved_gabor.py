#!/usr/bin/env python3
"""
개선된 Gabor 특징 추출기 테스트
OpenCV 의존성 해결 후 성능 검증
"""

import numpy as np
import logging
import time
from pathlib import Path

# 프로젝트 모듈 import
import sys
sys.path.append('src')

from feature_extraction.gabor_extractor import (
    GaborFeatureExtractor, 
    AdaptiveGaborExtractor,
    CV2_AVAILABLE,
    SKIMAGE_GABOR_AVAILABLE
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_sonar_image(size=(96, 96), target_present=True):
    """테스트용 소나 이미지 생성"""
    np.random.seed(42)
    image = np.random.normal(0.5, 0.1, size)
    
    if target_present:
        # 기뢰 형태 시뮬레이션 (타원형 + 그림자)
        center_y, center_x = size[0] // 2, size[1] // 2
        
        # 타원형 객체
        y, x = np.ogrid[:size[0], :size[1]]
        mask = ((x - center_x) / 12)**2 + ((y - center_y) / 8)**2 < 1
        image[mask] += 0.3
        
        # 음향 그림자
        shadow_start = center_x + 15
        shadow_end = min(shadow_start + 20, size[1])
        shadow_y_start = max(0, center_y - 10)
        shadow_y_end = min(center_y + 10, size[0])
        
        image[shadow_y_start:shadow_y_end, shadow_start:shadow_end] -= 0.2
        
        # 텍스처 추가
        texture = np.random.normal(0, 0.05, size)
        texture = np.abs(np.fft.ifft2(np.fft.fft2(texture) * 
                                     np.exp(-0.1 * np.arange(size[0]).reshape(-1, 1))))
        image += texture.real
    
    # 클리핑 및 정규화
    image = np.clip(image, 0, 1)
    return image.astype(np.float64)


def test_gabor_extractors():
    """Gabor 추출기들 테스트"""
    logger.info("=== 개선된 Gabor 특징 추출기 테스트 ===")
    logger.info(f"OpenCV 사용 가능: {CV2_AVAILABLE}")
    logger.info(f"scikit-image Gabor 사용 가능: {SKIMAGE_GABOR_AVAILABLE}")
    
    # 테스트 이미지 생성
    positive_images = [create_test_sonar_image(target_present=True) for _ in range(5)]
    negative_images = [create_test_sonar_image(target_present=False) for _ in range(5)]
    all_images = positive_images + negative_images
    
    results = {}
    
    # 1. 기본 Gabor 특징 추출기 테스트
    logger.info("\n1. 기본 Gabor 특징 추출기 테스트")
    try:
        extractor = GaborFeatureExtractor(
            n_frequencies=4,  # 의존성 이슈로 줄여서 테스트
            n_orientations=6,
            patch_size=16
        )
        
        extraction_times = []
        feature_counts = []
        success_count = 0
        
        for i, image in enumerate(all_images):
            try:
                start_time = time.time()
                features = extractor.extract_comprehensive_features(image)
                extraction_time = (time.time() - start_time) * 1000
                
                extraction_times.append(extraction_time)
                feature_counts.append(len(features))
                success_count += 1
                
                logger.info(f"  이미지 {i+1}: {len(features)}개 특징, {extraction_time:.1f}ms")
                
            except Exception as e:
                logger.warning(f"  이미지 {i+1} 처리 실패: {e}")
        
        if success_count > 0:
            results['GaborFeatureExtractor'] = {
                'success_rate': success_count / len(all_images),
                'avg_feature_count': np.mean(feature_counts) if feature_counts else 0,
                'avg_extraction_time': np.mean(extraction_times) if extraction_times else 0,
                'description': '기본 Gabor 특징 추출기'
            }
        
    except Exception as e:
        logger.error(f"기본 Gabor 추출기 초기화 실패: {e}")
        results['GaborFeatureExtractor'] = {'error': str(e)}
    
    # 2. 적응형 Gabor 특징 추출기 테스트
    logger.info("\n2. 적응형 Gabor 특징 추출기 테스트")
    try:
        adaptive_extractor = AdaptiveGaborExtractor()
        
        extraction_times = []
        feature_counts = []
        success_count = 0
        
        for i, image in enumerate(all_images):
            try:
                start_time = time.time()
                features = adaptive_extractor.extract_adaptive_features(image)
                extraction_time = (time.time() - start_time) * 1000
                
                extraction_times.append(extraction_time)
                feature_counts.append(len(features))
                success_count += 1
                
                logger.info(f"  이미지 {i+1}: {len(features)}개 특징, {extraction_time:.1f}ms")
                
            except Exception as e:
                logger.warning(f"  이미지 {i+1} 처리 실패: {e}")
        
        if success_count > 0:
            results['AdaptiveGaborExtractor'] = {
                'success_rate': success_count / len(all_images),
                'avg_feature_count': np.mean(feature_counts) if feature_counts else 0,
                'avg_extraction_time': np.mean(extraction_times) if extraction_times else 0,
                'description': '적응형 Gabor 특징 추출기'
            }
        
    except Exception as e:
        logger.error(f"적응형 Gabor 추출기 초기화 실패: {e}")
        results['AdaptiveGaborExtractor'] = {'error': str(e)}
    
    return results


def main():
    """메인 테스트 함수"""
    logger.info("OpenCV 의존성 해결 후 Gabor 추출기 성능 테스트 시작")
    
    # 테스트 실행
    results = test_gabor_extractors()
    
    # 결과 출력
    logger.info("\n=== 테스트 결과 요약 ===")
    for extractor_name, result in results.items():
        if 'error' in result:
            logger.error(f"{extractor_name}: 실행 실패 - {result['error']}")
        else:
            logger.info(f"{extractor_name}:")
            logger.info(f"  성공률: {result['success_rate']:.1%}")
            logger.info(f"  평균 특징 수: {result['avg_feature_count']:.0f}개")
            logger.info(f"  평균 처리 시간: {result['avg_extraction_time']:.1f}ms")
            logger.info(f"  설명: {result['description']}")
    
    # JSON 결과 저장
    import json
    results_dir = Path('data/results/gabor_dependency_test')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'gabor_dependency_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n테스트 결과가 {results_dir / 'gabor_dependency_test_results.json'}에 저장되었습니다.")
    
    return results


if __name__ == "__main__":
    main()