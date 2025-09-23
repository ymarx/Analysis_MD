#!/usr/bin/env python3
"""
실제 소나 이미지 데이터 특징 추출 성능 평가

실제 기뢰 이미지와 해저 배경 이미지를 사용하여 특징 추출기의 성능을 평가합니다.
"""

import numpy as np
import logging
from pathlib import Path
import json
import sys
import time
from typing import Dict, List, Any, Tuple, Optional
import os

# 프로젝트 모듈 import
sys.path.append('src')

from feature_extraction.hog_extractor import MultiScaleHOGExtractor
from feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from feature_extraction.gabor_extractor import GaborFeatureExtractor
from data_simulation.scenario_generator import ScenarioDataGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image_safely(image_path: Path) -> Optional[np.ndarray]:
    """이미지 안전하게 로드 (다양한 라이브러리 시도)"""
    try:
        # PIL 시도
        from PIL import Image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except ImportError:
        pass
    
    try:
        # OpenCV 시도
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    
    try:
        # matplotlib 시도
        import matplotlib.pyplot as plt
        img = plt.imread(str(image_path))
        return img
    except ImportError:
        pass
    
    logger.warning(f"이미지 로드 실패: {image_path}")
    return None


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """RGB를 그레이스케일로 변환"""
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # ITU-R BT.709 표준 가중치
        return np.dot(image[...,:3], [0.2126, 0.7152, 0.0722])
    else:
        return image


class RealImageFeatureEvaluator:
    """실제 이미지 데이터 특징 추출 평가기"""
    
    def __init__(self):
        """초기화"""
        # 특징 추출기들 초기화
        self.extractors = {}
        self._initialize_extractors()
        
        logger.info("실제 이미지 데이터 특징 추출 평가기 초기화 완료")
    
    def _initialize_extractors(self):
        """특징 추출기들 초기화"""
        try:
            # 1. HOG 추출기
            self.extractors['MultiScale_HOG'] = {
                'extractor': MultiScaleHOGExtractor(),
                'description': '다중 스케일 HOG 특징 추출기'
            }
            
            # 2. LBP 추출기
            self.extractors['Comprehensive_LBP'] = {
                'extractor': ComprehensiveLBPExtractor(),
                'description': '종합 LBP 특징 추출기'
            }
            
            # 3. Gabor 추출기
            self.extractors['Advanced_Gabor'] = {
                'extractor': GaborFeatureExtractor(n_frequencies=4, n_orientations=6),
                'description': '고급 Gabor 특징 추출기'
            }
            
            logger.info(f"{len(self.extractors)}개 특징 추출기 초기화 완료")
            
        except Exception as e:
            logger.error(f"특징 추출기 초기화 실패: {e}")
    
    def load_real_images(self, base_path: Path, max_images: int = 20) -> Dict[str, List[np.ndarray]]:
        """실제 이미지 데이터 로드"""
        logger.info(f"실제 이미지 데이터 로드 시작: {base_path}")
        
        image_data = {
            'mine_images': [],
            'background_images': []
        }
        
        # 기뢰 이미지 찾기
        mine_paths = list(base_path.glob("**/mine/*.jpg")) + list(base_path.glob("**/mine/*.png"))
        background_paths = list(base_path.glob("**/background/*.jpg")) + list(base_path.glob("**/background/*.png"))
        
        # 다른 패턴도 시도
        if not mine_paths:
            mine_paths = list(base_path.glob("**/*mine*.jpg")) + list(base_path.glob("**/*mine*.png"))
        
        if not background_paths:
            background_paths = list(base_path.glob("**/*background*.jpg")) + list(base_path.glob("**/*background*.png"))
            if not background_paths:
                # crop 폴더에서 mine이 아닌 다른 폴더들 찾기
                crop_dirs = list(base_path.glob("**/crop"))
                for crop_dir in crop_dirs:
                    for subdir in crop_dir.iterdir():
                        if subdir.is_dir() and 'mine' not in subdir.name.lower():
                            background_paths.extend(list(subdir.glob("*.jpg")))
                            background_paths.extend(list(subdir.glob("*.png")))
        
        logger.info(f"발견된 기뢰 이미지: {len(mine_paths)}개")
        logger.info(f"발견된 배경 이미지: {len(background_paths)}개")
        
        # 기뢰 이미지 로드
        for i, img_path in enumerate(mine_paths[:max_images]):
            image = load_image_safely(img_path)
            if image is not None:
                # 그레이스케일 변환 및 정규화
                gray_image = rgb_to_grayscale(image)
                normalized = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-10)
                image_data['mine_images'].append(normalized)
                
                if i < 3:  # 처음 몇 개만 로그
                    logger.info(f"기뢰 이미지 로드: {img_path.name} ({gray_image.shape})")
        
        # 배경 이미지 로드
        for i, img_path in enumerate(background_paths[:max_images]):
            image = load_image_safely(img_path)
            if image is not None:
                gray_image = rgb_to_grayscale(image)
                normalized = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-10)
                image_data['background_images'].append(normalized)
                
                if i < 3:  # 처음 몇 개만 로그
                    logger.info(f"배경 이미지 로드: {img_path.name} ({gray_image.shape})")
        
        logger.info(f"최종 로드된 이미지: 기뢰 {len(image_data['mine_images'])}개, "
                   f"배경 {len(image_data['background_images'])}개")
        
        return image_data
    
    def evaluate_extractor_on_images(self, extractor_name: str, images: List[np.ndarray], 
                                   image_type: str = "unknown") -> Dict[str, Any]:
        """특정 추출기로 이미지들에서 특징 추출 평가"""
        if extractor_name not in self.extractors:
            return {'error': f'알 수 없는 추출기: {extractor_name}'}
        
        extractor_info = self.extractors[extractor_name]
        extractor = extractor_info['extractor']
        
        results = {
            'extractor_name': extractor_name,
            'description': extractor_info['description'],
            'image_type': image_type,
            'total_images': len(images),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_times': [],
            'feature_counts': [],
            'feature_statistics': {},
            'errors': []
        }
        
        logger.info(f"{extractor_name} 특징 추출 평가 시작: {len(images)}개 {image_type} 이미지")
        
        all_features = []
        
        for i, image in enumerate(images):
            try:
                # 특징 추출 시간 측정
                start_time = time.time()
                
                # 특징 추출 실행
                features = extractor.extract_comprehensive_features(image)
                
                extraction_time = (time.time() - start_time) * 1000  # ms
                
                if features is not None and len(features) > 0:
                    results['successful_extractions'] += 1
                    results['extraction_times'].append(extraction_time)
                    results['feature_counts'].append(len(features))
                    all_features.append(features)
                    
                    if i % 5 == 0:  # 5개마다 로그
                        logger.debug(f"{extractor_name} {image_type} {i+1}: "
                                   f"{len(features)}개 특징, {extraction_time:.1f}ms")
                else:
                    results['failed_extractions'] += 1
                    results['errors'].append(f"이미지 {i}: 특징 추출 결과가 None 또는 빈 배열")
                
            except Exception as e:
                results['failed_extractions'] += 1
                results['errors'].append(f"이미지 {i}: {str(e)}")
                logger.warning(f"{extractor_name} {image_type} 이미지 {i} 처리 실패: {e}")
        
        # 통계 계산
        if results['successful_extractions'] > 0:
            results['success_rate'] = results['successful_extractions'] / results['total_images']
            results['avg_extraction_time'] = np.mean(results['extraction_times'])
            results['avg_feature_count'] = np.mean(results['feature_counts'])
            results['std_extraction_time'] = np.std(results['extraction_times'])
            results['std_feature_count'] = np.std(results['feature_counts'])
            
            # 전체 특징들의 통계
            if all_features:
                combined_features = np.concatenate(all_features)
                results['feature_statistics'] = {
                    'mean': float(np.mean(combined_features)),
                    'std': float(np.std(combined_features)),
                    'min': float(np.min(combined_features)),
                    'max': float(np.max(combined_features)),
                    'total_features': len(combined_features)
                }
        else:
            results['success_rate'] = 0.0
            results['avg_extraction_time'] = 0.0
            results['avg_feature_count'] = 0.0
        
        logger.info(f"{extractor_name} {image_type} 평가 완료: 성공률 {results['success_rate']:.1%}, "
                   f"평균 특징수 {results.get('avg_feature_count', 0):.0f}개")
        
        return results
    
    def compare_mine_vs_background_features(self, mine_results: Dict, bg_results: Dict) -> Dict[str, Any]:
        """기뢰 이미지와 배경 이미지의 특징 차이 분석"""
        comparison = {
            'extractor_name': mine_results['extractor_name'],
            'mine_vs_background_analysis': {}
        }
        
        # 각 지표별 비교
        metrics_to_compare = [
            'success_rate', 'avg_extraction_time', 'avg_feature_count'
        ]
        
        for metric in metrics_to_compare:
            mine_val = mine_results.get(metric, 0)
            bg_val = bg_results.get(metric, 0)
            
            comparison['mine_vs_background_analysis'][metric] = {
                'mine_value': mine_val,
                'background_value': bg_val,
                'difference': mine_val - bg_val,
                'ratio': mine_val / (bg_val + 1e-10)
            }
        
        # 특징 분포 비교
        mine_stats = mine_results.get('feature_statistics', {})
        bg_stats = bg_results.get('feature_statistics', {})
        
        if mine_stats and bg_stats:
            comparison['feature_distribution_analysis'] = {
                'mean_difference': mine_stats.get('mean', 0) - bg_stats.get('mean', 0),
                'std_ratio': mine_stats.get('std', 0) / (bg_stats.get('std', 1) + 1e-10),
                'range_difference': (mine_stats.get('max', 0) - mine_stats.get('min', 0)) - 
                                   (bg_stats.get('max', 0) - bg_stats.get('min', 0)),
                'discriminative_power': abs(mine_stats.get('mean', 0) - bg_stats.get('mean', 0)) / 
                                      (mine_stats.get('std', 1) + bg_stats.get('std', 1) + 1e-10)
            }
        
        return comparison
    
    def compare_with_synthetic_data(self, real_results: Dict[str, Any]) -> Dict[str, Any]:
        """실제 데이터 결과를 모의데이터와 비교"""
        logger.info("실제 데이터와 모의데이터 특징 추출 비교")
        
        # 모의데이터 생성
        generator = ScenarioDataGenerator()
        synthetic_images = []
        
        for scenario in ['A_deep_ocean', 'C_medium_depth', 'B_shallow_coastal']:
            for i in range(5):  # 각 시나리오당 5개씩
                sample = generator.generate_scenario_sample(
                    scenario,
                    target_present=(i % 2 == 0),
                    image_size=(64, 64)
                )
                synthetic_images.append(sample['image'])
        
        # 모의데이터 특징 추출
        synthetic_results = {}
        for extractor_name in self.extractors.keys():
            synth_result = self.evaluate_extractor_on_images(
                extractor_name, synthetic_images, "synthetic"
            )
            synthetic_results[extractor_name] = synth_result
        
        # 비교 분석
        comparison = {
            'real_data_results': real_results,
            'synthetic_data_results': synthetic_results,
            'comparison_analysis': {}
        }
        
        for extractor_name in self.extractors.keys():
            if extractor_name in real_results and extractor_name in synthetic_results:
                real_ext = real_results[extractor_name]
                synth_ext = synthetic_results[extractor_name]
                
                comparison['comparison_analysis'][extractor_name] = {
                    'success_rate_difference': real_ext.get('success_rate', 0) - synth_ext.get('success_rate', 0),
                    'feature_count_ratio': real_ext.get('avg_feature_count', 0) / (synth_ext.get('avg_feature_count', 1) + 1e-10),
                    'extraction_time_ratio': real_ext.get('avg_extraction_time', 0) / (synth_ext.get('avg_extraction_time', 1) + 1e-10),
                    'adaptability_score': self._calculate_adaptability_score(real_ext, synth_ext)
                }
        
        return comparison
    
    def _calculate_adaptability_score(self, real_result: Dict, synth_result: Dict) -> float:
        """실제 데이터에 대한 적응성 점수 계산 (0-1, 높을수록 좋음)"""
        # 성공률 유사성
        success_similarity = 1.0 - abs(real_result.get('success_rate', 0) - synth_result.get('success_rate', 0))
        
        # 특징수 안정성 (너무 큰 차이가 나지 않는 것이 좋음)
        feature_ratio = real_result.get('avg_feature_count', 0) / (synth_result.get('avg_feature_count', 1) + 1e-10)
        feature_stability = 1.0 / (1.0 + abs(feature_ratio - 1.0))
        
        # 종합 점수
        adaptability = (success_similarity + feature_stability) / 2.0
        return float(adaptability)


def main():
    """메인 실행 함수"""
    logger.info("실제 소나 이미지 데이터 특징 추출 성능 평가 시작")
    
    # 출력 디렉토리 생성
    output_dir = Path("data/results/real_image_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평가기 초기화
    evaluator = RealImageFeatureEvaluator()
    
    # 데이터 경로 설정
    base_path = Path("datasets")
    
    # 실제 이미지 데이터 로드
    logger.info("\n" + "="*50)
    logger.info("실제 이미지 데이터 로드")
    logger.info("="*50)
    
    image_data = evaluator.load_real_images(base_path, max_images=15)
    
    if not image_data['mine_images'] and not image_data['background_images']:
        logger.error("이미지를 로드할 수 없습니다. 데이터 경로를 확인하세요.")
        return
    
    # 특징 추출 평가
    logger.info("\n" + "="*50)
    logger.info("특징 추출 성능 평가")
    logger.info("="*50)
    
    results = {
        'analysis_type': 'real_image_feature_extraction',
        'data_summary': {
            'mine_images': len(image_data['mine_images']),
            'background_images': len(image_data['background_images']),
            'total_images': len(image_data['mine_images']) + len(image_data['background_images'])
        },
        'extractor_results': {},
        'mine_vs_background_comparison': {},
        'synthetic_data_comparison': {}
    }
    
    # 각 추출기별로 기뢰 이미지와 배경 이미지 평가
    for extractor_name in evaluator.extractors.keys():
        logger.info(f"\n{extractor_name} 평가 중...")
        
        extractor_results = {}
        
        # 기뢰 이미지 평가
        if image_data['mine_images']:
            mine_result = evaluator.evaluate_extractor_on_images(
                extractor_name, image_data['mine_images'], "mine"
            )
            extractor_results['mine'] = mine_result
        
        # 배경 이미지 평가
        if image_data['background_images']:
            bg_result = evaluator.evaluate_extractor_on_images(
                extractor_name, image_data['background_images'], "background"
            )
            extractor_results['background'] = bg_result
        
        results['extractor_results'][extractor_name] = extractor_results
        
        # 기뢰 vs 배경 비교
        if 'mine' in extractor_results and 'background' in extractor_results:
            comparison = evaluator.compare_mine_vs_background_features(
                extractor_results['mine'], extractor_results['background']
            )
            results['mine_vs_background_comparison'][extractor_name] = comparison
    
    # 모의데이터와 비교
    logger.info("\n" + "="*50)
    logger.info("모의데이터와의 비교 분석")
    logger.info("="*50)
    
    # 기뢰 이미지 결과만 모의데이터와 비교
    mine_results = {}
    for extractor_name, extractor_data in results['extractor_results'].items():
        if 'mine' in extractor_data:
            mine_results[extractor_name] = extractor_data['mine']
    
    if mine_results:
        synthetic_comparison = evaluator.compare_with_synthetic_data(mine_results)
        results['synthetic_data_comparison'] = synthetic_comparison
    
    # 결과 저장
    def json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_safe(v) for v in obj]
        else:
            return obj
    
    with open(output_dir / 'real_image_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)
    
    # 결과 요약 출력
    logger.info("\n" + "="*60)
    logger.info("실제 소나 이미지 특징 추출 평가 결과 요약")
    logger.info("="*60)
    
    logger.info(f"처리된 이미지: 기뢰 {results['data_summary']['mine_images']}개, "
               f"배경 {results['data_summary']['background_images']}개")
    
    # 추출기별 성능 요약
    for extractor_name, extractor_data in results['extractor_results'].items():
        logger.info(f"\n{extractor_name}:")
        
        if 'mine' in extractor_data:
            mine_perf = extractor_data['mine']
            logger.info(f"  기뢰 이미지: 성공률 {mine_perf.get('success_rate', 0):.1%}, "
                       f"평균 특징수 {mine_perf.get('avg_feature_count', 0):.0f}개, "
                       f"평균 시간 {mine_perf.get('avg_extraction_time', 0):.1f}ms")
        
        if 'background' in extractor_data:
            bg_perf = extractor_data['background']
            logger.info(f"  배경 이미지: 성공률 {bg_perf.get('success_rate', 0):.1%}, "
                       f"평균 특징수 {bg_perf.get('avg_feature_count', 0):.0f}개, "
                       f"평균 시간 {bg_perf.get('avg_extraction_time', 0):.1f}ms")
        
        # 판별력 분석
        if extractor_name in results['mine_vs_background_comparison']:
            comparison = results['mine_vs_background_comparison'][extractor_name]
            if 'feature_distribution_analysis' in comparison:
                discriminative_power = comparison['feature_distribution_analysis'].get('discriminative_power', 0)
                logger.info(f"  기뢰-배경 판별력: {discriminative_power:.3f}")
    
    # 모의데이터 비교 결과
    synth_comp = results.get('synthetic_data_comparison', {}).get('comparison_analysis', {})
    if synth_comp:
        logger.info(f"\n실제 데이터 vs 모의데이터 적응성:")
        for extractor_name, comp in synth_comp.items():
            logger.info(f"  {extractor_name}: 적응성 점수 {comp.get('adaptability_score', 0):.3f}")
    
    logger.info(f"\n상세 결과가 {output_dir}에 저장되었습니다.")
    
    return results


if __name__ == "__main__":
    main()