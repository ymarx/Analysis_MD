#!/usr/bin/env python3
"""
실제 XTF 데이터 특징 추출 성능 평가 (비라벨 데이터)

라벨 정보 없이 실제 XTF 데이터에서 특징 추출기들의 성능과 안정성을 평가합니다.
"""

import numpy as np
import logging
from pathlib import Path
import json
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

# 프로젝트 모듈 import
sys.path.append('src')

from data_processing.xtf_reader import XTFReader, BatchXTFProcessor
from data_processing.preprocessor import Preprocessor
from feature_extraction.hog_extractor import MultiScaleHOGExtractor
from feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from feature_extraction.gabor_extractor import GaborFeatureExtractor
from data_simulation.scenario_generator import ScenarioDataGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealXTFFeatureEvaluator:
    """실제 XTF 데이터 특징 추출 평가기"""
    
    def __init__(self):
        """초기화"""
        self.xtf_reader = XTFReader()
        self.preprocessor = Preprocessor()
        
        # 특징 추출기들 초기화
        self.extractors = {}
        self._initialize_extractors()
        
        logger.info("실제 XTF 데이터 특징 추출 평가기 초기화 완료")
    
    def _initialize_extractors(self):
        """특징 추출기들 초기화"""
        try:
            # 1. HOG 추출기
            self.extractors['MultiScale_HOG'] = {
                'extractor': MultiScaleHOGExtractor(),
                'description': '다중 스케일 HOG 특징 추출기',
                'extract_method': 'extract_comprehensive_features'
            }
            
            # 2. LBP 추출기
            self.extractors['Comprehensive_LBP'] = {
                'extractor': ComprehensiveLBPExtractor(),
                'description': '종합 LBP 특징 추출기',
                'extract_method': 'extract_comprehensive_features'
            }
            
            # 3. Gabor 추출기
            self.extractors['Advanced_Gabor'] = {
                'extractor': GaborFeatureExtractor(),
                'description': '고급 Gabor 특징 추출기',
                'extract_method': 'extract_comprehensive_features'
            }
            
            logger.info(f"{len(self.extractors)}개 특징 추출기 초기화 완료")
            
        except Exception as e:
            logger.error(f"특징 추출기 초기화 실패: {e}")
    
    def load_and_preprocess_xtf(self, xtf_path: Path, max_pings: int = 50) -> Optional[np.ndarray]:
        """
        XTF 파일 로드 및 전처리
        
        Args:
            xtf_path: XTF 파일 경로
            max_pings: 최대 처리할 ping 수
            
        Returns:
            Optional[np.ndarray]: 전처리된 이미지 패치들 또는 None
        """
        try:
            logger.info(f"XTF 파일 로드 중: {xtf_path.name}")
            
            # XTF 데이터 읽기
            ping_data = self.xtf_reader.read_xtf_file(str(xtf_path))
            if not ping_data:
                logger.warning(f"XTF 파일에서 데이터를 읽을 수 없음: {xtf_path}")
                return None
            
            logger.info(f"총 {len(ping_data)} ping 데이터 로드됨")
            
            # ping 수 제한
            if len(ping_data) > max_pings:
                ping_data = ping_data[:max_pings]
                logger.info(f"처리 ping 수를 {max_pings}개로 제한")
            
            # 소나 이미지 생성
            sonar_image = self.xtf_reader.create_sonar_image(ping_data)
            if sonar_image is None or sonar_image.size == 0:
                logger.warning("소나 이미지 생성 실패")
                return None
            
            logger.info(f"소나 이미지 생성됨: {sonar_image.shape}")
            
            # 전처리
            preprocessed = self.preprocessor.comprehensive_preprocessing(sonar_image)
            
            # 패치 추출 (64x64 크기로)
            patches = self._extract_patches(preprocessed, patch_size=64, max_patches=20)
            
            logger.info(f"{len(patches)}개 패치 추출 완료")
            return np.array(patches)
            
        except Exception as e:
            logger.error(f"XTF 파일 처리 실패 {xtf_path}: {e}")
            return None
    
    def _extract_patches(self, image: np.ndarray, patch_size: int = 64, max_patches: int = 20) -> List[np.ndarray]:
        """이미지에서 패치들을 추출"""
        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        patches = []
        
        # 균등한 간격으로 패치 추출
        step_h = max(1, h // int(np.sqrt(max_patches)))
        step_w = max(1, w // int(np.sqrt(max_patches)))
        
        for i in range(0, h - patch_size, step_h):
            for j in range(0, w - patch_size, step_w):
                if len(patches) >= max_patches:
                    break
                
                if len(image.shape) == 2:
                    patch = image[i:i+patch_size, j:j+patch_size]
                else:
                    patch = image[i:i+patch_size, j:j+patch_size, 0]  # 첫 번째 채널만
                
                # 패치 품질 확인
                if self._is_valid_patch(patch):
                    patches.append(patch)
            
            if len(patches) >= max_patches:
                break
        
        return patches
    
    def _is_valid_patch(self, patch: np.ndarray) -> bool:
        """패치가 특징 추출에 유효한지 확인"""
        # 기본적인 품질 체크
        if patch.std() < 0.01:  # 너무 균일한 패치 제외
            return False
        
        if np.sum(patch == 0) > patch.size * 0.5:  # 절반 이상이 0인 패치 제외
            return False
        
        return True
    
    def evaluate_extractor_on_patches(self, extractor_name: str, patches: np.ndarray) -> Dict[str, Any]:
        """특정 추출기로 패치들에서 특징 추출 평가"""
        if extractor_name not in self.extractors:
            return {'error': f'알 수 없는 추출기: {extractor_name}'}
        
        extractor_info = self.extractors[extractor_name]
        extractor = extractor_info['extractor']
        extract_method = extractor_info['extract_method']
        
        results = {
            'extractor_name': extractor_name,
            'description': extractor_info['description'],
            'total_patches': len(patches),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_times': [],
            'feature_counts': [],
            'feature_statistics': {},
            'errors': []
        }
        
        logger.info(f"{extractor_name} 특징 추출 평가 시작: {len(patches)}개 패치")
        
        all_features = []
        
        for i, patch in enumerate(patches):
            try:
                # 패치를 0-1 범위로 정규화
                normalized_patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-10)
                
                # 특징 추출 시간 측정
                start_time = time.time()
                
                # 특징 추출 실행
                if hasattr(extractor, extract_method):
                    features = getattr(extractor, extract_method)(normalized_patch)
                else:
                    logger.warning(f"{extractor_name}에서 {extract_method} 메서드를 찾을 수 없음")
                    continue
                
                extraction_time = (time.time() - start_time) * 1000  # ms
                
                if features is not None and len(features) > 0:
                    results['successful_extractions'] += 1
                    results['extraction_times'].append(extraction_time)
                    results['feature_counts'].append(len(features))
                    all_features.append(features)
                    
                    if i % 5 == 0:  # 5개마다 로그
                        logger.debug(f"{extractor_name} 패치 {i+1}: {len(features)}개 특징, {extraction_time:.1f}ms")
                else:
                    results['failed_extractions'] += 1
                    results['errors'].append(f"패치 {i}: 특징 추출 결과가 None 또는 빈 배열")
                
            except Exception as e:
                results['failed_extractions'] += 1
                results['errors'].append(f"패치 {i}: {str(e)}")
                logger.warning(f"{extractor_name} 패치 {i} 처리 실패: {e}")
        
        # 통계 계산
        if results['successful_extractions'] > 0:
            results['success_rate'] = results['successful_extractions'] / results['total_patches']
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
        
        logger.info(f"{extractor_name} 평가 완료: 성공률 {results['success_rate']:.1%}, "
                   f"평균 특징수 {results.get('avg_feature_count', 0):.0f}개")
        
        return results
    
    def evaluate_all_extractors_on_xtf(self, xtf_path: Path) -> Dict[str, Any]:
        """단일 XTF 파일에 대해 모든 추출기 평가"""
        logger.info(f"XTF 파일 종합 평가 시작: {xtf_path.name}")
        
        # XTF 데이터 로드 및 전처리
        patches = self.load_and_preprocess_xtf(xtf_path)
        if patches is None or len(patches) == 0:
            return {
                'xtf_file': str(xtf_path),
                'error': 'XTF 파일 로드 또는 패치 추출 실패',
                'extractors': {}
            }
        
        results = {
            'xtf_file': str(xtf_path),
            'file_name': xtf_path.name,
            'patch_count': len(patches),
            'patch_shape': list(patches[0].shape),
            'extractors': {}
        }
        
        # 각 추출기별 평가
        for extractor_name in self.extractors.keys():
            logger.info(f"{extractor_name} 평가 중...")
            extractor_result = self.evaluate_extractor_on_patches(extractor_name, patches)
            results['extractors'][extractor_name] = extractor_result
        
        return results
    
    def compare_with_synthetic_data(self, real_results: Dict[str, Any]) -> Dict[str, Any]:
        """실제 데이터 결과를 모의데이터와 비교"""
        logger.info("실제 데이터와 모의데이터 특징 추출 비교 분석")
        
        # 모의데이터 생성 및 특징 추출
        synthetic_results = self._generate_synthetic_comparison_data()
        
        comparison = {
            'real_data': real_results,
            'synthetic_data': synthetic_results,
            'comparison_analysis': {}
        }
        
        # 각 추출기별 비교
        for extractor_name in self.extractors.keys():
            if (extractor_name in real_results.get('extractors', {}) and
                extractor_name in synthetic_results.get('extractors', {})):
                
                real_ext = real_results['extractors'][extractor_name]
                synth_ext = synthetic_results['extractors'][extractor_name]
                
                # 비교 지표 계산
                comparison_metrics = {
                    'success_rate_diff': real_ext.get('success_rate', 0) - synth_ext.get('success_rate', 0),
                    'feature_count_ratio': real_ext.get('avg_feature_count', 0) / (synth_ext.get('avg_feature_count', 1) + 1e-10),
                    'extraction_time_ratio': real_ext.get('avg_extraction_time', 0) / (synth_ext.get('avg_extraction_time', 1) + 1e-10),
                    'feature_distribution_similarity': self._calculate_feature_similarity(
                        real_ext.get('feature_statistics', {}),
                        synth_ext.get('feature_statistics', {})
                    )
                }
                
                comparison['comparison_analysis'][extractor_name] = comparison_metrics
        
        return comparison
    
    def _generate_synthetic_comparison_data(self) -> Dict[str, Any]:
        """비교용 모의데이터 생성 및 특징 추출"""
        generator = ScenarioDataGenerator()
        
        # 중간 복잡도 시나리오로 비교 데이터 생성
        synthetic_patches = []
        for i in range(20):
            sample = generator.generate_scenario_sample(
                'C_medium_depth',
                target_present=(i % 2 == 0),  # 절반은 양성, 절반은 음성
                image_size=(64, 64)
            )
            synthetic_patches.append(sample['image'])
        
        synthetic_patches = np.array(synthetic_patches)
        
        # 모의데이터에 대해 특징 추출 평가
        synthetic_results = {
            'data_type': 'synthetic',
            'patch_count': len(synthetic_patches),
            'extractors': {}
        }
        
        for extractor_name in self.extractors.keys():
            extractor_result = self.evaluate_extractor_on_patches(extractor_name, synthetic_patches)
            synthetic_results['extractors'][extractor_name] = extractor_result
        
        return synthetic_results
    
    def _calculate_feature_similarity(self, real_stats: Dict, synth_stats: Dict) -> float:
        """실제 데이터와 모의데이터 특징 분포의 유사성 계산"""
        if not real_stats or not synth_stats:
            return 0.0
        
        # 평균과 표준편차 비교
        mean_diff = abs(real_stats.get('mean', 0) - synth_stats.get('mean', 0))
        std_diff = abs(real_stats.get('std', 0) - synth_stats.get('std', 0))
        
        # 유사도 점수 계산 (0-1, 높을수록 유사)
        similarity = 1.0 / (1.0 + mean_diff + std_diff)
        return float(similarity)


def analyze_multiple_xtf_files(xtf_files: List[Path], max_files: int = 3) -> Dict[str, Any]:
    """여러 XTF 파일에 대한 종합 분석"""
    logger.info(f"{len(xtf_files)} 개 XTF 파일 중 최대 {max_files}개 분석 시작")
    
    evaluator = RealXTFFeatureEvaluator()
    
    # 원본 데이터만 선택 (시뮬레이션 데이터 제외)
    original_files = [f for f in xtf_files if '/original/' in str(f)]
    if not original_files:
        # original 폴더가 없으면 simulation이 없는 파일들 사용
        original_files = [f for f in xtf_files if '/simulation/' not in str(f)]
    
    # 파일 수 제한
    selected_files = original_files[:max_files]
    
    results = {
        'analysis_type': 'real_xtf_feature_extraction',
        'total_files': len(xtf_files),
        'original_files_found': len(original_files),
        'files_analyzed': len(selected_files),
        'individual_results': {},
        'aggregate_analysis': {},
        'synthetic_comparison': {}
    }
    
    individual_results = []
    
    # 각 파일별 분석
    for i, xtf_file in enumerate(selected_files):
        logger.info(f"\n{'='*50}")
        logger.info(f"파일 {i+1}/{len(selected_files)}: {xtf_file.name}")
        logger.info(f"{'='*50}")
        
        file_result = evaluator.evaluate_all_extractors_on_xtf(xtf_file)
        results['individual_results'][xtf_file.name] = file_result
        
        if 'error' not in file_result:
            individual_results.append(file_result)
    
    # 종합 통계 계산
    if individual_results:
        results['aggregate_analysis'] = calculate_aggregate_statistics(individual_results)
        
        # 모의데이터와 비교 (첫 번째 성공한 결과 사용)
        results['synthetic_comparison'] = evaluator.compare_with_synthetic_data(individual_results[0])
    
    return results


def calculate_aggregate_statistics(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """개별 결과들의 종합 통계 계산"""
    aggregate = {
        'files_processed': len(individual_results),
        'total_patches': sum(r.get('patch_count', 0) for r in individual_results),
        'extractor_performance': {}
    }
    
    # 추출기별 종합 성능 계산
    extractor_names = []
    if individual_results:
        extractor_names = list(individual_results[0].get('extractors', {}).keys())
    
    for extractor_name in extractor_names:
        success_rates = []
        feature_counts = []
        extraction_times = []
        
        for result in individual_results:
            ext_result = result.get('extractors', {}).get(extractor_name, {})
            if 'success_rate' in ext_result:
                success_rates.append(ext_result['success_rate'])
                feature_counts.append(ext_result.get('avg_feature_count', 0))
                extraction_times.append(ext_result.get('avg_extraction_time', 0))
        
        if success_rates:
            aggregate['extractor_performance'][extractor_name] = {
                'avg_success_rate': np.mean(success_rates),
                'std_success_rate': np.std(success_rates),
                'avg_feature_count': np.mean(feature_counts),
                'avg_extraction_time': np.mean(extraction_times),
                'files_successful': len([r for r in success_rates if r > 0])
            }
    
    return aggregate


def main():
    """메인 실행 함수"""
    logger.info("실제 XTF 데이터 특징 추출 성능 평가 시작")
    
    # 출력 디렉토리 생성
    output_dir = Path("data/results/real_xtf_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # XTF 파일들 찾기
    xtf_files = list(Path(".").glob("**/*.xtf"))
    logger.info(f"총 {len(xtf_files)}개 XTF 파일 발견")
    
    if not xtf_files:
        logger.error("XTF 파일을 찾을 수 없습니다")
        return
    
    # 종합 분석 실행
    results = analyze_multiple_xtf_files(xtf_files, max_files=3)
    
    # 결과 저장
    with open(output_dir / 'real_xtf_evaluation_results.json', 'w', encoding='utf-8') as f:
        # JSON 안전 변환
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
        
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)
    
    # 요약 리포트 출력
    logger.info("\n" + "="*60)
    logger.info("실제 XTF 데이터 특징 추출 평가 결과 요약")
    logger.info("="*60)
    
    logger.info(f"분석된 파일 수: {results['files_analyzed']}")
    logger.info(f"총 처리된 패치 수: {results.get('aggregate_analysis', {}).get('total_patches', 0)}")
    
    # 추출기별 성능 요약
    extractor_perf = results.get('aggregate_analysis', {}).get('extractor_performance', {})
    for name, perf in extractor_perf.items():
        logger.info(f"\n{name}:")
        logger.info(f"  평균 성공률: {perf['avg_success_rate']:.1%}")
        logger.info(f"  평균 특징수: {perf['avg_feature_count']:.0f}개")
        logger.info(f"  평균 추출시간: {perf['avg_extraction_time']:.1f}ms")
        logger.info(f"  성공한 파일수: {perf['files_successful']}/{results['files_analyzed']}")
    
    # 모의데이터 비교 결과
    comparison = results.get('synthetic_comparison', {}).get('comparison_analysis', {})
    if comparison:
        logger.info(f"\n실제 데이터 vs 모의데이터 비교:")
        for name, comp in comparison.items():
            logger.info(f"  {name}:")
            logger.info(f"    성공률 차이: {comp['success_rate_diff']:.3f}")
            logger.info(f"    특징수 비율: {comp['feature_count_ratio']:.2f}")
            logger.info(f"    분포 유사도: {comp['feature_distribution_similarity']:.3f}")
    
    logger.info(f"\n상세 결과가 {output_dir}에 저장되었습니다.")
    
    return results


if __name__ == "__main__":
    main()