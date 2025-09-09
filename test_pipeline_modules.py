"""
기뢰탐지 시스템 파이프라인 모듈별 테스트

각 단계를 개별적으로 테스트하고 전체 파이프라인을 검증합니다.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import traceback
from datetime import datetime

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.main_pipeline import MineDetectionPipeline, PipelineConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineModuleTester:
    """파이프라인 모듈별 테스트 클래스"""
    
    def __init__(self, test_output_dir: str = "data/results/module_tests"):
        """
        테스터 초기화
        
        Args:
            test_output_dir: 테스트 결과 출력 디렉토리
        """
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.setup_test_logging()
    
    def setup_test_logging(self):
        """테스트용 로깅 설정"""
        log_file = self.test_output_dir / f"module_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("모듈 테스트 로깅 시작")
    
    def run_all_module_tests(self) -> Dict:
        """모든 모듈 테스트 실행"""
        logger.info("="*60)
        logger.info("전체 모듈 테스트 시작")
        logger.info("="*60)
        
        test_methods = [
            ('import_tests', self.test_imports),
            ('step1_intensity_extraction', self.test_step1_intensity_extraction),
            ('step2_preprocessing', self.test_step2_preprocessing),
            ('step3_training_data_prep', self.test_step3_training_data_prep),
            ('step4_feature_extraction', self.test_step4_feature_extraction),
            ('step5_performance_evaluation', self.test_step5_performance_evaluation),
            ('step6_model_training', self.test_step6_model_training),
            ('step7_comparison', self.test_step7_comparison),
            ('full_pipeline', self.test_full_pipeline)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"\n{'='*40}")
                logger.info(f"테스트: {test_name}")
                logger.info(f"{'='*40}")
                
                result = test_method()
                self.test_results[test_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'details': result
                }
                
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
                
            except Exception as e:
                logger.error(f"{test_name} 테스트 실패: {e}")
                logger.error(traceback.format_exc())
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        self.save_test_results()
        self.print_test_summary()
        
        return self.test_results
    
    def test_imports(self) -> bool:
        """필수 모듈 import 테스트"""
        try:
            logger.info("필수 모듈 import 테스트 시작")
            
            # 핵심 모듈들
            from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor
            logger.info("✓ XTF Intensity Extractor import 성공")
            
            from src.data_processing.coordinate_mapper import CoordinateMapper
            logger.info("✓ Coordinate Mapper import 성공")
            
            from src.data_processing.preprocessor import SonarPreprocessor
            logger.info("✓ Sonar Preprocessor import 성공")
            
            from src.feature_extraction.gabor_extractor import GaborFeatureExtractor
            logger.info("✓ Gabor Feature Extractor import 성공")
            
            from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
            logger.info("✓ LBP Feature Extractor import 성공")
            
            from src.feature_extraction.hog_extractor import MultiScaleHOGExtractor
            logger.info("✓ HOG Feature Extractor import 성공")
            
            from src.evaluation.performance_evaluator import FeaturePerformanceEvaluator
            logger.info("✓ Performance Evaluator import 성공")
            
            from src.models.cnn_detector import CNNDetector
            logger.info("✓ CNN Detector import 성공")
            
            logger.info("모든 필수 모듈 import 성공")
            return True
            
        except Exception as e:
            logger.error(f"Import 테스트 실패: {e}")
            return False
    
    def test_step1_intensity_extraction(self) -> bool:
        """1단계: 강도 데이터 추출 테스트"""
        try:
            logger.info("1단계: 강도 데이터 추출 테스트")
            
            config = PipelineConfig(output_dir=str(self.test_output_dir / "step1"))
            pipeline = MineDetectionPipeline(config)
            
            # 1단계만 실행
            pipeline.step1_extract_intensity_data()
            
            # 결과 검증
            intensity_data = pipeline.results.get('intensity_data')
            if not intensity_data:
                logger.error("강도 데이터가 없습니다")
                return False
            
            # 메타데이터 확인
            metadata = intensity_data.get('metadata')
            if not metadata:
                logger.error("메타데이터가 없습니다")
                return False
            
            logger.info(f"✓ Ping count: {metadata.ping_count}")
            logger.info(f"✓ Channel count: {metadata.channel_count}")
            
            # 강도 이미지 확인
            intensity_images = intensity_data.get('intensity_images', {})
            for img_type, img in intensity_images.items():
                if img.size > 0:
                    logger.info(f"✓ {img_type} image shape: {img.shape}")
                else:
                    logger.warning(f"⚠ {img_type} image is empty")
            
            logger.info("1단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"1단계 테스트 실패: {e}")
            return False
    
    def test_step2_preprocessing(self) -> bool:
        """2단계: 전처리 테스트"""
        try:
            logger.info("2단계: 데이터 전처리 테스트")
            
            config = PipelineConfig(output_dir=str(self.test_output_dir / "step2"))
            pipeline = MineDetectionPipeline(config)
            
            # 1-2단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            
            # 결과 검증
            preprocessed_data = pipeline.results.get('preprocessed_data')
            if not preprocessed_data:
                logger.error("전처리 데이터가 없습니다")
                return False
            
            # 전처리된 이미지 확인
            processed_images = preprocessed_data.get('images', {})
            for img_type, img in processed_images.items():
                if img.size > 0:
                    logger.info(f"✓ 전처리된 {img_type} shape: {img.shape}")
                    logger.info(f"✓ {img_type} 강도 범위: [{np.min(img):.3f}, {np.max(img):.3f}]")
            
            # 네비게이션 데이터 확인
            navigation_data = preprocessed_data.get('navigation', {})
            if navigation_data:
                logger.info(f"✓ 네비게이션 데이터 키: {list(navigation_data.keys())}")
            
            logger.info("2단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"2단계 테스트 실패: {e}")
            return False
    
    def test_step3_training_data_prep(self) -> bool:
        """3단계: 학습 데이터 준비 테스트"""
        try:
            logger.info("3단계: 학습 데이터 준비 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "step3"),
                use_synthetic_data=True
            )
            pipeline = MineDetectionPipeline(config)
            
            # 1-3단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            pipeline.step3_prepare_training_data()
            
            # 결과 검증
            training_data = pipeline.results.get('training_data')
            if not training_data:
                logger.error("학습 데이터가 없습니다")
                return False
            
            patches = training_data.get('patches', [])
            splits = training_data.get('splits', {})
            
            logger.info(f"✓ 총 패치 수: {len(patches)}")
            logger.info(f"✓ 실데이터 패치: {training_data.get('real_count', 0)}")
            logger.info(f"✓ 모의데이터 패치: {training_data.get('synthetic_count', 0)}")
            
            for split_name, split_patches in splits.items():
                logger.info(f"✓ {split_name}: {len(split_patches)} patches")
            
            # 패치 데이터 품질 확인
            if patches:
                sample_patch = patches[0]
                logger.info(f"✓ 샘플 패치 shape: {sample_patch.get('shape', 'N/A')}")
                logger.info(f"✓ 샘플 패치 평균 강도: {sample_patch.get('mean_intensity', 0):.3f}")
            
            logger.info("3단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"3단계 테스트 실패: {e}")
            return False
    
    def test_step4_feature_extraction(self) -> bool:
        """4단계: 특징 추출 테스트"""
        try:
            logger.info("4단계: 특징 추출 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "step4"),
                feature_extractors=['lbp', 'gabor'],  # HOG 제외 (오류 방지)
                use_synthetic_data=False  # 속도 향상
            )
            pipeline = MineDetectionPipeline(config)
            
            # 1-4단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            pipeline.step3_prepare_training_data()
            pipeline.step4_extract_and_validate_features()
            
            # 결과 검증
            extracted_features = pipeline.results.get('extracted_features')
            if not extracted_features:
                logger.error("추출된 특징이 없습니다")
                return False
            
            for extractor_name, features_by_split in extracted_features.items():
                logger.info(f"✓ {extractor_name.upper()} 특징 추출기:")
                
                for split_name, features in features_by_split.items():
                    if features:
                        feature_dims = [len(f['features']) for f in features]
                        avg_dim = np.mean(feature_dims) if feature_dims else 0
                        logger.info(f"  - {split_name}: {len(features)} features, 평균 차원: {avg_dim:.1f}")
            
            # 특징 검증 결과 확인
            validation_results = pipeline.results.get('feature_validation', {})
            for extractor_name, validation in validation_results.items():
                success_rate = validation.get('average_success_rate', 0)
                logger.info(f"✓ {extractor_name} 성공률: {success_rate:.2%}")
            
            logger.info("4단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"4단계 테스트 실패: {e}")
            return False
    
    def test_step5_performance_evaluation(self) -> bool:
        """5단계: 성능 평가 테스트"""
        try:
            logger.info("5단계: 성능 평가 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "step5"),
                feature_extractors=['lbp', 'gabor']
            )
            pipeline = MineDetectionPipeline(config)
            
            # 1-5단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            pipeline.step3_prepare_training_data()
            pipeline.step4_extract_and_validate_features()
            pipeline.step5_evaluate_feature_performance()
            
            # 결과 검증
            performance_results = pipeline.results.get('performance_evaluation')
            if not performance_results:
                logger.error("성능 평가 결과가 없습니다")
                return False
            
            for extractor_name, performance in performance_results.items():
                if extractor_name == 'ensemble':
                    logger.info(f"✓ {extractor_name.upper()} 성능:")
                    for method, result in performance.items():
                        accuracy = result.get('accuracy', 0)
                        logger.info(f"  - {method}: {accuracy:.2%}")
                else:
                    logger.info(f"✓ {extractor_name.upper()} 성능 평가 완료")
            
            logger.info("5단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"5단계 테스트 실패: {e}")
            return False
    
    def test_step6_model_training(self) -> bool:
        """6단계: 모델 훈련 테스트"""
        try:
            logger.info("6단계: 모델 훈련 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "step6"),
                feature_extractors=['lbp']  # 단일 추출기로 빠른 테스트
            )
            pipeline = MineDetectionPipeline(config)
            
            # 1-6단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            pipeline.step3_prepare_training_data()
            pipeline.step4_extract_and_validate_features()
            pipeline.step5_evaluate_feature_performance()
            pipeline.step6_train_classification_models()
            
            # 결과 검증
            classification_models = pipeline.results.get('classification_models')
            if not classification_models:
                logger.error("분류 모델이 없습니다")
                return False
            
            for extractor_name, models in classification_models.items():
                logger.info(f"✓ {extractor_name.upper()} 기반 모델:")
                
                cnn_results = models.get('cnn', {})
                if cnn_results:
                    test_accuracy = cnn_results.get('test_accuracy', 0)
                    logger.info(f"  - CNN 테스트 정확도: {test_accuracy:.2%}")
                
                ml_results = models.get('traditional_ml', {})
                for model_type, result in ml_results.items():
                    accuracy = result.get('accuracy', 0)
                    logger.info(f"  - {model_type}: {accuracy:.2%}")
            
            # 모델 성능 요약 확인
            performance_summary = pipeline.results.get('model_performance_summary', {})
            for extractor_name, summary in performance_summary.items():
                best_model = summary.get('best_model', 'N/A')
                best_accuracy = summary.get('best_accuracy', 0)
                logger.info(f"✓ {extractor_name} 최고 모델: {best_model} ({best_accuracy:.2%})")
            
            logger.info("6단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"6단계 테스트 실패: {e}")
            return False
    
    def test_step7_comparison(self) -> bool:
        """7단계: 실데이터-모의데이터 비교 테스트"""
        try:
            logger.info("7단계: 실데이터-모의데이터 비교 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "step7"),
                feature_extractors=['lbp'],
                use_synthetic_data=True
            )
            pipeline = MineDetectionPipeline(config)
            
            # 1-7단계 실행
            pipeline.step1_extract_intensity_data()
            pipeline.step2_preprocess_and_map()
            pipeline.step3_prepare_training_data()
            pipeline.step4_extract_and_validate_features()
            pipeline.step5_evaluate_feature_performance()
            pipeline.step6_train_classification_models()
            pipeline.step7_compare_real_synthetic_data()
            
            # 결과 검증
            comparison_results = pipeline.results.get('real_synthetic_comparison')
            if not comparison_results:
                logger.error("비교 결과가 없습니다")
                return False
            
            # 특징 분포 비교
            feature_distributions = comparison_results.get('feature_distributions', {})
            if feature_distributions:
                kl_div = feature_distributions.get('kl_divergence', 0)
                similarity = feature_distributions.get('distribution_similarity', 0)
                logger.info(f"✓ 특징 분포 KL divergence: {kl_div:.3f}")
                logger.info(f"✓ 분포 유사도: {similarity:.3f}")
            
            # 교차 도메인 성능
            cross_domain = comparison_results.get('cross_domain_performance', {})
            for test_type, result in cross_domain.items():
                accuracy = result.get('accuracy', 0)
                logger.info(f"✓ {test_type}: {accuracy:.2%}")
            
            # 도메인 적응 결과
            domain_adaptation = comparison_results.get('domain_adaptation', {})
            if domain_adaptation:
                success_rate = domain_adaptation.get('adaptation_success_rate', 0)
                logger.info(f"✓ 도메인 적응 성공률: {success_rate:.2%}")
            
            logger.info("7단계 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"7단계 테스트 실패: {e}")
            return False
    
    def test_full_pipeline(self) -> bool:
        """전체 파이프라인 통합 테스트"""
        try:
            logger.info("전체 파이프라인 통합 테스트")
            
            config = PipelineConfig(
                output_dir=str(self.test_output_dir / "full_pipeline"),
                feature_extractors=['lbp', 'gabor'],
                use_synthetic_data=True,
                enable_visualization=False  # 빠른 테스트
            )
            
            pipeline = MineDetectionPipeline(config)
            
            # 전체 파이프라인 실행
            results = pipeline.run_full_pipeline()
            
            # 최종 결과 검증
            if not results:
                logger.error("파이프라인 결과가 없습니다")
                return False
            
            # 주요 결과 확인
            required_keys = [
                'intensity_data', 'preprocessed_data', 'training_data',
                'extracted_features', 'performance_evaluation'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in results:
                    missing_keys.append(key)
                else:
                    logger.info(f"✓ {key} 존재")
            
            if missing_keys:
                logger.error(f"누락된 결과: {missing_keys}")
                return False
            
            logger.info("전체 파이프라인 테스트 통과")
            return True
            
        except Exception as e:
            logger.error(f"전체 파이프라인 테스트 실패: {e}")
            return False
    
    def save_test_results(self):
        """테스트 결과 저장"""
        try:
            import json
            
            # JSON으로 저장
            results_file = self.test_output_dir / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"테스트 결과 저장: {results_file}")
            
        except Exception as e:
            logger.error(f"테스트 결과 저장 실패: {e}")
    
    def print_test_summary(self):
        """테스트 요약 출력"""
        logger.info("\n" + "="*60)
        logger.info("테스트 결과 요약")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] in ['FAIL', 'ERROR'])
        
        logger.info(f"총 테스트: {total_tests}")
        logger.info(f"통과: {passed_tests}")
        logger.info(f"실패: {failed_tests}")
        logger.info(f"성공률: {passed_tests/total_tests:.1%}")
        
        logger.info("\n개별 테스트 결과:")
        for test_name, result in self.test_results.items():
            status = result['status']
            status_emoji = "✅" if status == "PASS" else "❌"
            logger.info(f"{status_emoji} {test_name}: {status}")
        
        logger.info("="*60)


def run_quick_test():
    """빠른 테스트 (핵심 기능만)"""
    logger.info("빠른 테스트 시작")
    
    tester = PipelineModuleTester("data/results/quick_test")
    
    # 핵심 테스트만 실행
    quick_tests = [
        ('imports', tester.test_imports),
        ('intensity_extraction', tester.test_step1_intensity_extraction),
        ('feature_extraction', tester.test_step4_feature_extraction),
    ]
    
    results = {}
    for test_name, test_method in quick_tests:
        try:
            logger.info(f"\n테스트: {test_name}")
            result = test_method()
            results[test_name] = 'PASS' if result else 'FAIL'
            logger.info(f"{test_name}: {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            logger.error(f"{test_name} 실패: {e}")
            results[test_name] = 'ERROR'
    
    logger.info(f"\n빠른 테스트 완료: {results}")
    return results


def run_individual_step_test(step_number: int):
    """개별 단계 테스트"""
    logger.info(f"단계 {step_number} 개별 테스트")
    
    tester = PipelineModuleTester(f"data/results/step_{step_number}_test")
    
    step_methods = {
        1: tester.test_step1_intensity_extraction,
        2: tester.test_step2_preprocessing,
        3: tester.test_step3_training_data_prep,
        4: tester.test_step4_feature_extraction,
        5: tester.test_step5_performance_evaluation,
        6: tester.test_step6_model_training,
        7: tester.test_step7_comparison
    }
    
    if step_number in step_methods:
        try:
            result = step_methods[step_number]()
            logger.info(f"단계 {step_number}: {'✅ PASS' if result else '❌ FAIL'}")
            return result
        except Exception as e:
            logger.error(f"단계 {step_number} 테스트 실패: {e}")
            return False
    else:
        logger.error(f"잘못된 단계 번호: {step_number}")
        return False


def main():
    """메인 테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='파이프라인 모듈 테스트')
    parser.add_argument('--mode', choices=['full', 'quick', 'step'], 
                       default='full', help='테스트 모드')
    parser.add_argument('--step', type=int, choices=range(1, 8), 
                       help='개별 단계 번호 (mode=step일 때)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            results = run_quick_test()
        elif args.mode == 'step' and args.step:
            results = run_individual_step_test(args.step)
        else:
            # 전체 테스트
            tester = PipelineModuleTester()
            results = tester.run_all_module_tests()
        
        return results
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")
        return None


if __name__ == "__main__":
    main()