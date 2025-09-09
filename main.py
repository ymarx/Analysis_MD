#!/usr/bin/env python3
"""
사이드스캔 소나 기물 탐지 시스템 메인 실행 파일

사용법:
    python main.py --mode sample  # 샘플 데이터 분석
    python main.py --mode batch   # 배치 데이터 처리
    python main.py --mode interactive  # 인터랙티브 모드
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 프로젝트 모듈 import
from src.data_processing.xtf_reader import XTFReader, BatchXTFProcessor
from src.data_processing.coordinate_mapper import (
    CoordinateTransformer, 
    TargetLocationLoader, 
    CoordinateMapper
)
from src.data_processing.preprocessor import Preprocessor, PreprocessingConfig
from config.settings import *
from config.paths import path_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(path_manager.logs / 'main.log')
    ]
)
logger = logging.getLogger(__name__)


class SidescanAnalysisSystem:
    """
    사이드스캔 소나 기물 탐지 분석 시스템 메인 클래스
    """
    
    def __init__(self):
        """시스템 초기화"""
        self.coord_transformer = None
        self.target_loader = None
        self.preprocessor = None
        
        logger.info("사이드스캔 분석 시스템 초기화")
        self._setup_components()
    
    def _setup_components(self):
        """시스템 구성요소 초기화"""
        try:
            # 좌표 변환기
            self.coord_transformer = CoordinateTransformer(
                utm_zone=COORDINATE_CONFIG['utm_zone']
            )
            
            # 기물 위치 로더
            self.target_loader = TargetLocationLoader(self.coord_transformer)
            
            # 전처리기
            preprocess_config = PreprocessingConfig(
                remove_water_column=True,
                normalize_intensity=True,
                apply_denoising=True,
                enhance_contrast=True,
                terrain_adaptive=True
            )
            self.preprocessor = Preprocessor(preprocess_config)
            
            logger.info("시스템 구성요소 초기화 완료")
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            raise
    
    def analyze_sample_data(self) -> bool:
        """
        샘플 데이터 분석 수행
        
        Returns:
            bool: 분석 성공 여부
        """
        try:
            logger.info("=== 샘플 데이터 분석 시작 ===")
            
            # 1. XTF 파일 로드
            xtf_filename = XTF_CONFIG['sample_file']
            xtf_filepath = path_manager.get_sample_file(xtf_filename)
            
            if not xtf_filepath.exists():
                logger.error(f"XTF 파일을 찾을 수 없습니다: {xtf_filepath}")
                return False
            
            logger.info(f"XTF 파일 로드: {xtf_filename}")
            xtf_reader = XTFReader(xtf_filepath, max_pings=XTF_CONFIG['max_pings_per_load'])
            
            if not xtf_reader.load_file():
                logger.error("XTF 파일 로드 실패")
                return False
            
            ping_data = xtf_reader.parse_pings()
            logger.info(f"Ping 데이터 파싱 완료: {len(ping_data)} pings")
            
            # 2. 기물 위치 정보 로드
            location_file = path_manager.get_sample_file(COORDINATE_CONFIG['location_file'])
            if location_file.exists():
                success = self.target_loader.load_from_excel(
                    location_file,
                    lat_col='latitude',  # 실제 컬럼명으로 수정 필요
                    lon_col='longitude'  # 실제 컬럼명으로 수정 필요
                )
                if success:
                    logger.info(f"기물 위치 로드 완료: {len(self.target_loader.target_locations)} 위치")
                else:
                    logger.warning("기물 위치 로드 실패")
            
            # 3. 좌표 매핑 설정
            coord_mapper = CoordinateMapper(self.coord_transformer)
            port_intensity, port_geo = xtf_reader.get_channel_data(XTF_CONFIG['channels']['port'])
            
            if port_intensity.size > 0:
                coord_mapper.set_sonar_data(
                    ping_coordinates=port_geo[['latitude', 'longitude', 'ping_number']],
                    intensity_shape=port_intensity.shape
                )
                logger.info(f"좌표 매핑 설정 완료: {port_intensity.shape}")
            
            # 4. 전처리 수행
            logger.info("전처리 시작")
            processing_result = self.preprocessor.process(port_intensity)
            logger.info(f"전처리 완료 - 품질 메트릭: {processing_result.quality_metrics}")
            
            # 5. 기물 마스크 생성
            if self.target_loader.target_locations:
                coord_bounds = xtf_reader.get_summary()['coordinate_bounds']
                targets_in_area = self.target_loader.get_targets_in_bounds(
                    min_lat=coord_bounds['lat'][0],
                    max_lat=coord_bounds['lat'][1],
                    min_lon=coord_bounds['lon'][0],
                    max_lon=coord_bounds['lon'][1]
                )
                
                if targets_in_area:
                    target_mask = coord_mapper.create_target_mask(targets_in_area)
                    bboxes = coord_mapper.get_target_bounding_boxes(targets_in_area)
                    logger.info(f"기물 탐지 완료: {len(bboxes)} 바운딩 박스")
            
            # 6. 결과 저장
            self._save_analysis_results(processing_result, xtf_reader, coord_mapper)
            
            logger.info("=== 샘플 데이터 분석 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"샘플 데이터 분석 실패: {e}")
            return False
    
    def process_batch_data(self) -> bool:
        """
        배치 데이터 처리 수행
        
        Returns:
            bool: 처리 성공 여부
        """
        try:
            logger.info("=== 배치 데이터 처리 시작 ===")
            
            # 데이터셋 목록 확인
            datasets = path_manager.list_datasets()
            if not datasets:
                logger.warning("처리할 데이터셋이 없습니다")
                return False
            
            logger.info(f"발견된 데이터셋: {len(datasets)}개")
            
            # 각 데이터셋 처리
            for dataset_name in datasets[:3]:  # 처음 3개만 처리
                logger.info(f"데이터셋 처리 시작: {dataset_name}")
                
                # XTF 파일 찾기
                dataset_path = path_manager.get_dataset_path(dataset_name, 'original')
                xtf_files = list(dataset_path.glob('*.xtf'))
                
                if not xtf_files:
                    logger.warning(f"XTF 파일을 찾을 수 없습니다: {dataset_path}")
                    continue
                
                # 배치 프로세서로 처리
                batch_processor = BatchXTFProcessor(
                    file_paths=xtf_files,
                    max_pings_per_file=500  # 메모리 절약
                )
                
                readers = batch_processor.process_all()
                logger.info(f"처리 완료: {len(readers)}/{len(xtf_files)} 파일")
                
                # 처리 결과 저장
                for filename, reader in readers.items():
                    self._save_batch_results(reader, dataset_name, filename)
            
            logger.info("=== 배치 데이터 처리 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"배치 데이터 처리 실패: {e}")
            return False
    
    def _save_analysis_results(self, processing_result, xtf_reader, coord_mapper):
        """분석 결과 저장"""
        try:
            # 전처리된 데이터 저장
            import numpy as np
            import json
            
            np.save(
                path_manager.processed_data / 'sample_processed_intensity.npy', 
                processing_result.processed_data
            )
            
            # 메타데이터 저장
            metadata = {
                'processing_steps': processing_result.processing_steps,
                'quality_metrics': processing_result.quality_metrics,
                'data_shape': processing_result.processed_data.shape,
                'xtf_summary': xtf_reader.get_summary()
            }
            
            with open(path_manager.processed_data / 'sample_analysis_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("분석 결과 저장 완료")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    def _save_batch_results(self, reader, dataset_name, filename):
        """배치 처리 결과 저장"""
        try:
            # 결과 디렉토리 생성
            result_dir = path_manager.processed_data / dataset_name
            result_dir.mkdir(exist_ok=True)
            
            # intensity 데이터 내보내기
            output_path = result_dir / f"{Path(filename).stem}_intensity.npy"
            reader.export_intensity_data(output_path, format='npy')
            
            # 요약 정보 저장
            summary = reader.get_summary()
            import json
            with open(result_dir / f"{Path(filename).stem}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"배치 결과 저장: {dataset_name}/{filename}")
            
        except Exception as e:
            logger.error(f"배치 결과 저장 실패: {e}")
    
    def run_training_pipeline(self) -> bool:
        """
        통합 학습 파이프라인 실행
        
        Returns:
            bool: 훈련 성공 여부
        """
        try:
            logger.info("=== 통합 학습 파이프라인 실행 시작 ===")
            
            from src.training.integrated_pipeline import IntegratedPipeline, PipelineConfig
            
            # 파이프라인 설정
            config = PipelineConfig(
                use_traditional_ml=True,
                use_deep_learning=True,
                batch_size=32,
                num_epochs=50,
                learning_rate=0.001,
                augmentation_strength=0.6
            )
            
            # 출력 디렉토리 설정
            output_dir = path_manager.processed_data / 'training_results'
            
            # 파이프라인 실행
            pipeline = IntegratedPipeline(config)
            
            # 샘플 데이터로 실행 (실제 환경에서는 실제 데이터 로드)
            from src.training.integrated_pipeline import PipelineRunner
            runner = PipelineRunner()
            results = runner.run_with_sample_data(None, output_dir)
            
            # 리포트 생성
            pipeline.generate_report(output_dir)
            
            logger.info("=== 통합 학습 파이프라인 실행 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"학습 파이프라인 실행 실패: {e}")
            return False
    
    def run_evaluation(self) -> bool:
        """
        모델 성능 평가 실행
        
        Returns:
            bool: 평가 성공 여부
        """
        try:
            logger.info("=== 모델 성능 평가 시작 ===")
            
            from src.evaluation.performance_evaluator import run_comprehensive_evaluation
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            import numpy as np
            
            # 가상 테스트 데이터 생성 (실제로는 저장된 모델과 테스트 데이터 로드)
            np.random.seed(42)
            X_test = np.random.random((100, 50))
            y_test = np.random.randint(0, 2, 100)
            
            # 모델 생성 및 훈련 (간단한 예제)
            models = {
                'RandomForest': (RandomForestClassifier(n_estimators=50, random_state=42), 'sklearn'),
                'SVM': (SVC(probability=True, random_state=42), 'sklearn')
            }
            
            # 모델 훈련
            X_train = np.random.random((200, 50))
            y_train = np.random.randint(0, 2, 200)
            
            for name, (model, model_type) in models.items():
                model.fit(X_train, y_train)
            
            # 출력 디렉토리
            output_dir = path_manager.processed_data / 'evaluation_results'
            
            # 종합 평가 실행
            results = run_comprehensive_evaluation(models, X_test, y_test, output_dir)
            
            logger.info("=== 모델 성능 평가 완료 ===")
            print(f"\\n평가 결과가 {output_dir}에 저장되었습니다.")
            
            # 간단한 결과 출력
            for model_name, metrics in results.items():
                print(f"\\n{model_name}:")
                print(f"  - 정확도: {metrics.accuracy:.4f}")
                print(f"  - F1 점수: {metrics.f1_score:.4f}")
                print(f"  - AUC: {metrics.auc_score:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 성능 평가 실패: {e}")
            return False
    
    def run_interactive_mode(self):
        """인터랙티브 모드 실행"""
        logger.info("=== 인터랙티브 모드 시작 ===")
        
        print("\\n사이드스캔 소나 기물 탐지 시스템")
        print("=" * 40)
        
        while True:
            print("\\n다음 중 선택하세요:")
            print("1. 샘플 데이터 분석")
            print("2. 배치 데이터 처리") 
            print("3. 딥러닝 모델 훈련")
            print("4. 모델 성능 평가")
            print("5. 시스템 정보 확인")
            print("6. 종료")
            
            try:
                choice = input("\\n선택 (1-6): ").strip()
                
                if choice == '1':
                    print("\\n샘플 데이터 분석을 시작합니다...")
                    success = self.analyze_sample_data()
                    if success:
                        print("✓ 샘플 데이터 분석이 완료되었습니다.")
                    else:
                        print("✗ 샘플 데이터 분석이 실패했습니다.")
                
                elif choice == '2':
                    print("\\n배치 데이터 처리를 시작합니다...")
                    success = self.process_batch_data()
                    if success:
                        print("✓ 배치 데이터 처리가 완료되었습니다.")
                    else:
                        print("✗ 배치 데이터 처리가 실패했습니다.")
                
                elif choice == '3':
                    print("\\n딥러닝 모델 훈련을 시작합니다...")
                    success = self.run_training_pipeline()
                    if success:
                        print("✓ 모델 훈련이 완료되었습니다.")
                    else:
                        print("✗ 모델 훈련이 실패했습니다.")
                
                elif choice == '4':
                    print("\\n모델 성능 평가를 시작합니다...")
                    success = self.run_evaluation()
                    if success:
                        print("✓ 성능 평가가 완료되었습니다.")
                    else:
                        print("✗ 성능 평가가 실패했습니다.")
                
                elif choice == '5':
                    self._show_system_info()
                
                elif choice == '6':
                    print("\\n시스템을 종료합니다.")
                    break
                
                else:
                    print("잘못된 선택입니다. 다시 입력해주세요.")
                    
            except KeyboardInterrupt:
                print("\\n\\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                logger.error(f"인터랙티브 모드 오류: {e}")
    
    def _show_system_info(self):
        """시스템 정보 출력"""
        print("\\n=== 시스템 정보 ===")
        print(f"프로젝트 루트: {path_manager.project_root}")
        print(f"샘플 데이터: {path_manager.sample_data}")
        print(f"처리된 데이터: {path_manager.processed_data}")
        print(f"모델 저장소: {path_manager.models}")
        
        # 샘플 파일 확인
        sample_files = path_manager.list_sample_files()
        print(f"\\n샘플 파일 ({len(sample_files)}개):")
        for name, path in sample_files.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {name}: {size_mb:.1f} MB")
        
        # 데이터셋 확인
        datasets = path_manager.list_datasets()
        print(f"\\n데이터셋 ({len(datasets)}개):")
        for dataset in datasets:
            print(f"  - {dataset}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="사이드스캔 소나 기물 탐지 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['sample', 'batch', 'interactive', 'train', 'evaluate'],
        default='interactive',
        help='실행 모드 선택 (기본값: interactive)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='로그 레벨 설정 (기본값: INFO)'
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # 시스템 초기화
        system = SidescanAnalysisSystem()
        
        # 모드에 따라 실행
        if args.mode == 'sample':
            success = system.analyze_sample_data()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'batch':
            success = system.process_batch_data()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'interactive':
            system.run_interactive_mode()
            
        elif args.mode == 'train':
            success = system.run_training_pipeline()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'evaluate':
            success = system.run_evaluation()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()