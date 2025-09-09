#!/usr/bin/env python3
"""
샘플 데이터 탐색적 분석 및 전처리 성능 검증

이 스크립트는 샘플 데이터를 이용해 다음을 수행합니다:
1. XTF 데이터 로드 및 기본 분석
2. 전처리 파이프라인 성능 검증
3. 결과 시각화 및 품질 평가
4. 2단계 진행을 위한 권장사항 도출
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent))

from src.data_processing.xtf_reader import XTFReader
from src.data_processing.coordinate_mapper import CoordinateTransformer, TargetLocationLoader, CoordinateMapper
from src.data_processing.preprocessor import Preprocessor, PreprocessingConfig
from config.settings import *
from config.paths import path_manager

# 로깅 설정
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleDataAnalyzer:
    """샘플 데이터 탐색적 분석 클래스"""
    
    def __init__(self):
        self.results = {}
        self.figures_dir = path_manager.figures
        self.processed_dir = path_manager.processed_data
        
        logger.info("샘플 데이터 분석기 초기화 완료")
    
    def run_complete_analysis(self):
        """전체 분석 파이프라인 실행"""
        print("="*60)
        print("사이드스캔 소나 샘플 데이터 탐색적 분석")
        print("="*60)
        
        try:
            # Phase 1: 데이터 로드 및 기본 분석
            self.phase1_data_loading()
            
            # Phase 2: 전처리 성능 분석
            self.phase2_preprocessing_analysis()
            
            # Phase 3: 좌표 매핑 검증
            self.phase3_coordinate_mapping()
            
            # Phase 4: 결과 종합 및 평가
            self.phase4_comprehensive_evaluation()
            
            # Phase 5: 2단계 진행 계획
            self.phase5_next_phase_planning()
            
            print("\\n" + "="*60)
            print("분석 완료! 결과는 다음 위치에 저장되었습니다:")
            print(f"- 그림: {self.figures_dir}")
            print(f"- 처리된 데이터: {self.processed_dir}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"분석 실행 중 오류: {e}")
            raise
    
    def phase1_data_loading(self):
        """Phase 1: 데이터 로드 및 기본 분석"""
        print("\\n🔍 Phase 1: 데이터 로드 및 기본 분석")
        print("-" * 40)
        
        # XTF 파일 로드
        xtf_filename = XTF_CONFIG['sample_file']
        xtf_filepath = path_manager.get_sample_file(xtf_filename)
        
        if not xtf_filepath.exists():
            logger.error(f"XTF 파일을 찾을 수 없습니다: {xtf_filepath}")
            print(f"❌ XTF 파일 없음: {xtf_filename}")
            return False
        
        print(f"📄 XTF 파일 로드: {xtf_filename} ({xtf_filepath.stat().st_size / (1024*1024):.1f} MB)")
        
        # XTF Reader 초기화
        self.xtf_reader = XTFReader(xtf_filepath, max_pings=XTF_CONFIG['max_pings_per_load'])
        
        if not self.xtf_reader.load_file():
            logger.error("XTF 파일 로드 실패")
            print("❌ XTF 파일 로드 실패")
            return False
        
        # Ping 데이터 파싱
        self.ping_data = self.xtf_reader.parse_pings()
        print(f"✅ {len(self.ping_data)} pings 로드 완료")
        
        # 기본 정보 출력
        summary = self.xtf_reader.get_summary()
        self.results['basic_info'] = summary
        
        print(f"\\n📊 기본 정보:")
        print(f"   - 총 ping 수: {summary['total_pings']:,}")
        print(f"   - 소나 채널 수: {summary['num_sonar_channels']}")
        print(f"   - 주파수 정보: {summary['frequency_info']}")
        
        if summary['coordinate_bounds']['lat'][0]:
            print(f"   - 위도 범위: {summary['coordinate_bounds']['lat'][0]:.6f} ~ {summary['coordinate_bounds']['lat'][1]:.6f}")
            print(f"   - 경도 범위: {summary['coordinate_bounds']['lon'][0]:.6f} ~ {summary['coordinate_bounds']['lon'][1]:.6f}")
        else:
            print("   - ⚠️  위치 정보 없음")
        
        # Intensity 데이터 추출 및 분석
        self.port_intensity, self.port_geo = self.xtf_reader.get_channel_data(0)
        self.starboard_intensity, self.starboard_geo = self.xtf_reader.get_channel_data(1)
        
        print(f"\\n📈 Intensity 데이터:")
        print(f"   - Port 채널: {self.port_intensity.shape} ({self.port_intensity.nbytes / (1024*1024):.1f} MB)")
        print(f"   - Starboard 채널: {self.starboard_intensity.shape} ({self.starboard_intensity.nbytes / (1024*1024):.1f} MB)")
        
        # 데이터 품질 분석
        quality_metrics = self._analyze_data_quality(self.port_intensity)
        self.results['data_quality'] = quality_metrics
        
        print(f"\\n⚡ 데이터 품질:")
        print(f"   - 동적 범위: {quality_metrics['dynamic_range']:.2f}")
        print(f"   - 평균 강도: {quality_metrics['mean_intensity']:.2f}")
        print(f"   - 표준편차: {quality_metrics['std_intensity']:.2f}")
        print(f"   - 결측치 비율: {quality_metrics['missing_ratio']:.1%}")
        print(f"   - SNR 추정: {quality_metrics['estimated_snr']:.1f} dB")
        
        # 초기 시각화
        self._visualize_raw_data()
        
        return True
    
    def phase2_preprocessing_analysis(self):
        """Phase 2: 전처리 성능 분석"""
        print("\\n🔧 Phase 2: 전처리 성능 분석")
        print("-" * 40)
        
        # 다양한 전처리 설정으로 성능 비교
        preprocessing_configs = {
            'basic': PreprocessingConfig(
                remove_water_column=True,
                normalize_intensity=True,
                apply_denoising=False,
                enhance_contrast=False,
                terrain_adaptive=False
            ),
            'standard': PreprocessingConfig(
                remove_water_column=True,
                normalize_intensity=True,
                normalization_method='minmax',
                apply_denoising=True,
                denoising_method='gaussian',
                enhance_contrast=True,
                contrast_method='clahe',
                terrain_adaptive=False
            ),
            'advanced': PreprocessingConfig(
                remove_water_column=True,
                normalize_intensity=True,
                normalization_method='minmax',
                apply_denoising=True,
                denoising_method='bilateral',
                enhance_contrast=True,
                contrast_method='clahe',
                terrain_adaptive=True
            )
        }
        
        preprocessing_results = {}
        
        for config_name, config in preprocessing_configs.items():
            print(f"\\n🔄 {config_name.upper()} 전처리 실행...")
            
            preprocessor = Preprocessor(config)
            result = preprocessor.process(self.port_intensity)
            
            preprocessing_results[config_name] = {
                'result': result,
                'config': config,
                'processing_time': getattr(result, 'processing_time', 0)
            }
            
            print(f"   ✅ 처리 단계: {len(result.processing_steps)}")
            print(f"   📊 SNR: {result.quality_metrics['snr']:.1f} dB")
            print(f"   🎯 대비 개선: {result.quality_metrics['contrast_improvement']:.2f}x")
            print(f"   🔀 엣지 보존: {result.quality_metrics['edge_preservation']:.3f}")
        
        self.results['preprocessing'] = preprocessing_results
        
        # 전처리 비교 시각화
        self._visualize_preprocessing_comparison(preprocessing_results)
        
        # 최적 전처리 방법 선정
        best_config = self._select_best_preprocessing(preprocessing_results)
        self.results['best_preprocessing'] = best_config
        
        print(f"\\n🏆 최적 전처리 방법: {best_config}")
        
        return preprocessing_results
    
    def phase3_coordinate_mapping(self):
        """Phase 3: 좌표 매핑 검증"""
        print("\\n🗺️  Phase 3: 좌표 매핑 검증")
        print("-" * 40)
        
        # 기물 위치 정보 로드 시도
        location_file = path_manager.get_sample_file(COORDINATE_CONFIG['location_file'])
        
        if not location_file.exists():
            print(f"⚠️  위치 파일 없음: {location_file.name}")
            self.results['coordinate_mapping'] = {'status': 'no_location_file'}
            return False
        
        # 좌표 변환기 초기화
        coord_transformer = CoordinateTransformer(utm_zone=COORDINATE_CONFIG['utm_zone'])
        target_loader = TargetLocationLoader(coord_transformer)
        
        try:
            # 엑셀 파일 구조 먼저 확인
            df_preview = pd.read_excel(location_file)
            print(f"📋 위치 파일 구조 확인:")
            print(f"   - 행 수: {len(df_preview)}")
            print(f"   - 컬럼: {list(df_preview.columns)}")
            print(f"   - 첫 5행:")
            print(df_preview.head().to_string(index=False))
            
            # 적절한 컬럼명 찾기
            lat_col, lon_col = self._find_coordinate_columns(df_preview.columns)
            
            if lat_col and lon_col:
                success = target_loader.load_from_excel(
                    location_file,
                    lat_col=lat_col,
                    lon_col=lon_col
                )
                
                if success:
                    targets_df = target_loader.get_targets_dataframe()
                    print(f"\\n✅ 기물 위치 로드 성공: {len(targets_df)} 개")
                    
                    # 좌표 매핑기 설정
                    coord_mapper = CoordinateMapper(coord_transformer)
                    coord_mapper.set_sonar_data(
                        self.port_geo[['latitude', 'longitude', 'ping_number']],
                        self.port_intensity.shape
                    )
                    
                    # 좌표 매핑 검증
                    mapping_validation = self._validate_coordinate_mapping(coord_mapper, target_loader)
                    self.results['coordinate_mapping'] = mapping_validation
                    
                    print(f"\\n📍 좌표 매핑 검증:")
                    print(f"   - 데이터 영역 내 기물: {mapping_validation['targets_in_area']}")
                    print(f"   - 평균 매핑 오차: {mapping_validation['avg_mapping_error']:.1f}m")
                    print(f"   - 매핑 성공률: {mapping_validation['mapping_success_rate']:.1%}")
                    
                else:
                    print("❌ 기물 위치 로드 실패")
                    self.results['coordinate_mapping'] = {'status': 'load_failed'}
                    
            else:
                print("❌ 적절한 좌표 컬럼을 찾을 수 없음")
                self.results['coordinate_mapping'] = {'status': 'no_coord_columns'}
                
        except Exception as e:
            logger.error(f"좌표 매핑 실패: {e}")
            print(f"❌ 좌표 매핑 오류: {e}")
            self.results['coordinate_mapping'] = {'status': 'error', 'error': str(e)}
            
        return True
    
    def phase4_comprehensive_evaluation(self):
        """Phase 4: 결과 종합 및 평가"""
        print("\\n📊 Phase 4: 결과 종합 및 평가")
        print("-" * 40)
        
        evaluation = {}
        
        # 데이터 품질 평가
        data_quality = self.results['data_quality']
        if data_quality['estimated_snr'] > 15:
            quality_grade = 'A (우수)'
        elif data_quality['estimated_snr'] > 10:
            quality_grade = 'B (양호)'
        elif data_quality['estimated_snr'] > 5:
            quality_grade = 'C (보통)'
        else:
            quality_grade = 'D (개선 필요)'
        
        evaluation['data_quality_grade'] = quality_grade
        
        # 전처리 효과 평가
        if 'preprocessing' in self.results:
            best_preprocessing = self.results['preprocessing'][self.results['best_preprocessing']]
            preprocessing_effectiveness = best_preprocessing['result'].quality_metrics
            
            if preprocessing_effectiveness['snr'] > 20:
                preprocessing_grade = 'A (매우 효과적)'
            elif preprocessing_effectiveness['snr'] > 15:
                preprocessing_grade = 'B (효과적)'
            elif preprocessing_effectiveness['snr'] > 10:
                preprocessing_grade = 'C (보통)'
            else:
                preprocessing_grade = 'D (개선 필요)'
                
            evaluation['preprocessing_grade'] = preprocessing_grade
        
        # 좌표 매핑 평가
        if 'coordinate_mapping' in self.results:
            mapping_result = self.results['coordinate_mapping']
            if mapping_result.get('status') == 'success':
                if mapping_result['mapping_success_rate'] > 0.8:
                    mapping_grade = 'A (정확)'
                elif mapping_result['mapping_success_rate'] > 0.6:
                    mapping_grade = 'B (양호)'
                else:
                    mapping_grade = 'C (개선 필요)'
            else:
                mapping_grade = 'F (실패)'
            evaluation['mapping_grade'] = mapping_grade
        
        # 전체 시스템 준비도 평가
        grades = [evaluation.get('data_quality_grade', 'D'),
                 evaluation.get('preprocessing_grade', 'D'),
                 evaluation.get('mapping_grade', 'F')]
        
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        avg_score = np.mean([grade_scores[g.split()[0]] for g in grades])
        
        if avg_score >= 3.5:
            overall_readiness = 'Phase 2 진행 준비 완료'
        elif avg_score >= 2.5:
            overall_readiness = 'Phase 2 진행 가능 (일부 개선 권장)'
        elif avg_score >= 1.5:
            overall_readiness = 'Phase 2 진행 전 개선 필요'
        else:
            overall_readiness = 'Phase 1 재점검 필요'
        
        evaluation['overall_readiness'] = overall_readiness
        evaluation['readiness_score'] = avg_score
        
        self.results['comprehensive_evaluation'] = evaluation
        
        print(f"\\n🎯 종합 평가 결과:")
        print(f"   - 데이터 품질: {quality_grade}")
        if 'preprocessing_grade' in evaluation:
            print(f"   - 전처리 효과: {preprocessing_grade}")
        if 'mapping_grade' in evaluation:
            print(f"   - 좌표 매핑: {mapping_grade}")
        print(f"   - 전체 준비도: {overall_readiness}")
        print(f"   - 준비도 점수: {avg_score:.1f}/4.0")
        
        return evaluation
    
    def phase5_next_phase_planning(self):
        """Phase 5: 2단계 진행 계획"""
        print("\\n📋 Phase 5: 2단계 진행 계획")
        print("-" * 40)
        
        evaluation = self.results['comprehensive_evaluation']
        readiness_score = evaluation['readiness_score']
        
        # 권장사항 생성
        recommendations = []
        immediate_tasks = []
        medium_term_tasks = []
        
        # 데이터 품질 기반 권장사항
        data_quality = self.results['data_quality']
        if data_quality['estimated_snr'] < 10:
            recommendations.append("데이터 품질 개선을 위한 추가 전처리 기법 도입")
            immediate_tasks.append("노이즈 제거 알고리즘 최적화")
        
        if data_quality['missing_ratio'] > 0.1:
            recommendations.append("결측치 처리 방법 개선")
            immediate_tasks.append("워터컬럼 처리 알고리즘 튜닝")
        
        # 좌표 매핑 기반 권장사항
        if 'coordinate_mapping' in self.results:
            mapping_result = self.results['coordinate_mapping']
            if mapping_result.get('status') != 'success':
                recommendations.append("좌표 매핑 시스템 재구성 필요")
                immediate_tasks.append("위치 데이터 형식 및 좌표계 검토")
            elif mapping_result.get('mapping_success_rate', 0) < 0.7:
                recommendations.append("좌표 매핑 정확도 개선")
                immediate_tasks.append("UTM 존 설정 및 변환 알고리즘 최적화")
        
        # 2단계 진행 계획 수립
        if readiness_score >= 3.0:
            phase2_plan = self._create_advanced_phase2_plan()
        elif readiness_score >= 2.0:
            phase2_plan = self._create_standard_phase2_plan()
        else:
            phase2_plan = self._create_basic_phase2_plan()
        
        self.results['phase2_plan'] = phase2_plan
        self.results['recommendations'] = recommendations
        self.results['immediate_tasks'] = immediate_tasks
        self.results['medium_term_tasks'] = medium_term_tasks
        
        print(f"\\n✨ 2단계 계획: {phase2_plan['plan_type']}")
        print(f"\\n🎯 즉시 수행 과제:")
        for i, task in enumerate(immediate_tasks[:5], 1):
            print(f"   {i}. {task}")
        
        print(f"\\n📅 중기 수행 과제:")
        for i, task in enumerate(medium_term_tasks[:3], 1):
            print(f"   {i}. {task}")
        
        print(f"\\n💡 주요 권장사항:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        # 상세 계획 저장
        self._save_detailed_plan(phase2_plan, recommendations)
        
        return phase2_plan
    
    def _analyze_data_quality(self, intensity_data):
        """데이터 품질 분석"""
        metrics = {
            'dynamic_range': float(np.max(intensity_data) - np.min(intensity_data)),
            'mean_intensity': float(np.mean(intensity_data)),
            'std_intensity': float(np.std(intensity_data)),
            'missing_ratio': float(np.sum(intensity_data == 0) / intensity_data.size),
            'estimated_snr': float(self._estimate_snr(intensity_data))
        }
        return metrics
    
    def _estimate_snr(self, data):
        """SNR 추정"""
        # 신호: 상위 25% 데이터의 평균
        signal = np.mean(data[data > np.percentile(data, 75)])
        # 노이즈: 하위 25% 데이터의 표준편차
        noise = np.std(data[data < np.percentile(data, 25)])
        
        if noise > 0:
            snr_db = 20 * np.log10(signal / noise)
        else:
            snr_db = 40  # 기본값
        
        return max(0, min(40, snr_db))  # 0-40 dB 범위로 제한
    
    def _visualize_raw_data(self):
        """원본 데이터 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Port 채널 이미지
        im1 = axes[0, 0].imshow(self.port_intensity, aspect='auto', cmap='gray')
        axes[0, 0].set_title('Port 채널 (원본)', fontsize=14)
        axes[0, 0].set_xlabel('샘플 번호')
        axes[0, 0].set_ylabel('Ping 번호')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Starboard 채널 이미지
        im2 = axes[0, 1].imshow(self.starboard_intensity, aspect='auto', cmap='gray')
        axes[0, 1].set_title('Starboard 채널 (원본)', fontsize=14)
        axes[0, 1].set_xlabel('샘플 번호')
        axes[0, 1].set_ylabel('Ping 번호')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Port 채널 히스토그램
        axes[0, 2].hist(self.port_intensity.flatten(), bins=100, alpha=0.7, color='blue', density=True)
        axes[0, 2].set_title('Port 채널 분포', fontsize=14)
        axes[0, 2].set_xlabel('Intensity 값')
        axes[0, 2].set_ylabel('밀도')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Starboard 채널 히스토그램
        axes[1, 0].hist(self.starboard_intensity.flatten(), bins=100, alpha=0.7, color='red', density=True)
        axes[1, 0].set_title('Starboard 채널 분포', fontsize=14)
        axes[1, 0].set_xlabel('Intensity 값')
        axes[1, 0].set_ylabel('밀도')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 채널별 평균 프로파일
        port_mean_profile = np.mean(self.port_intensity, axis=0)
        starboard_mean_profile = np.mean(self.starboard_intensity, axis=0)
        
        axes[1, 1].plot(port_mean_profile, label='Port', alpha=0.8)
        axes[1, 1].plot(starboard_mean_profile, label='Starboard', alpha=0.8)
        axes[1, 1].set_title('채널별 평균 프로파일', fontsize=14)
        axes[1, 1].set_xlabel('샘플 번호')
        axes[1, 1].set_ylabel('평균 Intensity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 통계 요약
        stats_text = f"""데이터 통계 요약
        
Port 채널:
- 크기: {self.port_intensity.shape}
- 평균: {np.mean(self.port_intensity):.2f}
- 표준편차: {np.std(self.port_intensity):.2f}
- 최솟값: {np.min(self.port_intensity):.2f}
- 최댓값: {np.max(self.port_intensity):.2f}

Starboard 채널:
- 크기: {self.starboard_intensity.shape}
- 평균: {np.mean(self.starboard_intensity):.2f}
- 표준편차: {np.std(self.starboard_intensity):.2f}
- 최솟값: {np.min(self.starboard_intensity):.2f}
- 최댓값: {np.max(self.starboard_intensity):.2f}"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / '01_raw_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_preprocessing_comparison(self, preprocessing_results):
        """전처리 결과 비교 시각화"""
        n_configs = len(preprocessing_results)
        fig, axes = plt.subplots(2, n_configs + 1, figsize=(6 * (n_configs + 1), 12))
        
        # 원본 데이터
        im_orig = axes[0, 0].imshow(self.port_intensity, aspect='auto', cmap='gray')
        axes[0, 0].set_title('원본 데이터', fontsize=14)
        plt.colorbar(im_orig, ax=axes[0, 0])
        
        axes[1, 0].hist(self.port_intensity.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('원본 분포')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 각 전처리 결과
        for i, (config_name, result_data) in enumerate(preprocessing_results.items(), 1):
            processed_data = result_data['result'].processed_data
            
            # 이미지
            im = axes[0, i].imshow(processed_data, aspect='auto', cmap='gray')
            axes[0, i].set_title(f'{config_name.upper()}\\n전처리 결과', fontsize=14)
            plt.colorbar(im, ax=axes[0, i])
            
            # 히스토그램
            axes[1, i].hist(processed_data.flatten(), bins=50, alpha=0.7, density=True)
            axes[1, i].set_title(f'{config_name.upper()} 분포')
            axes[1, i].grid(True, alpha=0.3)
            
            # 품질 메트릭 표시
            metrics = result_data['result'].quality_metrics
            metric_text = f"SNR: {metrics['snr']:.1f}dB\\n대비: {metrics['contrast_improvement']:.2f}x\\n엣지: {metrics['edge_preservation']:.3f}"
            axes[1, i].text(0.02, 0.98, metric_text, transform=axes[1, i].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / '02_preprocessing_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _select_best_preprocessing(self, preprocessing_results):
        """최적 전처리 방법 선정"""
        scores = {}
        
        for config_name, result_data in preprocessing_results.items():
            metrics = result_data['result'].quality_metrics
            
            # 복합 점수 계산 (SNR 50%, 대비개선 30%, 엣지보존 20%)
            score = (metrics['snr'] * 0.5 + 
                    metrics['contrast_improvement'] * 10 * 0.3 +
                    metrics['edge_preservation'] * 20 * 0.2)
            
            scores[config_name] = score
        
        best_config = max(scores.keys(), key=lambda k: scores[k])
        return best_config
    
    def _find_coordinate_columns(self, columns):
        """좌표 컬럼명 자동 감지"""
        lat_keywords = ['lat', 'latitude', '위도', 'y']
        lon_keywords = ['lon', 'long', 'longitude', '경도', 'x']
        
        lat_col = None
        lon_col = None
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in lat_keywords):
                lat_col = col
            elif any(keyword in col_lower for keyword in lon_keywords):
                lon_col = col
        
        return lat_col, lon_col
    
    def _validate_coordinate_mapping(self, coord_mapper, target_loader):
        """좌표 매핑 검증"""
        validation = {}
        
        # 데이터 영역 내 기물 수 확인
        coord_bounds = self.xtf_reader.get_summary()['coordinate_bounds']
        targets_in_area = target_loader.get_targets_in_bounds(
            min_lat=coord_bounds['lat'][0],
            max_lat=coord_bounds['lat'][1],
            min_lon=coord_bounds['lon'][0],
            max_lon=coord_bounds['lon'][1]
        )
        
        validation['targets_in_area'] = len(targets_in_area)
        
        if targets_in_area:
            # 매핑 정확도 테스트
            mapping_errors = []
            successful_mappings = 0
            
            for target in targets_in_area[:min(10, len(targets_in_area))]:  # 최대 10개 테스트
                # 위경도 -> 픽셀 -> 위경도 변환 테스트
                pixel_coords = coord_mapper.geo_to_pixel(target.longitude, target.latitude)
                
                if pixel_coords[0] >= 0 and pixel_coords[1] >= 0:
                    reverse_coords = coord_mapper.pixel_to_geo(pixel_coords[0], pixel_coords[1])
                    
                    # 거리 오차 계산 (대략적)
                    lat_diff = abs(target.latitude - reverse_coords[1])
                    lon_diff = abs(target.longitude - reverse_coords[0])
                    error_meters = np.sqrt((lat_diff * 111000)**2 + (lon_diff * 111000 * np.cos(np.radians(target.latitude)))**2)
                    
                    mapping_errors.append(error_meters)
                    successful_mappings += 1
            
            if mapping_errors:
                validation['avg_mapping_error'] = np.mean(mapping_errors)
                validation['max_mapping_error'] = np.max(mapping_errors)
                validation['mapping_success_rate'] = successful_mappings / len(targets_in_area)
                validation['status'] = 'success'
            else:
                validation['status'] = 'no_valid_mappings'
        else:
            validation['status'] = 'no_targets_in_area'
        
        return validation
    
    def _create_advanced_phase2_plan(self):
        """고급 Phase 2 계획 (준비도 높음)"""
        return {
            'plan_type': '고급 특징 추출 및 딥러닝',
            'priority_tasks': [
                '다중 특징 추출 시스템 구현 (HOG, LBP, Gabor, SfS)',
                'CNN 기반 자동 특징 학습',
                '앙상블 모델 구성',
                '실시간 처리 파이프라인 최적화'
            ],
            'timeline': '4-6주',
            'expected_accuracy': '>90%',
            'next_milestones': [
                '특징 추출 모듈 완성 (2주)',
                'CNN 모델 학습 완료 (3주)', 
                '앙상블 시스템 구축 (4주)',
                '성능 최적화 완료 (6주)'
            ]
        }
    
    def _create_standard_phase2_plan(self):
        """표준 Phase 2 계획 (준비도 보통)"""
        return {
            'plan_type': '전통적 특징 추출 + 머신러닝',
            'priority_tasks': [
                '핵심 특징 추출 알고리즘 구현 (HOG, LBP)',
                '데이터 증강 시스템 구축',
                'SVM/Random Forest 분류 모델',
                '하이퍼파라미터 최적화'
            ],
            'timeline': '6-8주',
            'expected_accuracy': '75-85%',
            'next_milestones': [
                'HOG/LBP 특징 추출 (2주)',
                '데이터 증강 완료 (3주)',
                '분류 모델 학습 (5주)',
                '성능 평가 완료 (8주)'
            ]
        }
    
    def _create_basic_phase2_plan(self):
        """기본 Phase 2 계획 (준비도 낮음)"""
        return {
            'plan_type': '기초 시스템 안정화 + 단순 탐지',
            'priority_tasks': [
                '데이터 품질 개선',
                '좌표 매핑 시스템 재구축',
                '임계값 기반 단순 탐지',
                '전처리 파라미터 최적화'
            ],
            'timeline': '8-12주',
            'expected_accuracy': '60-70%',
            'next_milestones': [
                '데이터 문제 해결 (3주)',
                '좌표 시스템 개선 (5주)',
                '기본 탐지 알고리즘 (8주)',
                '시스템 안정화 (12주)'
            ]
        }
    
    def _save_detailed_plan(self, phase2_plan, recommendations):
        """상세 계획 저장"""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'sample_data_info': self.results['basic_info'],
            'data_quality_assessment': self.results['data_quality'],
            'preprocessing_evaluation': {
                'best_method': self.results['best_preprocessing'],
                'quality_improvements': self.results['preprocessing'][self.results['best_preprocessing']]['result'].quality_metrics
            },
            'coordinate_mapping_status': self.results.get('coordinate_mapping', {}),
            'overall_evaluation': self.results['comprehensive_evaluation'],
            'phase2_plan': phase2_plan,
            'recommendations': recommendations,
            'immediate_tasks': self.results['immediate_tasks'],
            'medium_term_tasks': self.results['medium_term_tasks']
        }
        
        # JSON 저장
        with open(self.processed_dir / 'sample_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 요약 리포트 저장
        self._save_summary_report(report)
    
    def _save_summary_report(self, report):
        """요약 리포트 저장"""
        with open(self.processed_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("사이드스캔 소나 샘플 데이터 분석 요약 리포트\\n")
            f.write("=" * 60 + "\\n\\n")
            
            f.write(f"분석 일시: {report['analysis_date'][:19]}\\n")
            f.write(f"데이터 파일: {report['sample_data_info']['filename']}\\n\\n")
            
            f.write("📊 데이터 품질 평가\\n")
            f.write("-" * 30 + "\\n")
            quality = report['data_quality_assessment']
            f.write(f"- 동적 범위: {quality['dynamic_range']:.2f}\\n")
            f.write(f"- 추정 SNR: {quality['estimated_snr']:.1f} dB\\n")
            f.write(f"- 결측치 비율: {quality['missing_ratio']:.1%}\\n\\n")
            
            f.write("🔧 전처리 성능\\n")
            f.write("-" * 30 + "\\n")
            preprocessing = report['preprocessing_evaluation']
            f.write(f"- 최적 방법: {preprocessing['best_method']}\\n")
            f.write(f"- SNR 개선: {preprocessing['quality_improvements']['snr']:.1f} dB\\n")
            f.write(f"- 대비 향상: {preprocessing['quality_improvements']['contrast_improvement']:.2f}x\\n\\n")
            
            f.write("🎯 종합 평가\\n")
            f.write("-" * 30 + "\\n")
            evaluation = report['overall_evaluation']
            f.write(f"- 준비도: {evaluation['overall_readiness']}\\n")
            f.write(f"- 점수: {evaluation['readiness_score']:.1f}/4.0\\n\\n")
            
            f.write("📋 2단계 계획\\n")
            f.write("-" * 30 + "\\n")
            plan = report['phase2_plan']
            f.write(f"- 계획 유형: {plan['plan_type']}\\n")
            f.write(f"- 예상 기간: {plan['timeline']}\\n")
            f.write(f"- 목표 정확도: {plan['expected_accuracy']}\\n\\n")
            
            f.write("💡 즉시 수행 과제\\n")
            f.write("-" * 30 + "\\n")
            for i, task in enumerate(report['immediate_tasks'], 1):
                f.write(f"{i}. {task}\\n")


if __name__ == "__main__":
    analyzer = SampleDataAnalyzer()
    analyzer.run_complete_analysis()