#!/usr/bin/env python3
"""
샘플 데이터를 활용한 특징 추출 성능 평가

실제 사이드스캔 소나 데이터를 대신하여 시뮬레이션된 데이터로 
각 특징 추출기의 성능을 종합적으로 평가합니다.
"""

import sys
import numpy as np
# import cv2  # OpenCV는 선택사항으로 처리
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from src.feature_extraction.hog_extractor import MultiScaleHOGExtractor, AdaptiveHOGExtractor
from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from src.feature_extraction.gabor_extractor import GaborFeatureExtractor, AdaptiveGaborExtractor
from src.feature_extraction.sfs_extractor import EnhancedSfSExtractor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """특징 추출 성능 평가기"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 특징 추출기 초기화
        self.extractors = {
            'HOG_MultiScale': MultiScaleHOGExtractor(),
            'HOG_Adaptive': AdaptiveHOGExtractor(),
            'LBP_Comprehensive': ComprehensiveLBPExtractor(),
            'Gabor_Standard': GaborFeatureExtractor(n_frequencies=4, n_orientations=6),
            'Gabor_Adaptive': AdaptiveGaborExtractor(),
            'SfS_Enhanced': EnhancedSfSExtractor()
        }
        
        self.results = {}
        
        logger.info(f"특징 추출 평가기 초기화 - {len(self.extractors)}개 추출기")
    
    def generate_sample_sonar_data(self, num_samples: int = 50) -> Tuple[List[np.ndarray], List[int]]:
        """
        시뮬레이션된 사이드스캔 소나 데이터 생성
        
        Args:
            num_samples: 생성할 샘플 수
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: (이미지 리스트, 라벨 리스트)
        """
        logger.info(f"시뮬레이션 소나 데이터 생성: {num_samples}개 샘플")
        
        np.random.seed(42)  # 재현 가능한 결과를 위해
        images = []
        labels = []
        
        # 양성 샘플 생성 (기물 포함)
        for i in range(num_samples // 2):
            # 베이스 해저면 생성 (낮은 반사강도)
            base_intensity = 0.2 + 0.1 * np.random.random()
            image = np.random.normal(base_intensity, 0.05, (128, 128))
            
            # 기물 추가 (높은 반사강도의 원형/타원형 객체)
            center_x = np.random.randint(30, 98)
            center_y = np.random.randint(30, 98)
            
            # 기물 크기와 형태 랜덤화
            radius_x = np.random.randint(8, 20)
            radius_y = np.random.randint(8, 20)
            
            # 타원형 마스크 생성
            y, x = np.ogrid[:128, :128]
            mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1
            
            # 기물 반사강도 (0.6-0.9 범위)
            object_intensity = 0.6 + 0.3 * np.random.random()
            image[mask] = object_intensity + np.random.normal(0, 0.1, np.sum(mask))
            
            # 음향 그림자 추가 (기물 뒤쪽)
            shadow_length = np.random.randint(15, 35)
            shadow_start_y = center_y + radius_y + 2
            shadow_end_y = min(128, shadow_start_y + shadow_length)
            shadow_x_start = max(0, center_x - radius_x//2)
            shadow_x_end = min(128, center_x + radius_x//2)
            
            if shadow_end_y < 128:
                shadow_intensity = 0.05 + 0.05 * np.random.random()
                image[shadow_start_y:shadow_end_y, shadow_x_start:shadow_x_end] = shadow_intensity
            
            # 노이즈 추가
            noise = np.random.normal(0, 0.02, image.shape)
            image += noise
            image = np.clip(image, 0, 1)
            
            images.append(image)
            labels.append(1)  # 기물 있음
        
        # 음성 샘플 생성 (배경만)
        for i in range(num_samples - num_samples // 2):
            # 다양한 해저면 타입 시뮬레이션
            terrain_type = np.random.choice(['sand', 'mud', 'rock'])
            
            if terrain_type == 'sand':
                base_intensity = 0.4 + 0.2 * np.random.random()
                texture_noise = np.random.normal(0, 0.08, (128, 128))
            elif terrain_type == 'mud':
                base_intensity = 0.2 + 0.1 * np.random.random()
                texture_noise = np.random.normal(0, 0.03, (128, 128))
            else:  # rock
                base_intensity = 0.3 + 0.3 * np.random.random()
                texture_noise = np.random.normal(0, 0.12, (128, 128))
            
            image = np.full((128, 128), base_intensity) + texture_noise
            
            # 자연스러운 지형 변화 추가
            for _ in range(np.random.randint(2, 6)):
                blob_center_x = np.random.randint(10, 118)
                blob_center_y = np.random.randint(10, 118)
                blob_radius = np.random.randint(5, 15)
                
                y, x = np.ogrid[:128, :128]
                blob_mask = (x - blob_center_x)**2 + (y - blob_center_y)**2 <= blob_radius**2
                
                intensity_variation = np.random.uniform(-0.1, 0.1)
                image[blob_mask] += intensity_variation
            
            # 전체 노이즈
            noise = np.random.normal(0, 0.015, image.shape)
            image += noise
            image = np.clip(image, 0, 1)
            
            images.append(image)
            labels.append(0)  # 기물 없음
        
        # 데이터 섞기
        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images, labels = zip(*combined)
        
        logger.info(f"데이터 생성 완료 - 양성: {labels.count(1)}, 음성: {labels.count(0)}")
        
        return list(images), list(labels)
    
    def extract_features_with_timing(self, extractor_name: str, extractor, images: List[np.ndarray]) -> Dict:
        """
        타이밍을 포함한 특징 추출
        
        Args:
            extractor_name: 추출기 이름
            extractor: 특징 추출기 객체
            images: 입력 이미지 리스트
            
        Returns:
            Dict: 추출 결과 및 성능 지표
        """
        logger.info(f"{extractor_name} 특징 추출 시작")
        
        features_list = []
        extraction_times = []
        successful_extractions = 0
        
        start_total_time = time.time()
        
        for i, image in enumerate(images):
            try:
                start_time = time.time()
                
                # 추출기별로 다른 메서드 호출
                if 'HOG' in extractor_name:
                    if 'Adaptive' in extractor_name:
                        features = extractor.extract_adaptive_features(image)
                    else:
                        features = extractor.extract_combined_features(image)
                elif 'LBP' in extractor_name:
                    features = extractor.extract_comprehensive_features(image)
                elif 'Gabor' in extractor_name:
                    if 'Adaptive' in extractor_name:
                        features = extractor.extract_adaptive_features(image)
                    else:
                        features = extractor.extract_comprehensive_features(image)
                elif 'SfS' in extractor_name:
                    features = extractor.extract_comprehensive_sfs_features(image)
                else:
                    logger.warning(f"알 수 없는 추출기 타입: {extractor_name}")
                    continue
                
                end_time = time.time()
                extraction_time = (end_time - start_time) * 1000  # ms
                
                if len(features) > 0:
                    features_list.append(features)
                    extraction_times.append(extraction_time)
                    successful_extractions += 1
                else:
                    logger.warning(f"{extractor_name}: 빈 특징 벡터 - 이미지 {i}")
                    
            except Exception as e:
                logger.error(f"{extractor_name} 추출 실패 - 이미지 {i}: {e}")
                continue
        
        total_time = (time.time() - start_total_time) * 1000  # ms
        
        # 성능 지표 계산
        if features_list:
            feature_matrix = np.array(features_list)
            
            results = {
                'extractor_name': extractor_name,
                'successful_extractions': successful_extractions,
                'total_images': len(images),
                'success_rate': successful_extractions / len(images),
                'feature_dimensions': feature_matrix.shape[1],
                'avg_extraction_time_ms': np.mean(extraction_times),
                'std_extraction_time_ms': np.std(extraction_times),
                'total_time_ms': total_time,
                'feature_statistics': {
                    'mean': np.mean(feature_matrix, axis=0).tolist(),
                    'std': np.std(feature_matrix, axis=0).tolist(),
                    'min': np.min(feature_matrix, axis=0).tolist(),
                    'max': np.max(feature_matrix, axis=0).tolist()
                },
                'features': feature_matrix
            }
            
            logger.info(f"{extractor_name} 완료 - 성공률: {results['success_rate']:.2%}, "
                       f"특징 차원: {results['feature_dimensions']}, "
                       f"평균 시간: {results['avg_extraction_time_ms']:.2f}ms")
        else:
            results = {
                'extractor_name': extractor_name,
                'successful_extractions': 0,
                'total_images': len(images),
                'success_rate': 0.0,
                'error': 'No features extracted'
            }
            logger.error(f"{extractor_name}: 모든 특징 추출 실패")
        
        return results
    
    def evaluate_feature_quality(self, features: np.ndarray, labels: List[int]) -> Dict:
        """
        특징 품질 평가
        
        Args:
            features: 특징 행렬
            labels: 클래스 레이블
            
        Returns:
            Dict: 품질 지표
        """
        from sklearn.metrics import silhouette_score
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        quality_metrics = {}
        
        try:
            # 클러스터 분리도 (실루엣 점수)
            if len(np.unique(labels)) > 1 and len(features) > 1:
                silhouette = silhouette_score(features, labels)
                quality_metrics['silhouette_score'] = silhouette
            
            # 선형 판별 가능성
            if len(np.unique(labels)) > 1:
                lda = LinearDiscriminantAnalysis()
                try:
                    lda.fit(features, labels)
                    quality_metrics['lda_score'] = lda.score(features, labels)
                except Exception as e:
                    quality_metrics['lda_score'] = 0.0
                    logger.warning(f"LDA 평가 실패: {e}")
            
            # 랜덤 포레스트 교차 검증
            if len(features) >= 10:  # 최소 10개 샘플 필요
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                cv_scores = cross_val_score(rf, features, labels, cv=min(5, len(features)//2))
                quality_metrics['rf_cv_mean'] = np.mean(cv_scores)
                quality_metrics['rf_cv_std'] = np.std(cv_scores)
            
            # 특징 다양성 (분산 기반)
            feature_variance = np.var(features, axis=0)
            quality_metrics['feature_diversity'] = np.mean(feature_variance)
            quality_metrics['feature_stability'] = 1.0 / (1.0 + np.std(feature_variance))
            
            # 정규성 테스트 (Shapiro-Wilk 테스트의 간소화 버전)
            # 특징이 너무 치우쳐 있지 않은지 확인
            skewness_scores = []
            for i in range(min(10, features.shape[1])):  # 처음 10개 특징만 테스트
                feature_col = features[:, i]
                mean_val = np.mean(feature_col)
                std_val = np.std(feature_col)
                if std_val > 0:
                    skewness = np.mean(((feature_col - mean_val) / std_val) ** 3)
                    skewness_scores.append(abs(skewness))
            
            if skewness_scores:
                quality_metrics['avg_skewness'] = np.mean(skewness_scores)
            
        except Exception as e:
            logger.error(f"특징 품질 평가 실패: {e}")
            quality_metrics['evaluation_error'] = str(e)
        
        return quality_metrics
    
    def run_comprehensive_evaluation(self, num_samples: int = 50):
        """종합적인 특징 추출 성능 평가 실행"""
        logger.info("=== 종합적인 특징 추출 성능 평가 시작 ===")
        
        # 1. 샘플 데이터 생성
        images, labels = self.generate_sample_sonar_data(num_samples)
        
        # 2. 각 추출기별 성능 평가
        for extractor_name, extractor in self.extractors.items():
            logger.info(f"\n--- {extractor_name} 평가 시작 ---")
            
            # 특징 추출 및 타이밍
            extraction_result = self.extract_features_with_timing(extractor_name, extractor, images)
            
            # 특징 품질 평가 (성공한 경우에만)
            if extraction_result.get('features') is not None:
                features = extraction_result['features']
                # 성공한 특징에 대응하는 라벨만 사용
                successful_labels = labels[:len(features)]
                
                quality_metrics = self.evaluate_feature_quality(features, successful_labels)
                extraction_result['quality_metrics'] = quality_metrics
            
            self.results[extractor_name] = extraction_result
        
        # 3. 결과 저장 및 요약
        self.save_results()
        self.generate_comparison_report()
        self.create_visualizations()
        
        logger.info("=== 특징 추출 성능 평가 완료 ===")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        results_to_save = {}
        
        for name, result in self.results.items():
            # NumPy 배열은 제외하고 저장
            save_result = {k: v for k, v in result.items() if k != 'features'}
            
            # 통계 정보는 간소화
            if 'feature_statistics' in save_result:
                stats = save_result['feature_statistics']
                save_result['feature_statistics'] = {
                    'mean_avg': float(np.mean(stats['mean'])) if stats['mean'] else 0,
                    'std_avg': float(np.mean(stats['std'])) if stats['std'] else 0,
                    'range_avg': float(np.mean(np.array(stats['max']) - np.array(stats['min']))) if stats['max'] and stats['min'] else 0
                }
            
            results_to_save[name] = save_result
        
        # 결과 저장
        results_file = self.output_dir / 'feature_extraction_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"결과 저장 완료: {results_file}")
    
    def generate_comparison_report(self):
        """비교 리포트 생성"""
        report_file = self.output_dir / 'feature_extraction_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 특징 추출 성능 평가 리포트\n\n")
            f.write(f"**평가 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 요약 테이블
            f.write("## 📊 성능 요약\n\n")
            f.write("| 추출기 | 성공률 | 특징 차원 | 평균 시간(ms) | 실루엣 점수 | RF 교차검증 |\n")
            f.write("|--------|-------|-----------|-------------|------------|------------|\n")
            
            for name, result in self.results.items():
                success_rate = result.get('success_rate', 0) * 100
                dimensions = result.get('feature_dimensions', 0)
                avg_time = result.get('avg_extraction_time_ms', 0)
                
                quality = result.get('quality_metrics', {})
                silhouette = quality.get('silhouette_score', 0)
                rf_cv = quality.get('rf_cv_mean', 0)
                
                f.write(f"| {name} | {success_rate:.1f}% | {dimensions} | {avg_time:.2f} | {silhouette:.3f} | {rf_cv:.3f} |\n")
            
            f.write("\n")
            
            # 상세 결과
            f.write("## 🔍 상세 평가 결과\n\n")
            
            for name, result in self.results.items():
                f.write(f"### {name}\n\n")
                
                if 'error' in result:
                    f.write(f"❌ **오류 발생**: {result['error']}\n\n")
                    continue
                
                f.write(f"- **성공률**: {result.get('success_rate', 0)*100:.1f}%\n")
                f.write(f"- **특징 차원**: {result.get('feature_dimensions', 0):,}개\n")
                f.write(f"- **평균 추출 시간**: {result.get('avg_extraction_time_ms', 0):.2f} ± {result.get('std_extraction_time_ms', 0):.2f}ms\n")
                f.write(f"- **총 처리 시간**: {result.get('total_time_ms', 0):.2f}ms\n")
                
                # 품질 지표
                quality = result.get('quality_metrics', {})
                if quality:
                    f.write(f"- **실루엣 점수**: {quality.get('silhouette_score', 0):.3f}\n")
                    f.write(f"- **LDA 점수**: {quality.get('lda_score', 0):.3f}\n")
                    f.write(f"- **RF 교차검증**: {quality.get('rf_cv_mean', 0):.3f} ± {quality.get('rf_cv_std', 0):.3f}\n")
                    f.write(f"- **특징 다양성**: {quality.get('feature_diversity', 0):.6f}\n")
                    f.write(f"- **특징 안정성**: {quality.get('feature_stability', 0):.3f}\n")
                
                f.write("\n")
            
            # 권장사항
            f.write("## 💡 권장사항\n\n")
            
            # 최고 성능 추출기 찾기
            best_overall = None
            best_score = -1
            
            for name, result in self.results.items():
                if 'quality_metrics' in result:
                    quality = result['quality_metrics']
                    # 종합 점수 계산 (실루엣 + RF 교차검증)
                    score = quality.get('silhouette_score', 0) + quality.get('rf_cv_mean', 0)
                    if score > best_score:
                        best_score = score
                        best_overall = name
            
            if best_overall:
                f.write(f"🏆 **최고 성능**: {best_overall}\n")
                
            # 속도별 추천
            fastest = min(self.results.items(), 
                         key=lambda x: x[1].get('avg_extraction_time_ms', float('inf')))
            f.write(f"⚡ **최고 속도**: {fastest[0]} ({fastest[1].get('avg_extraction_time_ms', 0):.2f}ms)\n")
            
            # 차원별 추천
            highest_dim = max(self.results.items(), 
                             key=lambda x: x[1].get('feature_dimensions', 0))
            f.write(f"📊 **최고 차원**: {highest_dim[0]} ({highest_dim[1].get('feature_dimensions', 0)}차원)\n")
        
        logger.info(f"리포트 생성 완료: {report_file}")
    
    def create_visualizations(self):
        """성능 비교 시각화 생성"""
        try:
            # 한글 폰트 설정
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 성능 비교 차트
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            extractors = []
            success_rates = []
            dimensions = []
            times = []
            silhouette_scores = []
            
            for name, result in self.results.items():
                if 'error' not in result:
                    extractors.append(name.replace('_', '\n'))
                    success_rates.append(result.get('success_rate', 0) * 100)
                    dimensions.append(result.get('feature_dimensions', 0))
                    times.append(result.get('avg_extraction_time_ms', 0))
                    
                    quality = result.get('quality_metrics', {})
                    silhouette_scores.append(quality.get('silhouette_score', 0))
            
            # 성공률
            ax1.bar(extractors, success_rates, color='skyblue')
            ax1.set_title('Feature Extraction Success Rate')
            ax1.set_ylabel('Success Rate (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 특징 차원
            ax2.bar(extractors, dimensions, color='lightgreen')
            ax2.set_title('Feature Dimensions')
            ax2.set_ylabel('Number of Features')
            ax2.tick_params(axis='x', rotation=45)
            
            # 추출 시간
            ax3.bar(extractors, times, color='orange')
            ax3.set_title('Average Extraction Time')
            ax3.set_ylabel('Time (ms)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 실루엣 점수
            ax4.bar(extractors, silhouette_scores, color='pink')
            ax4.set_title('Silhouette Score (Feature Quality)')
            ax4.set_ylabel('Silhouette Score')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_extraction_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("시각화 생성 완료")
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")


def main():
    """메인 실행 함수"""
    # 출력 디렉토리 설정
    output_dir = Path("data/results/feature_evaluation")
    
    # 평가기 생성 및 실행
    evaluator = FeatureEvaluator(output_dir)
    evaluator.run_comprehensive_evaluation(num_samples=60)
    
    print(f"\n🎉 특징 추출 성능 평가 완료!")
    print(f"📊 결과 확인: {output_dir}")
    print(f"📈 리포트: {output_dir / 'feature_extraction_report.md'}")
    print(f"🔍 상세 결과: {output_dir / 'feature_extraction_results.json'}")


if __name__ == "__main__":
    main()