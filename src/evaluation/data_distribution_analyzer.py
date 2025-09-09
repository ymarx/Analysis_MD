#!/usr/bin/env python3
"""
실데이터와 모의데이터 분포 비교 분석기

다양한 통계적 지표를 통해 실제 데이터와 시뮬레이션 데이터의 분포 특성을 비교 분석합니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from scipy import stats
import warnings

# Wasserstein distance 대체 구현
def simple_wasserstein_distance(u_values, v_values):
    """간단한 Wasserstein 거리 구현"""
    try:
        # 히스토그램 기반 근사
        all_values = np.concatenate([u_values, v_values])
        bins = np.linspace(np.min(all_values), np.max(all_values), 50)
        
        hist_u, _ = np.histogram(u_values, bins=bins, density=True)
        hist_v, _ = np.histogram(v_values, bins=bins, density=True)
        
        # 누적 분포 계산
        cdf_u = np.cumsum(hist_u) / np.sum(hist_u)
        cdf_v = np.cumsum(hist_v) / np.sum(hist_v)
        
        # L1 거리 계산
        return np.sum(np.abs(cdf_u - cdf_v))
    except:
        return 0.0

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


def json_safe_convert(obj):
    """JSON 직렬화 안전 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: json_safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_convert(v) for v in obj]
    else:
        return obj


class DataDistributionAnalyzer:
    """데이터 분포 분석기"""
    
    def __init__(self):
        """초기화"""
        logger.info("데이터 분포 분석기 초기화")
    
    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        통계적 특징 추출
        
        Args:
            data: 분석할 데이터 (이미지 배열)
            
        Returns:
            Dict[str, float]: 통계적 특징들
        """
        flattened = data.flatten()
        
        # 기본 통계량
        basic_stats = {
            'mean': float(np.mean(flattened)),
            'std': float(np.std(flattened)),
            'var': float(np.var(flattened)),
            'min': float(np.min(flattened)),
            'max': float(np.max(flattened)),
            'median': float(np.median(flattened)),
            'range': float(np.max(flattened) - np.min(flattened))
        }
        
        # 고차 모멘트
        try:
            skewness = float(stats.skew(flattened))
            kurtosis = float(stats.kurtosis(flattened))
        except:
            skewness = 0.0
            kurtosis = 0.0
        
        higher_moments = {
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # 백분위수
        percentiles = {
            'p10': float(np.percentile(flattened, 10)),
            'p25': float(np.percentile(flattened, 25)),
            'p75': float(np.percentile(flattened, 75)),
            'p90': float(np.percentile(flattened, 90)),
            'p95': float(np.percentile(flattened, 95)),
            'p99': float(np.percentile(flattened, 99))
        }
        
        # IQR (Interquartile Range)
        iqr = {
            'iqr': percentiles['p75'] - percentiles['p25'],
            'iqr_normalized': (percentiles['p75'] - percentiles['p25']) / (percentiles['p90'] - percentiles['p10'] + 1e-10)
        }
        
        # 모든 특징 결합
        features = {**basic_stats, **higher_moments, **percentiles, **iqr}
        
        return features
    
    def extract_texture_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        텍스처 특징 추출
        
        Args:
            data: 이미지 데이터
            
        Returns:
            Dict[str, float]: 텍스처 특징들
        """
        features = {}
        
        if len(data.shape) == 2:
            # 단일 이미지
            image = data
        elif len(data.shape) == 3:
            # 여러 이미지의 평균
            image = np.mean(data, axis=0)
        else:
            logger.warning("지원하지 않는 데이터 형태")
            return {}
        
        h, w = image.shape
        
        # 1. 그래디언트 기반 특징
        try:
            # 수평/수직 그래디언트
            grad_x = np.diff(image, axis=1)  # 수평 그래디언트
            grad_y = np.diff(image, axis=0)  # 수직 그래디언트
            
            # 패딩하여 원본 크기 유지
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.update({
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'gradient_max': float(np.max(gradient_magnitude)),
                'edge_density': float(np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size)
            })
        except Exception as e:
            logger.warning(f"그래디언트 특징 추출 실패: {e}")
            features.update({
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'gradient_max': 0.0,
                'edge_density': 0.0
            })
        
        # 2. 지역적 분산 (Local Variance)
        try:
            kernel_size = 5
            local_variance = []
            
            for i in range(0, h - kernel_size, kernel_size):
                for j in range(0, w - kernel_size, kernel_size):
                    patch = image[i:i+kernel_size, j:j+kernel_size]
                    local_variance.append(np.var(patch))
            
            if local_variance:
                features.update({
                    'local_var_mean': float(np.mean(local_variance)),
                    'local_var_std': float(np.std(local_variance)),
                    'local_var_max': float(np.max(local_variance))
                })
            else:
                features.update({
                    'local_var_mean': 0.0,
                    'local_var_std': 0.0,
                    'local_var_max': 0.0
                })
        except Exception as e:
            logger.warning(f"지역 분산 계산 실패: {e}")
            features.update({
                'local_var_mean': 0.0,
                'local_var_std': 0.0,
                'local_var_max': 0.0
            })
        
        # 3. 방향성 특징
        try:
            # 수평/수직 방향성 분석
            horizontal_profile = np.mean(image, axis=0)  # 각 열의 평균
            vertical_profile = np.mean(image, axis=1)    # 각 행의 평균
            
            features.update({
                'horizontal_uniformity': float(1.0 / (np.std(horizontal_profile) + 1e-10)),
                'vertical_uniformity': float(1.0 / (np.std(vertical_profile) + 1e-10)),
                'directional_ratio': float(np.std(horizontal_profile) / (np.std(vertical_profile) + 1e-10))
            })
        except Exception as e:
            logger.warning(f"방향성 특징 계산 실패: {e}")
            features.update({
                'horizontal_uniformity': 0.0,
                'vertical_uniformity': 0.0,
                'directional_ratio': 1.0
            })
        
        return features
    
    def extract_spatial_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        공간적 특징 추출
        
        Args:
            data: 이미지 데이터
            
        Returns:
            Dict[str, float]: 공간적 특징들
        """
        features = {}
        
        if len(data.shape) == 2:
            image = data
        elif len(data.shape) == 3:
            image = np.mean(data, axis=0)
        else:
            return {}
        
        h, w = image.shape
        
        # 1. 중심 영역과 주변 영역 비교
        try:
            center_h, center_w = h // 4, w // 4
            center_region = image[center_h:3*center_h, center_w:3*center_w]
            
            # 경계 영역
            top_region = image[:center_h, :]
            bottom_region = image[3*center_h:, :]
            left_region = image[:, :center_w]
            right_region = image[:, 3*center_w:]
            
            center_mean = np.mean(center_region)
            edge_mean = np.mean([np.mean(top_region), np.mean(bottom_region), 
                               np.mean(left_region), np.mean(right_region)])
            
            features.update({
                'center_edge_contrast': float(abs(center_mean - edge_mean)),
                'center_mean': float(center_mean),
                'edge_mean': float(edge_mean)
            })
        except Exception as e:
            logger.warning(f"중심-경계 특징 계산 실패: {e}")
            features.update({
                'center_edge_contrast': 0.0,
                'center_mean': 0.0,
                'edge_mean': 0.0
            })
        
        # 2. 영역별 균일성
        try:
            # 4분할 영역의 분산 비교
            q1 = image[:h//2, :w//2]  # 좌상
            q2 = image[:h//2, w//2:]  # 우상
            q3 = image[h//2:, :w//2]  # 좌하
            q4 = image[h//2:, w//2:]  # 우하
            
            quadrant_means = [np.mean(q1), np.mean(q2), np.mean(q3), np.mean(q4)]
            quadrant_stds = [np.std(q1), np.std(q2), np.std(q3), np.std(q4)]
            
            features.update({
                'quadrant_mean_var': float(np.var(quadrant_means)),
                'quadrant_std_mean': float(np.mean(quadrant_stds)),
                'spatial_uniformity': float(1.0 / (np.var(quadrant_means) + 1e-10))
            })
        except Exception as e:
            logger.warning(f"영역별 균일성 계산 실패: {e}")
            features.update({
                'quadrant_mean_var': 0.0,
                'quadrant_std_mean': 0.0,
                'spatial_uniformity': 1.0
            })
        
        return features
    
    def compute_comprehensive_features(self, data: np.ndarray, 
                                     data_type: str = "unknown") -> Dict[str, Any]:
        """
        종합적인 데이터 특징 추출
        
        Args:
            data: 분석할 데이터
            data_type: 데이터 타입 ("real", "synthetic", "scenario_A", etc.)
            
        Returns:
            Dict[str, Any]: 종합 특징 딕셔너리
        """
        logger.info(f"{data_type} 데이터 특징 추출 시작")
        
        features = {
            'data_type': data_type,
            'data_shape': list(data.shape),
            'sample_count': data.shape[0] if len(data.shape) > 2 else 1
        }
        
        # 1. 통계적 특징
        try:
            statistical_features = self.extract_statistical_features(data)
            features['statistical'] = statistical_features
        except Exception as e:
            logger.error(f"통계적 특징 추출 실패: {e}")
            features['statistical'] = {}
        
        # 2. 텍스처 특징
        try:
            texture_features = self.extract_texture_features(data)
            features['texture'] = texture_features
        except Exception as e:
            logger.error(f"텍스처 특징 추출 실패: {e}")
            features['texture'] = {}
        
        # 3. 공간적 특징
        try:
            spatial_features = self.extract_spatial_features(data)
            features['spatial'] = spatial_features
        except Exception as e:
            logger.error(f"공간적 특징 추출 실패: {e}")
            features['spatial'] = {}
        
        logger.info(f"{data_type} 데이터 특징 추출 완료")
        return features
    
    def compare_distributions(self, data1: np.ndarray, data2: np.ndarray,
                            name1: str = "Data1", name2: str = "Data2") -> Dict[str, float]:
        """
        두 데이터셋 분포 비교
        
        Args:
            data1: 첫 번째 데이터셋
            data2: 두 번째 데이터셋
            name1: 첫 번째 데이터셋 이름
            name2: 두 번째 데이터셋 이름
            
        Returns:
            Dict[str, float]: 비교 결과 지표들
        """
        logger.info(f"{name1} vs {name2} 분포 비교 시작")
        
        flat1 = data1.flatten()
        flat2 = data2.flatten()
        
        comparison = {
            'dataset1': name1,
            'dataset2': name2,
            'size1': len(flat1),
            'size2': len(flat2)
        }
        
        # 1. 기본 통계량 차이
        try:
            mean_diff = abs(np.mean(flat1) - np.mean(flat2))
            std_diff = abs(np.std(flat1) - np.std(flat2))
            var_ratio = np.var(flat1) / (np.var(flat2) + 1e-10)
            
            comparison.update({
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'variance_ratio': float(var_ratio)
            })
        except Exception as e:
            logger.warning(f"기본 통계량 비교 실패: {e}")
        
        # 2. 분포 형태 비교 (KS 테스트)
        try:
            # 샘플링으로 성능 최적화
            sample_size = min(10000, len(flat1), len(flat2))
            sample1 = np.random.choice(flat1, sample_size, replace=False)
            sample2 = np.random.choice(flat2, sample_size, replace=False)
            
            ks_statistic, ks_p_value = stats.ks_2samp(sample1, sample2)
            
            comparison.update({
                'ks_statistic': float(ks_statistic),
                'ks_p_value': float(ks_p_value),
                'distributions_similar': bool(ks_p_value > 0.05)  # 5% 유의수준
            })
        except Exception as e:
            logger.warning(f"KS 테스트 실패: {e}")
            comparison.update({
                'ks_statistic': 0.0,
                'ks_p_value': 1.0,
                'distributions_similar': bool(False)
            })
        
        # 3. 히스토그램 비교
        try:
            bins = np.linspace(min(np.min(flat1), np.min(flat2)),
                             max(np.max(flat1), np.max(flat2)), 50)
            
            hist1, _ = np.histogram(flat1, bins=bins, density=True)
            hist2, _ = np.histogram(flat2, bins=bins, density=True)
            
            # 히스토그램 간 유클리드 거리
            hist_distance = np.sqrt(np.sum((hist1 - hist2)**2))
            
            # 히스토그램 상관계수
            hist_correlation = np.corrcoef(hist1, hist2)[0, 1]
            if np.isnan(hist_correlation):
                hist_correlation = 0.0
            
            comparison.update({
                'histogram_distance': float(hist_distance),
                'histogram_correlation': float(hist_correlation)
            })
        except Exception as e:
            logger.warning(f"히스토그램 비교 실패: {e}")
            comparison.update({
                'histogram_distance': 0.0,
                'histogram_correlation': 0.0
            })
        
        # 4. Wasserstein 거리 (Earth Mover's Distance)
        try:
            # 샘플링으로 성능 최적화
            sample_size = min(5000, len(flat1), len(flat2))
            sample1 = np.random.choice(flat1, sample_size, replace=False)
            sample2 = np.random.choice(flat2, sample_size, replace=False)
            
            wasserstein_dist = simple_wasserstein_distance(sample1, sample2)
            
            comparison['wasserstein_distance'] = float(wasserstein_dist)
        except Exception as e:
            logger.warning(f"Wasserstein 거리 계산 실패: {e}")
            comparison['wasserstein_distance'] = 0.0
        
        # 5. 종합 유사도 점수 (0-1, 1이 가장 유사)
        try:
            # 여러 지표를 종합한 유사도
            scores = []
            
            # 평균 차이 (정규화)
            mean_score = 1.0 / (1.0 + comparison.get('mean_difference', 0) * 10)
            scores.append(mean_score)
            
            # 표준편차 차이 (정규화)
            std_score = 1.0 / (1.0 + comparison.get('std_difference', 0) * 10)
            scores.append(std_score)
            
            # KS 테스트 p-value
            ks_score = comparison.get('ks_p_value', 0)
            scores.append(ks_score)
            
            # 히스토그램 상관계수 (절댓값)
            hist_score = abs(comparison.get('histogram_correlation', 0))
            scores.append(hist_score)
            
            similarity_score = np.mean(scores)
            comparison['similarity_score'] = float(similarity_score)
            
        except Exception as e:
            logger.warning(f"유사도 점수 계산 실패: {e}")
            comparison['similarity_score'] = 0.0
        
        logger.info(f"{name1} vs {name2} 분포 비교 완료")
        return comparison
    
    def analyze_dataset_collection(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        다중 데이터셋 종합 분석
        
        Args:
            datasets: {이름: 데이터} 딕셔너리
            
        Returns:
            Dict[str, Any]: 종합 분석 결과
        """
        logger.info(f"다중 데이터셋 분석 시작: {len(datasets)}개 데이터셋")
        
        analysis_results = {
            'datasets': list(datasets.keys()),
            'dataset_count': len(datasets),
            'individual_features': {},
            'pairwise_comparisons': {},
            'summary': {}
        }
        
        # 1. 개별 데이터셋 특징 추출
        for name, data in datasets.items():
            features = self.compute_comprehensive_features(data, name)
            analysis_results['individual_features'][name] = features
        
        # 2. 쌍별 비교 (모든 조합)
        dataset_names = list(datasets.keys())
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                comparison_key = f"{name1}_vs_{name2}"
                
                comparison = self.compare_distributions(
                    datasets[name1], datasets[name2], name1, name2
                )
                analysis_results['pairwise_comparisons'][comparison_key] = comparison
        
        # 3. 요약 통계
        try:
            # 각 데이터셋의 기본 통계
            summary_stats = {}
            for name, data in datasets.items():
                flat_data = data.flatten()
                summary_stats[name] = {
                    'mean': float(np.mean(flat_data)),
                    'std': float(np.std(flat_data)),
                    'min': float(np.min(flat_data)),
                    'max': float(np.max(flat_data)),
                    'size': len(flat_data)
                }
            
            # 유사도 점수 요약
            similarity_scores = []
            for comp in analysis_results['pairwise_comparisons'].values():
                if 'similarity_score' in comp:
                    similarity_scores.append(comp['similarity_score'])
            
            analysis_results['summary'] = {
                'dataset_stats': summary_stats,
                'average_similarity': float(np.mean(similarity_scores)) if similarity_scores else 0.0,
                'similarity_range': [float(np.min(similarity_scores)), float(np.max(similarity_scores))] if similarity_scores else [0.0, 0.0],
                'most_similar_pair': max(analysis_results['pairwise_comparisons'].items(), 
                                       key=lambda x: x[1].get('similarity_score', 0))[0] if analysis_results['pairwise_comparisons'] else None,
                'least_similar_pair': min(analysis_results['pairwise_comparisons'].items(), 
                                        key=lambda x: x[1].get('similarity_score', 1))[0] if analysis_results['pairwise_comparisons'] else None
            }
        except Exception as e:
            logger.error(f"요약 통계 계산 실패: {e}")
            analysis_results['summary'] = {}
        
        logger.info(f"다중 데이터셋 분석 완료")
        return analysis_results


def main():
    """테스트 함수"""
    import sys
    sys.path.append('src')
    
    from data_simulation.scenario_generator import ScenarioDataGenerator
    
    logging.basicConfig(level=logging.INFO)
    
    # 분석기 초기화
    analyzer = DataDistributionAnalyzer()
    
    # 테스트 데이터 생성
    generator = ScenarioDataGenerator()
    
    # 각 시나리오별 데이터 생성
    datasets = {}
    
    logger.info("테스트용 데이터셋 생성 중...")
    
    for scenario_name in ['A_deep_ocean', 'B_shallow_coastal', 'C_medium_depth']:
        # 작은 데이터셋 생성
        dataset = generator.generate_scenario_dataset(
            scenario_name, 
            n_positive=10, 
            n_negative=10, 
            image_size=(64, 64)
        )
        
        # 이미지 배열로 변환
        images = np.array(dataset['images'])
        datasets[scenario_name] = images
    
    # 종합 분석 실행
    logger.info("종합 분포 분석 실행 중...")
    results = analyzer.analyze_dataset_collection(datasets)
    
    # 결과 저장
    output_dir = Path("data/results/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'distribution_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_safe_convert(results), f, ensure_ascii=False, indent=2)
    
    # 요약 리포트 출력
    logger.info("\n" + "="*50)
    logger.info("분포 분석 결과 요약")
    logger.info("="*50)
    
    summary = results.get('summary', {})
    
    if 'dataset_stats' in summary:
        logger.info("\n데이터셋 기본 통계:")
        for name, stats in summary['dataset_stats'].items():
            logger.info(f"  {name}: 평균={stats['mean']:.3f}, 표준편차={stats['std']:.3f}")
    
    if 'average_similarity' in summary:
        logger.info(f"\n평균 유사도 점수: {summary['average_similarity']:.3f}")
        logger.info(f"유사도 범위: {summary['similarity_range'][0]:.3f} ~ {summary['similarity_range'][1]:.3f}")
        
        if summary.get('most_similar_pair'):
            logger.info(f"가장 유사한 쌍: {summary['most_similar_pair']}")
        if summary.get('least_similar_pair'):
            logger.info(f"가장 다른 쌍: {summary['least_similar_pair']}")
    
    logger.info(f"\n상세 결과가 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()