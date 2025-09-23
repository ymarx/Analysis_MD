#!/usr/bin/env python3
"""
실데이터와 모의데이터 분포 비교 테스트

기존 모의데이터와 새로운 시나리오 기반 모의데이터의 분포 특성을 비교 분석합니다.
"""

import numpy as np
import logging
from pathlib import Path
import json
import sys
from typing import Dict, List, Any, Tuple

# 프로젝트 모듈 import
sys.path.append('src')

from data_simulation.scenario_generator import ScenarioDataGenerator
# 직접 import를 피하고 파일 경로로 로드
import importlib.util

# 직접 모듈 로드
spec = importlib.util.spec_from_file_location(
    "data_distribution_analyzer", 
    "src/evaluation/data_distribution_analyzer.py"
)
analyzer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer_module)

DataDistributionAnalyzer = analyzer_module.DataDistributionAnalyzer
json_safe_convert = analyzer_module.json_safe_convert

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_legacy_synthetic_data(n_samples: int = 20, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """기존 방식의 모의데이터 생성 (Phase 1 스타일)"""
    np.random.seed(42)
    images = []
    
    for i in range(n_samples):
        # 기본 배경 생성
        image = np.random.normal(0.5, 0.1, size)
        
        if i < n_samples // 2:  # 절반은 양성 샘플
            # 간단한 타원형 기뢰 추가
            center_y, center_x = size[0] // 2, size[1] // 2
            
            y, x = np.ogrid[:size[0], :size[1]]
            mask = ((x - center_x) / 12)**2 + ((y - center_y) / 8)**2 < 1
            image[mask] += 0.3
            
            # 간단한 그림자
            shadow_start = center_x + 15
            shadow_end = min(shadow_start + 20, size[1])
            shadow_y_start = max(0, center_y - 10)
            shadow_y_end = min(center_y + 10, size[0])
            
            image[shadow_y_start:shadow_y_end, shadow_start:shadow_end] -= 0.2
        
        image = np.clip(image, 0, 1)
        images.append(image)
    
    return np.array(images, dtype=np.float32)


def generate_scenario_comparison_datasets() -> Dict[str, np.ndarray]:
    """시나리오 비교용 데이터셋 생성"""
    logger.info("시나리오 비교용 데이터셋 생성 시작")
    
    datasets = {}
    generator = ScenarioDataGenerator()
    
    # 1. 기존 방식 모의데이터
    legacy_data = create_legacy_synthetic_data(n_samples=20, size=(64, 64))
    datasets['Legacy_Synthetic'] = legacy_data
    
    # 2. 시나리오별 모의데이터
    scenarios_to_test = [
        'A_deep_ocean',      # 깊은 바다
        'B_shallow_coastal', # 얕은 연안  
        'C_medium_depth',    # 중간 깊이
        'D_high_current',    # 강한 해류
        'E_sandy_rocky'      # 모래/암초
    ]
    
    for scenario_name in scenarios_to_test:
        logger.info(f"{scenario_name} 데이터셋 생성 중...")
        
        dataset = generator.generate_scenario_dataset(
            scenario_name,
            n_positive=10,
            n_negative=10,
            image_size=(64, 64)
        )
        
        images = np.array(dataset['images'])
        datasets[scenario_name] = images
    
    logger.info(f"총 {len(datasets)}개 데이터셋 생성 완료")
    return datasets


def analyze_data_quality_metrics(datasets: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """데이터 품질 지표 분석"""
    logger.info("데이터 품질 지표 분석 시작")
    
    quality_metrics = {}
    
    for name, data in datasets.items():
        logger.info(f"{name} 품질 분석 중...")
        
        flat_data = data.flatten()
        
        # 기본 품질 지표
        metrics = {
            'dynamic_range': float(np.max(flat_data) - np.min(flat_data)),
            'signal_noise_ratio': float(np.mean(flat_data) / (np.std(flat_data) + 1e-10)),
            'contrast_ratio': float(np.percentile(flat_data, 95) - np.percentile(flat_data, 5)),
            'data_sparsity': float(np.sum(flat_data < 0.1) / len(flat_data)),  # 매우 어두운 픽셀 비율
            'saturation_ratio': float(np.sum(flat_data > 0.9) / len(flat_data)),  # 포화 픽셀 비율
            'entropy': calculate_entropy(flat_data),
            'uniformity_score': 1.0 / (np.std(flat_data) + 1e-10)  # 높을수록 균일함
        }
        
        # 텍스처 복잡도
        if len(data.shape) >= 3:
            sample_image = data[0]
        else:
            sample_image = data
        
        # 그래디언트 기반 복잡도
        grad_x = np.diff(sample_image, axis=1)
        grad_y = np.diff(sample_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
        
        metrics.update({
            'texture_complexity': float(np.mean(gradient_magnitude)),
            'edge_sharpness': float(np.percentile(gradient_magnitude, 95)),
            'texture_uniformity': float(1.0 / (np.std(gradient_magnitude) + 1e-10))
        })
        
        quality_metrics[name] = metrics
    
    return quality_metrics


def calculate_entropy(data: np.ndarray, bins: int = 256) -> float:
    """히스토그램 엔트로피 계산"""
    try:
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist + 1e-10  # log(0) 방지
        entropy = -np.sum(hist * np.log2(hist)) / np.log2(bins)  # 정규화
        return float(entropy)
    except:
        return 0.0


def create_distribution_report(analysis_results: Dict[str, Any], 
                             quality_metrics: Dict[str, Any]) -> str:
    """분포 분석 리포트 생성"""
    
    report_lines = [
        "# 🔍 실데이터와 모의데이터 분포 비교 분석 리포트",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}" if 'pd' in globals() else "**분석 일시**: 2025-09-09",
        "",
        "---",
        "",
        "## 📊 데이터셋 개요",
        ""
    ]
    
    # 데이터셋 기본 정보
    if 'summary' in analysis_results and 'dataset_stats' in analysis_results['summary']:
        report_lines.append("| 데이터셋 | 평균 | 표준편차 | 최소값 | 최대값 | 샘플 수 |")
        report_lines.append("|---------|------|----------|--------|--------|---------|")
        
        for name, stats in analysis_results['summary']['dataset_stats'].items():
            report_lines.append(
                f"| **{name}** | {stats['mean']:.3f} | {stats['std']:.3f} | "
                f"{stats['min']:.3f} | {stats['max']:.3f} | {stats['size']//4096:,} |"
            )
        report_lines.append("")
    
    # 품질 지표
    report_lines.extend([
        "## 🎯 데이터 품질 지표",
        "",
        "| 데이터셋 | 동적범위 | S/N 비 | 대비도 | 텍스처복잡도 | 엔트로피 |",
        "|---------|----------|--------|--------|--------------|----------|"
    ])
    
    for name, metrics in quality_metrics.items():
        report_lines.append(
            f"| **{name}** | {metrics['dynamic_range']:.3f} | "
            f"{metrics['signal_noise_ratio']:.1f} | {metrics['contrast_ratio']:.3f} | "
            f"{metrics['texture_complexity']:.4f} | {metrics['entropy']:.3f} |"
        )
    
    report_lines.extend(["", "## 🔄 분포 유사도 비교", ""])
    
    # 유사도 분석
    if 'pairwise_comparisons' in analysis_results:
        report_lines.append("| 비교 쌍 | 평균 차이 | KS 통계량 | 히스토그램 상관 | 유사도 점수 |")
        report_lines.append("|---------|-----------|-----------|-----------------|-------------|")
        
        for pair_name, comparison in analysis_results['pairwise_comparisons'].items():
            report_lines.append(
                f"| **{pair_name.replace('_vs_', ' vs ')}** | "
                f"{comparison.get('mean_difference', 0):.4f} | "
                f"{comparison.get('ks_statistic', 0):.3f} | "
                f"{comparison.get('histogram_correlation', 0):.3f} | "
                f"{comparison.get('similarity_score', 0):.3f} |"
            )
        report_lines.append("")
    
    # 요약 및 권장사항
    summary = analysis_results.get('summary', {})
    
    report_lines.extend([
        "## 💡 분석 결과 요약",
        "",
        f"- **전체 평균 유사도**: {summary.get('average_similarity', 0):.3f}",
        f"- **유사도 범위**: {summary.get('similarity_range', [0, 0])[0]:.3f} ~ {summary.get('similarity_range', [0, 0])[1]:.3f}",
    ])
    
    if summary.get('most_similar_pair'):
        report_lines.append(f"- **가장 유사한 쌍**: {summary['most_similar_pair'].replace('_vs_', ' vs ')}")
    if summary.get('least_similar_pair'):
        report_lines.append(f"- **가장 다른 쌍**: {summary['least_similar_pair'].replace('_vs_', ' vs ')}")
    
    report_lines.extend([
        "",
        "## 🚀 권장사항",
        "",
        "### 시나리오별 특성"
    ])
    
    # 시나리오별 분석
    scenario_analysis = {
        'A_deep_ocean': "깊은 바다 환경 - 낮은 노이즈, 높은 기뢰 가시성",
        'B_shallow_coastal': "얕은 연안 환경 - 높은 노이즈, 복잡한 텍스처", 
        'C_medium_depth': "중간 깊이 환경 - 균형잡힌 특성",
        'D_high_current': "강한 해류 환경 - 동적 왜곡, 낮은 기뢰 가시성",
        'E_sandy_rocky': "모래/암초 환경 - 높은 텍스처 복잡도"
    }
    
    for scenario, description in scenario_analysis.items():
        if scenario in quality_metrics:
            metrics = quality_metrics[scenario]
            report_lines.append(f"- **{scenario}**: {description}")
            report_lines.append(f"  - 텍스처 복잡도: {metrics['texture_complexity']:.4f}")
            report_lines.append(f"  - 대비도: {metrics['contrast_ratio']:.3f}")
            report_lines.append(f"  - 엔트로피: {metrics['entropy']:.3f}")
    
    report_lines.extend([
        "",
        "### 모델 훈련 권장사항",
        "",
        "1. **다양성 확보**: 여러 시나리오를 조합하여 강건한 모델 구축",
        "2. **균형 조정**: 각 환경별 샘플 비율을 실제 운용 환경에 맞게 조정",
        "3. **점진적 훈련**: 단순한 환경(깊은 바다)부터 복잡한 환경(연안)으로 점진적 학습",
        "4. **전이 학습**: 시나리오간 지식 전이를 통한 효율적 학습"
    ])
    
    return "\n".join(report_lines)


def main():
    """메인 분석 함수"""
    logger.info("실데이터와 모의데이터 분포 비교 분석 시작")
    
    # 출력 디렉토리 생성
    output_dir = Path("data/results/data_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터셋 생성
    logger.info("\n" + "="*50)
    logger.info("1. 데이터셋 생성")
    logger.info("="*50)
    
    datasets = generate_scenario_comparison_datasets()
    
    # 2. 분포 분석
    logger.info("\n" + "="*50)
    logger.info("2. 분포 특성 분석")
    logger.info("="*50)
    
    analyzer = DataDistributionAnalyzer()
    analysis_results = analyzer.analyze_dataset_collection(datasets)
    
    # 3. 품질 지표 분석
    logger.info("\n" + "="*50)
    logger.info("3. 데이터 품질 지표 분석")
    logger.info("="*50)
    
    quality_metrics = analyze_data_quality_metrics(datasets)
    
    # 4. 결과 저장
    logger.info("\n" + "="*50)
    logger.info("4. 결과 저장")
    logger.info("="*50)
    
    # JSON 결과 저장
    complete_results = {
        'distribution_analysis': analysis_results,
        'quality_metrics': quality_metrics,
        'metadata': {
            'analysis_type': 'real_vs_synthetic_comparison',
            'datasets_analyzed': list(datasets.keys()),
            'total_samples': {name: data.shape[0] for name, data in datasets.items()}
        }
    }
    
    # json_safe_convert는 이미 위에서 import됨
    
    with open(output_dir / 'data_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_safe_convert(complete_results), f, ensure_ascii=False, indent=2)
    
    # 텍스트 리포트 생성
    report_text = create_distribution_report(analysis_results, quality_metrics)
    
    with open(output_dir / 'data_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 5. 요약 출력
    logger.info("\n" + "="*50)
    logger.info("5. 분석 결과 요약")
    logger.info("="*50)
    
    summary = analysis_results.get('summary', {})
    
    logger.info(f"분석 데이터셋: {len(datasets)}개")
    for name in datasets.keys():
        logger.info(f"  - {name}: {datasets[name].shape[0]}개 샘플")
    
    logger.info(f"\n평균 유사도 점수: {summary.get('average_similarity', 0):.3f}")
    logger.info(f"유사도 범위: {summary.get('similarity_range', [0, 0])[0]:.3f} ~ {summary.get('similarity_range', [0, 0])[1]:.3f}")
    
    if summary.get('most_similar_pair'):
        logger.info(f"가장 유사한 쌍: {summary['most_similar_pair']}")
    if summary.get('least_similar_pair'):
        logger.info(f"가장 다른 쌍: {summary['least_similar_pair']}")
    
    # 품질 지표 요약
    logger.info(f"\n데이터 품질 지표:")
    for name, metrics in quality_metrics.items():
        logger.info(f"  {name}:")
        logger.info(f"    동적범위: {metrics['dynamic_range']:.3f}")
        logger.info(f"    텍스처복잡도: {metrics['texture_complexity']:.4f}")
        logger.info(f"    엔트로피: {metrics['entropy']:.3f}")
    
    logger.info(f"\n결과가 {output_dir}에 저장되었습니다:")
    logger.info(f"  - JSON: data_comparison_results.json")
    logger.info(f"  - 리포트: data_comparison_report.md")
    
    return complete_results


if __name__ == "__main__":
    main()