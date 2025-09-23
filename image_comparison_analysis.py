#!/usr/bin/env python3
"""
Image Comparison Analysis
=========================
PH_annotation.bmp와 XTF 이미지를 비교하여 동일한 기뢰 위치를 나타내는지 분석
180도 회전, 좌우 반전 포함하여 형상 유사도 검증

Author: YMARX
Date: 2025-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
from skimage import measure
from scipy import ndimage
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_images():
    """두 이미지 파일 로드"""
    logger.info("Loading images for comparison")

    # Image file paths
    annotation_path = Path("datasets/PH_annotation.bmp")
    xtf_image_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_IMG_00.BMP")

    images = {}

    # Load annotation image
    if annotation_path.exists():
        try:
            annotation_img = cv2.imread(str(annotation_path))
            if annotation_img is not None:
                # Convert BGR to RGB for matplotlib
                annotation_img = cv2.cvtColor(annotation_img, cv2.COLOR_BGR2RGB)
                images['annotation'] = {
                    'image': annotation_img,
                    'path': str(annotation_path),
                    'shape': annotation_img.shape,
                    'size_mb': annotation_path.stat().st_size / (1024*1024)
                }
                logger.info(f"Loaded PH_annotation.bmp: {annotation_img.shape}, {images['annotation']['size_mb']:.1f}MB")
            else:
                logger.error("Failed to load PH_annotation.bmp")
        except Exception as e:
            logger.error(f"Error loading PH_annotation.bmp: {e}")
    else:
        logger.error(f"PH_annotation.bmp not found at {annotation_path}")

    # Load XTF image
    if xtf_image_path.exists():
        try:
            xtf_img = cv2.imread(str(xtf_image_path))
            if xtf_img is not None:
                # Convert BGR to RGB for matplotlib
                xtf_img = cv2.cvtColor(xtf_img, cv2.COLOR_BGR2RGB)
                images['xtf'] = {
                    'image': xtf_img,
                    'path': str(xtf_image_path),
                    'shape': xtf_img.shape,
                    'size_mb': xtf_image_path.stat().st_size / (1024*1024)
                }
                logger.info(f"Loaded XTF image: {xtf_img.shape}, {images['xtf']['size_mb']:.1f}MB")
            else:
                logger.error("Failed to load XTF image")
        except Exception as e:
            logger.error(f"Error loading XTF image: {e}")
    else:
        logger.error(f"XTF image not found at {xtf_image_path}")

    return images


def apply_transformations(image):
    """이미지에 다양한 변환 적용"""
    transformations = {}

    # Original
    transformations['original'] = image.copy()

    # 180도 회전
    transformations['rotate_180'] = np.rot90(image, 2)

    # 좌우 반전
    transformations['flip_horizontal'] = np.fliplr(image)

    # 상하 반전
    transformations['flip_vertical'] = np.flipud(image)

    # 180도 회전 + 좌우 반전
    transformations['rotate_flip'] = np.fliplr(np.rot90(image, 2))

    # 90도 회전
    transformations['rotate_90'] = np.rot90(image, 1)

    # 270도 회전
    transformations['rotate_270'] = np.rot90(image, 3)

    return transformations


def convert_to_grayscale(image):
    """이미지를 그레이스케일로 변환"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def calculate_similarity_metrics(img1, img2):
    """두 이미지 간의 유사도 메트릭 계산"""
    # 크기를 같게 맞추기
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 더 작은 크기로 리사이즈
    target_h = min(h1, h2)
    target_w = min(w1, w2)

    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))

    # 그레이스케일 변환
    gray1 = convert_to_grayscale(img1_resized)
    gray2 = convert_to_grayscale(img2_resized)

    # 정규화
    gray1 = gray1.astype(np.float32) / 255.0
    gray2 = gray2.astype(np.float32) / 255.0

    # 유사도 메트릭 계산
    metrics = {}

    # 1. Mean Squared Error (낮을수록 유사)
    metrics['mse'] = mean_squared_error(gray1.flatten(), gray2.flatten())

    # 2. Normalized Cross-Correlation (높을수록 유사)
    correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    metrics['ncc'] = correlation.max()

    # 3. Structural Similarity Index (높을수록 유사)
    try:
        from skimage.metrics import structural_similarity as ssim
        metrics['ssim'] = ssim(gray1, gray2)
    except ImportError:
        logger.warning("SSIM calculation requires scikit-image")
        metrics['ssim'] = None

    # 4. Histogram correlation (높을수록 유사)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 1])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 1])
    metrics['hist_corr'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return metrics


def analyze_shape_features(image):
    """이미지의 형상 특징 분석"""
    gray = convert_to_grayscale(image)

    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        'num_contours': len(contours),
        'total_area': 0,
        'perimeter': 0,
        'aspect_ratios': [],
        'solidity': []
    }

    if contours:
        # 가장 큰 윤곽선들 분석
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        for i, contour in enumerate(contours_sorted[:5]):  # 상위 5개 윤곽선
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            features['total_area'] += area
            features['perimeter'] += perimeter

            # 경계 상자
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            features['aspect_ratios'].append(aspect_ratio)

            # 볼록 껍질
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            features['solidity'].append(solidity)

    return features


def perform_comprehensive_comparison(images):
    """종합적인 이미지 비교 분석"""
    logger.info("Performing comprehensive image comparison")

    if 'annotation' not in images or 'xtf' not in images:
        logger.error("Both images are required for comparison")
        return None

    annotation_img = images['annotation']['image']
    xtf_img = images['xtf']['image']

    # 변환 적용
    logger.info("Applying transformations to XTF image")
    xtf_transformations = apply_transformations(xtf_img)

    # 각 변환에 대해 유사도 계산
    comparison_results = {}

    for transform_name, transformed_img in xtf_transformations.items():
        logger.info(f"Comparing with transformation: {transform_name}")

        # 유사도 메트릭 계산
        similarity_metrics = calculate_similarity_metrics(annotation_img, transformed_img)

        # 형상 특징 분석
        annotation_features = analyze_shape_features(annotation_img)
        transformed_features = analyze_shape_features(transformed_img)

        comparison_results[transform_name] = {
            'similarity_metrics': similarity_metrics,
            'annotation_features': annotation_features,
            'transformed_features': transformed_features,
            'transformed_image': transformed_img
        }

    return comparison_results


def create_comparison_visualization(images, comparison_results):
    """비교 결과 시각화"""
    logger.info("Creating comparison visualization")

    if not comparison_results:
        logger.error("No comparison results to visualize")
        return

    # 전체 비교 시각화
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    # 원본 이미지들
    annotation_img = images['annotation']['image']
    xtf_img = images['xtf']['image']

    # 원본 annotation 이미지
    axes[0].imshow(annotation_img)
    axes[0].set_title('PH_annotation.bmp\n(Reference)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 원본 XTF 이미지
    axes[1].imshow(xtf_img)
    axes[1].set_title('XTF Original', fontsize=12)
    axes[1].axis('off')

    # 각 변환 결과
    transform_names = ['rotate_180', 'flip_horizontal', 'flip_vertical', 'rotate_flip', 'rotate_90', 'rotate_270']

    for i, transform_name in enumerate(transform_names):
        if transform_name in comparison_results:
            ax_idx = i + 2
            if ax_idx < len(axes):
                transformed_img = comparison_results[transform_name]['transformed_image']
                metrics = comparison_results[transform_name]['similarity_metrics']

                axes[ax_idx].imshow(transformed_img)

                # 유사도 정보 표시
                title = f'{transform_name.replace("_", " ").title()}\n'
                if metrics['ssim'] is not None:
                    title += f'SSIM: {metrics["ssim"]:.3f}\n'
                title += f'NCC: {metrics["ncc"]:.3f}'

                axes[ax_idx].set_title(title, fontsize=10)
                axes[ax_idx].axis('off')

    # 남은 축 숨기기
    for i in range(len(transform_names) + 2, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_file = Path("analysis_results/image_comparison/comparison_visualization.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison visualization to: {output_file}")


def create_detailed_similarity_plot(comparison_results):
    """상세 유사도 분석 플롯"""
    logger.info("Creating detailed similarity analysis plot")

    # 유사도 메트릭 수집
    transforms = list(comparison_results.keys())
    mse_values = [comparison_results[t]['similarity_metrics']['mse'] for t in transforms]
    ncc_values = [comparison_results[t]['similarity_metrics']['ncc'] for t in transforms]
    hist_corr_values = [comparison_results[t]['similarity_metrics']['hist_corr'] for t in transforms]
    ssim_values = [comparison_results[t]['similarity_metrics']['ssim'] for t in transforms if comparison_results[t]['similarity_metrics']['ssim'] is not None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # MSE (낮을수록 좋음)
    bars1 = ax1.bar(transforms, mse_values, color='lightcoral')
    ax1.set_title('Mean Squared Error (Lower = More Similar)')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)

    # 최솟값 표시
    min_mse_idx = np.argmin(mse_values)
    bars1[min_mse_idx].set_color('darkred')
    ax1.text(min_mse_idx, mse_values[min_mse_idx], f'{mse_values[min_mse_idx]:.4f}',
             ha='center', va='bottom', fontweight='bold')

    # NCC (높을수록 좋음)
    bars2 = ax2.bar(transforms, ncc_values, color='lightblue')
    ax2.set_title('Normalized Cross-Correlation (Higher = More Similar)')
    ax2.set_ylabel('NCC')
    ax2.tick_params(axis='x', rotation=45)

    # 최댓값 표시
    max_ncc_idx = np.argmax(ncc_values)
    bars2[max_ncc_idx].set_color('darkblue')
    ax2.text(max_ncc_idx, ncc_values[max_ncc_idx], f'{ncc_values[max_ncc_idx]:.4f}',
             ha='center', va='bottom', fontweight='bold')

    # Histogram Correlation (높을수록 좋음)
    bars3 = ax3.bar(transforms, hist_corr_values, color='lightgreen')
    ax3.set_title('Histogram Correlation (Higher = More Similar)')
    ax3.set_ylabel('Histogram Correlation')
    ax3.tick_params(axis='x', rotation=45)

    # 최댓값 표시
    max_hist_idx = np.argmax(hist_corr_values)
    bars3[max_hist_idx].set_color('darkgreen')
    ax3.text(max_hist_idx, hist_corr_values[max_hist_idx], f'{hist_corr_values[max_hist_idx]:.4f}',
             ha='center', va='bottom', fontweight='bold')

    # SSIM (있는 경우)
    if ssim_values and len(ssim_values) == len(transforms):
        bars4 = ax4.bar(transforms, ssim_values, color='lightyellow')
        ax4.set_title('Structural Similarity Index (Higher = More Similar)')
        ax4.set_ylabel('SSIM')
        ax4.tick_params(axis='x', rotation=45)

        # 최댓값 표시
        max_ssim_idx = np.argmax(ssim_values)
        bars4[max_ssim_idx].set_color('orange')
        ax4.text(max_ssim_idx, ssim_values[max_ssim_idx], f'{ssim_values[max_ssim_idx]:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'SSIM not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')

    plt.tight_layout()
    output_file = Path("analysis_results/image_comparison/similarity_metrics.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved similarity metrics plot to: {output_file}")


def generate_comparison_report(images, comparison_results):
    """이미지 비교 분석 보고서 생성"""
    logger.info("Generating comparison analysis report")

    report_lines = []
    report_lines.append("# 이미지 비교 분석 보고서")
    report_lines.append(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**분석자**: YMARX")
    report_lines.append("")

    # 개요
    report_lines.append("## 🎯 **분석 목적**")
    report_lines.append("PH_annotation.bmp와 XTF 이미지가 같은 기뢰 위치를 나타내는지 확인")
    report_lines.append("180도 회전, 좌우 반전 등 다양한 변환을 적용하여 형상 유사성 검증")
    report_lines.append("")

    # 이미지 정보
    report_lines.append("## 📁 **분석 대상 이미지**")
    report_lines.append("")

    for img_type, img_data in images.items():
        report_lines.append(f"### {img_type.title()} Image")
        report_lines.append(f"- **파일 경로**: `{img_data['path']}`")
        report_lines.append(f"- **이미지 크기**: {img_data['shape']}")
        report_lines.append(f"- **파일 크기**: {img_data['size_mb']:.1f} MB")
        report_lines.append("")

    # 변환 및 유사도 분석
    report_lines.append("## 🔄 **변환별 유사도 분석**")
    report_lines.append("")

    # 최고 유사도 찾기
    best_transform = None
    best_score = -1
    best_metric = None

    similarity_summary = []

    for transform_name, result in comparison_results.items():
        metrics = result['similarity_metrics']

        # 종합 점수 계산 (NCC + Hist_Corr + (1-MSE) + SSIM)
        score = metrics['ncc'] + metrics['hist_corr'] + (1 - metrics['mse'])
        if metrics['ssim'] is not None:
            score += metrics['ssim']

        if score > best_score:
            best_score = score
            best_transform = transform_name
            best_metric = metrics

        similarity_summary.append({
            'transform': transform_name,
            'score': score,
            'metrics': metrics
        })

    # 유사도 결과 테이블
    report_lines.append("| 변환 타입 | MSE ↓ | NCC ↑ | Hist Corr ↑ | SSIM ↑ | 종합 점수 |")
    report_lines.append("|-----------|-------|-------|-------------|---------|----------|")

    for item in sorted(similarity_summary, key=lambda x: x['score'], reverse=True):
        transform = item['transform']
        metrics = item['metrics']
        score = item['score']

        ssim_str = f"{metrics['ssim']:.3f}" if metrics['ssim'] is not None else "N/A"

        # 최고 점수 표시
        marker = "🥇" if transform == best_transform else ""

        report_lines.append(f"| {marker} {transform.replace('_', ' ').title()} | {metrics['mse']:.4f} | {metrics['ncc']:.3f} | {metrics['hist_corr']:.3f} | {ssim_str} | {score:.3f} |")

    report_lines.append("")

    # 분석 결과
    report_lines.append("## 📊 **분석 결과**")
    report_lines.append("")

    if best_transform and best_metric:
        report_lines.append(f"### ✅ **최적 변환**: `{best_transform.replace('_', ' ').title()}`")
        report_lines.append("")
        report_lines.append("**유사도 지표**:")
        report_lines.append(f"- MSE (Mean Squared Error): {best_metric['mse']:.4f} (낮을수록 유사)")
        report_lines.append(f"- NCC (Normalized Cross-Correlation): {best_metric['ncc']:.3f} (높을수록 유사)")
        report_lines.append(f"- Histogram Correlation: {best_metric['hist_corr']:.3f} (높을수록 유사)")
        if best_metric['ssim'] is not None:
            report_lines.append(f"- SSIM (Structural Similarity): {best_metric['ssim']:.3f} (높을수록 유사)")
        report_lines.append(f"- **종합 점수**: {best_score:.3f}")
        report_lines.append("")

        # 형상 특징 비교
        if best_transform in comparison_results:
            ann_features = comparison_results[best_transform]['annotation_features']
            trans_features = comparison_results[best_transform]['transformed_features']

            report_lines.append("**형상 특징 비교**:")
            report_lines.append(f"- 윤곽선 개수: Annotation({ann_features['num_contours']}) vs Transformed({trans_features['num_contours']})")
            report_lines.append(f"- 총 면적: Annotation({ann_features['total_area']:.0f}) vs Transformed({trans_features['total_area']:.0f})")

            if ann_features['aspect_ratios'] and trans_features['aspect_ratios']:
                ann_aspect = np.mean(ann_features['aspect_ratios'])
                trans_aspect = np.mean(trans_features['aspect_ratios'])
                report_lines.append(f"- 평균 종횡비: Annotation({ann_aspect:.2f}) vs Transformed({trans_aspect:.2f})")

    report_lines.append("")

    # 결론
    report_lines.append("## 🎯 **결론**")
    report_lines.append("")

    # 유사도 임계값 설정
    similarity_threshold = {
        'ncc': 0.3,      # NCC > 0.3
        'hist_corr': 0.5, # Histogram correlation > 0.5
        'mse': 0.1,      # MSE < 0.1
        'ssim': 0.5      # SSIM > 0.5 (if available)
    }

    if best_metric:
        is_similar = (
            best_metric['ncc'] > similarity_threshold['ncc'] and
            best_metric['hist_corr'] > similarity_threshold['hist_corr'] and
            best_metric['mse'] < similarity_threshold['mse']
        )

        if best_metric['ssim'] is not None:
            is_similar = is_similar and best_metric['ssim'] > similarity_threshold['ssim']

        if is_similar:
            report_lines.append("### ✅ **두 이미지는 같은 기뢰 위치를 나타내는 것으로 판단됩니다**")
            report_lines.append(f"**최적 변환**: `{best_transform.replace('_', ' ').title()}`")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 모든 유사도 지표가 임계값을 충족")
            report_lines.append(f"- 특히 {best_transform.replace('_', ' ')} 변환 적용 시 높은 유사성 확인")
            report_lines.append("")
            report_lines.append("**의미**:")
            report_lines.append("- 이전의 \"180도 회전/좌우 반전으로 문제 해결\" 주장이 **실제로 정확함**")
            report_lines.append("- XTF 이미지와 annotation 이미지가 실제로 동일한 영역을 나타냄")
            report_lines.append("- 좌표 매핑 시 해당 변환을 적용하면 정확한 매핑 가능")
        else:
            report_lines.append("### ❌ **두 이미지는 다른 위치를 나타내는 것으로 판단됩니다**")
            report_lines.append("")
            report_lines.append("**근거**:")
            report_lines.append(f"- 최고 유사도에서도 임계값 미달")
            report_lines.append(f"- 최적 변환: {best_transform.replace('_', ' ').title()}")
            report_lines.append(f"- 최고 NCC: {best_metric['ncc']:.3f} (임계값: {similarity_threshold['ncc']})")
            report_lines.append(f"- 최고 Hist Corr: {best_metric['hist_corr']:.3f} (임계값: {similarity_threshold['hist_corr']})")
            report_lines.append("")
            report_lines.append("**의미**:")
            report_lines.append("- XTF 이미지와 annotation 이미지가 서로 다른 영역")
            report_lines.append("- 이전 좌표 분석과 일치: 지리적으로 다른 지역의 데이터")
            report_lines.append("- 좌표 변환만으로는 해결할 수 없는 근본적인 데이터 불일치")

    # 기술적 세부사항
    report_lines.append("")
    report_lines.append("## 🛠️ **기술적 세부사항**")
    report_lines.append("")
    report_lines.append("**적용된 변환**:")
    report_lines.append("- Original (변환 없음)")
    report_lines.append("- 180도 회전")
    report_lines.append("- 좌우 반전")
    report_lines.append("- 상하 반전")
    report_lines.append("- 180도 회전 + 좌우 반전")
    report_lines.append("- 90도 회전")
    report_lines.append("- 270도 회전")
    report_lines.append("")
    report_lines.append("**유사도 측정 방법**:")
    report_lines.append("- MSE (Mean Squared Error): 픽셀 간 차이의 제곱 평균")
    report_lines.append("- NCC (Normalized Cross-Correlation): 정규화된 교차 상관계수")
    report_lines.append("- Histogram Correlation: 히스토그램 간 상관계수")
    report_lines.append("- SSIM (Structural Similarity Index): 구조적 유사성 지수")

    # 보고서 저장
    output_file = Path("analysis_results/image_comparison/IMAGE_COMPARISON_REPORT.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved comparison report to: {output_file}")

    # JSON 데이터 저장
    json_data = {
        'images': images,
        'best_transform': best_transform,
        'best_score': best_score,
        'similarity_summary': similarity_summary,
        'analysis_timestamp': datetime.now().isoformat()
    }

    # numpy arrays는 JSON 직렬화할 수 없으므로 제거
    for img_type in json_data['images']:
        json_data['images'][img_type].pop('image', None)

    json_file = Path("analysis_results/image_comparison/comparison_data.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info(f"Saved comparison data to: {json_file}")


def main():
    """메인 실행 함수"""
    logger.info("Starting Image Comparison Analysis")

    try:
        # 이미지 로드
        images = load_images()

        if len(images) < 2:
            logger.error("Need both annotation and XTF images for comparison")
            return 1

        # 종합 비교 분석
        comparison_results = perform_comprehensive_comparison(images)

        if not comparison_results:
            logger.error("Failed to perform comparison analysis")
            return 1

        # 시각화 생성
        create_comparison_visualization(images, comparison_results)
        create_detailed_similarity_plot(comparison_results)

        # 보고서 생성
        generate_comparison_report(images, comparison_results)

        # 결과 요약 출력
        print("\n" + "="*80)
        print("이미지 비교 분석 결과 요약")
        print("="*80)

        # 최고 유사도 찾기
        best_transform = None
        best_score = -1

        for transform_name, result in comparison_results.items():
            metrics = result['similarity_metrics']
            score = metrics['ncc'] + metrics['hist_corr'] + (1 - metrics['mse'])
            if metrics['ssim'] is not None:
                score += metrics['ssim']

            if score > best_score:
                best_score = score
                best_transform = transform_name

        print(f"📁 분석된 이미지:")
        print(f"   - PH_annotation.bmp: {images['annotation']['shape']}")
        print(f"   - XTF 이미지: {images['xtf']['shape']}")

        print(f"\n🔄 최적 변환: {best_transform.replace('_', ' ').title()}")
        print(f"📊 종합 유사도 점수: {best_score:.3f}")

        best_metrics = comparison_results[best_transform]['similarity_metrics']
        print(f"\n📈 상세 유사도 지표:")
        print(f"   - MSE: {best_metrics['mse']:.4f} (낮을수록 유사)")
        print(f"   - NCC: {best_metrics['ncc']:.3f} (높을수록 유사)")
        print(f"   - Hist Corr: {best_metrics['hist_corr']:.3f} (높을수록 유사)")
        if best_metrics['ssim'] is not None:
            print(f"   - SSIM: {best_metrics['ssim']:.3f} (높을수록 유사)")

        # 유사성 판단
        is_similar = (
            best_metrics['ncc'] > 0.3 and
            best_metrics['hist_corr'] > 0.5 and
            best_metrics['mse'] < 0.1
        )

        if best_metrics['ssim'] is not None:
            is_similar = is_similar and best_metrics['ssim'] > 0.5

        print(f"\n🎯 **최종 결론**:")
        if is_similar:
            print("   ✅ 두 이미지는 **동일한 기뢰 위치**를 나타내는 것으로 판단됩니다!")
            print(f"   → {best_transform.replace('_', ' ')} 변환 적용 시 높은 유사성 확인")
            print("   → 이전 \"180도 회전/좌우 반전 해결\" 주장이 **실제로 정확함**")
            print("   → 해당 변환을 적용하면 정확한 좌표 매핑 가능")
        else:
            print("   ❌ 두 이미지는 **다른 위치**를 나타내는 것으로 판단됩니다")
            print("   → 이전 좌표 분석과 일치: 지리적으로 다른 지역의 데이터")
            print("   → 좌표 변환만으로는 해결할 수 없는 근본적인 데이터 불일치")

        return 0

    except Exception as e:
        logger.error(f"Image comparison analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())