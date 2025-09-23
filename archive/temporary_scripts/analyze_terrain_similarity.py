#!/usr/bin/env python3
"""
BMP 이미지와 PH_annotation 이미지 간 지형 유사도 분석 스크립트

사용법:
    python analyze_terrain_similarity.py

목적:
    1. Original BMP 이미지와 PH_annotation 이미지 간 지형 유사도 분석
    2. 이미지 변환(180도 회전, 좌우 반전) 고려한 비교
    3. 지형 형상 및 그림자 패턴 분석
    4. Annotation의 빨간색 바운딩박스 제외한 순수 지형 비교
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import json
from skimage import feature, measure, filters, morphology
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_original_bmp_images() -> List[str]:
    """Original 폴더의 BMP 이미지 파일 찾기"""

    logger.info("Original BMP 이미지 파일 검색 중...")

    bmp_files = []
    base_path = Path("datasets")

    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            original_dir = dataset_dir / "original"
            if original_dir.exists():
                for bmp_file in original_dir.glob("*.BMP"):
                    bmp_files.append(str(bmp_file))

    logger.info(f"발견된 Original BMP 파일: {len(bmp_files)}개")
    return bmp_files

def find_annotation_images() -> List[str]:
    """PH_annotation 이미지 파일 찾기"""

    logger.info("PH_annotation 이미지 파일 검색 중...")

    annotation_files = []

    # PH_annotation.bmp와 PH_annotation.png 찾기
    for ext in ['bmp', 'png']:
        annotation_path = Path(f"datasets/PH_annotation.{ext}")
        if annotation_path.exists():
            annotation_files.append(str(annotation_path))

    logger.info(f"발견된 Annotation 파일: {len(annotation_files)}개")
    return annotation_files

def load_and_preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """이미지 로드 및 전처리"""

    try:
        # 이미지 로드
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"이미지 로드 실패: {image_path}")
            return None

        # BGR to RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 크기 조정 (필요시)
        if target_size:
            img = cv2.resize(img, target_size)

        logger.info(f"이미지 로드 완료: {image_path} - 크기: {img.shape}")
        return img

    except Exception as e:
        logger.error(f"이미지 전처리 실패 {image_path}: {e}")
        return None

def remove_annotation_elements(img: np.ndarray) -> np.ndarray:
    """Annotation 요소(빨간색 바운딩박스, 숫자) 제거"""

    try:
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 빨간색 범위 정의 (바운딩박스)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # 빨간색 마스크 생성
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # 형태학적 연산으로 마스크 정제
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # 빨간색 영역을 주변 색상으로 대체 (인페인팅)
        result = cv2.inpaint(img, red_mask, 3, cv2.INPAINT_TELEA)

        # 텍스트 영역 제거 (작은 고대비 영역들)
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 작은 연결 성분들 찾기 (텍스트일 가능성)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_mask = np.zeros(gray.shape, dtype=np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # 텍스트 크기 범위
                cv2.fillPoly(text_mask, [contour], 255)

        # 텍스트 영역 인페인팅
        if np.any(text_mask):
            result = cv2.inpaint(result, text_mask, 3, cv2.INPAINT_TELEA)

        logger.info("Annotation 요소 제거 완료")
        return result

    except Exception as e:
        logger.error(f"Annotation 제거 실패: {e}")
        return img

def extract_terrain_features(img: np.ndarray) -> Dict:
    """지형 특징 추출"""

    try:
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

        # 1. 엣지 검출 (지형 경계)
        edges = cv2.Canny(gray, 30, 100)

        # 2. 텍스처 특징 (LBP - Local Binary Pattern)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')

        # 3. 그림자 영역 검출 (어두운 영역)
        _, shadow_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        shadow_area = np.sum(shadow_mask == 0) / (shadow_mask.shape[0] * shadow_mask.shape[1])

        # 4. 밝기 분포
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)

        # 5. 구조적 패턴 (연결 성분)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        num_objects = len(regions)
        object_sizes = [region.area for region in regions if region.area > 50]
        avg_object_size = np.mean(object_sizes) if object_sizes else 0

        features = {
            'edges': edges,
            'lbp': lbp,
            'shadow_area_ratio': shadow_area,
            'brightness_histogram': hist,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'num_objects': num_objects,
            'avg_object_size': avg_object_size,
            'edge_density': np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        }

        logger.info("지형 특징 추출 완료")
        return features

    except Exception as e:
        logger.error(f"지형 특징 추출 실패: {e}")
        return {}

def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray, features1: Dict, features2: Dict) -> Dict:
    """두 이미지 간의 유사도 계산"""

    try:
        # 이미지 크기 맞추기
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_size = (min(w1, w2), min(h1, h2))

        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)

        # 그레이스케일 변환
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY) if len(img1_resized.shape) == 3 else img1_resized
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY) if len(img2_resized.shape) == 3 else img2_resized

        # 1. 구조적 유사도 (SSIM)
        ssim_score = ssim(gray1, gray2)

        # 2. 정규화 상호 상관 (Normalized Cross Correlation)
        correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0, 0]

        # 3. 히스토그램 유사도
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # 4. 특징 기반 유사도
        feature_similarity = calculate_feature_similarity(features1, features2)

        similarity_scores = {
            'ssim': float(ssim_score),
            'correlation': float(correlation),
            'histogram_correlation': float(hist_correlation),
            'feature_similarity': feature_similarity,
            'overall_similarity': float((ssim_score + correlation + hist_correlation + feature_similarity['overall']) / 4)
        }

        return similarity_scores

    except Exception as e:
        logger.error(f"유사도 계산 실패: {e}")
        return {}

def calculate_feature_similarity(features1: Dict, features2: Dict) -> Dict:
    """특징 기반 유사도 계산"""

    try:
        similarities = {}

        # 그림자 영역 비율 유사도
        shadow_diff = abs(features1.get('shadow_area_ratio', 0) - features2.get('shadow_area_ratio', 0))
        similarities['shadow_similarity'] = 1.0 - shadow_diff

        # 밝기 통계 유사도
        brightness_diff = abs(features1.get('brightness_mean', 0) - features2.get('brightness_mean', 0)) / 255.0
        similarities['brightness_similarity'] = 1.0 - brightness_diff

        # 엣지 밀도 유사도
        edge_diff = abs(features1.get('edge_density', 0) - features2.get('edge_density', 0))
        similarities['edge_similarity'] = 1.0 - edge_diff

        # 객체 수 유사도
        obj1 = features1.get('num_objects', 0)
        obj2 = features2.get('num_objects', 0)
        max_objects = max(obj1, obj2) if max(obj1, obj2) > 0 else 1
        similarities['object_similarity'] = 1.0 - abs(obj1 - obj2) / max_objects

        # 전체 유사도
        similarities['overall'] = np.mean(list(similarities.values()))

        return similarities

    except Exception as e:
        logger.error(f"특징 유사도 계산 실패: {e}")
        return {'overall': 0.0}

def apply_image_transformations(img: np.ndarray) -> Dict[str, np.ndarray]:
    """이미지 변환 적용 (회전, 반전)"""

    transformations = {
        'original': img,
        'rotate_90': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        'rotate_180': cv2.rotate(img, cv2.ROTATE_180),
        'rotate_270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        'flip_horizontal': cv2.flip(img, 1),
        'flip_vertical': cv2.flip(img, 0),
        'flip_both': cv2.flip(img, -1)
    }

    logger.info("이미지 변환 완료")
    return transformations

def analyze_terrain_similarity():
    """지형 유사도 분석 메인 함수"""

    logger.info("지형 유사도 분석 시작")
    print("="*70)
    print("BMP 이미지와 PH_annotation 이미지 간 지형 유사도 분석")
    print("="*70)

    # 이미지 파일 찾기
    original_bmps = find_original_bmp_images()
    annotation_files = find_annotation_images()

    if not original_bmps:
        logger.error("Original BMP 파일을 찾을 수 없습니다.")
        return False

    if not annotation_files:
        logger.error("PH_annotation 파일을 찾을 수 없습니다.")
        return False

    analysis_results = []

    # 각 Original BMP와 Annotation 이미지 비교
    for bmp_path in original_bmps:
        for annotation_path in annotation_files:
            result = compare_terrain_images(bmp_path, annotation_path)
            if result:
                analysis_results.append(result)

    # 종합 분석
    comprehensive_analysis = perform_comprehensive_terrain_analysis(analysis_results)

    # 결과 저장
    save_terrain_analysis_results(analysis_results, comprehensive_analysis)

    return True

def compare_terrain_images(bmp_path: str, annotation_path: str) -> Dict:
    """두 이미지의 지형 유사도 비교"""

    logger.info(f"지형 비교: {os.path.basename(bmp_path)} vs {os.path.basename(annotation_path)}")

    try:
        # 이미지 로드
        bmp_img = load_and_preprocess_image(bmp_path)
        annotation_img = load_and_preprocess_image(annotation_path)

        if bmp_img is None or annotation_img is None:
            return None

        # Annotation 이미지에서 빨간색 바운딩박스 제거
        annotation_clean = remove_annotation_elements(annotation_img.copy())

        # 지형 특징 추출
        bmp_features = extract_terrain_features(bmp_img)
        annotation_features = extract_terrain_features(annotation_clean)

        # 이미지 변환들 적용
        bmp_transformations = apply_image_transformations(bmp_img)

        best_similarity = 0.0
        best_transformation = 'original'
        all_similarities = {}

        # 각 변환에 대해 유사도 계산
        for transform_name, transformed_bmp in bmp_transformations.items():
            transformed_features = extract_terrain_features(transformed_bmp)
            similarity = calculate_image_similarity(
                transformed_bmp, annotation_clean,
                transformed_features, annotation_features
            )

            all_similarities[transform_name] = similarity

            if similarity.get('overall_similarity', 0) > best_similarity:
                best_similarity = similarity.get('overall_similarity', 0)
                best_transformation = transform_name

        result = {
            'bmp_file': bmp_path,
            'annotation_file': annotation_path,
            'bmp_filename': os.path.basename(bmp_path),
            'annotation_filename': os.path.basename(annotation_path),
            'best_transformation': best_transformation,
            'best_similarity_score': best_similarity,
            'all_transformations': all_similarities,
            'bmp_features': bmp_features,
            'annotation_features': annotation_features,
            'analysis_timestamp': datetime.now().isoformat()
        }

        print(f"\n📊 {os.path.basename(bmp_path)} vs {os.path.basename(annotation_path)}")
        print(f"   최고 유사도: {best_similarity:.3f} (변환: {best_transformation})")
        print(f"   SSIM: {all_similarities[best_transformation].get('ssim', 0):.3f}")
        print(f"   상관계수: {all_similarities[best_transformation].get('correlation', 0):.3f}")

        return result

    except Exception as e:
        logger.error(f"지형 비교 실패: {e}")
        return None

def perform_comprehensive_terrain_analysis(results: List[Dict]) -> Dict:
    """종합 지형 분석"""

    logger.info("종합 지형 분석 수행 중...")

    if not results:
        return {}

    # 최고 유사도 결과들
    best_matches = []
    transformation_stats = {}

    for result in results:
        best_similarity = result.get('best_similarity_score', 0)
        best_transformation = result.get('best_transformation', 'original')

        best_matches.append({
            'files': f"{result['bmp_filename']} vs {result['annotation_filename']}",
            'similarity': best_similarity,
            'transformation': best_transformation
        })

        # 변환 통계
        if best_transformation not in transformation_stats:
            transformation_stats[best_transformation] = 0
        transformation_stats[best_transformation] += 1

    # 유사도 순으로 정렬
    best_matches.sort(key=lambda x: x['similarity'], reverse=True)

    # 통계 계산
    similarities = [match['similarity'] for match in best_matches]

    comprehensive = {
        'total_comparisons': len(results),
        'best_matches': best_matches[:5],  # 상위 5개
        'similarity_statistics': {
            'max': float(np.max(similarities)),
            'min': float(np.min(similarities)),
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities))
        },
        'transformation_statistics': transformation_stats,
        'high_similarity_count': len([s for s in similarities if s > 0.7]),
        'medium_similarity_count': len([s for s in similarities if 0.4 <= s <= 0.7]),
        'low_similarity_count': len([s for s in similarities if s < 0.4]),
        'analysis_conclusion': generate_terrain_conclusion(similarities, transformation_stats)
    }

    return comprehensive

def generate_terrain_conclusion(similarities: List[float], transformation_stats: Dict) -> Dict:
    """지형 분석 결론 생성"""

    max_sim = max(similarities) if similarities else 0
    mean_sim = np.mean(similarities) if similarities else 0

    # 가장 많이 사용된 변환
    best_transform = max(transformation_stats.items(), key=lambda x: x[1])[0] if transformation_stats else 'original'

    # 유사도 기준 판단
    if max_sim > 0.8:
        similarity_level = "매우 높음"
        terrain_match = "동일한 지형일 가능성이 매우 높음"
    elif max_sim > 0.6:
        similarity_level = "높음"
        terrain_match = "유사한 지형일 가능성이 높음"
    elif max_sim > 0.4:
        similarity_level = "보통"
        terrain_match = "부분적으로 유사한 지형 특징 존재"
    else:
        similarity_level = "낮음"
        terrain_match = "서로 다른 지형일 가능성이 높음"

    conclusion = {
        'max_similarity': max_sim,
        'mean_similarity': mean_sim,
        'similarity_level': similarity_level,
        'terrain_match_assessment': terrain_match,
        'optimal_transformation': best_transform,
        'coordinate_mismatch_explanation': analyze_coordinate_mismatch(similarities, transformation_stats)
    }

    return conclusion

def analyze_coordinate_mismatch(similarities: List[float], transformation_stats: Dict) -> str:
    """좌표 불일치에 대한 설명"""

    max_sim = max(similarities) if similarities else 0

    if max_sim > 0.6:
        if 'rotate_180' in transformation_stats or 'flip_horizontal' in transformation_stats:
            return ("좌표는 다르지만 지형이 유사함. 이미지 방향/변환으로 인한 차이일 가능성. "
                   "실제로는 같은 지역의 다른 시점 또는 다른 각도에서 촬영된 데이터일 수 있음.")
        else:
            return ("좌표는 다르지만 지형이 유사함. 좌표 시스템 차이, 투영법 차이, "
                   "또는 시간 차이로 인한 좌표 오차일 가능성.")
    else:
        return ("좌표 차이와 지형 차이가 모두 존재. 실제로 서로 다른 지역의 데이터일 가능성이 높음.")

def save_terrain_analysis_results(results: List[Dict], comprehensive: Dict):
    """지형 분석 결과 저장"""

    # 출력 디렉토리 생성
    output_dir = Path("analysis_results/terrain_similarity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 상세 결과 저장
    detail_file = output_dir / "terrain_similarity_detail.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        # NumPy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만듦
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # NumPy 배열 제거 (용량이 큰 특징들)
            if 'bmp_features' in serializable_result:
                bmp_features = serializable_result['bmp_features'].copy()
                for key in ['edges', 'lbp', 'brightness_histogram']:
                    if key in bmp_features:
                        del bmp_features[key]
                serializable_result['bmp_features'] = bmp_features

            if 'annotation_features' in serializable_result:
                annotation_features = serializable_result['annotation_features'].copy()
                for key in ['edges', 'lbp', 'brightness_histogram']:
                    if key in annotation_features:
                        del annotation_features[key]
                serializable_result['annotation_features'] = annotation_features

            serializable_results.append(serializable_result)

        json.dump({
            'individual_results': serializable_results,
            'comprehensive_analysis': comprehensive,
            'analysis_timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    # 요약 보고서 생성
    report_file = output_dir / "TERRAIN_SIMILARITY_ANALYSIS_REPORT.md"
    generate_terrain_similarity_report(results, comprehensive, report_file)

    logger.info(f"지형 분석 결과 저장 완료: {output_dir}")
    print(f"\n📁 분석 결과 저장: {output_dir}/")

def generate_terrain_similarity_report(results: List[Dict], comprehensive: Dict, output_file: Path):
    """지형 유사도 분석 보고서 생성"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""# BMP와 PH_annotation 지형 유사도 분석 보고서
**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석자**: YMARX

## 🎯 **분석 목적**
- Original BMP 이미지와 PH_annotation 이미지 간 지형 유사도 분석
- 이미지 변환(180도 회전, 좌우 반전) 고려한 비교
- 지형 형상 및 그림자 패턴 분석으로 동일 지역 여부 판단

## 📊 **분석 개요**
- **총 비교 횟수**: {comprehensive.get('total_comparisons', 0)}
- **분석 방법**: 구조적 유사도(SSIM), 상관계수, 히스토그램 비교, 특징 기반 유사도
- **변환 고려**: 원본, 90도/180도/270도 회전, 수평/수직/양방향 반전

## 🏆 **최고 유사도 결과**

""")

        best_matches = comprehensive.get('best_matches', [])
        for i, match in enumerate(best_matches, 1):
            f.write(f"""### {i}. {match['files']}
- **유사도 점수**: {match['similarity']:.3f}
- **최적 변환**: {match['transformation']}
- **평가**: {'매우 유사' if match['similarity'] > 0.8 else '유사' if match['similarity'] > 0.6 else '보통' if match['similarity'] > 0.4 else '낮음'}

""")

        # 통계 정보
        stats = comprehensive.get('similarity_statistics', {})
        f.write(f"""## 📈 **유사도 통계**
- **최고 유사도**: {stats.get('max', 0):.3f}
- **평균 유사도**: {stats.get('mean', 0):.3f}
- **최저 유사도**: {stats.get('min', 0):.3f}
- **표준편차**: {stats.get('std', 0):.3f}
- **중간값**: {stats.get('median', 0):.3f}

### 유사도 분포
- **높은 유사도 (>0.7)**: {comprehensive.get('high_similarity_count', 0)}개
- **중간 유사도 (0.4-0.7)**: {comprehensive.get('medium_similarity_count', 0)}개
- **낮은 유사도 (<0.4)**: {comprehensive.get('low_similarity_count', 0)}개

## 🔄 **변환 통계**
""")

        transform_stats = comprehensive.get('transformation_statistics', {})
        for transform, count in transform_stats.items():
            korean_transform = {
                'original': '원본',
                'rotate_90': '90도 회전',
                'rotate_180': '180도 회전',
                'rotate_270': '270도 회전',
                'flip_horizontal': '수평 반전',
                'flip_vertical': '수직 반전',
                'flip_both': '양방향 반전'
            }.get(transform, transform)

            f.write(f"- **{korean_transform}**: {count}회 최적\n")

        # 결론
        conclusion = comprehensive.get('analysis_conclusion', {})
        f.write(f"""
## 🎯 **분석 결론**

### 지형 유사도 평가
- **최고 유사도**: {conclusion.get('max_similarity', 0):.3f}
- **평균 유사도**: {conclusion.get('mean_similarity', 0):.3f}
- **유사도 수준**: {conclusion.get('similarity_level', 'Unknown')}
- **지형 일치 평가**: {conclusion.get('terrain_match_assessment', 'Unknown')}
- **최적 변환**: {conclusion.get('optimal_transformation', 'Unknown')}

### 좌표 불일치 설명
{conclusion.get('coordinate_mismatch_explanation', 'No explanation available')}

## 💡 **종합 판단**

""")

        max_sim = conclusion.get('max_similarity', 0)
        if max_sim > 0.7:
            f.write("""### ✅ **지형 유사성 높음**
- BMP 이미지와 PH_annotation 이미지가 **유사한 지형 특징**을 보임
- 좌표상 차이에도 불구하고 **동일하거나 인접한 지역**일 가능성 높음
- 이미지 변환을 통해 유사도가 향상되어 **촬영 각도나 방향 차이** 존재

**가능한 시나리오**:
1. 동일 지역의 다른 시점 촬영 데이터
2. 좌표 시스템 또는 투영법 차이로 인한 좌표 오차
3. 시간 차이로 인한 좌표 드리프트

**권장사항**:
- 메타데이터 기반 촬영 시점 및 좌표계 확인
- 지형 특징점 기반 정밀 매칭 수행
- 실제 현장 위치 재확인""")

        elif max_sim > 0.4:
            f.write("""### ⚠️ **부분적 지형 유사성**
- BMP 이미지와 PH_annotation 이미지가 **부분적으로 유사한 특징**을 보임
- 인접 지역이거나 **유사한 해저 환경**일 가능성 존재
- 완전히 다른 지역이지만 **비슷한 지질학적 특성**을 가질 수 있음

**가능한 시나리오**:
1. 인접한 조사 구역의 데이터
2. 유사한 해저 지형을 가진 다른 지역
3. 데이터 처리 과정에서의 변형

**권장사항**:
- 지질학적 특성 기반 추가 분석
- 더 넓은 범위의 지형 비교
- 다른 독립적인 위치 확인 방법 활용""")

        else:
            f.write("""### ❌ **지형 유사성 낮음**
- BMP 이미지와 PH_annotation 이미지가 **서로 다른 지형 특징**을 보임
- **실제로 다른 지역**의 데이터일 가능성이 높음
- 좌표 차이와 지형 차이가 **모두 일치**하는 결과

**결론**:
- Original 데이터와 Annotation 데이터는 서로 다른 지역의 조사 결과
- 좌표 분석 결과(55km 거리 차이)와 일치
- 각각 독립적인 조사 지역으로 판단됨

**권장사항**:
- 데이터 출처 및 목적 재확인
- 각 데이터셋의 독립적 분석 수행
- 혼동 방지를 위한 명확한 라벨링""")

        f.write(f"""

## 📋 **분석 방법론**

### 이미지 전처리
1. **Annotation 요소 제거**: 빨간색 바운딩박스 및 숫자 제거
2. **이미지 정규화**: 크기 및 색상 정규화
3. **노이즈 제거**: 형태학적 연산 적용

### 특징 추출
1. **엣지 검출**: Canny 엣지 검출로 지형 경계 추출
2. **텍스처 분석**: LBP(Local Binary Pattern)로 표면 텍스처 분석
3. **그림자 분석**: 명도 기반 그림자 영역 검출
4. **구조적 패턴**: 연결 성분 분석으로 객체 분포 파악

### 유사도 측정
1. **구조적 유사도(SSIM)**: 이미지 구조 비교
2. **정규화 상호 상관**: 패턴 매칭 정확도
3. **히스토그램 상관**: 밝기 분포 유사성
4. **특징 기반 유사도**: 추출된 특징들의 종합 비교

### 변환 적용
- 원본, 90°/180°/270° 회전, 수평/수직/양방향 반전 총 7가지 변환 적용
- 각 변환에 대해 독립적으로 유사도 계산
- 최고 유사도를 보인 변환을 최적 변환으로 선정
""")

    logger.info(f"지형 유사도 분석 보고서 생성 완료: {output_file}")

def main():
    """메인 실행 함수"""

    print("BMP와 PH_annotation 지형 유사도 분석을 시작합니다...")

    try:
        success = analyze_terrain_similarity()

        if success:
            print(f"\n{'='*70}")
            print("🎉 지형 유사도 분석이 성공적으로 완료되었습니다!")
            print(f"📁 상세 결과: analysis_results/terrain_similarity_analysis/")
        else:
            print("\n❌ 지형 유사도 분석 중 오류가 발생했습니다.")
            return False

    except Exception as e:
        logger.error(f"분석 실행 실패: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)