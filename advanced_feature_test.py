#!/usr/bin/env python3
"""
고급 특징 추출 성능 테스트

Phase 2에서 개발한 고급 특징 추출기들의 성능을 평가합니다.
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_realistic_sonar_data(num_samples=30):
    """더 현실적인 시뮬레이션 소나 데이터 생성"""
    logger.info(f"현실적인 소나 데이터 생성: {num_samples}개 샘플")
    
    np.random.seed(42)
    images = []
    labels = []
    
    # 양성 샘플 (기물 포함)
    for i in range(num_samples // 2):
        # 다양한 해저면 배경
        terrain_type = np.random.choice(['sand', 'mud', 'rock'])
        
        if terrain_type == 'sand':
            base_intensity = 0.4 + 0.2 * np.random.random()
            texture_scale = 0.08
        elif terrain_type == 'mud':
            base_intensity = 0.2 + 0.1 * np.random.random()
            texture_scale = 0.04
        else:  # rock
            base_intensity = 0.3 + 0.3 * np.random.random()
            texture_scale = 0.12
        
        # 베이스 이미지 생성
        image = np.random.normal(base_intensity, texture_scale, (96, 96))
        
        # 텍스처 패턴 추가
        for _ in range(np.random.randint(3, 8)):
            blob_x = np.random.randint(10, 86)
            blob_y = np.random.randint(10, 86) 
            blob_size = np.random.randint(3, 12)
            
            y, x = np.ogrid[:96, :96]
            blob_mask = (x - blob_x)**2 + (y - blob_y)**2 <= blob_size**2
            
            intensity_change = np.random.uniform(-0.15, 0.15)
            image[blob_mask] += intensity_change
        
        # 기물 추가 - 다양한 형태
        object_type = np.random.choice(['circular', 'elongated', 'irregular'])
        
        center_x = np.random.randint(20, 76)
        center_y = np.random.randint(20, 76)
        
        # 변수 초기화
        radius = 10
        radius_x = 6
        radius_y = 10
        
        if object_type == 'circular':
            radius = np.random.randint(6, 15)
            y, x = np.ogrid[:96, :96]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
        elif object_type == 'elongated':
            radius_x = np.random.randint(4, 8)
            radius_y = np.random.randint(10, 20)
            y, x = np.ogrid[:96, :96]
            mask = ((x - center_x) / radius_x)**2 + ((y - center_y) / radius_y)**2 <= 1
            
        else:  # irregular
            base_radius = np.random.randint(6, 12)
            radius = base_radius  # 그림자 계산용
            y, x = np.ogrid[:96, :96]
            base_mask = (x - center_x)**2 + (y - center_y)**2 <= base_radius**2
            
            # 불규칙한 형태를 위한 변형
            noise_mask = np.random.random((96, 96)) > 0.3
            mask = base_mask & noise_mask
        
        # 기물 반사강도 적용
        object_intensity = 0.7 + 0.2 * np.random.random()
        image[mask] = object_intensity + np.random.normal(0, 0.05, np.sum(mask))
        
        # 음향 그림자 추가
        shadow_length = np.random.randint(12, 25)
        shadow_start_y = center_y + (radius if object_type == 'circular' else max(radius_x, radius_y)) + 2
        shadow_end_y = min(96, shadow_start_y + shadow_length)
        
        if shadow_end_y < 96:
            shadow_x_start = max(0, center_x - 6)
            shadow_x_end = min(96, center_x + 6)
            
            shadow_intensity = 0.05 + 0.05 * np.random.random()
            image[shadow_start_y:shadow_end_y, shadow_x_start:shadow_x_end] = shadow_intensity
        
        # 전체 노이즈 및 클리핑
        noise = np.random.normal(0, 0.02, image.shape)
        image += noise
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(1)
    
    # 음성 샘플 (배경만 - 더 복잡한 배경)
    for i in range(num_samples - num_samples // 2):
        terrain_type = np.random.choice(['sand', 'mud', 'rock', 'mixed'])
        
        if terrain_type == 'mixed':
            # 혼합 지형
            image = np.zeros((96, 96))
            
            # 여러 지형 패치를 합성
            for region in range(3):
                region_type = np.random.choice(['sand', 'mud', 'rock'])
                
                if region_type == 'sand':
                    base = 0.4 + 0.1 * np.random.random()
                    noise_scale = 0.06
                elif region_type == 'mud':
                    base = 0.2 + 0.08 * np.random.random()
                    noise_scale = 0.03
                else:  # rock
                    base = 0.35 + 0.2 * np.random.random()
                    noise_scale = 0.1
                
                # 영역 마스크
                region_center_x = np.random.randint(20, 76)
                region_center_y = np.random.randint(20, 76)
                region_radius = np.random.randint(15, 30)
                
                y, x = np.ogrid[:96, :96]
                region_mask = (x - region_center_x)**2 + (y - region_center_y)**2 <= region_radius**2
                
                region_image = np.random.normal(base, noise_scale, (96, 96))
                image[region_mask] = region_image[region_mask]
        
        else:
            # 단일 지형
            if terrain_type == 'sand':
                base_intensity = 0.4 + 0.15 * np.random.random()
                texture_noise = np.random.normal(0, 0.07, (96, 96))
            elif terrain_type == 'mud':
                base_intensity = 0.25 + 0.1 * np.random.random()
                texture_noise = np.random.normal(0, 0.04, (96, 96))
            else:  # rock
                base_intensity = 0.35 + 0.25 * np.random.random()
                texture_noise = np.random.normal(0, 0.11, (96, 96))
            
            image = np.full((96, 96), base_intensity) + texture_noise
        
        # 자연스러운 지형 변화 및 구조물 (기물이 아닌)
        for _ in range(np.random.randint(4, 10)):
            structure_x = np.random.randint(5, 91)
            structure_y = np.random.randint(5, 91)
            structure_size = np.random.randint(3, 8)
            
            y, x = np.ogrid[:96, :96]
            structure_mask = (x - structure_x)**2 + (y - structure_y)**2 <= structure_size**2
            
            intensity_variation = np.random.uniform(-0.08, 0.08)
            image[structure_mask] += intensity_variation
        
        # 전체 노이즈
        noise = np.random.normal(0, 0.015, image.shape)
        image += noise
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(0)
    
    # 데이터 섞기
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    images, labels = zip(*combined)
    
    return list(images), list(labels)


def test_advanced_hog_features(images, labels):
    """고급 HOG 특징 추출 테스트"""
    logger.info("고급 HOG 특징 추출 테스트...")
    
    try:
        from src.feature_extraction.hog_extractor import MultiScaleHOGExtractor
        
        extractor = MultiScaleHOGExtractor()
        
        start_time = time.time()
        features_list = []
        successful_extractions = 0
        
        for image in images:
            try:
                features = extractor.extract_combined_features(image)
                if len(features) > 0:
                    features_list.append(features)
                    successful_extractions += 1
                else:
                    logger.warning("빈 HOG 특징 벡터")
            except Exception as e:
                logger.warning(f"HOG 추출 실패: {e}")
        
        extraction_time = (time.time() - start_time) * 1000
        
        if features_list:
            feature_matrix = np.array(features_list)
            
            # 분류 성능 테스트
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                successful_labels = labels[:len(features_list)]
                rf = RandomForestClassifier(n_estimators=20, random_state=42)
                scores = cross_val_score(rf, feature_matrix, successful_labels, cv=3)
                
                return {
                    'name': 'MultiScale_HOG',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'classification_accuracy': np.mean(scores),
                    'accuracy_std': np.std(scores),
                    'description': '다중 스케일 HOG (3가지 스케일)'
                }
            except Exception as e:
                logger.warning(f"HOG 분류 평가 실패: {e}")
                return {
                    'name': 'MultiScale_HOG',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'description': '다중 스케일 HOG (3가지 스케일)',
                    'classification_error': str(e)
                }
        else:
            return {'name': 'MultiScale_HOG', 'error': '모든 HOG 추출 실패'}
            
    except ImportError as e:
        return {'name': 'MultiScale_HOG', 'error': f'모듈 임포트 실패: {e}'}
    except Exception as e:
        return {'name': 'MultiScale_HOG', 'error': f'HOG 테스트 실패: {e}'}


def test_advanced_lbp_features(images, labels):
    """고급 LBP 특징 추출 테스트"""
    logger.info("고급 LBP 특징 추출 테스트...")
    
    try:
        from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
        
        extractor = ComprehensiveLBPExtractor()
        
        start_time = time.time()
        features_list = []
        successful_extractions = 0
        
        for image in images:
            try:
                features = extractor.extract_comprehensive_features(image)
                if len(features) > 0:
                    features_list.append(features)
                    successful_extractions += 1
            except Exception as e:
                logger.warning(f"LBP 추출 실패: {e}")
        
        extraction_time = (time.time() - start_time) * 1000
        
        if features_list:
            feature_matrix = np.array(features_list)
            
            # 분류 성능 테스트
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                successful_labels = labels[:len(features_list)]
                rf = RandomForestClassifier(n_estimators=20, random_state=42)
                scores = cross_val_score(rf, feature_matrix, successful_labels, cv=3)
                
                return {
                    'name': 'Comprehensive_LBP',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'classification_accuracy': np.mean(scores),
                    'accuracy_std': np.std(scores),
                    'description': '종합 LBP (지형 적응형 + 회전 불변)'
                }
            except Exception as e:
                return {
                    'name': 'Comprehensive_LBP',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'description': '종합 LBP (지형 적응형 + 회전 불변)',
                    'classification_error': str(e)
                }
        else:
            return {'name': 'Comprehensive_LBP', 'error': '모든 LBP 추출 실패'}
            
    except ImportError as e:
        return {'name': 'Comprehensive_LBP', 'error': f'모듈 임포트 실패: {e}'}
    except Exception as e:
        return {'name': 'Comprehensive_LBP', 'error': f'LBP 테스트 실패: {e}'}


def test_advanced_gabor_features(images, labels):
    """고급 Gabor 특징 추출 테스트"""
    logger.info("고급 Gabor 특징 추출 테스트...")
    
    try:
        from src.feature_extraction.gabor_extractor import GaborFeatureExtractor
        
        extractor = GaborFeatureExtractor(n_frequencies=4, n_orientations=6)
        
        start_time = time.time()
        features_list = []
        successful_extractions = 0
        
        for image in images:
            try:
                features = extractor.extract_comprehensive_features(image)
                if len(features) > 0:
                    features_list.append(features)
                    successful_extractions += 1
            except Exception as e:
                logger.warning(f"Gabor 추출 실패: {e}")
        
        extraction_time = (time.time() - start_time) * 1000
        
        if features_list:
            feature_matrix = np.array(features_list)
            
            # 분류 성능 테스트
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                successful_labels = labels[:len(features_list)]
                rf = RandomForestClassifier(n_estimators=20, random_state=42)
                scores = cross_val_score(rf, feature_matrix, successful_labels, cv=3)
                
                return {
                    'name': 'Advanced_Gabor',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'classification_accuracy': np.mean(scores),
                    'accuracy_std': np.std(scores),
                    'description': '고급 Gabor 필터 뱅크 (4주파수 × 6방향)'
                }
            except Exception as e:
                return {
                    'name': 'Advanced_Gabor',
                    'success_rate': successful_extractions / len(images),
                    'feature_count': feature_matrix.shape[1],
                    'extraction_time_ms': extraction_time,
                    'description': '고급 Gabor 필터 뱅크 (4주파수 × 6방향)',
                    'classification_error': str(e)
                }
        else:
            return {'name': 'Advanced_Gabor', 'error': '모든 Gabor 추출 실패'}
            
    except ImportError as e:
        return {'name': 'Advanced_Gabor', 'error': f'모듈 임포트 실패: {e}'}
    except Exception as e:
        return {'name': 'Advanced_Gabor', 'error': f'Gabor 테스트 실패: {e}'}


def run_advanced_feature_evaluation():
    """고급 특징 추출 성능 평가 실행"""
    logger.info("=== 고급 특징 추출 성능 평가 시작 ===")
    
    output_dir = Path("data/results/advanced_feature_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 더 현실적인 테스트 데이터 생성
    images, labels = create_realistic_sonar_data(30)
    
    results = {}
    
    # 고급 특징 추출기들 테스트
    test_functions = [
        test_advanced_hog_features,
        test_advanced_lbp_features,
        test_advanced_gabor_features
    ]
    
    for test_func in test_functions:
        try:
            result = test_func(images, labels)
            results[result['name']] = result
            
            if 'error' not in result:
                logger.info(f"{result['name']} 완료 - "
                          f"성공률: {result.get('success_rate', 0)*100:.1f}%, "
                          f"특징수: {result.get('feature_count', 0)}, "
                          f"시간: {result.get('extraction_time_ms', 0):.1f}ms")
            else:
                logger.error(f"{result['name']} 실패: {result['error']}")
                
        except Exception as e:
            logger.error(f"테스트 함수 실행 실패: {e}")
    
    # 메타데이터 추가
    results['metadata'] = {
        'test_date': datetime.now().isoformat(),
        'num_samples': len(images),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels),
        'image_size': f"{images[0].shape[0]}x{images[0].shape[1]}",
        'test_type': 'advanced_features'
    }
    
    # 결과 저장
    with open(output_dir / 'advanced_feature_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 리포트 생성
    generate_advanced_report(results, output_dir)
    
    logger.info("=== 고급 특징 추출 성능 평가 완료 ===")
    
    return results


def generate_advanced_report(results, output_dir):
    """고급 특징 추출 리포트 생성"""
    with open(output_dir / 'advanced_feature_report.md', 'w', encoding='utf-8') as f:
        f.write("# 고급 특징 추출 성능 평가 리포트\n\n")
        f.write(f"**평가 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        metadata = results.get('metadata', {})
        f.write(f"**샘플 수**: {metadata.get('num_samples', 0)}개\n")
        f.write(f"**양성 샘플**: {metadata.get('positive_samples', 0)}개\n")
        f.write(f"**음성 샘플**: {metadata.get('negative_samples', 0)}개\n")
        f.write(f"**이미지 크기**: {metadata.get('image_size', 'N/A')}\n\n")
        
        f.write("## 📊 성능 요약\n\n")
        f.write("| 특징 추출기 | 성공률 | 특징 수 | 추출 시간(ms) | 분류 정확도 | 설명 |\n")
        f.write("|-------------|--------|---------|---------------|-------------|------|\n")
        
        for name, result in results.items():
            if name == 'metadata' or 'error' in result:
                continue
                
            success_rate = result.get('success_rate', 0) * 100
            feature_count = result.get('feature_count', 0)
            extraction_time = result.get('extraction_time_ms', 0)
            accuracy = result.get('classification_accuracy', 0)
            description = result.get('description', '')
            
            f.write(f"| {name} | {success_rate:.1f}% | {feature_count} | {extraction_time:.1f} | {accuracy:.3f} | {description} |\n")
        
        f.write("\n## 🔍 상세 분석\n\n")
        
        for name, result in results.items():
            if name == 'metadata':
                continue
                
            f.write(f"### {name}\n\n")
            
            if 'error' in result:
                f.write(f"❌ **실행 실패**: {result['error']}\n\n")
                continue
            
            f.write(f"- **성공률**: {result.get('success_rate', 0)*100:.1f}%\n")
            f.write(f"- **특징 차원**: {result.get('feature_count', 0):,}개\n")
            f.write(f"- **추출 시간**: {result.get('extraction_time_ms', 0):.2f}ms\n")
            
            if 'classification_accuracy' in result:
                f.write(f"- **분류 정확도**: {result['classification_accuracy']:.3f} ± {result.get('accuracy_std', 0):.3f}\n")
            elif 'classification_error' in result:
                f.write(f"- **분류 오류**: {result['classification_error']}\n")
            
            f.write(f"- **설명**: {result.get('description', '')}\n\n")
        
        # 결론
        f.write("## 💡 결론 및 권장사항\n\n")
        
        # 성공적으로 실행된 추출기들 중에서 최고 성능 찾기
        successful_results = {k: v for k, v in results.items() 
                            if k != 'metadata' and 'error' not in v and 'classification_accuracy' in v}
        
        if successful_results:
            best_accuracy = max(successful_results.items(), 
                              key=lambda x: x[1].get('classification_accuracy', 0))
            fastest = min(successful_results.items(), 
                         key=lambda x: x[1].get('extraction_time_ms', float('inf')))
            most_features = max(successful_results.items(), 
                              key=lambda x: x[1].get('feature_count', 0))
            
            f.write(f"🏆 **최고 분류 성능**: {best_accuracy[0]} (정확도: {best_accuracy[1]['classification_accuracy']:.3f})\n")
            f.write(f"⚡ **최고 속도**: {fastest[0]} ({fastest[1]['extraction_time_ms']:.1f}ms)\n")
            f.write(f"📊 **최다 특징**: {most_features[0]} ({most_features[1]['feature_count']:,}차원)\n\n")
        
        f.write("### 추천 사항\n\n")
        f.write("1. **실용성**: 빠른 추출과 안정적인 성능을 위해 성공률 90% 이상인 추출기 선택\n")
        f.write("2. **정확도**: 분류 성능이 0.8 이상인 추출기를 우선 고려\n")
        f.write("3. **특징 차원**: 과도한 차원은 과적합 위험이 있으므로 적절한 차원수 유지\n")
        f.write("4. **처리 시간**: 실시간 적용을 위해서는 100ms 이하의 추출 시간 권장\n\n")


def main():
    """메인 실행 함수"""
    results = run_advanced_feature_evaluation()
    
    output_dir = Path("data/results/advanced_feature_test")
    
    print(f"\n🎉 고급 특징 추출 성능 평가 완료!")
    print(f"📊 결과 디렉토리: {output_dir}")
    print(f"📈 리포트: {output_dir / 'advanced_feature_report.md'}")
    print(f"🔍 상세 결과: {output_dir / 'advanced_feature_results.json'}")
    
    print("\n📋 요약:")
    for name, result in results.items():
        if name == 'metadata':
            continue
        
        if 'error' in result:
            print(f"  ❌ {name}: 실행 실패 - {result['error']}")
        else:
            success_rate = result.get('success_rate', 0) * 100
            feature_count = result.get('feature_count', 0)
            extraction_time = result.get('extraction_time_ms', 0)
            accuracy = result.get('classification_accuracy', 0)
            
            print(f"  ✅ {name}: {feature_count}개 특징, {extraction_time:.1f}ms, "
                  f"성공률 {success_rate:.1f}%, 정확도 {accuracy:.3f}")


if __name__ == "__main__":
    main()