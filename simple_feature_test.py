#!/usr/bin/env python3
"""
간단한 특징 추출 테스트

현재 환경에서 실행 가능한 기본적인 특징 추출 성능 평가를 수행합니다.
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_sonar_images(num_samples=20):
    """간단한 시뮬레이션 소나 이미지 생성"""
    logger.info(f"시뮬레이션 소나 데이터 생성: {num_samples}개 샘플")
    
    np.random.seed(42)
    images = []
    labels = []
    
    # 양성 샘플 (기물 포함)
    for i in range(num_samples // 2):
        # 베이스 이미지
        image = np.random.normal(0.3, 0.1, (64, 64))
        
        # 기물 추가 (밝은 원형 영역)
        center_x, center_y = np.random.randint(15, 50), np.random.randint(15, 50)
        radius = np.random.randint(5, 12)
        
        y, x = np.ogrid[:64, :64]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = 0.8 + 0.2 * np.random.random()
        
        # 노이즈 추가
        image += np.random.normal(0, 0.05, image.shape)
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(1)
    
    # 음성 샘플 (배경만)
    for i in range(num_samples - num_samples // 2):
        image = np.random.normal(0.2, 0.08, (64, 64))
        image += np.random.normal(0, 0.03, image.shape)
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(0)
    
    return images, labels


def test_basic_features():
    """기본적인 특징 추출 테스트"""
    logger.info("=== 기본 특징 추출 테스트 시작 ===")
    
    # 테스트 데이터 생성
    images, labels = create_simple_sonar_images(20)
    
    results = {}
    output_dir = Path("data/results/simple_feature_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 기본 통계 특징
    logger.info("기본 통계 특징 추출 테스트...")
    start_time = time.time()
    
    statistical_features = []
    for image in images:
        features = [
            np.mean(image),          # 평균
            np.std(image),           # 표준편차  
            np.max(image),           # 최대값
            np.min(image),           # 최소값
            np.median(image),        # 중간값
            len(np.unique(image)),   # 고유값 개수
        ]
        
        # 히스토그램 특징
        hist, _ = np.histogram(image, bins=10, range=(0, 1))
        hist = hist / hist.sum()  # 정규화
        features.extend(hist)
        
        statistical_features.append(features)
    
    stat_time = (time.time() - start_time) * 1000
    
    results['statistical'] = {
        'feature_count': len(statistical_features[0]),
        'extraction_time_ms': stat_time,
        'success_rate': 1.0,
        'description': '기본 통계 + 히스토그램 특징'
    }
    
    # 2. 간단한 HOG 특징 (scikit-image 사용)
    logger.info("간단한 HOG 특징 추출 테스트...")
    
    try:
        from skimage.feature import hog
        from skimage import exposure
        
        start_time = time.time()
        hog_features = []
        
        for image in images:
            # 이미지 정규화
            normalized = exposure.equalize_adapthist(image)
            
            # HOG 특징 추출
            features = hog(
                normalized,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True
            )
            
            hog_features.append(features)
        
        hog_time = (time.time() - start_time) * 1000
        
        results['hog_simple'] = {
            'feature_count': len(hog_features[0]),
            'extraction_time_ms': hog_time,
            'success_rate': 1.0,
            'description': '기본 HOG 특징 (8방향, 8x8셀)'
        }
        
    except Exception as e:
        logger.error(f"HOG 특징 추출 실패: {e}")
        results['hog_simple'] = {'error': str(e)}
    
    # 3. 간단한 텍스처 특징
    logger.info("간단한 텍스처 특징 추출 테스트...")
    
    start_time = time.time()
    texture_features = []
    
    for image in images:
        # 기울기 기반 텍스처 특징
        grad_y, grad_x = np.gradient(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features = [
            np.mean(gradient_magnitude),     # 평균 기울기 크기
            np.std(gradient_magnitude),      # 기울기 크기 분산
            np.max(gradient_magnitude),      # 최대 기울기
            np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size,  # 강한 엣지 비율
        ]
        
        # 지역 분산
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(image, size=3)
        local_variance = uniform_filter(image**2, size=3) - local_mean**2
        
        features.extend([
            np.mean(local_variance),         # 평균 지역 분산
            np.std(local_variance),          # 지역 분산의 분산
        ])
        
        texture_features.append(features)
    
    texture_time = (time.time() - start_time) * 1000
    
    results['texture_simple'] = {
        'feature_count': len(texture_features[0]),
        'extraction_time_ms': texture_time,
        'success_rate': 1.0,
        'description': '기울기 + 지역 분산 기반 텍스처'
    }
    
    # 4. 특징 품질 평가
    logger.info("특징 품질 평가...")
    
    all_features = {
        'statistical': np.array(statistical_features),
        'texture_simple': np.array(texture_features)
    }
    
    if 'hog_simple' in results and 'error' not in results['hog_simple']:
        all_features['hog_simple'] = np.array(hog_features)
    
    # 각 특징별 분류 성능 간단 평가
    for feature_name, feature_matrix in all_features.items():
        try:
            # 간단한 분류 성능 측정
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            if len(feature_matrix) >= 10:  # 최소 샘플 수 확인
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                scores = cross_val_score(rf, feature_matrix, labels, cv=3)
                
                results[feature_name]['classification_accuracy'] = np.mean(scores)
                results[feature_name]['accuracy_std'] = np.std(scores)
            
        except Exception as e:
            logger.warning(f"{feature_name} 분류 평가 실패: {e}")
    
    # 5. 결과 저장
    results['metadata'] = {
        'test_date': datetime.now().isoformat(),
        'num_samples': len(images),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels),
        'image_size': f"{images[0].shape[0]}x{images[0].shape[1]}"
    }
    
    # JSON 저장
    with open(output_dir / 'simple_feature_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 마크다운 리포트 생성
    with open(output_dir / 'simple_feature_report.md', 'w', encoding='utf-8') as f:
        f.write("# 간단한 특징 추출 성능 테스트 결과\n\n")
        f.write(f"**테스트 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**샘플 수**: {len(images)}개 (양성: {sum(labels)}, 음성: {len(labels) - sum(labels)})\n\n")
        
        f.write("## 📊 특징별 성능 요약\n\n")
        f.write("| 특징 타입 | 특징 수 | 추출 시간(ms) | 분류 정확도 | 설명 |\n")
        f.write("|-----------|---------|---------------|-------------|------|\n")
        
        for name, result in results.items():
            if name == 'metadata' or 'error' in result:
                continue
                
            feature_count = result.get('feature_count', 0)
            extraction_time = result.get('extraction_time_ms', 0)
            accuracy = result.get('classification_accuracy', 0)
            description = result.get('description', '')
            
            f.write(f"| {name} | {feature_count} | {extraction_time:.2f} | {accuracy:.3f} | {description} |\n")
        
        f.write("\n## 🔍 상세 결과\n\n")
        
        for name, result in results.items():
            if name == 'metadata':
                continue
                
            f.write(f"### {name}\n\n")
            
            if 'error' in result:
                f.write(f"❌ **오류 발생**: {result['error']}\n\n")
                continue
            
            f.write(f"- **특징 개수**: {result.get('feature_count', 0)}개\n")
            f.write(f"- **추출 시간**: {result.get('extraction_time_ms', 0):.2f}ms\n")
            f.write(f"- **성공률**: {result.get('success_rate', 0)*100:.1f}%\n")
            
            if 'classification_accuracy' in result:
                f.write(f"- **분류 정확도**: {result['classification_accuracy']:.3f} ± {result.get('accuracy_std', 0):.3f}\n")
            
            f.write(f"- **설명**: {result.get('description', '')}\n\n")
        
        f.write("## 💡 결론\n\n")
        
        # 최고 성능 특징 찾기
        best_feature = None
        best_accuracy = 0
        
        for name, result in results.items():
            if 'classification_accuracy' in result:
                accuracy = result['classification_accuracy']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = name
        
        if best_feature:
            f.write(f"🏆 **최고 성능**: {best_feature} (정확도: {best_accuracy:.3f})\n")
        
        # 최고 속도
        fastest_feature = None
        fastest_time = float('inf')
        
        for name, result in results.items():
            if 'extraction_time_ms' in result:
                time_val = result['extraction_time_ms']
                if time_val < fastest_time:
                    fastest_time = time_val
                    fastest_feature = name
        
        if fastest_feature:
            f.write(f"⚡ **최고 속도**: {fastest_feature} ({fastest_time:.2f}ms)\n")
    
    logger.info("=== 기본 특징 추출 테스트 완료 ===")
    
    # 결과 출력
    print("\n🎉 간단한 특징 추출 테스트 완료!")
    print(f"📊 결과 디렉토리: {output_dir}")
    print(f"📈 리포트: {output_dir / 'simple_feature_report.md'}")
    print(f"🔍 상세 결과: {output_dir / 'simple_feature_results.json'}")
    
    print("\n📋 요약:")
    for name, result in results.items():
        if name == 'metadata' or 'error' in result:
            continue
        
        feature_count = result.get('feature_count', 0)
        extraction_time = result.get('extraction_time_ms', 0)
        accuracy = result.get('classification_accuracy', 0)
        
        print(f"  {name}: {feature_count}개 특징, {extraction_time:.1f}ms, 정확도 {accuracy:.3f}")


if __name__ == "__main__":
    test_basic_features()