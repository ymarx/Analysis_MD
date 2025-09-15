#!/usr/bin/env python3
"""
특징 추출 파이프라인

매핑된 객체 위치에서 다양한 특징을 추출하여
기물 탐지를 위한 머신러닝 학습 데이터를 준비합니다.
"""

import sys
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from skimage import measure
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 기존 모듈 임포트
from config.paths import path_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """특징 추출 파이프라인"""
    
    def __init__(self):
        self.datasets_path = path_manager.datasets
        self.output_path = path_manager.processed_data
        self.figures_path = path_manager.figures
        
        # 데이터 구조
        self.annotation_image = None
        self.coordinate_mappings = []
        self.extracted_features = []
        
        logger.info("특징 추출 파이프라인 초기화 완료")
    
    def load_annotation_image(self) -> bool:
        """어노테이션 이미지 로드"""
        annotation_file = self.datasets_path / 'PH_annotation.png'
        
        try:
            image = cv2.imread(str(annotation_file))
            if image is not None:
                # RGB로 변환하고 그레이스케일도 준비
                self.annotation_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.annotation_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logger.info(f"어노테이션 이미지 로드 완료: {self.annotation_image.shape}")
                return True
            else:
                logger.error("어노테이션 이미지 로드 실패")
                return False
                
        except Exception as e:
            logger.error(f"어노테이션 이미지 로드 실패: {e}")
            return False
    
    def load_coordinate_mappings(self) -> bool:
        """좌표 매핑 정보 로드"""
        mapping_file = self.output_path / 'coordinate_mappings' / 'pixel_gps_mappings.json'
        
        try:
            with open(mapping_file, 'r') as f:
                self.coordinate_mappings = json.load(f)
            
            logger.info(f"좌표 매핑 로드 완료: {len(self.coordinate_mappings)}개")
            return True
            
        except Exception as e:
            logger.error(f"좌표 매핑 로드 실패: {e}")
            return False
    
    def extract_roi_patches(self, patch_size: Tuple[int, int] = (64, 64)) -> List[Dict]:
        """관심 영역(ROI) 패치 추출"""
        patches = []
        
        try:
            for mapping in self.coordinate_mappings:
                bbox = mapping['bbox']
                
                # 바운딩 박스 중심에서 패치 추출
                center_x = mapping['pixel_x']
                center_y = mapping['pixel_y']
                
                # 패치 크기의 절반
                half_w, half_h = patch_size[0] // 2, patch_size[1] // 2
                
                # 패치 좌표 계산
                x1 = max(0, center_x - half_w)
                y1 = max(0, center_y - half_h)
                x2 = min(self.annotation_image.shape[1], center_x + half_w)
                y2 = min(self.annotation_image.shape[0], center_y + half_h)
                
                # 패치 추출 (RGB와 그레이스케일)
                rgb_patch = self.annotation_image[y1:y2, x1:x2]
                gray_patch = self.annotation_gray[y1:y2, x1:x2]
                
                # 패치를 지정된 크기로 리사이즈
                if rgb_patch.shape[:2] != patch_size:
                    rgb_patch = cv2.resize(rgb_patch, patch_size)
                    gray_patch = cv2.resize(gray_patch, patch_size)
                
                patch_data = {
                    'object_id': mapping['object_id'],
                    'gps_point_id': mapping['gps_point_id'],
                    'center_coords': (center_x, center_y),
                    'patch_coords': (x1, y1, x2, y2),
                    'rgb_patch': rgb_patch,
                    'gray_patch': gray_patch,
                    'patch_size': patch_size,
                    'latitude': mapping['latitude'],
                    'longitude': mapping['longitude']
                }
                
                patches.append(patch_data)
            
            logger.info(f"ROI 패치 추출 완료: {len(patches)}개 ({patch_size[0]}x{patch_size[1]})")
            return patches
            
        except Exception as e:
            logger.error(f"ROI 패치 추출 실패: {e}")
            return []
    
    def extract_hog_features(self, gray_patch: np.ndarray) -> np.ndarray:
        """HOG 특징 추출"""
        try:
            # HOG 파라미터
            orientations = 9
            pixels_per_cell = (8, 8)
            cells_per_block = (2, 2)
            
            features = hog(
                gray_patch,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                feature_vector=True
            )
            
            return features
            
        except Exception as e:
            logger.error(f"HOG 특징 추출 실패: {e}")
            return np.array([])
    
    def extract_lbp_features(self, gray_patch: np.ndarray) -> np.ndarray:
        """LBP 특징 추출"""
        try:
            # LBP 파라미터
            radius = 3
            n_points = 8 * radius
            method = 'uniform'
            
            lbp = local_binary_pattern(gray_patch, n_points, radius, method)
            
            # LBP 히스토그램 계산
            n_bins = n_points + 2  # uniform patterns + non-uniform
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            
            return hist
            
        except Exception as e:
            logger.error(f"LBP 특징 추출 실패: {e}")
            return np.array([])
    
    def extract_gabor_features(self, gray_patch: np.ndarray) -> np.ndarray:
        """Gabor 필터 특징 추출"""
        try:
            features = []
            
            # 다양한 주파수와 각도로 Gabor 필터 적용
            frequencies = [0.1, 0.3, 0.5]
            angles = [0, 45, 90, 135]
            
            for freq in frequencies:
                for angle in angles:
                    # 각도를 라디안으로 변환
                    theta = np.deg2rad(angle)
                    
                    # Gabor 필터 적용
                    filt_real, filt_imag = gabor(gray_patch, frequency=freq, theta=theta)
                    
                    # 응답의 통계적 특징 추출
                    features.extend([
                        np.mean(filt_real),
                        np.std(filt_real),
                        np.mean(filt_imag),
                        np.std(filt_imag)
                    ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Gabor 특징 추출 실패: {e}")
            return np.array([])
    
    def extract_texture_features(self, gray_patch: np.ndarray) -> np.ndarray:
        """텍스처 통계 특징 추출"""
        try:
            features = []
            
            # 기본 통계 특징
            features.extend([
                np.mean(gray_patch),
                np.std(gray_patch),
                np.var(gray_patch),
                np.min(gray_patch),
                np.max(gray_patch)
            ])
            
            # GLCM 기반 특징 (간단화된 버전)
            # 히스토그램 특징
            hist, _ = np.histogram(gray_patch, bins=16, range=(0, 256), density=True)
            features.extend([
                np.sum(hist * np.arange(16)),  # 평균 그레이 레벨
                np.sum(hist * (np.arange(16) ** 2)),  # 분산
                -np.sum(hist * np.log(hist + 1e-10))  # 엔트로피
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"텍스처 특징 추출 실패: {e}")
            return np.array([])
    
    def extract_geometric_features(self, patch_data: Dict) -> np.ndarray:
        """기하학적 특징 추출"""
        try:
            bbox = patch_data['patch_coords']
            gray_patch = patch_data['gray_patch']
            
            # 바운딩 박스 특징
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # 이진화된 이미지에서 모멘트 계산
            _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 연결된 컴포넌트 분석
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 컨투어 특징
                contour_area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = (perimeter ** 2) / (4 * np.pi * contour_area) if contour_area > 0 else 0
                
                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(largest_contour)
                extent = contour_area / (w * h) if w * h > 0 else 0
                
                features = [
                    width, height, area, aspect_ratio,
                    contour_area, perimeter, compactness, extent
                ]
            else:
                features = [width, height, area, aspect_ratio, 0, 0, 0, 0]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"기하학적 특징 추출 실패: {e}")
            return np.array([0] * 8)
    
    def extract_all_features(self, patches: List[Dict]) -> List[Dict]:
        """모든 특징을 종합적으로 추출"""
        feature_data = []
        
        for i, patch_data in enumerate(patches):
            try:
                gray_patch = patch_data['gray_patch']
                
                # 각 종류별 특징 추출
                hog_features = self.extract_hog_features(gray_patch)
                lbp_features = self.extract_lbp_features(gray_patch)
                gabor_features = self.extract_gabor_features(gray_patch)
                texture_features = self.extract_texture_features(gray_patch)
                geometric_features = self.extract_geometric_features(patch_data)
                
                # 특징 벡터 결합
                combined_features = np.concatenate([
                    hog_features,
                    lbp_features,
                    gabor_features,
                    texture_features,
                    geometric_features
                ])
                
                feature_record = {
                    'object_id': patch_data['object_id'],
                    'gps_point_id': patch_data['gps_point_id'],
                    'latitude': patch_data['latitude'],
                    'longitude': patch_data['longitude'],
                    'center_coords': patch_data['center_coords'],
                    'features': combined_features.tolist(),
                    'feature_dimensions': {
                        'hog': len(hog_features),
                        'lbp': len(lbp_features),
                        'gabor': len(gabor_features),
                        'texture': len(texture_features),
                        'geometric': len(geometric_features),
                        'total': len(combined_features)
                    }
                }
                
                feature_data.append(feature_record)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"특징 추출 진행: {i + 1}/{len(patches)}")
                
            except Exception as e:
                logger.error(f"객체 {patch_data['object_id']} 특징 추출 실패: {e}")
        
        logger.info(f"전체 특징 추출 완료: {len(feature_data)}개 객체")
        
        if feature_data:
            total_dims = feature_data[0]['feature_dimensions']['total']
            logger.info(f"특징 벡터 차원: {total_dims}차원")
        
        return feature_data
    
    def save_features(self, feature_data: List[Dict]) -> bool:
        """추출된 특징을 저장"""
        try:
            features_dir = self.output_path / 'features'
            features_dir.mkdir(exist_ok=True)
            
            # JSON 형태로 저장
            features_file = features_dir / 'extracted_features.json'
            with open(features_file, 'w') as f:
                json.dump(feature_data, f, indent=2, ensure_ascii=False)
            
            # CSV 형태로도 저장 (메타데이터 + 특징 일부)
            csv_data = []
            for record in feature_data:
                row = {
                    'object_id': record['object_id'],
                    'gps_point_id': record['gps_point_id'],
                    'latitude': record['latitude'],
                    'longitude': record['longitude'],
                    'center_x': record['center_coords'][0],
                    'center_y': record['center_coords'][1],
                    'feature_dim_total': record['feature_dimensions']['total'],
                    'feature_dim_hog': record['feature_dimensions']['hog'],
                    'feature_dim_lbp': record['feature_dimensions']['lbp'],
                    'feature_dim_gabor': record['feature_dimensions']['gabor'],
                    'feature_dim_texture': record['feature_dimensions']['texture'],
                    'feature_dim_geometric': record['feature_dimensions']['geometric']
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_file = features_dir / 'feature_metadata.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            # NumPy 배열로 특징 벡터만 저장 (ML 학습용)
            feature_matrix = np.array([record['features'] for record in feature_data])
            labels = np.array([record['object_id'] for record in feature_data])
            
            np.savez_compressed(
                features_dir / 'feature_matrix.npz',
                features=feature_matrix,
                labels=labels,
                metadata=csv_data
            )
            
            logger.info(f"특징 데이터 저장 완료: {features_dir}")
            logger.info(f"특징 매트릭스 크기: {feature_matrix.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"특징 데이터 저장 실패: {e}")
            return False
    
    def create_feature_visualization(self, feature_data: List[Dict], patches: List[Dict]) -> bool:
        """특징 추출 결과 시각화"""
        try:
            viz_dir = self.figures_path / 'feature_extraction'
            viz_dir.mkdir(exist_ok=True)
            
            # 1. 패치 샘플 시각화
            self._visualize_patch_samples(patches, viz_dir)
            
            # 2. 특징 차원 분석
            self._visualize_feature_dimensions(feature_data, viz_dir)
            
            # 3. 특징 분포 분석
            self._visualize_feature_distributions(feature_data, viz_dir)
            
            logger.info(f"특징 추출 시각화 생성 완료: {viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"특징 시각화 생성 실패: {e}")
            return False
    
    def _visualize_patch_samples(self, patches: List[Dict], output_dir: Path):
        """패치 샘플 시각화"""
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))
        fig.suptitle('ROI Patch Samples', fontsize=16)
        
        sample_indices = np.linspace(0, len(patches)-1, 24, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            row, col = i // 6, i % 6
            
            patch = patches[idx]
            axes[row, col].imshow(patch['rgb_patch'])
            axes[row, col].set_title(f"ID:{patch['object_id']} GPS:{patch['gps_point_id']}", fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'roi_patch_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_dimensions(self, feature_data: List[Dict], output_dir: Path):
        """특징 차원 분석 시각화"""
        if not feature_data:
            return
        
        dims = feature_data[0]['feature_dimensions']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 특징 타입별 차원 수
        feature_types = list(dims.keys())[:-1]  # 'total' 제외
        dim_counts = [dims[ft] for ft in feature_types]
        
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
        bars = ax1.bar(feature_types, dim_counts, color=colors)
        ax1.set_title('Feature Dimensions by Type', fontsize=14)
        ax1.set_ylabel('Number of Dimensions')
        ax1.tick_params(axis='x', rotation=45)
        
        # 각 막대에 값 표시
        for bar, count in zip(bars, dim_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # 파이 차트
        ax2.pie(dim_counts, labels=feature_types, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Feature Dimension Distribution', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_dimensions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_distributions(self, feature_data: List[Dict], output_dir: Path):
        """특징 분포 시각화"""
        if not feature_data:
            return
        
        # 특징 매트릭스 구성
        feature_matrix = np.array([record['features'] for record in feature_data])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 특징 평균값 분포
        feature_means = np.mean(feature_matrix, axis=0)
        axes[0, 0].hist(feature_means, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of Feature Means')
        axes[0, 0].set_xlabel('Feature Mean Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 특징 표준편차 분포
        feature_stds = np.std(feature_matrix, axis=0)
        axes[0, 1].hist(feature_stds, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Distribution of Feature Standard Deviations')
        axes[0, 1].set_xlabel('Feature Std Value')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 샘플별 특징 벡터 크기
        feature_norms = np.linalg.norm(feature_matrix, axis=1)
        axes[1, 0].hist(feature_norms, bins=20, alpha=0.7, color='red')
        axes[1, 0].set_title('Distribution of Feature Vector Norms')
        axes[1, 0].set_xlabel('L2 Norm')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. 특징 상관관계 히트맵 (샘플링)
        if feature_matrix.shape[1] > 50:
            # 너무 많은 특징이 있으면 샘플링
            sample_indices = np.random.choice(feature_matrix.shape[1], 50, replace=False)
            sample_features = feature_matrix[:, sample_indices]
        else:
            sample_features = feature_matrix
        
        correlation_matrix = np.corrcoef(sample_features.T)
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Feature Correlation Matrix (Sample)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_complete_pipeline(self, patch_size: Tuple[int, int] = (64, 64)) -> bool:
        """전체 특징 추출 파이프라인 실행"""
        logger.info("=== 특징 추출 파이프라인 실행 시작 ===")
        
        try:
            # 1. 데이터 로드
            logger.info("1. 필요 데이터 로드")
            if not self.load_annotation_image():
                return False
                
            if not self.load_coordinate_mappings():
                return False
            
            # 2. ROI 패치 추출
            logger.info("2. ROI 패치 추출")
            patches = self.extract_roi_patches(patch_size)
            if not patches:
                logger.error("패치 추출 실패")
                return False
            
            # 3. 특징 추출
            logger.info("3. 특징 추출")
            feature_data = self.extract_all_features(patches)
            if not feature_data:
                logger.error("특징 추출 실패")
                return False
            
            # 4. 특징 저장
            logger.info("4. 특징 데이터 저장")
            if not self.save_features(feature_data):
                return False
            
            # 5. 시각화 생성
            logger.info("5. 특징 추출 시각화")
            if not self.create_feature_visualization(feature_data, patches):
                return False
            
            logger.info("=== 특징 추출 파이프라인 실행 완료 ===")
            self._print_summary(feature_data)
            
            return True
            
        except Exception as e:
            logger.error(f"특징 추출 파이프라인 실행 실패: {e}")
            return False
    
    def _print_summary(self, feature_data: List[Dict]):
        """특징 추출 결과 요약 출력"""
        print("\n" + "="*60)
        print("특징 추출 파이프라인 결과 요약")
        print("="*60)
        
        if feature_data:
            dims = feature_data[0]['feature_dimensions']
            feature_matrix = np.array([record['features'] for record in feature_data])
            
            print(f"🎯 처리된 객체 수: {len(feature_data)}개")
            print(f"📏 총 특징 차원: {dims['total']}차원")
            print(f"   - HOG: {dims['hog']}차원")
            print(f"   - LBP: {dims['lbp']}차원")
            print(f"   - Gabor: {dims['gabor']}차원")
            print(f"   - Texture: {dims['texture']}차원")
            print(f"   - Geometric: {dims['geometric']}차원")
            
            print(f"📊 특징 매트릭스 크기: {feature_matrix.shape}")
            print(f"📈 특징 값 범위: [{feature_matrix.min():.3f}, {feature_matrix.max():.3f}]")
            print(f"📉 특징 평균: {feature_matrix.mean():.3f} ± {feature_matrix.std():.3f}")
        
        print(f"\n💾 출력 위치:")
        print(f"- 특징 데이터: {self.output_path / 'features'}")
        print(f"- 시각화: {self.figures_path / 'feature_extraction'}")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    pipeline = FeatureExtractionPipeline()
    
    success = pipeline.run_complete_pipeline(patch_size=(64, 64))
    
    if success:
        print("✅ 특징 추출 파이프라인이 성공적으로 완료되었습니다!")
        return 0
    else:
        print("❌ 특징 추출 파이프라인 실행 중 오류가 발생했습니다.")
        return 1


if __name__ == "__main__":
    exit(main())