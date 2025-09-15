#!/usr/bin/env python3
"""
실제 데이터를 이용한 기뢰 탐지 파이프라인

기존 샘플 데이터 대신 datasets 폴더의 실제 데이터를 사용하여
intensity data 패킷 추출, 기물 위치 매핑, 특징 추출을 수행합니다.
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import json
import logging
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 기존 모듈 임포트
try:
    from config.paths import PathManager
    from src.data_processing.xtf_reader import XTFReader
    from src.data_processing.sonar_data_processor import SonarDataProcessor
    from src.feature_extraction.basic_features import BasicFeatureExtractor
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealDataPipeline:
    """실제 데이터 파이프라인"""
    
    def __init__(self):
        self.path_manager = PathManager()
        self.datasets_path = self.path_manager.datasets
        self.output_path = self.path_manager.processed_data
        
        # GPS 데이터와 어노테이션 이미지 로드
        self.gps_data = self._load_gps_data()
        self.annotation_image = self._load_annotation_image()
        self.object_locations = self._extract_object_locations()
        
    def _load_gps_data(self) -> Optional[pd.DataFrame]:
        """GPS 데이터 로드"""
        gps_file = self.datasets_path / 'Location_MDGPS.xlsx'
        try:
            df = pd.read_excel(gps_file)
            logger.info(f"GPS 데이터 로드 완료: {df.shape[0]}개 좌표")
            return df
        except Exception as e:
            logger.error(f"GPS 데이터 로드 실패: {e}")
            return None
    
    def _load_annotation_image(self) -> Optional[np.ndarray]:
        """어노테이션 이미지 로드"""
        # BMP를 PNG로 변환했으므로 PNG 파일 사용
        annotation_file = self.datasets_path / 'PH_annotation.png'
        try:
            image = cv2.imread(str(annotation_file))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logger.info(f"어노테이션 이미지 로드 완료: {image.shape}")
                return image
            else:
                logger.error("어노테이션 이미지 로드 실패")
                return None
        except Exception as e:
            logger.error(f"어노테이션 이미지 로드 실패: {e}")
            return None
    
    def _extract_object_locations(self) -> List[Dict]:
        """어노테이션 이미지에서 객체 위치 추출"""
        if self.annotation_image is None:
            return []
        
        locations = []
        try:
            # 빨간색 박스 감지 (BGR에서 RGB로 변환했으므로 RGB 기준)
            # 빨간색 범위 설정
            red_lower = np.array([200, 0, 0])
            red_upper = np.array([255, 100, 100])
            
            # 빨간색 마스크 생성
            mask = cv2.inRange(self.annotation_image, red_lower, red_upper)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # 최소 크기 필터
                    locations.append({
                        'id': i + 1,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
            
            logger.info(f"감지된 객체 수: {len(locations)}")
            return locations
            
        except Exception as e:
            logger.error(f"객체 위치 추출 실패: {e}")
            return []
    
    def list_available_datasets(self) -> List[str]:
        """사용 가능한 데이터셋 목록"""
        datasets = []
        for dataset_dir in self.datasets_path.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                datasets.append(dataset_dir.name)
        return datasets
    
    def extract_intensity_data(self, dataset_name: str, data_type: str = 'original') -> Dict:
        """Intensity 데이터 추출"""
        logger.info(f"Intensity 데이터 추출 시작: {dataset_name} ({data_type})")
        
        dataset_path = self.datasets_path / dataset_name / data_type
        if not dataset_path.exists():
            logger.error(f"데이터셋 경로 없음: {dataset_path}")
            return {'error': f'Dataset path not found: {dataset_path}'}
        
        # XTF 파일 찾기
        xtf_files = list(dataset_path.glob('*.xtf'))
        if not xtf_files:
            logger.error(f"XTF 파일을 찾을 수 없음: {dataset_path}")
            return {'error': f'No XTF files found in {dataset_path}'}
        
        results = {}
        for xtf_file in xtf_files:
            try:
                logger.info(f"XTF 파일 처리 중: {xtf_file.name}")
                
                # XTF 리더 사용 (사용 가능한 경우)
                try:
                    reader = XTFReader(str(xtf_file))
                    intensity_data = reader.read_intensity_data()
                    
                    if intensity_data:
                        results[xtf_file.name] = {
                            'file_path': str(xtf_file),
                            'data_shape': intensity_data.shape if hasattr(intensity_data, 'shape') else 'unknown',
                            'extraction_time': datetime.now().isoformat(),
                            'success': True
                        }
                        
                        # 데이터 저장
                        output_file = self.output_path / f"{xtf_file.stem}_intensity.npy"
                        np.save(output_file, intensity_data)
                        results[xtf_file.name]['output_file'] = str(output_file)
                        
                    else:
                        results[xtf_file.name] = {'error': 'No intensity data extracted', 'success': False}
                        
                except Exception as e:
                    logger.warning(f"XTF 리더 실패, 기본 처리로 전환: {e}")
                    # 기본 처리 방법
                    results[xtf_file.name] = {
                        'error': f'XTF reader failed: {e}',
                        'file_path': str(xtf_file),
                        'success': False
                    }
                    
            except Exception as e:
                logger.error(f"파일 처리 실패 {xtf_file.name}: {e}")
                results[xtf_file.name] = {'error': str(e), 'success': False}
        
        return results
    
    def map_objects_to_pixels(self) -> List[Dict]:
        """GPS 좌표를 픽셀 좌표로 매핑"""
        if self.gps_data is None or not self.object_locations:
            logger.error("GPS 데이터 또는 객체 위치 정보 없음")
            return []
        
        mapped_objects = []
        
        # GPS 데이터와 감지된 객체 매칭
        for i, (_, gps_row) in enumerate(self.gps_data.iterrows()):
            if i < len(self.object_locations):
                obj_loc = self.object_locations[i]
                
                mapped_objects.append({
                    'point_id': gps_row['정점'],
                    'gps_lat': gps_row['위도'],
                    'gps_lon': gps_row['경도'],
                    'pixel_center': obj_loc['center'],
                    'pixel_bbox': obj_loc['bbox'],
                    'annotation_id': obj_loc['id']
                })
        
        logger.info(f"매핑된 객체 수: {len(mapped_objects)}")
        return mapped_objects
    
    def visualize_objects(self, save_path: Optional[Path] = None) -> None:
        """객체 위치 시각화"""
        if self.annotation_image is None or not self.object_locations:
            logger.error("시각화할 데이터 없음")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 20))
        ax.imshow(self.annotation_image)
        ax.set_title('기물 위치 및 바운딩 박스', fontsize=16)
        
        # 바운딩 박스 그리기
        for obj_loc in self.object_locations:
            x, y, w, h = obj_loc['bbox']
            center_x, center_y = obj_loc['center']
            
            # 바운딩 박스
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            
            # 중심점
            ax.plot(center_x, center_y, 'ro', markersize=8)
            ax.text(center_x + 10, center_y, f"ID:{obj_loc['id']}", 
                   fontsize=10, color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"시각화 결과 저장: {save_path}")
        
        plt.show()
    
    def extract_features_from_objects(self, intensity_data: np.ndarray) -> List[Dict]:
        """감지된 객체들로부터 특징 추출"""
        if not self.object_locations:
            logger.error("객체 위치 정보 없음")
            return []
        
        features = []
        try:
            feature_extractor = BasicFeatureExtractor()
            
            for obj_loc in self.object_locations:
                x, y, w, h = obj_loc['bbox']
                
                # 바운딩 박스 영역 추출 (안전한 범위 내에서)
                y1, y2 = max(0, y), min(intensity_data.shape[0], y + h)
                x1, x2 = max(0, x), min(intensity_data.shape[1], x + w)
                
                if y2 > y1 and x2 > x1:
                    roi = intensity_data[y1:y2, x1:x2]
                    
                    # 96x96으로 리사이즈 (기존 코드와 호환)
                    roi_resized = cv2.resize(roi, (96, 96))
                    
                    # 특징 추출
                    obj_features = feature_extractor.extract_features(roi_resized)
                    
                    features.append({
                        'object_id': obj_loc['id'],
                        'bbox': obj_loc['bbox'],
                        'features': obj_features,
                        'roi_shape': roi.shape
                    })
            
            logger.info(f"특징 추출 완료: {len(features)}개 객체")
            return features
            
        except Exception as e:
            logger.error(f"특징 추출 실패: {e}")
            return []
    
    def run_complete_pipeline(self) -> Dict:
        """전체 파이프라인 실행"""
        logger.info("=== 실제 데이터 파이프라인 시작 ===")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'datasets_processed': [],
            'object_mapping': [],
            'feature_extraction': [],
            'errors': []
        }
        
        try:
            # 1. 사용 가능한 데이터셋 목록
            datasets = self.list_available_datasets()
            logger.info(f"발견된 데이터셋: {len(datasets)}개")
            
            # 2. 각 데이터셋에서 intensity 데이터 추출
            for dataset in datasets:
                if 'Location_MDGPS' in dataset or 'PH_annotation' in dataset:
                    continue  # 메타데이터 파일들은 스킵
                
                logger.info(f"데이터셋 처리 중: {dataset}")
                
                # Original 데이터 처리
                original_results = self.extract_intensity_data(dataset, 'original')
                
                # Simulation 데이터 처리 (존재하는 경우)
                simulation_results = {}
                simulation_path = self.datasets_path / dataset / 'simulation'
                if simulation_path.exists():
                    for sim_dir in simulation_path.iterdir():
                        if sim_dir.is_dir():
                            sim_results = self.extract_intensity_data(dataset, f'simulation/{sim_dir.name}')
                            simulation_results[sim_dir.name] = sim_results
                
                dataset_result = {
                    'dataset_name': dataset,
                    'original': original_results,
                    'simulation': simulation_results,
                    'processing_time': datetime.now().isoformat()
                }
                
                results['datasets_processed'].append(dataset_result)
            
            # 3. 객체 위치 매핑
            object_mapping = self.map_objects_to_pixels()
            results['object_mapping'] = object_mapping
            
            # 4. 시각화 생성
            viz_path = self.output_path / 'object_visualization.png'
            self.visualize_objects(viz_path)
            results['visualization_path'] = str(viz_path)
            
            # 5. 결과 저장
            results_file = self.output_path / 'real_data_pipeline_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            results['results_file'] = str(results_file)
            results['end_time'] = datetime.now().isoformat()
            
            logger.info("=== 실제 데이터 파이프라인 완료 ===")
            
        except Exception as e:
            error_msg = f"파이프라인 실행 실패: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results


def main():
    """메인 실행 함수"""
    try:
        pipeline = RealDataPipeline()
        
        print("🚀 실제 데이터 파이프라인 시작")
        print(f"📂 데이터셋 경로: {pipeline.datasets_path}")
        print(f"📊 GPS 데이터: {pipeline.gps_data.shape[0] if pipeline.gps_data is not None else 0}개 좌표")
        print(f"🎯 감지된 객체: {len(pipeline.object_locations)}개")
        
        # 파이프라인 실행
        results = pipeline.run_complete_pipeline()
        
        # 결과 요약
        print(f"\n✅ 파이프라인 완료")
        print(f"📋 처리된 데이터셋: {len(results['datasets_processed'])}개")
        print(f"🎯 매핑된 객체: {len(results['object_mapping'])}개")
        
        if results['errors']:
            print(f"⚠️ 오류: {len(results['errors'])}개")
            for error in results['errors']:
                print(f"   - {error}")
        
        print(f"\n📁 결과 파일: {results.get('results_file', 'N/A')}")
        print(f"🖼️ 시각화: {results.get('visualization_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        logger.error(f"메인 실행 실패: {e}")


if __name__ == "__main__":
    main()