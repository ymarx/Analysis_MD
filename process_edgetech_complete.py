#!/usr/bin/env python3
"""
XTF Reader와 Intensity Extractor 연계 처리 파이프라인

Edgetech4205 시스템의 original 데이터 한 세트를 
XTF Reader -> Intensity Extractor 순으로 완전 처리합니다.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json
import matplotlib.pyplot as plt

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.xtf_reader import XTFReader
from data_processing.xtf_intensity_extractor import XTFIntensityExtractor, IntensityPing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XTFPipeline:
    """XTF Reader와 Intensity Extractor 연계 처리 파이프라인"""
    
    def __init__(self, output_dir: str = "data/processed/xtf_extracted/pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reader = None
        self.extractor = XTFIntensityExtractor()
        self.results = {}
    
    def process_xtf_file(self, xtf_path: Path, max_pings: int = None):
        """XTF 파일을 Reader -> Extractor 파이프라인으로 처리"""
        print(f"\n{'='*80}")
        print(f"XTF 파이프라인 처리: {xtf_path.name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Step 1: XTF Reader로 데이터 추출
        print(f"\n🔍 Step 1: XTF Reader로 기본 데이터 추출")
        reader_result = self._process_with_reader(xtf_path, max_pings)
        
        if not reader_result:
            print("❌ XTF Reader 처리 실패")
            return None
        
        # Step 2: Reader 결과를 Extractor 형태로 변환
        print(f"\n🔄 Step 2: Reader 데이터를 Extractor 입력으로 변환")
        extractor_input = self._convert_reader_to_extractor_input(reader_result)
        
        # Step 3: Intensity Extractor로 고급 처리
        print(f"\n🎨 Step 3: Intensity Extractor로 이미지 및 특징 데이터 생성")
        extractor_result = self._process_with_extractor_pipeline(extractor_input, xtf_path)
        
        # Step 4: 결합된 결과 생성
        print(f"\n📊 Step 4: 최종 결과 생성 및 저장")
        combined_result = self._combine_results(reader_result, extractor_result, xtf_path)
        
        processing_time = time.time() - start_time
        print(f"\n✅ 파이프라인 처리 완료 ({processing_time:.2f}초)")
        
        return combined_result
    
    def _process_with_reader(self, xtf_path: Path, max_pings: int = None):
        """XTF Reader로 데이터 추출"""
        try:
            self.reader = XTFReader(xtf_path, max_pings=max_pings)
            
            # 파일 로드
            if not self.reader.load_file():
                return None
            
            print(f"  ✅ 파일 로드 성공")
            
            # ping 데이터 파싱
            ping_data = self.reader.parse_pings()
            print(f"  ✅ Ping 파싱: {len(ping_data)}개")
            
            # 강도 매트릭스 추출
            intensity_matrix = self.reader.extract_intensity_matrix()
            print(f"  ✅ 강도 매트릭스: {intensity_matrix.shape}")
            
            # 위치 정보 추출
            geo_df = self.reader.get_georeferenced_data()
            print(f"  ✅ 위치 정보: {len(geo_df)}개 레코드")
            
            # 요약 정보
            summary = self.reader.get_summary()
            print(f"  ✅ 메타데이터: {summary['total_pings']} 전체 pings")
            
            return {
                'reader': self.reader,
                'ping_data': ping_data,
                'intensity_matrix': intensity_matrix,
                'geo_df': geo_df,
                'summary': summary
            }
            
        except Exception as e:
            print(f"  ❌ Reader 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_reader_to_extractor_input(self, reader_result):
        """Reader 결과를 Extractor 입력 형태로 변환"""
        try:
            ping_data = reader_result['ping_data']
            intensity_pings = []
            
            for i, ping in enumerate(ping_data):
                # Reader의 PingData를 Extractor의 IntensityPing으로 변환
                # PORT/STARBOARD 채널 분리 (ping.data는 이미 결합된 데이터)
                data_len = len(ping.data)
                mid_point = data_len // 2
                
                port_intensity = ping.data[:mid_point]
                starboard_intensity = ping.data[mid_point:]
                
                intensity_ping = IntensityPing(
                    ping_number=ping.ping_number,
                    timestamp=ping.timestamp.timestamp() if ping.timestamp else 0.0,
                    latitude=ping.latitude,
                    longitude=ping.longitude,
                    heading=0.0,  # Reader에서는 heading 정보가 없음
                    port_intensity=port_intensity,
                    starboard_intensity=starboard_intensity,
                    port_range=np.arange(len(port_intensity), dtype=np.float32),
                    starboard_range=np.arange(len(starboard_intensity), dtype=np.float32)
                )
                
                intensity_pings.append(intensity_ping)
            
            print(f"  ✅ 변환 완료: {len(intensity_pings)}개 IntensityPing 객체")
            return intensity_pings
            
        except Exception as e:
            print(f"  ❌ 데이터 변환 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _process_with_extractor_pipeline(self, intensity_pings, xtf_path: Path):
        """변환된 데이터로 Extractor 기능 수행"""
        try:
            # IntensityPing 객체들로부터 이미지 생성
            images = self.extractor._create_intensity_images(intensity_pings)
            print(f"  ✅ 강도 이미지 생성:")
            for img_type, img in images.items():
                if img.size > 0:
                    print(f"    - {img_type}: {img.shape}")
            
            # 네비게이션 데이터 추출
            nav_data = self.extractor._extract_navigation_data(intensity_pings)
            if nav_data:
                print(f"  ✅ 네비게이션 데이터: {len(nav_data)} 종류")
            
            # 메타데이터 생성
            metadata = self._create_pipeline_metadata(intensity_pings, xtf_path)
            print(f"  ✅ 메타데이터 생성 완료")
            
            return {
                'intensity_images': images,
                'navigation_data': nav_data,
                'metadata': metadata,
                'ping_data': intensity_pings
            }
            
        except Exception as e:
            print(f"  ❌ Extractor 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_pipeline_metadata(self, intensity_pings, xtf_path: Path):
        """파이프라인용 메타데이터 생성"""
        if not intensity_pings:
            return None
        
        # 시간 범위
        timestamps = [ping.timestamp for ping in intensity_pings if ping.timestamp > 0]
        time_range = (min(timestamps), max(timestamps)) if timestamps else (0.0, 0.0)
        
        # 좌표 범위
        lats = [ping.latitude for ping in intensity_pings if ping.latitude != 0]
        lons = [ping.longitude for ping in intensity_pings if ping.longitude != 0]
        
        coord_bounds = None
        if lats and lons:
            coord_bounds = {
                'latitude': (min(lats), max(lats)),
                'longitude': (min(lons), max(lons))
            }
        
        # 강도 범위 계산
        all_port = np.concatenate([ping.port_intensity for ping in intensity_pings if len(ping.port_intensity) > 0])
        all_starboard = np.concatenate([ping.starboard_intensity for ping in intensity_pings if len(ping.starboard_intensity) > 0])
        all_intensity = np.concatenate([all_port, all_starboard]) if len(all_port) > 0 and len(all_starboard) > 0 else np.array([])
        
        intensity_range = (float(all_intensity.min()), float(all_intensity.max())) if len(all_intensity) > 0 else (0.0, 0.0)
        
        return {
            'file_path': str(xtf_path),
            'ping_count': len(intensity_pings),
            'timestamp_range': time_range,
            'coordinate_bounds': coord_bounds,
            'intensity_range': intensity_range,
            'processing_method': 'XTF_Reader_to_Intensity_Extractor_Pipeline'
        }
    
    def _combine_results(self, reader_result, extractor_result, xtf_path: Path):
        """Reader와 Extractor 결과 결합"""
        if not reader_result or not extractor_result:
            return None
        
        combined = {
            'file_info': {
                'filename': xtf_path.name,
                'filepath': str(xtf_path),
                'processing_method': 'pipeline'
            },
            'reader_data': {
                'summary': reader_result['summary'],
                'ping_count': len(reader_result['ping_data']),
                'intensity_matrix_shape': list(reader_result['intensity_matrix'].shape),
                'coordinate_stats': self._get_coordinate_stats(reader_result['geo_df'])
            },
            'extractor_data': {
                'metadata': extractor_result['metadata'],
                'image_info': {img_type: list(img.shape) for img_type, img in extractor_result['intensity_images'].items() if img.size > 0},
                'navigation_data_keys': list(extractor_result['navigation_data'].keys()) if extractor_result['navigation_data'] else []
            },
            'images': extractor_result['intensity_images'],
            'navigation_data': extractor_result['navigation_data'],
            'raw_ping_data': extractor_result['ping_data']
        }
        
        # 결과 저장
        self._save_results(combined, xtf_path)
        
        return combined
    
    def _get_coordinate_stats(self, geo_df):
        """좌표 통계 계산"""
        if geo_df.empty:
            return {}
        
        return {
            'latitude_range': [float(geo_df['latitude'].min()), float(geo_df['latitude'].max())],
            'longitude_range': [float(geo_df['longitude'].min()), float(geo_df['longitude'].max())],
            'total_records': len(geo_df)
        }
    
    def _save_results(self, combined_result, xtf_path: Path):
        """결과를 파일로 저장"""
        base_name = xtf_path.stem
        
        try:
            # 메타데이터 저장 (JSON)
            metadata_file = self.output_dir / f"{base_name}_pipeline_metadata.json"
            metadata_dict = {
                'file_info': combined_result['file_info'],
                'reader_data': combined_result['reader_data'],
                'extractor_data': {
                    'metadata': combined_result['extractor_data']['metadata'],
                    'image_info': combined_result['extractor_data']['image_info'],
                    'navigation_data_keys': combined_result['extractor_data']['navigation_data_keys']
                }
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            # 강도 이미지 저장 (NumPy)
            for img_type, img_array in combined_result['images'].items():
                if img_array.size > 0:
                    img_file = self.output_dir / f"{base_name}_pipeline_{img_type}_intensity.npy"
                    np.save(img_file, img_array)
            
            # 네비게이션 데이터 저장 (NumPy)
            if combined_result['navigation_data']:
                nav_file = self.output_dir / f"{base_name}_pipeline_navigation.npz"
                np.savez(nav_file, **combined_result['navigation_data'])
            
            # 시각화 이미지 생성 및 저장
            self._create_visualization(combined_result, base_name)
            
            print(f"  ✅ 결과 저장 완료: {self.output_dir}")
            
        except Exception as e:
            print(f"  ❌ 저장 중 오류: {e}")
    
    def _create_visualization(self, combined_result, base_name):
        """결과 시각화 생성"""
        try:
            images = combined_result['images']
            
            # 3개 이미지 (combined, port, starboard) 시각화
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, (img_type, img_array) in enumerate(images.items()):
                if img_array.size > 0 and i < 3:
                    axes[i].imshow(img_array, cmap='gray', aspect='auto')
                    axes[i].set_title(f'{img_type.capitalize()} Channel')
                    axes[i].set_xlabel('Samples')
                    axes[i].set_ylabel('Pings')
            
            plt.tight_layout()
            viz_file = self.output_dir / f"{base_name}_pipeline_visualization.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 시각화 저장: {viz_file.name}")
            
        except Exception as e:
            print(f"  ⚠️  시각화 생성 실패: {e}")


def find_edgetech_original_file():
    """Edgetech4205 시스템의 original 파일 찾기"""
    datasets_path = Path('datasets')
    
    for dataset_dir in datasets_path.iterdir():
        if dataset_dir.is_dir() and 'Edgetech4205' in dataset_dir.name:
            original_path = dataset_dir / 'original'
            if original_path.exists():
                for xtf_file in original_path.glob('*.xtf'):
                    return xtf_file
    
    return None


def main():
    """메인 처리 함수"""
    print("="*100)
    print("XTF Reader → Intensity Extractor 연계 파이프라인 테스트")
    print("Edgetech4205 Original Data 완전 처리")
    print("="*100)
    
    # Edgetech4205 파일 찾기
    xtf_file = find_edgetech_original_file()
    
    if not xtf_file:
        print("❌ Edgetech4205 original 파일을 찾을 수 없습니다")
        return
    
    print(f"\n📁 처리 대상 파일:")
    print(f"  {xtf_file}")
    
    # 파이프라인 생성 및 처리
    pipeline = XTFPipeline()
    
    # 전체 데이터가 크므로 처음 200 ping만 처리
    result = pipeline.process_xtf_file(xtf_file, max_pings=200)
    
    if result:
        print(f"\n{'='*100}")
        print("파이프라인 처리 결과 요약")
        print(f"{'='*100}")
        
        reader_data = result['reader_data']
        extractor_data = result['extractor_data']
        
        print(f"\n📊 XTF Reader 결과:")
        print(f"  - 처리된 ping 수: {reader_data['ping_count']}")
        print(f"  - 강도 매트릭스: {reader_data['intensity_matrix_shape']}")
        print(f"  - 좌표 범위: {reader_data['coordinate_stats']}")
        
        print(f"\n🎨 Intensity Extractor 결과:")
        print(f"  - 생성된 이미지: {list(extractor_data['image_info'].keys())}")
        for img_type, shape in extractor_data['image_info'].items():
            print(f"    - {img_type}: {shape}")
        print(f"  - 네비게이션 데이터: {extractor_data['navigation_data_keys']}")
        
        print(f"\n💾 저장된 파일 위치:")
        output_dir = pipeline.output_dir
        print(f"  - 메인 디렉토리: {output_dir}")
        
        # 저장된 파일 목록
        saved_files = list(output_dir.glob(f"{xtf_file.stem}_pipeline_*"))
        for file in saved_files:
            print(f"    - {file.name}")
        
        print(f"\n🎉 XTF Reader → Intensity Extractor 파이프라인 처리 완료!")
        print(f"📂 결과 확인: {output_dir}")
        
    else:
        print("❌ 파이프라인 처리 실패")


if __name__ == "__main__":
    main()