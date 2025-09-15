#!/usr/bin/env python3
"""
세 개의 original XTF 파일 전체 테스트 스크립트

datasets 폴더에 있는 모든 original XTF 파일에서 핑 데이터를 추출하고
메타데이터와 함께 결과를 저장/확인합니다.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.xtf_reader import XTFReader
from data_processing.xtf_intensity_extractor import XTFIntensityExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_original_xtf_files():
    """datasets 폴더에서 original .xtf 파일만 찾기"""
    datasets_path = Path('datasets')
    
    xtf_files = []
    if datasets_path.exists():
        for dataset_dir in datasets_path.iterdir():
            if dataset_dir.is_dir():
                # original 폴더에서 .xtf 파일 찾기
                original_path = dataset_dir / 'original'
                if original_path.exists():
                    for xtf_file in original_path.glob('*.xtf'):
                        xtf_files.append(xtf_file)
    
    return sorted(xtf_files)


def test_xtf_reader(xtf_file_path):
    """XTF Reader로 데이터 추출 및 테스트"""
    print(f"\n{'='*80}")
    print(f"XTF Reader 테스트: {xtf_file_path.name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # XTF Reader 초기화 (처리 시간을 위해 첫 100 ping만)
        reader = XTFReader(xtf_file_path, max_pings=100)
        
        # 파일 로드
        if not reader.load_file():
            print("❌ 파일 로드 실패")
            return None
        
        print(f"✅ 파일 로드 성공 ({time.time() - start_time:.2f}초)")
        
        # 파일 요약 정보
        summary = reader.get_summary()
        print(f"\n📋 파일 요약:")
        print(f"  - 파일명: {summary['filename']}")
        print(f"  - 총 핑 수: {summary['total_pings']}")
        print(f"  - 소나 채널 수: {summary['num_sonar_channels']}")
        print(f"  - 주파수 정보: {summary['frequency_info']}")
        print(f"  - 좌표 범위: {summary['coordinate_bounds']}")
        
        # ping 데이터 파싱
        ping_start = time.time()
        ping_data = reader.parse_pings()
        print(f"✅ Ping 파싱 완료: {len(ping_data)}개 ({time.time() - ping_start:.2f}초)")
        
        if len(ping_data) > 0:
            # 첫 번째와 마지막 ping 정보
            first_ping = ping_data[0]
            last_ping = ping_data[-1]
            
            print(f"\n📊 데이터 샘플:")
            print(f"  첫 번째 ping:")
            print(f"    - ping_number: {first_ping.ping_number}")
            print(f"    - 좌표: ({first_ping.latitude:.6f}, {first_ping.longitude:.6f})")
            print(f"    - 데이터 크기: {len(first_ping.data)}")
            print(f"    - 데이터 범위: [{first_ping.data.min():.2f}, {first_ping.data.max():.2f}]")
            
            print(f"  마지막 ping:")
            print(f"    - ping_number: {last_ping.ping_number}")
            print(f"    - 좌표: ({last_ping.latitude:.6f}, {last_ping.longitude:.6f})")
            print(f"    - 데이터 크기: {len(last_ping.data)}")
            
            # 강도 매트릭스 추출
            matrix_start = time.time()
            intensity_matrix = reader.extract_intensity_matrix()
            print(f"✅ 강도 매트릭스: {intensity_matrix.shape} ({time.time() - matrix_start:.2f}초)")
            
            # 위치 정보 데이터프레임
            geo_df = reader.get_georeferenced_data()
            print(f"✅ 위치 정보: {len(geo_df)}개 레코드")
            
            # 통계 정보
            print(f"\n📈 통계:")
            print(f"  - 평균 강도: {intensity_matrix.mean():.2f}")
            print(f"  - 강도 범위: [{intensity_matrix.min():.2f}, {intensity_matrix.max():.2f}]")
            print(f"  - 위도 범위: [{geo_df['latitude'].min():.6f}, {geo_df['latitude'].max():.6f}]")
            print(f"  - 경도 범위: [{geo_df['longitude'].min():.6f}, {geo_df['longitude'].max():.6f}]")
            
            return {
                'reader': reader,
                'ping_data': ping_data,
                'intensity_matrix': intensity_matrix,
                'geo_df': geo_df,
                'processing_time': time.time() - start_time
            }
        else:
            print("❌ ping 데이터 파싱 실패")
            return None
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_intensity_extractor(xtf_file_path):
    """Intensity Extractor로 데이터 추출 및 테스트"""
    print(f"\n{'='*80}")
    print(f"Intensity Extractor 테스트: {xtf_file_path.name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        extractor = XTFIntensityExtractor()
        
        # 강도 데이터 추출 (첫 50 ping만)
        result = extractor.extract_intensity_data(str(xtf_file_path), ping_range=(0, 50))
        
        print(f"✅ 추출 완료 ({time.time() - start_time:.2f}초)")
        
        metadata = result['metadata']
        print(f"\n📋 메타데이터:")
        print(f"  - ping 수: {metadata.ping_count}")
        print(f"  - 채널 수: {metadata.channel_count}")
        print(f"  - 주파수: {metadata.frequency}")
        print(f"  - 시간 범위: {metadata.timestamp_range}")
        print(f"  - 좌표 경계: {metadata.coordinate_bounds}")
        
        ping_data = result['ping_data']
        print(f"  - 실제 추출된 ping 수: {len(ping_data)}")
        
        if len(ping_data) > 0:
            first_ping = ping_data[0]
            print(f"\n📊 첫 번째 ping:")
            print(f"  - Port 데이터: {len(first_ping.port_intensity)} 샘플")
            print(f"  - Starboard 데이터: {len(first_ping.starboard_intensity)} 샘플")
            print(f"  - 좌표: ({first_ping.latitude:.6f}, {first_ping.longitude:.6f})")
            
            # 강도 이미지 확인
            images = result['intensity_images']
            print(f"\n🖼️ 강도 이미지:")
            for img_type, img_array in images.items():
                if img_array.size > 0:
                    print(f"  - {img_type}: {img_array.shape}, 범위: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            # 네비게이션 데이터 확인
            nav_data = result['navigation_data']
            if nav_data:
                print(f"\n🧭 네비게이션 데이터:")
                for key, arr in nav_data.items():
                    if len(arr) > 0:
                        print(f"  - {key}: {len(arr)}개, 범위: [{np.min(arr):.6f}, {np.max(arr):.6f}]")
            
            return result
        else:
            print("❌ ping 데이터 추출 실패")
            return None
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_test_results(results, output_path):
    """테스트 결과를 파일로 저장"""
    summary = {
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': len(results),
        'successful_files': len([r for r in results if r['success']]),
        'files': []
    }
    
    for result in results:
        file_info = {
            'filename': result['filename'],
            'success': result['success'],
            'processing_time': result.get('processing_time', 0)
        }
        
        if result['success'] and 'reader_result' in result:
            reader_result = result['reader_result']
            file_info.update({
                'ping_count': len(reader_result['ping_data']),
                'matrix_shape': list(reader_result['intensity_matrix'].shape),
                'coordinate_range': {
                    'lat_min': float(reader_result['geo_df']['latitude'].min()),
                    'lat_max': float(reader_result['geo_df']['latitude'].max()),
                    'lon_min': float(reader_result['geo_df']['longitude'].min()),
                    'lon_max': float(reader_result['geo_df']['longitude'].max())
                }
            })
        
        summary['files'].append(file_info)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 테스트 결과 저장: {output_path}")


def main():
    """메인 테스트 함수"""
    print("="*100)
    print("세 개 original XTF 파일 전체 테스트")
    print("="*100)
    
    # original .xtf 파일 찾기
    xtf_files = find_original_xtf_files()
    
    if not xtf_files:
        print("❌ datasets 폴더에서 original .xtf 파일을 찾을 수 없습니다")
        return
    
    print(f"발견된 original .xtf 파일: {len(xtf_files)}개")
    for i, xtf_file in enumerate(xtf_files, 1):
        print(f"  {i}. {xtf_file}")
    
    results = []
    
    for i, xtf_file in enumerate(xtf_files, 1):
        print(f"\n\n{'='*100}")
        print(f"파일 {i}/{len(xtf_files)} 처리 중")
        print(f"{'='*100}")
        
        start_time = time.time()
        result = {
            'filename': xtf_file.name,
            'filepath': str(xtf_file),
            'success': False
        }
        
        # XTF Reader 테스트
        reader_result = test_xtf_reader(xtf_file)
        if reader_result:
            result['reader_result'] = reader_result
            result['success'] = True
            
            # Intensity Extractor 테스트
            extractor_result = test_intensity_extractor(xtf_file)
            if extractor_result:
                result['extractor_result'] = extractor_result
        
        result['processing_time'] = time.time() - start_time
        results.append(result)
        
        print(f"파일 {i} 처리 완료: {'✅ 성공' if result['success'] else '❌ 실패'} ({result['processing_time']:.2f}초)")
    
    # 전체 결과 요약
    print(f"\n\n{'='*100}")
    print("전체 테스트 결과 요약")
    print(f"{'='*100}")
    
    successful = len([r for r in results if r['success']])
    total_time = sum(r['processing_time'] for r in results)
    
    print(f"총 파일 수: {len(results)}")
    print(f"성공한 파일: {successful}")
    print(f"실패한 파일: {len(results) - successful}")
    print(f"총 처리 시간: {total_time:.2f}초")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['filename']} ({result['processing_time']:.2f}초)")
    
    # 결과 저장
    output_path = Path("data/processed/xtf_extracted/test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_test_results(results, output_path)
    
    print(f"\n📂 추출된 데이터 확인 위치:")
    print(f"  - 메인 디렉토리: data/processed/xtf_extracted/")
    print(f"  - 테스트 결과: {output_path}")
    
    if successful == len(results):
        print("\n🎉 모든 XTF 파일 처리 성공!")
    else:
        print(f"\n⚠️  {len(results) - successful}개 파일에서 오류가 발생했습니다.")


if __name__ == "__main__":
    main()