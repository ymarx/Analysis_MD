#!/usr/bin/env python3
"""
수정된 XTF 리더와 추출기 테스트 스크립트

datasets 폴더에서 .xtf 확장자 파일만 처리하여 
소나 데이터 추출이 올바르게 작동하는지 확인합니다.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.xtf_reader import XTFReader, BatchXTFProcessor
from data_processing.xtf_intensity_extractor import XTFIntensityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_xtf_files():
    """datasets 폴더에서 .xtf 파일만 찾기"""
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
    
    return xtf_files


def test_xtf_reader(xtf_file_path):
    """XTF Reader 테스트"""
    print(f"\n=== XTF Reader 테스트: {xtf_file_path.name} ===")
    
    try:
        reader = XTFReader(xtf_file_path)
        
        # 파일 로드
        if not reader.load_file():
            print("❌ 파일 로드 실패")
            return False
        
        print(f"✅ 파일 로드 성공")
        print(f"  총 패킷 수: {len(reader.packets)}")
        
        # ping 데이터 파싱
        ping_data = reader.parse_pings()
        print(f"  파싱된 ping 수: {len(ping_data)}")
        
        if len(ping_data) > 0:
            # 첫 번째 ping 데이터 검사
            first_ping = ping_data[0]
            print(f"  첫 번째 ping:")
            print(f"    - ping_number: {first_ping.ping_number}")
            print(f"    - 채널: {first_ping.channel}")
            print(f"    - 데이터 길이: {len(first_ping.data)}")
            print(f"    - 좌표: ({first_ping.latitude:.6f}, {first_ping.longitude:.6f})")
            print(f"    - 데이터 범위: [{first_ping.data.min():.2f}, {first_ping.data.max():.2f}]")
            
            # 채널별 강도 매트릭스 추출
            port_matrix = reader.extract_intensity_matrix(channel=0)
            starboard_matrix = reader.extract_intensity_matrix(channel=1)
            
            print(f"  Port 매트릭스: {port_matrix.shape}")
            print(f"  Starboard 매트릭스: {starboard_matrix.shape}")
            
            return True
        else:
            print("❌ ping 데이터 파싱 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intensity_extractor(xtf_file_path):
    """Intensity Extractor 테스트"""
    print(f"\n=== Intensity Extractor 테스트: {xtf_file_path.name} ===")
    
    try:
        extractor = XTFIntensityExtractor()
        
        # 강도 데이터 추출
        result = extractor.extract_intensity_data(str(xtf_file_path))
        
        metadata = result['metadata']
        print(f"✅ 메타데이터:")
        print(f"  - ping 수: {metadata.ping_count}")
        print(f"  - 채널 수: {metadata.channel_count}")
        print(f"  - 주파수: {metadata.frequency}")
        
        ping_data = result['ping_data']
        print(f"  - 추출된 ping 수: {len(ping_data)}")
        
        if len(ping_data) > 0:
            first_ping = ping_data[0]
            print(f"  첫 번째 ping:")
            print(f"    - Port 데이터: {len(first_ping.port_intensity)} 샘플")
            print(f"    - Starboard 데이터: {len(first_ping.starboard_intensity)} 샘플")
            print(f"    - 좌표: ({first_ping.latitude:.6f}, {first_ping.longitude:.6f})")
            
            # 강도 이미지 확인
            images = result['intensity_images']
            print(f"  강도 이미지:")
            for img_type, img_array in images.items():
                if img_array.size > 0:
                    print(f"    - {img_type}: {img_array.shape}")
            
            return True
        else:
            print("❌ ping 데이터 추출 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("=== 수정된 XTF 처리 시스템 테스트 ===")
    
    # .xtf 파일 찾기
    xtf_files = find_xtf_files()
    
    if not xtf_files:
        print("❌ datasets 폴더에서 .xtf 파일을 찾을 수 없습니다")
        return
    
    print(f"발견된 .xtf 파일: {len(xtf_files)}개")
    for xtf_file in xtf_files:
        print(f"  - {xtf_file}")
    
    # 첫 번째 파일로 테스트 (처리 시간 고려)
    test_file = xtf_files[0]
    print(f"\n테스트 대상: {test_file}")
    
    # XTF Reader 테스트
    reader_success = test_xtf_reader(test_file)
    
    # Intensity Extractor 테스트  
    extractor_success = test_intensity_extractor(test_file)
    
    # 결과 요약
    print(f"\n=== 테스트 결과 요약 ===")
    print(f"XTF Reader: {'✅ 성공' if reader_success else '❌ 실패'}")
    print(f"Intensity Extractor: {'✅ 성공' if extractor_success else '❌ 실패'}")
    
    if reader_success and extractor_success:
        print("🎉 모든 테스트 통과! XTF 처리 시스템이 정상 작동합니다.")
    else:
        print("⚠️  일부 테스트 실패. 추가 디버깅이 필요합니다.")


if __name__ == "__main__":
    main()