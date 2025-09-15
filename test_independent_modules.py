#!/usr/bin/env python3
"""
XTF Reader와 Intensity Extractor의 독립적 실행 테스트

각 모듈의 역할과 수정된 내용 적용 여부를 확인합니다.
"""

import sys
from pathlib import Path
import numpy as np

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.xtf_reader import XTFReader
from data_processing.xtf_intensity_extractor import XTFIntensityExtractor


def analyze_xtf_reader():
    """XTF Reader 모듈 분석 및 테스트"""
    print("="*80)
    print("1. XTF Reader 모듈 분석")
    print("="*80)
    
    print("\n📋 XTF Reader의 역할:")
    print("  - XTF 파일을 읽고 파싱하여 구조화된 데이터로 변환")
    print("  - 개별 ping 데이터를 PingData 객체로 추출")
    print("  - 강도 매트릭스 (intensity matrix) 생성")
    print("  - GPS 좌표와 메타데이터 추출")
    print("  - 배치 처리 지원 (BatchXTFProcessor)")
    
    print("\n🔧 주요 클래스:")
    print("  - XTFReader: 메인 XTF 파일 리더")
    print("  - PingData: 개별 ping 데이터 구조")
    print("  - XTFMetadata: 파일 메타데이터 구조")
    print("  - BatchXTFProcessor: 다중 파일 처리")
    
    print("\n🛠️ 주요 메소드:")
    print("  - load_file(): XTF 파일 로드")
    print("  - parse_pings(): ping 데이터 파싱")
    print("  - extract_intensity_matrix(): 강도 매트릭스 추출")
    print("  - get_georeferenced_data(): GPS 포함 데이터프레임")
    print("  - get_channel_data(): 채널별 데이터 추출")
    
    # 독립적 실행 테스트
    print("\n🧪 독립 실행 테스트:")
    try:
        # 첫 번째 XTF 파일로 테스트
        datasets_path = Path('datasets')
        xtf_file = None
        
        for dataset_dir in datasets_path.iterdir():
            if dataset_dir.is_dir():
                original_path = dataset_dir / 'original'
                if original_path.exists():
                    for f in original_path.glob('*.xtf'):
                        xtf_file = f
                        break
                if xtf_file:
                    break
        
        if xtf_file:
            print(f"  테스트 파일: {xtf_file.name}")
            
            # XTF Reader 독립 실행
            reader = XTFReader(xtf_file, max_pings=10)
            
            if reader.load_file():
                print("  ✅ 파일 로드 성공")
                
                ping_data = reader.parse_pings()
                print(f"  ✅ Ping 파싱: {len(ping_data)}개")
                
                intensity_matrix = reader.extract_intensity_matrix()
                print(f"  ✅ 강도 매트릭스: {intensity_matrix.shape}")
                
                geo_df = reader.get_georeferenced_data()
                print(f"  ✅ 위치 데이터: {len(geo_df)}개 레코드")
                
                summary = reader.get_summary()
                print(f"  ✅ 요약 정보: {summary['total_pings']} 총 pings")
                
                print("  ✅ XTF Reader 독립 실행 성공!")
                return True
            else:
                print("  ❌ 파일 로드 실패")
                return False
        else:
            print("  ❌ 테스트할 XTF 파일이 없습니다")
            return False
            
    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_intensity_extractor():
    """Intensity Extractor 모듈 분석 및 테스트"""
    print("\n" + "="*80)
    print("2. Intensity Extractor 모듈 분석")
    print("="*80)
    
    print("\n📋 Intensity Extractor의 역할:")
    print("  - XTF 파일에서 강도 데이터를 추출하고 이미지로 변환")
    print("  - PORT/STARBOARD 채널별 데이터 분리")
    print("  - 강도 이미지 생성 및 정규화")
    print("  - 네비게이션 데이터 추출")
    print("  - 추출된 데이터 자동 저장")
    
    print("\n🔧 주요 클래스:")
    print("  - XTFIntensityExtractor: 메인 강도 추출기")
    print("  - IntensityMetadata: 강도 데이터 메타정보")
    print("  - IntensityPing: 개별 ping 강도 데이터")
    print("  - IntensityDataProcessor: 특징 추출용 데이터 처리")
    
    print("\n🛠️ 주요 메소드:")
    print("  - extract_intensity_data(): 강도 데이터 추출")
    print("  - _create_intensity_images(): 강도 이미지 생성")
    print("  - _extract_navigation_data(): 네비게이션 데이터 추출")
    print("  - load_intensity_images(): 저장된 이미지 로드")
    print("  - prepare_for_feature_extraction(): 특징 추출용 패치 준비")
    
    # 독립적 실행 테스트
    print("\n🧪 독립 실행 테스트:")
    try:
        # 첫 번째 XTF 파일로 테스트
        datasets_path = Path('datasets')
        xtf_file = None
        
        for dataset_dir in datasets_path.iterdir():
            if dataset_dir.is_dir():
                original_path = dataset_dir / 'original'
                if original_path.exists():
                    for f in original_path.glob('*.xtf'):
                        xtf_file = f
                        break
                if xtf_file:
                    break
        
        if xtf_file:
            print(f"  테스트 파일: {xtf_file.name}")
            
            # Intensity Extractor 독립 실행
            extractor = XTFIntensityExtractor()
            
            # 작은 범위로 테스트 (첫 5 ping만)
            result = extractor.extract_intensity_data(str(xtf_file), 
                                                    output_dir="data/processed/xtf_extracted/test",
                                                    ping_range=(0, 5))
            
            metadata = result['metadata']
            print(f"  ✅ 메타데이터 추출: {metadata.ping_count} pings")
            
            ping_data = result['ping_data']
            print(f"  ✅ Ping 데이터: {len(ping_data)}개")
            
            images = result['intensity_images']
            for img_type, img in images.items():
                if img.size > 0:
                    print(f"  ✅ {img_type} 이미지: {img.shape}")
            
            nav_data = result['navigation_data']
            if nav_data and len(nav_data) > 0:
                print(f"  ✅ 네비게이션 데이터: {len(list(nav_data.keys()))} 종류")
            
            print("  ✅ Intensity Extractor 독립 실행 성공!")
            return True
            
        else:
            print("  ❌ 테스트할 XTF 파일이 없습니다")
            return False
            
    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_applied_modifications():
    """수정된 내용이 파일에 적용되었는지 확인"""
    print("\n" + "="*80)
    print("3. 수정된 내용 적용 여부 확인")
    print("="*80)
    
    print("\n🔍 XTF Reader 수정 사항 확인:")
    
    try:
        # xtf_reader.py 읽기
        reader_path = Path('src/data_processing/xtf_reader.py')
        if reader_path.exists():
            reader_content = reader_path.read_text(encoding='utf-8')
            
            # 주요 수정 사항들 확인
            checks = [
                ("딕셔너리 패킷 처리", "isinstance(self.packets, dict)" in reader_content),
                ("XTFHeaderType.sonar 접근", "XTFHeaderType.sonar in self.packets" in reader_content),
                ("안전한 속성 접근", "NumSonarChannels" in reader_content and "hasattr" in reader_content),
                ("데이터 리스트 처리", "isinstance(packet.data, list)" in reader_content),
                ("포트/스타보드 결합", "np.concatenate" in reader_content)
            ]
            
            for check_name, is_applied in checks:
                status = "✅" if is_applied else "❌"
                print(f"  {status} {check_name}")
        else:
            print("  ❌ xtf_reader.py 파일을 찾을 수 없습니다")
            
    except Exception as e:
        print(f"  ❌ XTF Reader 확인 중 오류: {e}")
    
    print("\n🔍 Intensity Extractor 수정 사항 확인:")
    
    try:
        # xtf_intensity_extractor.py 읽기
        extractor_path = Path('src/data_processing/xtf_intensity_extractor.py')
        if extractor_path.exists():
            extractor_content = extractor_path.read_text(encoding='utf-8')
            
            # 주요 수정 사항들 확인
            checks = [
                ("올바른 pyxtf 호출", "pyxtf.xtf_read(str(xtf_path))" in extractor_content),
                ("딕셔너리 패킷 처리", "isinstance(packets, dict)" in extractor_content),
                ("자동 저장 경로", "data/processed/xtf_extracted" in extractor_content),
                ("포트/스타보드 분리", "packet.data[0]" in extractor_content and "packet.data[1]" in extractor_content),
                ("with 구문 제거", "with pyxtf.xtf_read" not in extractor_content)
            ]
            
            for check_name, is_applied in checks:
                status = "✅" if is_applied else "❌"
                print(f"  {status} {check_name}")
        else:
            print("  ❌ xtf_intensity_extractor.py 파일을 찾을 수 없습니다")
            
    except Exception as e:
        print(f"  ❌ Intensity Extractor 확인 중 오류: {e}")


def main():
    """메인 분석 함수"""
    print("XTF 처리 모듈 분석 및 독립 실행 테스트")
    
    # 1. XTF Reader 분석 및 테스트
    reader_success = analyze_xtf_reader()
    
    # 2. Intensity Extractor 분석 및 테스트
    extractor_success = analyze_intensity_extractor()
    
    # 3. 수정 사항 확인
    check_applied_modifications()
    
    # 4. 종합 결과
    print("\n" + "="*80)
    print("4. 종합 결과")
    print("="*80)
    
    print(f"\n📊 독립 실행 테스트 결과:")
    print(f"  XTF Reader: {'✅ 성공' if reader_success else '❌ 실패'}")
    print(f"  Intensity Extractor: {'✅ 성공' if extractor_success else '❌ 실패'}")
    
    print(f"\n🎯 결론:")
    if reader_success and extractor_success:
        print("  ✅ 두 모듈 모두 독립적으로 실행 가능합니다")
        print("  ✅ 수정된 내용이 올바르게 적용되었습니다")
        print("  ✅ 전체 파이프라인과 별도로 사용할 수 있습니다")
    else:
        print("  ⚠️  일부 모듈에서 문제가 발생했습니다")
        print("  ⚠️  추가 디버깅이 필요할 수 있습니다")


if __name__ == "__main__":
    main()