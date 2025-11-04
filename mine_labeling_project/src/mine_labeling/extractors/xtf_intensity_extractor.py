"""
XTF 파일에서 강도 데이터 추출 및 이미지 변환 모듈

XTF 패킷에서 사이드스캔 소나의 강도 데이터를 추출하고,
분석 가능한 형태의 이미지로 변환합니다.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

try:
    import pyxtf
    PYXTF_AVAILABLE = True
except ImportError:
    PYXTF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IntensityMetadata:
    """강도 데이터 메타정보"""
    file_path: str
    ping_count: int
    channel_count: int
    frequency: float
    range_resolution: float
    sample_rate: float
    timestamp_range: Tuple[float, float]
    coordinate_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    intensity_range: Optional[Tuple[float, float]] = None


@dataclass  
class IntensityPing:
    """개별 ping의 강도 데이터"""
    ping_number: int
    timestamp: float
    latitude: float
    longitude: float
    heading: float
    port_intensity: np.ndarray
    starboard_intensity: np.ndarray
    port_range: np.ndarray
    starboard_range: np.ndarray


class XTFIntensityExtractor:
    """
    XTF 파일에서 강도 데이터를 추출하는 클래스
    
    사이드스캔 소나 데이터에서 음향 강도와 위치 정보를 추출하여
    분석 가능한 형태로 변환합니다.
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        """
        XTF 강도 추출기 초기화
        
        Args:
            max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self.max_memory_mb = max_memory_mb
        self.metadata_cache = {}
        
        if not PYXTF_AVAILABLE:
            logger.warning("pyxtf가 설치되지 않았습니다. XTF 파일 직접 처리가 제한됩니다.")
    
    def extract_intensity_data(self, xtf_path: str, 
                             output_dir: Optional[str] = None,
                             ping_range: Optional[Tuple[int, int]] = None) -> Dict[str, Union[np.ndarray, IntensityMetadata]]:
        """
        XTF 파일에서 강도 데이터 추출
        
        Args:
            xtf_path: XTF 파일 경로
            output_dir: 출력 디렉토리 (None이면 임시 저장하지 않음)
            ping_range: 추출할 ping 범위 (start, end)
            
        Returns:
            Dict: 강도 데이터 및 메타데이터
        """
        if not PYXTF_AVAILABLE:
            logger.error("pyxtf가 필요합니다. 'pip install pyxtf'로 설치해주세요.")
            return self._create_dummy_intensity_data(xtf_path)
        
        try:
            logger.info(f"XTF 파일에서 강도 데이터 추출 시작: {xtf_path}")
            
            # pyxtf를 사용하여 파일 읽기
            file_header, packets = pyxtf.xtf_read(str(xtf_path))
            
            # 메타데이터 추출
            metadata = self._extract_metadata(file_header, packets, xtf_path)
            
            # 강도 데이터 추출
            intensity_data = self._extract_ping_data(packets, ping_range)
            
            # 결과 구성
            result = {
                'metadata': metadata,
                'ping_data': intensity_data,
                'intensity_images': self._create_intensity_images(intensity_data),
                'navigation_data': self._extract_navigation_data(intensity_data)
            }
            
            # 자동으로 data/processed/xtf_extracted에 저장
            if not output_dir:
                output_dir = "data/processed/xtf_extracted"
            self._save_extracted_data(result, output_dir, xtf_path)
            
            logger.info(f"강도 데이터 추출 완료: {len(intensity_data)} pings")
            logger.info(f"데이터 저장 위치: {output_dir}")
            return result
                
        except Exception as e:
            logger.error(f"XTF 강도 데이터 추출 실패: {e}")
            return self._create_dummy_intensity_data(xtf_path)
    
    def _extract_metadata(self, file_header, packets, file_path: str) -> IntensityMetadata:
        """XTF 파일 메타데이터 추출"""
        try:
            # XTF 헤더에서 기본 정보 추출
            ping_count = 0
            timestamps = []
            coordinates = []
            frequency = 0.0
            range_resolution = 0.0
            sample_rate = 0.0

            # 파일 헤더에서 기본 정보 추출
            if file_header:
                frequency = getattr(file_header, 'frequency', 0.0)
                range_resolution = getattr(file_header, 'range_resolution', 0.0)
                sample_rate = getattr(file_header, 'sample_rate', 0.0)

            # 패킷에서 ping 정보 추출
            if isinstance(packets, dict):
                # pyxtf.XTFHeaderType.sonar 패킷 확인
                import pyxtf
                if pyxtf.XTFHeaderType.sonar in packets:
                    sonar_packets = packets[pyxtf.XTFHeaderType.sonar]
                    ping_count = len(sonar_packets)

                    for packet in sonar_packets[:100]:  # 처음 100개만 메타데이터용으로 확인
                        # 타임스탬프 수집
                        if hasattr(packet, 'TimeStamp'):
                            timestamps.append(packet.TimeStamp)
                        elif hasattr(packet, 'time'):
                            timestamps.append(packet.time)

                        # 좌표 수집
                        if hasattr(packet, 'SensorXcoordinate') and hasattr(packet, 'SensorYcoordinate'):
                            coordinates.append((packet.SensorXcoordinate, packet.SensorYcoordinate))
                        elif hasattr(packet, 'SensorX') and hasattr(packet, 'SensorY'):
                            coordinates.append((packet.SensorX, packet.SensorY))

                        # 주파수 정보 업데이트
                        if hasattr(packet, 'SonarFreq') and packet.SonarFreq > 0:
                            frequency = packet.SonarFreq
            elif isinstance(packets, list):
                ping_count = len(packets)
                for packet in packets[:100]:  # 처음 100개만 확인
                    if hasattr(packet, 'TimeStamp'):
                        timestamps.append(packet.TimeStamp)
                    if hasattr(packet, 'SensorXcoordinate') and hasattr(packet, 'SensorYcoordinate'):
                        coordinates.append((packet.SensorXcoordinate, packet.SensorYcoordinate))

            # 시간 범위
            time_range = (min(timestamps), max(timestamps)) if timestamps else (0.0, 0.0)

            # 좌표 범위
            coord_bounds = None
            if coordinates:
                lons, lats = zip(*coordinates)
                coord_bounds = {
                    'longitude': (min(lons), max(lons)),
                    'latitude': (min(lats), max(lats))
                }

            metadata = IntensityMetadata(
                file_path=file_path,
                ping_count=ping_count,
                channel_count=2,  # Port/Starboard
                frequency=frequency,
                range_resolution=range_resolution,
                sample_rate=sample_rate,
                timestamp_range=time_range,
                coordinate_bounds=coord_bounds
            )

            return metadata

        except Exception as e:
            logger.warning(f"메타데이터 추출 중 오류: {e}")
            return IntensityMetadata(
                file_path=file_path,
                ping_count=0,
                channel_count=2,
                frequency=0.0,
                range_resolution=0.0,
                sample_rate=0.0,
                timestamp_range=(0.0, 0.0)
            )
    
    def _extract_ping_data(self, packets, ping_range: Optional[Tuple[int, int]]) -> List[IntensityPing]:
        """개별 ping 데이터 추출"""
        ping_data = []
        ping_counter = 0
        
        try:
            # 소나 데이터 패킷만 필터링 (packets는 딕셔너리)
            sonar_packets = []
            if isinstance(packets, dict):
                # XTFHeaderType.sonar (키 0) 에서 소나 패킷 추출
                try:
                    import pyxtf
                    if pyxtf.XTFHeaderType.sonar in packets:
                        sonar_packets = packets[pyxtf.XTFHeaderType.sonar]
                except:
                    # 숫자 키로도 시도
                    if 0 in packets:
                        sonar_packets = packets[0]
            elif isinstance(packets, list):
                # 리스트인 경우
                for packet in packets:
                    if hasattr(packet, 'data') and packet.data is not None:
                        sonar_packets.append(packet)
            
            logger.info(f"소나 데이터 패킷 발견: {len(sonar_packets)}개")
            
            for packet in sonar_packets:
                # ping 범위 필터링
                if ping_range and (ping_counter < ping_range[0] or ping_counter >= ping_range[1]):
                    ping_counter += 1
                    continue
                try:
                    # 기본 정보 추출 - 다양한 속성명 시도
                    ping_num = getattr(packet, 'PingNumber', getattr(packet, 'ping_number', ping_counter))
                    
                    # 타임스탬프 추출
                    timestamp = 0.0
                    if hasattr(packet, 'TimeStamp'):
                        timestamp = packet.TimeStamp
                    elif hasattr(packet, 'time'):
                        timestamp = packet.time
                    
                    # 좌표 정보 추출
                    latitude = getattr(packet, 'SensorYcoordinate', getattr(packet, 'SensorY', 0.0))
                    longitude = getattr(packet, 'SensorXcoordinate', getattr(packet, 'SensorX', 0.0))
                    heading = getattr(packet, 'heading', getattr(packet, 'Heading', 0.0))
                    
                    # 강도 데이터 추출 - data는 리스트 형태 [PORT, STARBOARD]
                    port_intensity = np.array([], dtype=np.float32)
                    starboard_intensity = np.array([], dtype=np.float32)
                    
                    if hasattr(packet, 'data') and packet.data is not None:
                        if isinstance(packet.data, list) and len(packet.data) >= 2:
                            # PORT(0)과 STARBOARD(1) 채널 데이터 분리
                            if packet.data[0] is not None:
                                port_intensity = np.array(packet.data[0], dtype=np.float32)
                            if packet.data[1] is not None:
                                starboard_intensity = np.array(packet.data[1], dtype=np.float32)
                        else:
                            # 단일 데이터인 경우 절반으로 나누기
                            data_array = np.array(packet.data, dtype=np.float32)
                            mid_point = len(data_array) // 2
                            port_intensity = data_array[:mid_point]
                            starboard_intensity = data_array[mid_point:]
                    
                    # 거리 정보 (샘플 인덱스 기반)
                    port_range = np.arange(len(port_intensity), dtype=np.float32)
                    starboard_range = np.arange(len(starboard_intensity), dtype=np.float32)
                    
                    ping_obj = IntensityPing(
                        ping_number=ping_num,
                        timestamp=timestamp,
                        latitude=latitude,
                        longitude=longitude,
                        heading=heading if heading is not None else 0.0,  # 기본값 사용
                        port_intensity=port_intensity,
                        starboard_intensity=starboard_intensity,
                        port_range=port_range,
                        starboard_range=starboard_range
                    )
                    
                    ping_data.append(ping_obj)
                    
                except Exception as e:
                    logger.warning(f"Ping {ping_counter} 처리 중 오류: {e}")
                
                ping_counter += 1
                
                # 메모리 제한 확인
                if len(ping_data) > 0 and len(ping_data) % 100 == 0:
                    estimated_memory = self._estimate_memory_usage(ping_data)
                    if estimated_memory > self.max_memory_mb:
                        logger.warning(f"메모리 제한 도달 ({estimated_memory:.1f}MB), ping 추출 중단")
                        break
            
        except Exception as e:
            logger.error(f"Ping 데이터 추출 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return ping_data
    
    def _create_intensity_images(self, ping_data: List[IntensityPing]) -> Dict[str, np.ndarray]:
        """강도 데이터를 이미지로 변환"""
        if not ping_data:
            return {'combined': np.array([]), 'port': np.array([]), 'starboard': np.array([])}
        
        try:
            # 최대 샘플 수 결정
            max_port_samples = max(len(ping.port_intensity) for ping in ping_data)
            max_starboard_samples = max(len(ping.starboard_intensity) for ping in ping_data)
            max_samples = max(max_port_samples, max_starboard_samples)
            
            # 이미지 배열 초기화
            n_pings = len(ping_data)
            port_image = np.zeros((n_pings, max_samples), dtype=np.float32)
            starboard_image = np.zeros((n_pings, max_samples), dtype=np.float32)
            
            # 각 ping 데이터를 이미지로 변환
            for i, ping in enumerate(ping_data):
                # Port 채널
                port_len = len(ping.port_intensity)
                if port_len > 0:
                    port_image[i, :port_len] = ping.port_intensity
                
                # Starboard 채널  
                starboard_len = len(ping.starboard_intensity)
                if starboard_len > 0:
                    starboard_image[i, :starboard_len] = ping.starboard_intensity
            
            # 결합 이미지 생성 (Port + Starboard)
            combined_width = max_port_samples + max_starboard_samples
            combined_image = np.zeros((n_pings, combined_width), dtype=np.float32)
            combined_image[:, :max_port_samples] = port_image
            combined_image[:, max_port_samples:] = starboard_image
            
            # 정규화 (0-1 범위)
            for img in [port_image, starboard_image, combined_image]:
                if img.size > 0:
                    img_min, img_max = np.min(img), np.max(img)
                    if img_max > img_min:
                        img[:] = (img - img_min) / (img_max - img_min)
            
            return {
                'combined': combined_image,
                'port': port_image,
                'starboard': starboard_image
            }
            
        except Exception as e:
            logger.error(f"이미지 생성 중 오류: {e}")
            return {'combined': np.array([]), 'port': np.array([]), 'starboard': np.array([])}
    
    def _extract_navigation_data(self, ping_data: List[IntensityPing]) -> Dict[str, np.ndarray]:
        """네비게이션 데이터 추출"""
        if not ping_data:
            return {}
        
        try:
            timestamps = np.array([ping.timestamp for ping in ping_data])
            latitudes = np.array([ping.latitude for ping in ping_data])
            longitudes = np.array([ping.longitude for ping in ping_data])
            headings = np.array([ping.heading for ping in ping_data])
            
            return {
                'timestamps': timestamps,
                'latitudes': latitudes,
                'longitudes': longitudes,
                'headings': headings,
                'ping_numbers': np.array([ping.ping_number for ping in ping_data])
            }
            
        except Exception as e:
            logger.error(f"네비게이션 데이터 추출 중 오류: {e}")
            return {}
    
    def _estimate_memory_usage(self, ping_data: List[IntensityPing]) -> float:
        """메모리 사용량 추정 (MB)"""
        if not ping_data:
            return 0.0
        
        # 샘플 ping의 메모리 사용량 계산
        sample_ping = ping_data[0]
        sample_size = (
            sample_ping.port_intensity.nbytes + 
            sample_ping.starboard_intensity.nbytes +
            sample_ping.port_range.nbytes +
            sample_ping.starboard_range.nbytes
        )
        
        total_bytes = sample_size * len(ping_data)
        return total_bytes / (1024 * 1024)  # MB 변환
    
    def _save_extracted_data(self, data: Dict, output_dir: str, xtf_path: str):
        """추출된 데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(xtf_path).stem
        
        try:
            # 메타데이터 저장 (JSON)
            metadata_file = output_path / f"{base_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                # dataclass를 dict로 변환
                metadata_dict = {
                    'file_path': data['metadata'].file_path,
                    'ping_count': data['metadata'].ping_count,
                    'channel_count': data['metadata'].channel_count,
                    'frequency': data['metadata'].frequency,
                    'range_resolution': data['metadata'].range_resolution,
                    'sample_rate': data['metadata'].sample_rate,
                    'timestamp_range': data['metadata'].timestamp_range,
                    'coordinate_bounds': data['metadata'].coordinate_bounds,
                    'intensity_range': data['metadata'].intensity_range
                }
                json.dump(metadata_dict, f, indent=2)
            
            # 강도 이미지 저장 (NumPy)
            for img_type, img_array in data['intensity_images'].items():
                if img_array.size > 0:
                    img_file = output_path / f"{base_name}_{img_type}_intensity.npy"
                    np.save(img_file, img_array)
            
            # 네비게이션 데이터 저장 (NumPy)
            nav_data = data['navigation_data']
            if nav_data:
                nav_file = output_path / f"{base_name}_navigation.npz"
                np.savez(nav_file, **nav_data)
            
            # Raw ping 데이터 저장 (Pickle - 선택적)
            if len(data['ping_data']) < 1000:  # 큰 파일은 제외
                ping_file = output_path / f"{base_name}_ping_data.pkl"
                with open(ping_file, 'wb') as f:
                    pickle.dump(data['ping_data'], f)
            
            logger.info(f"추출된 데이터 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
    
    def _create_dummy_intensity_data(self, file_path: str) -> Dict:
        """Dummy 강도 데이터 생성 (pyxtf 없을 때)"""
        logger.warning("pyxtf 없이 더미 데이터 생성")
        
        # 가상의 강도 데이터 생성
        n_pings = 100
        samples_per_ping = 512
        
        combined_image = np.random.rand(n_pings, samples_per_ping * 2).astype(np.float32)
        port_image = combined_image[:, :samples_per_ping]
        starboard_image = combined_image[:, samples_per_ping:]
        
        # 가상 메타데이터
        metadata = IntensityMetadata(
            file_path=file_path,
            ping_count=n_pings,
            channel_count=2,
            frequency=100000.0,
            range_resolution=0.1,
            sample_rate=1000.0,
            timestamp_range=(0.0, 100.0)
        )
        
        return {
            'metadata': metadata,
            'ping_data': [],
            'intensity_images': {
                'combined': combined_image,
                'port': port_image,
                'starboard': starboard_image
            },
            'navigation_data': {
                'timestamps': np.linspace(0, 100, n_pings),
                'latitudes': np.random.uniform(35.0, 36.0, n_pings),
                'longitudes': np.random.uniform(129.0, 130.0, n_pings),
                'headings': np.random.uniform(0, 360, n_pings),
                'ping_numbers': np.arange(n_pings)
            }
        }
    
    def load_intensity_images(self, data_dir: str, base_name: str) -> Dict[str, np.ndarray]:
        """저장된 강도 이미지 로드"""
        data_path = Path(data_dir)
        images = {}
        
        for img_type in ['combined', 'port', 'starboard']:
            img_file = data_path / f"{base_name}_{img_type}_intensity.npy"
            if img_file.exists():
                try:
                    images[img_type] = np.load(img_file)
                    logger.info(f"강도 이미지 로드 완료: {img_type} - {images[img_type].shape}")
                except Exception as e:
                    logger.error(f"이미지 로드 실패 {img_type}: {e}")
        
        return images
    
    def create_visualization_image(self, intensity_image: np.ndarray, 
                                 colormap: str = 'gray',
                                 contrast_enhancement: bool = True) -> np.ndarray:
        """
        시각화를 위한 이미지 생성
        
        Args:
            intensity_image: 강도 이미지 (2D numpy array)
            colormap: 컬러맵 ('gray', 'jet', 'hot' 등)
            contrast_enhancement: 대비 향상 적용 여부
            
        Returns:
            np.ndarray: 시각화 이미지 (0-255 범위)
        """
        if intensity_image.size == 0:
            return np.array([])
        
        try:
            # 대비 향상
            if contrast_enhancement:
                # 히스토그램 평활화
                img_flat = intensity_image.flatten()
                hist, bins = np.histogram(img_flat, bins=256, range=(0, 1))
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                
                # 이미지에 적용
                enhanced = np.interp(img_flat, bins[:-1], cdf_normalized)
                enhanced_image = enhanced.reshape(intensity_image.shape)
            else:
                enhanced_image = intensity_image
            
            # 0-255 범위로 변환
            vis_image = (enhanced_image * 255).astype(np.uint8)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"시각화 이미지 생성 중 오류: {e}")
            return (intensity_image * 255).astype(np.uint8)


class IntensityDataProcessor:
    """
    추출된 강도 데이터를 특징 추출에 적합한 형태로 처리하는 클래스
    """
    
    def __init__(self):
        self.extractor = XTFIntensityExtractor()
    
    def prepare_for_feature_extraction(self, intensity_images: Dict[str, np.ndarray],
                                     patch_size: int = 64,
                                     overlap_ratio: float = 0.5) -> List[Dict]:
        """
        특징 추출을 위한 패치 데이터 준비
        
        Args:
            intensity_images: 강도 이미지 딕셔너리
            patch_size: 패치 크기
            overlap_ratio: 패치 간 겹침 비율
            
        Returns:
            List[Dict]: 패치 데이터 리스트
        """
        patches = []
        
        for img_type, img_array in intensity_images.items():
            if img_array.size == 0:
                continue
            
            # 패치 추출
            img_patches = self._extract_patches(img_array, patch_size, overlap_ratio)
            
            for i, patch in enumerate(img_patches):
                patch_info = {
                    'image_type': img_type,
                    'patch_id': f"{img_type}_{i:04d}",
                    'patch_data': patch,
                    'shape': patch.shape,
                    'mean_intensity': np.mean(patch),
                    'std_intensity': np.std(patch),
                    'dynamic_range': np.max(patch) - np.min(patch)
                }
                patches.append(patch_info)
        
        logger.info(f"특징 추출용 패치 준비 완료: {len(patches)}개")
        return patches
    
    def _extract_patches(self, image: np.ndarray, patch_size: int, 
                        overlap_ratio: float) -> List[np.ndarray]:
        """이미지에서 패치 추출"""
        patches = []
        h, w = image.shape
        
        step_size = int(patch_size * (1 - overlap_ratio))
        
        for i in range(0, h - patch_size + 1, step_size):
            for j in range(0, w - patch_size + 1, step_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return patches