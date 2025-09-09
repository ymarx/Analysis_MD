"""
XTF 파일 읽기 및 파싱 모듈

사이드스캔 소나 XTF 파일을 읽고 처리하는 클래스들을 포함합니다.
기물 탐지를 위한 intensity 데이터와 위치 정보를 추출합니다.
"""

import pyxtf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PingData:
    """개별 ping 데이터를 저장하는 데이터클래스"""
    ping_number: int
    timestamp: datetime
    latitude: float
    longitude: float
    frequency: float
    channel: int
    data: np.ndarray
    range_samples: int
    ship_x: float
    ship_y: float
    

@dataclass 
class XTFMetadata:
    """XTF 파일 메타데이터를 저장하는 데이터클래스"""
    filename: str
    num_sonar_channels: int
    num_bathymetry_channels: int
    total_pings: int
    frequency_info: Dict[int, float]
    time_range: Tuple[datetime, datetime]
    coordinate_bounds: Dict[str, Tuple[float, float]]


class XTFReader:
    """
    XTF 파일을 읽고 사이드스캔 소나 데이터를 추출하는 클래스
    
    주요 기능:
    - XTF 파일 파싱
    - ping 데이터 추출
    - 위경도 정보 추출
    - intensity 매트릭스 변환
    - 배치 처리 지원
    """
    
    def __init__(self, filepath: Union[str, Path], max_pings: Optional[int] = None):
        """
        XTF Reader 초기화
        
        Args:
            filepath: XTF 파일 경로
            max_pings: 로드할 최대 ping 수 (메모리 효율성)
        """
        self.filepath = Path(filepath)
        self.max_pings = max_pings
        
        # 내부 데이터 저장소
        self.file_header = None
        self.packets = None
        self.ping_data: List[PingData] = []
        self.metadata: Optional[XTFMetadata] = None
        
        # 상태 플래그
        self._is_loaded = False
        self._is_parsed = False
        
        logger.info(f"XTF Reader 초기화 완료: {self.filepath}")
    
    def load_file(self) -> bool:
        """
        XTF 파일을 메모리에 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not self.filepath.exists():
                raise FileNotFoundError(f"XTF 파일을 찾을 수 없습니다: {self.filepath}")
            
            logger.info(f"XTF 파일 로드 시작: {self.filepath}")
            
            # pyxtf를 사용하여 파일 읽기
            self.file_header, self.packets = pyxtf.xtf_read(str(self.filepath), verbose=False)
            
            self._is_loaded = True
            logger.info("XTF 파일 로드 완료")
            
            # 메타데이터 생성
            self._create_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"XTF 파일 로드 실패: {e}")
            return False
    
    def _create_metadata(self) -> None:
        """파일 메타데이터 생성"""
        if not self._is_loaded:
            return
        
        sonar_packets = self.packets.get(pyxtf.XTFHeaderType.sonar, [])
        
        # 주파수 정보 수집
        frequency_info = {}
        timestamps = []
        coordinates = {'lat': [], 'lon': []}
        
        for packet in sonar_packets:
            if hasattr(packet, 'SonarFreq'):
                frequency_info[packet.channel_number] = packet.SonarFreq
            
            # 타임스탬프 수집
            if hasattr(packet, 'ping_time_year'):
                try:
                    timestamp = datetime(
                        packet.ping_time_year,
                        packet.ping_time_month, 
                        packet.ping_time_day,
                        packet.ping_time_hour,
                        packet.ping_time_minute,
                        packet.ping_time_second
                    )
                    timestamps.append(timestamp)
                except:
                    pass
            
            # 좌표 정보 수집
            if hasattr(packet, 'SensorXcoordinate') and hasattr(packet, 'SensorYcoordinate'):
                coordinates['lat'].append(packet.SensorYcoordinate)
                coordinates['lon'].append(packet.SensorXcoordinate)
        
        # 메타데이터 생성
        self.metadata = XTFMetadata(
            filename=self.filepath.name,
            num_sonar_channels=self.file_header.NumSonarChannels if self.file_header else 0,
            num_bathymetry_channels=self.file_header.NumBathymetryChannels if self.file_header else 0,
            total_pings=len(sonar_packets),
            frequency_info=frequency_info,
            time_range=(min(timestamps), max(timestamps)) if timestamps else (None, None),
            coordinate_bounds={
                'lat': (min(coordinates['lat']), max(coordinates['lat'])) if coordinates['lat'] else (None, None),
                'lon': (min(coordinates['lon']), max(coordinates['lon'])) if coordinates['lon'] else (None, None)
            }
        )
        
        logger.info(f"메타데이터 생성 완료 - 총 {self.metadata.total_pings} pings")
    
    def parse_pings(self) -> List[PingData]:
        """
        소나 ping 데이터를 파싱하여 PingData 객체 리스트로 변환
        
        Returns:
            List[PingData]: 파싱된 ping 데이터 리스트
        """
        if not self._is_loaded:
            logger.error("파일이 로드되지 않았습니다. load_file()을 먼저 호출하세요.")
            return []
        
        if pyxtf.XTFHeaderType.sonar not in self.packets:
            logger.error("소나 데이터를 찾을 수 없습니다.")
            return []
        
        sonar_packets = self.packets[pyxtf.XTFHeaderType.sonar]
        
        # ping 수 제한 적용
        if self.max_pings:
            sonar_packets = sonar_packets[:self.max_pings]
        
        logger.info(f"Ping 데이터 파싱 시작 - {len(sonar_packets)} pings")
        
        self.ping_data = []
        
        for i, packet in enumerate(sonar_packets):
            try:
                # 타임스탬프 생성
                timestamp = None
                if hasattr(packet, 'ping_time_year'):
                    try:
                        timestamp = datetime(
                            packet.ping_time_year,
                            packet.ping_time_month,
                            packet.ping_time_day, 
                            packet.ping_time_hour,
                            packet.ping_time_minute,
                            packet.ping_time_second
                        )
                    except:
                        timestamp = datetime.now()
                
                # PingData 객체 생성
                ping_data = PingData(
                    ping_number=i,
                    timestamp=timestamp,
                    latitude=getattr(packet, 'SensorYcoordinate', 0.0),
                    longitude=getattr(packet, 'SensorXcoordinate', 0.0),
                    frequency=getattr(packet, 'SonarFreq', 0.0),
                    channel=getattr(packet, 'channel_number', 0),
                    data=np.array(packet.data) if hasattr(packet, 'data') else np.array([]),
                    range_samples=len(packet.data) if hasattr(packet, 'data') else 0,
                    ship_x=getattr(packet, 'SensorXcoordinate', 0.0),
                    ship_y=getattr(packet, 'SensorYcoordinate', 0.0)
                )
                
                self.ping_data.append(ping_data)
                
            except Exception as e:
                logger.warning(f"Ping {i} 파싱 실패: {e}")
                continue
        
        self._is_parsed = True
        logger.info(f"Ping 데이터 파싱 완료 - {len(self.ping_data)} pings 성공")
        
        return self.ping_data
    
    def extract_intensity_matrix(self, channel: Optional[int] = None) -> np.ndarray:
        """
        intensity 데이터를 2D 매트릭스로 추출
        
        Args:
            channel: 추출할 채널 번호 (None이면 모든 채널)
            
        Returns:
            np.ndarray: intensity 매트릭스 [pings, samples]
        """
        if not self._is_parsed:
            logger.error("Ping 데이터가 파싱되지 않았습니다. parse_pings()을 먼저 호출하세요.")
            return np.array([])
        
        # 채널별 데이터 분리
        if channel is not None:
            filtered_pings = [ping for ping in self.ping_data if ping.channel == channel]
        else:
            filtered_pings = self.ping_data
        
        if not filtered_pings:
            logger.warning(f"채널 {channel}에 대한 데이터가 없습니다.")
            return np.array([])
        
        # 매트릭스 차원 결정
        max_samples = max(ping.range_samples for ping in filtered_pings)
        num_pings = len(filtered_pings)
        
        # intensity 매트릭스 초기화
        intensity_matrix = np.zeros((num_pings, max_samples), dtype=np.float32)
        
        # 데이터 채우기
        for i, ping in enumerate(filtered_pings):
            if ping.range_samples > 0:
                intensity_matrix[i, :ping.range_samples] = ping.data[:max_samples]
        
        logger.info(f"Intensity 매트릭스 추출 완료 - Shape: {intensity_matrix.shape}")
        
        return intensity_matrix
    
    def get_georeferenced_data(self) -> pd.DataFrame:
        """
        위경도 정보가 포함된 데이터프레임 반환
        
        Returns:
            pd.DataFrame: 위경도와 ping 정보가 포함된 데이터프레임
        """
        if not self._is_parsed:
            logger.error("Ping 데이터가 파싱되지 않았습니다.")
            return pd.DataFrame()
        
        # 데이터프레임 생성
        data = {
            'ping_number': [ping.ping_number for ping in self.ping_data],
            'timestamp': [ping.timestamp for ping in self.ping_data],
            'latitude': [ping.latitude for ping in self.ping_data],
            'longitude': [ping.longitude for ping in self.ping_data],
            'frequency': [ping.frequency for ping in self.ping_data],
            'channel': [ping.channel for ping in self.ping_data],
            'range_samples': [ping.range_samples for ping in self.ping_data],
            'ship_x': [ping.ship_x for ping in self.ping_data],
            'ship_y': [ping.ship_y for ping in self.ping_data]
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"위치 정보 데이터프레임 생성 완료 - {len(df)} rows")
        
        return df
    
    def get_channel_data(self, channel: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        특정 채널의 데이터 반환
        
        Args:
            channel: 채널 번호 (0: port, 1: starboard)
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: (intensity 매트릭스, 위치 정보)
        """
        # intensity 매트릭스 추출
        intensity_matrix = self.extract_intensity_matrix(channel)
        
        # 해당 채널의 위치 정보 추출
        georef_df = self.get_georeferenced_data()
        channel_df = georef_df[georef_df['channel'] == channel].copy()
        
        return intensity_matrix, channel_df
    
    def get_summary(self) -> Dict:
        """
        XTF 파일 요약 정보 반환
        
        Returns:
            Dict: 요약 정보 딕셔너리
        """
        if not self.metadata:
            return {}
        
        summary = {
            'filename': self.metadata.filename,
            'total_pings': self.metadata.total_pings,
            'num_sonar_channels': self.metadata.num_sonar_channels,
            'frequency_info': self.metadata.frequency_info,
            'coordinate_bounds': self.metadata.coordinate_bounds,
            'time_range': self.metadata.time_range,
            'is_loaded': self._is_loaded,
            'is_parsed': self._is_parsed
        }
        
        return summary
    
    def export_intensity_data(self, output_path: Union[str, Path], 
                            channel: Optional[int] = None, 
                            format: str = 'npy') -> bool:
        """
        intensity 데이터를 파일로 내보내기
        
        Args:
            output_path: 출력 파일 경로
            channel: 내보낼 채널 (None이면 모든 채널)
            format: 출력 포맷 ('npy', 'csv')
            
        Returns:
            bool: 내보내기 성공 여부
        """
        try:
            intensity_matrix = self.extract_intensity_matrix(channel)
            
            if intensity_matrix.size == 0:
                logger.error("내보낼 데이터가 없습니다.")
                return False
            
            output_path = Path(output_path)
            
            if format == 'npy':
                np.save(output_path.with_suffix('.npy'), intensity_matrix)
            elif format == 'csv':
                pd.DataFrame(intensity_matrix).to_csv(output_path.with_suffix('.csv'), index=False)
            else:
                logger.error(f"지원하지 않는 포맷: {format}")
                return False
            
            logger.info(f"데이터 내보내기 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 내보내기 실패: {e}")
            return False


class BatchXTFProcessor:
    """
    여러 XTF 파일을 배치로 처리하는 클래스
    """
    
    def __init__(self, file_paths: List[Union[str, Path]], max_pings_per_file: Optional[int] = None):
        """
        배치 프로세서 초기화
        
        Args:
            file_paths: 처리할 XTF 파일 경로 리스트
            max_pings_per_file: 파일당 최대 ping 수
        """
        self.file_paths = [Path(fp) for fp in file_paths]
        self.max_pings_per_file = max_pings_per_file
        self.readers: List[XTFReader] = []
        
        logger.info(f"배치 프로세서 초기화 - {len(self.file_paths)} 파일")
    
    def process_all(self) -> Dict[str, XTFReader]:
        """
        모든 파일을 처리하고 XTF Reader 딕셔너리 반환
        
        Returns:
            Dict[str, XTFReader]: 파일명을 키로 하는 XTF Reader 딕셔너리
        """
        results = {}
        
        for file_path in self.file_paths:
            try:
                logger.info(f"처리 중: {file_path.name}")
                
                reader = XTFReader(file_path, self.max_pings_per_file)
                
                if reader.load_file():
                    reader.parse_pings()
                    results[file_path.name] = reader
                    self.readers.append(reader)
                    logger.info(f"처리 완료: {file_path.name}")
                else:
                    logger.error(f"처리 실패: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"파일 처리 중 오류 ({file_path.name}): {e}")
                continue
        
        logger.info(f"배치 처리 완료 - {len(results)}/{len(self.file_paths)} 파일 성공")
        
        return results
    
    def get_combined_summary(self) -> Dict:
        """
        모든 파일의 종합 요약 정보 반환
        
        Returns:
            Dict: 종합 요약 정보
        """
        if not self.readers:
            return {}
        
        total_pings = sum(len(reader.ping_data) for reader in self.readers)
        all_frequencies = set()
        all_channels = set()
        
        for reader in self.readers:
            if reader.metadata:
                all_frequencies.update(reader.metadata.frequency_info.values())
                all_channels.update(reader.metadata.frequency_info.keys())
        
        summary = {
            'total_files': len(self.readers),
            'total_pings': total_pings,
            'unique_frequencies': list(all_frequencies),
            'channels': list(all_channels),
            'file_summaries': {reader.filepath.name: reader.get_summary() 
                             for reader in self.readers}
        }
        
        return summary