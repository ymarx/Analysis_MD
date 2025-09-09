"""
좌표 매핑 시스템

사이드스캔 소나 데이터의 픽셀 좌표와 실제 위경도 좌표 간의 매핑을 처리합니다.
기물 위치 정보를 활용하여 레이블 마스크를 생성하고, 좌표계 변환을 수행합니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import pyproj
from scipy.spatial import distance_matrix
from scipy.interpolate import griddata
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TargetLocation:
    """기물 위치 정보를 저장하는 데이터클래스"""
    target_id: str
    latitude: float
    longitude: float
    utm_x: float
    utm_y: float
    description: Optional[str] = None


@dataclass
class CoordinateMapping:
    """좌표 매핑 결과를 저장하는 데이터클래스"""
    pixel_x: int
    pixel_y: int
    latitude: float
    longitude: float
    utm_x: float
    utm_y: float
    ping_number: int
    sample_number: int


class CoordinateTransformer:
    """
    좌표계 변환을 위한 유틸리티 클래스
    WGS84 <-> UTM 변환을 지원합니다.
    """
    
    def __init__(self, utm_zone: int = 52, hemisphere: str = 'north'):
        """
        좌표 변환기 초기화
        
        Args:
            utm_zone: UTM 존 번호 (한국은 52존)
            hemisphere: 반구 ('north' 또는 'south')
        """
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere
        
        # 좌표계 정의
        self.wgs84 = pyproj.CRS('EPSG:4326')  # WGS84
        self.utm = pyproj.CRS(f'EPSG:326{utm_zone}' if hemisphere == 'north' else f'EPSG:327{utm_zone}')
        
        # 변환기 생성
        self.wgs84_to_utm = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True)
        self.utm_to_wgs84 = pyproj.Transformer.from_crs(self.utm, self.wgs84, always_xy=True)
        
        logger.info(f"좌표 변환기 초기화 - UTM Zone {utm_zone}{hemisphere}")
    
    def wgs84_to_utm_coords(self, longitude: float, latitude: float) -> Tuple[float, float]:
        """
        WGS84 좌표를 UTM 좌표로 변환
        
        Args:
            longitude: 경도
            latitude: 위도
            
        Returns:
            Tuple[float, float]: (UTM_X, UTM_Y)
        """
        utm_x, utm_y = self.wgs84_to_utm.transform(longitude, latitude)
        return utm_x, utm_y
    
    def utm_to_wgs84_coords(self, utm_x: float, utm_y: float) -> Tuple[float, float]:
        """
        UTM 좌표를 WGS84 좌표로 변환
        
        Args:
            utm_x: UTM X 좌표
            utm_y: UTM Y 좌표
            
        Returns:
            Tuple[float, float]: (longitude, latitude)
        """
        longitude, latitude = self.utm_to_wgs84.transform(utm_x, utm_y)
        return longitude, latitude
    
    def batch_wgs84_to_utm(self, longitudes: np.ndarray, latitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 WGS84 -> UTM 변환
        
        Args:
            longitudes: 경도 배열
            latitudes: 위도 배열
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (UTM_X 배열, UTM_Y 배열)
        """
        utm_x, utm_y = self.wgs84_to_utm.transform(longitudes, latitudes)
        return utm_x, utm_y


class TargetLocationLoader:
    """
    기물 위치 정보를 로드하고 관리하는 클래스
    """
    
    def __init__(self, coordinate_transformer: CoordinateTransformer):
        """
        위치 로더 초기화
        
        Args:
            coordinate_transformer: 좌표 변환기
        """
        self.transformer = coordinate_transformer
        self.target_locations: List[TargetLocation] = []
        
        logger.info("기물 위치 로더 초기화 완료")
    
    def load_from_excel(self, filepath: Union[str, Path], 
                       lat_col: str = 'latitude', 
                       lon_col: str = 'longitude',
                       id_col: str = 'target_id',
                       desc_col: Optional[str] = None) -> bool:
        """
        엑셀 파일에서 기물 위치 정보 로드
        
        Args:
            filepath: 엑셀 파일 경로
            lat_col: 위도 컬럼명
            lon_col: 경도 컬럼명
            id_col: 기물 ID 컬럼명
            desc_col: 설명 컬럼명
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"파일을 찾을 수 없습니다: {filepath}")
                return False
            
            # 엑셀 파일 읽기
            df = pd.read_excel(filepath)
            
            logger.info(f"엑셀 파일 로드 완료: {len(df)} rows")
            logger.info(f"컬럼: {list(df.columns)}")
            
            # 필수 컬럼 확인
            required_cols = [lat_col, lon_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"필수 컬럼이 없습니다: {missing_cols}")
                return False
            
            # ID 컬럼이 없으면 자동 생성
            if id_col not in df.columns:
                df[id_col] = [f'target_{i:03d}' for i in range(len(df))]
                logger.info("기물 ID 자동 생성")
            
            # 기물 위치 정보 생성
            self.target_locations = []
            
            for idx, row in df.iterrows():
                try:
                    latitude = float(row[lat_col])
                    longitude = float(row[lon_col])
                    target_id = str(row[id_col])
                    
                    # UTM 좌표 변환
                    utm_x, utm_y = self.transformer.wgs84_to_utm_coords(longitude, latitude)
                    
                    # 설명 정보
                    description = str(row[desc_col]) if desc_col and desc_col in df.columns else None
                    
                    # TargetLocation 객체 생성
                    target_location = TargetLocation(
                        target_id=target_id,
                        latitude=latitude,
                        longitude=longitude,
                        utm_x=utm_x,
                        utm_y=utm_y,
                        description=description
                    )
                    
                    self.target_locations.append(target_location)
                    
                except Exception as e:
                    logger.warning(f"행 {idx} 처리 실패: {e}")
                    continue
            
            logger.info(f"기물 위치 로드 완료: {len(self.target_locations)} 위치")
            return True
            
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            return False
    
    def get_targets_in_bounds(self, min_lat: float, max_lat: float, 
                            min_lon: float, max_lon: float) -> List[TargetLocation]:
        """
        경계 내의 기물 위치 반환
        
        Args:
            min_lat, max_lat: 위도 범위
            min_lon, max_lon: 경도 범위
            
        Returns:
            List[TargetLocation]: 경계 내 기물 위치 리스트
        """
        targets_in_bounds = []
        
        for target in self.target_locations:
            if (min_lat <= target.latitude <= max_lat and 
                min_lon <= target.longitude <= max_lon):
                targets_in_bounds.append(target)
        
        logger.info(f"경계 내 기물: {len(targets_in_bounds)}/{len(self.target_locations)}")
        
        return targets_in_bounds
    
    def get_targets_dataframe(self) -> pd.DataFrame:
        """
        기물 위치를 데이터프레임으로 반환
        
        Returns:
            pd.DataFrame: 기물 위치 데이터프레임
        """
        if not self.target_locations:
            return pd.DataFrame()
        
        data = {
            'target_id': [t.target_id for t in self.target_locations],
            'latitude': [t.latitude for t in self.target_locations],
            'longitude': [t.longitude for t in self.target_locations],
            'utm_x': [t.utm_x for t in self.target_locations],
            'utm_y': [t.utm_y for t in self.target_locations],
            'description': [t.description for t in self.target_locations]
        }
        
        return pd.DataFrame(data)


class CoordinateMapper:
    """
    사이드스캔 소나 데이터의 픽셀 좌표와 실제 지리 좌표 간의 매핑을 처리하는 클래스
    """
    
    def __init__(self, coordinate_transformer: CoordinateTransformer):
        """
        좌표 매핑기 초기화
        
        Args:
            coordinate_transformer: 좌표 변환기
        """
        self.transformer = coordinate_transformer
        self.ping_coordinates: Optional[pd.DataFrame] = None
        self.intensity_shape: Optional[Tuple[int, int]] = None
        self.coordinate_grid: Optional[np.ndarray] = None
        
        logger.info("좌표 매핑기 초기화 완료")
    
    def set_sonar_data(self, ping_coordinates: pd.DataFrame, intensity_shape: Tuple[int, int]):
        """
        소나 데이터 설정
        
        Args:
            ping_coordinates: ping 좌표 정보 (DataFrame with lat, lon, ping_number)
            intensity_shape: intensity 매트릭스 모양 (pings, samples)
        """
        self.ping_coordinates = ping_coordinates.copy()
        self.intensity_shape = intensity_shape
        
        # UTM 좌표 추가
        utm_coords = self.transformer.batch_wgs84_to_utm(
            self.ping_coordinates['longitude'].values,
            self.ping_coordinates['latitude'].values
        )
        
        self.ping_coordinates['utm_x'] = utm_coords[0]
        self.ping_coordinates['utm_y'] = utm_coords[1]
        
        logger.info(f"소나 데이터 설정 완료 - Shape: {intensity_shape}, Pings: {len(ping_coordinates)}")
        
        # 좌표 그리드 생성
        self._create_coordinate_grid()
    
    def _create_coordinate_grid(self):
        """
        intensity 매트릭스의 각 픽셀에 대한 좌표 그리드 생성
        """
        if self.ping_coordinates is None or self.intensity_shape is None:
            logger.error("소나 데이터가 설정되지 않았습니다.")
            return
        
        num_pings, num_samples = self.intensity_shape
        
        # 각 ping의 UTM 좌표
        ping_utm_x = self.ping_coordinates['utm_x'].values
        ping_utm_y = self.ping_coordinates['utm_y'].values
        
        # 좌표 그리드 초기화 [pings, samples, 2] (utm_x, utm_y)
        self.coordinate_grid = np.zeros((num_pings, num_samples, 2))
        
        # 각 ping에 대해 샘플 위치 계산
        for ping_idx in range(min(num_pings, len(ping_utm_x))):
            # 현재 ping의 위치
            center_x = ping_utm_x[ping_idx]
            center_y = ping_utm_y[ping_idx]
            
            # 간단한 직선 매핑 (실제로는 더 복잡한 기하학적 계산 필요)
            # 여기서는 각 샘플이 ping 중심에서 일정 간격으로 떨어져 있다고 가정
            sample_spacing = 1.0  # 1미터 간격 가정
            
            for sample_idx in range(num_samples):
                # 거리 계산 (음의 범위부터 양의 범위)
                range_distance = (sample_idx - num_samples // 2) * sample_spacing
                
                # 좌우 스캔 방향으로 좌표 계산 (여기서는 X축 방향으로 가정)
                self.coordinate_grid[ping_idx, sample_idx, 0] = center_x + range_distance
                self.coordinate_grid[ping_idx, sample_idx, 1] = center_y
        
        logger.info("좌표 그리드 생성 완료")
    
    def pixel_to_geo(self, ping_idx: int, sample_idx: int) -> Tuple[float, float]:
        """
        픽셀 좌표를 위경도 좌표로 변환
        
        Args:
            ping_idx: ping 인덱스
            sample_idx: 샘플 인덱스
            
        Returns:
            Tuple[float, float]: (longitude, latitude)
        """
        if self.coordinate_grid is None:
            logger.error("좌표 그리드가 생성되지 않았습니다.")
            return 0.0, 0.0
        
        if (ping_idx >= self.coordinate_grid.shape[0] or 
            sample_idx >= self.coordinate_grid.shape[1]):
            logger.warning(f"인덱스 범위 초과: ping={ping_idx}, sample={sample_idx}")
            return 0.0, 0.0
        
        # UTM 좌표 추출
        utm_x = self.coordinate_grid[ping_idx, sample_idx, 0]
        utm_y = self.coordinate_grid[ping_idx, sample_idx, 1]
        
        # WGS84로 변환
        longitude, latitude = self.transformer.utm_to_wgs84_coords(utm_x, utm_y)
        
        return longitude, latitude
    
    def geo_to_pixel(self, longitude: float, latitude: float) -> Tuple[int, int]:
        """
        위경도 좌표를 가장 가까운 픽셀 좌표로 변환
        
        Args:
            longitude: 경도
            latitude: 위도
            
        Returns:
            Tuple[int, int]: (ping_index, sample_index)
        """
        if self.coordinate_grid is None:
            logger.error("좌표 그리드가 생성되지 않았습니다.")
            return -1, -1
        
        # UTM 좌표로 변환
        target_utm_x, target_utm_y = self.transformer.wgs84_to_utm_coords(longitude, latitude)
        
        # 가장 가까운 픽셀 찾기
        min_distance = float('inf')
        best_ping_idx = -1
        best_sample_idx = -1
        
        for ping_idx in range(self.coordinate_grid.shape[0]):
            for sample_idx in range(self.coordinate_grid.shape[1]):
                utm_x = self.coordinate_grid[ping_idx, sample_idx, 0]
                utm_y = self.coordinate_grid[ping_idx, sample_idx, 1]
                
                distance = np.sqrt((utm_x - target_utm_x)**2 + (utm_y - target_utm_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_ping_idx = ping_idx
                    best_sample_idx = sample_idx
        
        return best_ping_idx, best_sample_idx
    
    def create_target_mask(self, target_locations: List[TargetLocation], 
                          mask_radius: int = 5) -> np.ndarray:
        """
        기물 위치에 대한 바이너리 마스크 생성
        
        Args:
            target_locations: 기물 위치 리스트
            mask_radius: 마스크 반경 (픽셀)
            
        Returns:
            np.ndarray: 바이너리 마스크 [pings, samples]
        """
        if self.intensity_shape is None:
            logger.error("Intensity shape이 설정되지 않았습니다.")
            return np.array([])
        
        # 바이너리 마스크 초기화
        mask = np.zeros(self.intensity_shape, dtype=np.uint8)
        
        logger.info(f"기물 마스크 생성 시작 - {len(target_locations)} 기물")
        
        for target in target_locations:
            # 위경도를 픽셀 좌표로 변환
            ping_idx, sample_idx = self.geo_to_pixel(target.longitude, target.latitude)
            
            if ping_idx >= 0 and sample_idx >= 0:
                # 원형 마스크 생성
                y, x = np.ogrid[:self.intensity_shape[0], :self.intensity_shape[1]]
                distance_mask = (x - sample_idx)**2 + (y - ping_idx)**2 <= mask_radius**2
                mask[distance_mask] = 1
                
                logger.debug(f"기물 {target.target_id} 마스크 생성: ({ping_idx}, {sample_idx})")
            else:
                logger.warning(f"기물 {target.target_id} 위치를 픽셀로 변환할 수 없습니다.")
        
        target_pixels = np.sum(mask)
        total_pixels = mask.size
        
        logger.info(f"기물 마스크 생성 완료 - {target_pixels}/{total_pixels} 픽셀 ({target_pixels/total_pixels*100:.2f}%)")
        
        return mask
    
    def get_target_bounding_boxes(self, target_locations: List[TargetLocation], 
                                box_size: int = 20) -> List[Dict]:
        """
        기물 위치에 대한 바운딩 박스 생성
        
        Args:
            target_locations: 기물 위치 리스트
            box_size: 박스 크기 (픽셀)
            
        Returns:
            List[Dict]: 바운딩 박스 정보 리스트
        """
        bounding_boxes = []
        
        for target in target_locations:
            # 위경도를 픽셀 좌표로 변환
            ping_idx, sample_idx = self.geo_to_pixel(target.longitude, target.latitude)
            
            if ping_idx >= 0 and sample_idx >= 0:
                # 바운딩 박스 좌표 계산
                half_size = box_size // 2
                
                x1 = max(0, sample_idx - half_size)
                y1 = max(0, ping_idx - half_size)
                x2 = min(self.intensity_shape[1] - 1, sample_idx + half_size)
                y2 = min(self.intensity_shape[0] - 1, ping_idx + half_size)
                
                bbox = {
                    'target_id': target.target_id,
                    'center_x': sample_idx,
                    'center_y': ping_idx,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': x2 - x1 + 1,
                    'height': y2 - y1 + 1,
                    'latitude': target.latitude,
                    'longitude': target.longitude
                }
                
                bounding_boxes.append(bbox)
                
                logger.debug(f"바운딩 박스 생성: {target.target_id} - ({x1},{y1}) to ({x2},{y2})")
        
        logger.info(f"바운딩 박스 생성 완료: {len(bounding_boxes)} 박스")
        
        return bounding_boxes
    
    def export_coordinate_mapping(self, output_path: Union[str, Path]) -> bool:
        """
        좌표 매핑 정보를 파일로 내보내기
        
        Args:
            output_path: 출력 파일 경로
            
        Returns:
            bool: 내보내기 성공 여부
        """
        try:
            if self.coordinate_grid is None or self.ping_coordinates is None:
                logger.error("좌표 데이터가 없습니다.")
                return False
            
            # 매핑 데이터 생성
            mappings = []
            
            for ping_idx in range(self.coordinate_grid.shape[0]):
                for sample_idx in range(self.coordinate_grid.shape[1]):
                    utm_x = self.coordinate_grid[ping_idx, sample_idx, 0]
                    utm_y = self.coordinate_grid[ping_idx, sample_idx, 1]
                    
                    longitude, latitude = self.transformer.utm_to_wgs84_coords(utm_x, utm_y)
                    
                    mapping = CoordinateMapping(
                        pixel_x=sample_idx,
                        pixel_y=ping_idx,
                        latitude=latitude,
                        longitude=longitude,
                        utm_x=utm_x,
                        utm_y=utm_y,
                        ping_number=ping_idx,
                        sample_number=sample_idx
                    )
                    
                    mappings.append({
                        'pixel_x': mapping.pixel_x,
                        'pixel_y': mapping.pixel_y,
                        'latitude': mapping.latitude,
                        'longitude': mapping.longitude,
                        'utm_x': mapping.utm_x,
                        'utm_y': mapping.utm_y,
                        'ping_number': mapping.ping_number,
                        'sample_number': mapping.sample_number
                    })
            
            # 데이터프레임으로 변환 후 저장
            df = pd.DataFrame(mappings)
            
            output_path = Path(output_path)
            if output_path.suffix == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path.with_suffix('.csv'), index=False)
            
            logger.info(f"좌표 매핑 내보내기 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"좌표 매핑 내보내기 실패: {e}")
            return False