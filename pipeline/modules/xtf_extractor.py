"""
XTF Extractor Module
====================
XTF 데이터 추출 및 전처리 모듈
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing extraction modules
try:
    from data_processing.xtf_intensity_extractor import XTFIntensityExtractor
except ImportError:
    XTFIntensityExtractor = None

logger = logging.getLogger(__name__)


class XTFExtractor:
    """XTF 데이터 추출 클래스"""

    def __init__(self,
                 sample_rate: float = 0.05,
                 channels: List[str] = None,
                 normalize: bool = True):
        """
        Initialize XTF Extractor

        Args:
            sample_rate: 샘플링 비율 (0.0-1.0)
            channels: 추출할 채널 ['port', 'starboard']
            normalize: 강도 데이터 정규화 여부
        """
        self.sample_rate = sample_rate
        self.channels = channels or ['port', 'starboard']
        self.normalize = normalize
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize existing extractor if available
        if XTFIntensityExtractor is not None:
            self.intensity_extractor = XTFIntensityExtractor()
        else:
            self.intensity_extractor = None
            self.logger.warning("XTFIntensityExtractor not available. Using fallback.")

    def extract(self, xtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        XTF 데이터에서 강도 데이터 추출

        Args:
            xtf_data: XTF Reader에서 반환된 데이터

        Returns:
            추출된 데이터 딕셔너리
        """
        self.logger.info("Extracting intensity data from XTF")

        try:
            if self.intensity_extractor is not None:
                return self._extract_with_existing(xtf_data)
            else:
                return self._extract_fallback(xtf_data)

        except Exception as e:
            self.logger.error(f"Failed to extract XTF data: {e}")
            raise

    def _extract_with_existing(self, xtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """기존 XTFIntensityExtractor 사용"""
        file_path = xtf_data.get('file_path')
        if not file_path:
            raise ValueError("No file path in XTF data")

        self.logger.debug("Using existing XTFIntensityExtractor")

        # 기존 추출기 사용
        result = self.intensity_extractor.process_xtf_file(
            Path(file_path),
            sample_rate=self.sample_rate
        )

        return {
            'intensity': result.get('combined_intensity'),
            'port_intensity': result.get('port_intensity'),
            'starboard_intensity': result.get('starboard_intensity'),
            'navigation': result.get('navigation'),
            'metadata': result.get('metadata', {}),
            'processing_info': {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'normalized': self.normalize,
                'extractor_type': 'existing'
            }
        }

    def _extract_fallback(self, xtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback 추출 방법"""
        self.logger.debug("Using fallback extraction")

        pings = xtf_data.get('pings', [])
        channels = xtf_data.get('channels', {})

        # 샘플링
        if self.sample_rate < 1.0 and pings:
            sample_count = max(1, int(len(pings) * self.sample_rate))
            indices = np.linspace(0, len(pings) - 1, sample_count, dtype=int)
            sampled_pings = [pings[i] for i in indices]
        else:
            sampled_pings = pings

        # 강도 데이터 추출
        port_data = self._extract_channel_data(channels.get('port', []))
        starboard_data = self._extract_channel_data(channels.get('starboard', []))

        # 결합된 강도 데이터
        combined_intensity = self._combine_channels(port_data, starboard_data)

        # 네비게이션 데이터 추출
        navigation = self._extract_navigation_data(xtf_data.get('navigation', []))

        return {
            'intensity': combined_intensity,
            'port_intensity': port_data,
            'starboard_intensity': starboard_data,
            'navigation': navigation,
            'metadata': xtf_data.get('metadata', {}),
            'processing_info': {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'normalized': self.normalize,
                'extractor_type': 'fallback',
                'ping_count': len(sampled_pings)
            }
        }

    def _extract_channel_data(self, channel_data: List) -> Optional[np.ndarray]:
        """채널 데이터 추출 및 전처리"""
        if not channel_data:
            return None

        try:
            # 데이터를 numpy 배열로 변환
            if isinstance(channel_data[0], np.ndarray):
                intensity_data = np.vstack(channel_data)
            elif isinstance(channel_data[0], (list, tuple)):
                intensity_data = np.array(channel_data)
            else:
                # 단일 값들의 리스트인 경우
                intensity_data = np.array(channel_data).reshape(-1, 1)

            # 정규화
            if self.normalize:
                intensity_data = self._normalize_intensity(intensity_data)

            return intensity_data

        except Exception as e:
            self.logger.error(f"Failed to extract channel data: {e}")
            return None

    def _combine_channels(self, port_data: Optional[np.ndarray],
                         starboard_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """포트와 스타보드 채널 결합"""
        if port_data is None and starboard_data is None:
            return None

        if port_data is None:
            return starboard_data

        if starboard_data is None:
            return port_data

        try:
            # 크기가 다른 경우 작은 쪽에 맞춤
            min_rows = min(port_data.shape[0], starboard_data.shape[0])
            min_cols = min(port_data.shape[1] if port_data.ndim > 1 else 1,
                          starboard_data.shape[1] if starboard_data.ndim > 1 else 1)

            port_resized = port_data[:min_rows, :min_cols] if port_data.ndim > 1 else port_data[:min_rows]
            starboard_resized = starboard_data[:min_rows, :min_cols] if starboard_data.ndim > 1 else starboard_data[:min_rows]

            # 수평으로 결합
            combined = np.hstack([port_resized, starboard_resized])
            return combined

        except Exception as e:
            self.logger.error(f"Failed to combine channels: {e}")
            return port_data if port_data is not None else starboard_data

    def _normalize_intensity(self, data: np.ndarray) -> np.ndarray:
        """강도 데이터 정규화"""
        if data.size == 0:
            return data

        # Remove outliers (beyond 3 standard deviations)
        mean = np.mean(data)
        std = np.std(data)
        data_clipped = np.clip(data, mean - 3*std, mean + 3*std)

        # Min-max normalization to [0, 1]
        data_min = np.min(data_clipped)
        data_max = np.max(data_clipped)

        if data_max > data_min:
            normalized = (data_clipped - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data_clipped)

        return normalized

    def _extract_navigation_data(self, navigation_list: List[Dict]) -> Dict[str, np.ndarray]:
        """네비게이션 데이터 추출"""
        if not navigation_list:
            return {}

        nav_data = {
            'latitude': [],
            'longitude': [],
            'timestamp': [],
            'heading': [],
            'altitude': []
        }

        for nav in navigation_list:
            nav_data['latitude'].append(nav.get('latitude', 0.0))
            nav_data['longitude'].append(nav.get('longitude', 0.0))
            nav_data['timestamp'].append(nav.get('timestamp', 0))
            nav_data['heading'].append(nav.get('heading', 0.0))
            nav_data['altitude'].append(nav.get('altitude', 0.0))

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in nav_data.items() if v}

    def get_extraction_info(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """추출 정보 반환"""
        intensity = extracted_data.get('intensity')
        processing_info = extracted_data.get('processing_info', {})

        info = {
            'processing_info': processing_info,
            'data_shape': intensity.shape if intensity is not None else None,
            'data_type': type(intensity).__name__ if intensity is not None else None,
            'intensity_range': {
                'min': float(np.min(intensity)) if intensity is not None else None,
                'max': float(np.max(intensity)) if intensity is not None else None,
                'mean': float(np.mean(intensity)) if intensity is not None else None,
                'std': float(np.std(intensity)) if intensity is not None else None
            },
            'navigation_points': len(extracted_data.get('navigation', {}).get('latitude', [])),
            'channels_available': [
                'port' if extracted_data.get('port_intensity') is not None else None,
                'starboard' if extracted_data.get('starboard_intensity') is not None else None,
                'combined' if intensity is not None else None
            ]
        }

        # Remove None values
        info['channels_available'] = [ch for ch in info['channels_available'] if ch is not None]

        return info