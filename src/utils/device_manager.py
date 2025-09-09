"""
디바이스 관리 시스템

자동으로 하드웨어를 감지하고 최적의 실행 환경을 설정합니다.
CPU → GPU → 클라우드 환경을 자동 지원하며 기존 코드와 완벽 호환됩니다.
"""

import torch
import platform
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DeviceCapabilities:
    """디바이스 성능 정보"""
    device_type: str
    device_name: str
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    gpu_count: int = 1
    supports_mixed_precision: bool = False
    recommended_batch_size: int = 8
    max_workers: int = 2


@dataclass
class PerformanceConfig:
    """성능 최적화 설정"""
    batch_size: int
    num_workers: int
    pin_memory: bool
    mixed_precision: bool
    gradient_accumulation: int
    memory_limit_mb: int


class DeviceManager:
    """
    하드웨어 감지 및 최적화 설정을 자동으로 관리하는 클래스
    
    기존 코드와의 호환성을 완벽히 보장하며, GPU 및 클라우드 환경에서의
    성능 최적화를 자동으로 처리합니다.
    """
    
    def __init__(self, device: str = 'auto', force_cpu: bool = False):
        """
        디바이스 매니저 초기화
        
        Args:
            device: 'auto', 'cpu', 'cuda', 'mps' 중 선택
            force_cpu: True이면 GPU 무시하고 CPU 사용
        """
        self.force_cpu = force_cpu
        self.device = self._detect_optimal_device(device)
        self.capabilities = self._analyze_capabilities()
        self.config = self._create_performance_config()
        
        logger.info(f"디바이스 관리자 초기화 완료")
        logger.info(f"  - 디바이스: {self.device}")
        logger.info(f"  - 메모리: {self.capabilities.memory_available}MB 사용 가능")
        logger.info(f"  - 권장 배치 크기: {self.config.batch_size}")
    
    def _detect_optimal_device(self, requested_device: str) -> torch.device:
        """최적의 디바이스 자동 감지"""
        if self.force_cpu or requested_device == 'cpu':
            return torch.device('cpu')
        
        if requested_device != 'auto' and requested_device != 'cpu':
            # 특정 디바이스 요청
            try:
                device = torch.device(requested_device)
                if self._is_device_available(device):
                    return device
                else:
                    logger.warning(f"요청된 디바이스 {requested_device}를 사용할 수 없습니다. CPU로 대체합니다.")
                    return torch.device('cpu')
            except Exception:
                logger.warning(f"잘못된 디바이스 이름 {requested_device}. CPU로 대체합니다.")
                return torch.device('cpu')
        
        # 자동 감지 (우선순위: CUDA → MPS → CPU)
        try:
            # NVIDIA GPU (CUDA) 확인
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU 감지: {gpu_name} ({gpu_count}개)")
                return torch.device('cuda')
        except Exception as e:
            logger.debug(f"CUDA 확인 중 오류: {e}")
        
        try:
            # Apple Silicon (MPS) 확인
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple Silicon MPS 감지")
                return torch.device('mps')
        except Exception as e:
            logger.debug(f"MPS 확인 중 오류: {e}")
        
        # 기본값: CPU
        logger.info("CPU 모드로 실행")
        return torch.device('cpu')
    
    def _is_device_available(self, device: torch.device) -> bool:
        """디바이스 사용 가능 여부 확인"""
        try:
            if device.type == 'cuda':
                return torch.cuda.is_available()
            elif device.type == 'mps':
                return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            elif device.type == 'cpu':
                return True
            else:
                return False
        except Exception:
            return False
    
    def _analyze_capabilities(self) -> DeviceCapabilities:
        """디바이스 성능 분석"""
        if self.device.type == 'cuda':
            return self._analyze_cuda_capabilities()
        elif self.device.type == 'mps':
            return self._analyze_mps_capabilities()
        else:
            return self._analyze_cpu_capabilities()
    
    def _analyze_cuda_capabilities(self) -> DeviceCapabilities:
        """CUDA GPU 성능 분석"""
        try:
            gpu_props = torch.cuda.get_device_properties(self.device)
            memory_total = gpu_props.total_memory // (1024 * 1024)  # MB
            memory_available = memory_total - (torch.cuda.memory_reserved() // (1024 * 1024))
            
            # Compute Capability 확인
            compute_cap = f"{gpu_props.major}.{gpu_props.minor}"
            supports_mixed_precision = float(compute_cap) >= 7.0  # Tensor Cores
            
            # 배치 크기 추정 (메모리 기반)
            if memory_available > 8000:  # 8GB+
                batch_size = 32
                max_workers = 4
            elif memory_available > 4000:  # 4GB+
                batch_size = 16
                max_workers = 3
            else:
                batch_size = 8
                max_workers = 2
            
            return DeviceCapabilities(
                device_type='cuda',
                device_name=gpu_props.name,
                memory_total=memory_total,
                memory_available=memory_available,
                compute_capability=compute_cap,
                gpu_count=torch.cuda.device_count(),
                supports_mixed_precision=supports_mixed_precision,
                recommended_batch_size=batch_size,
                max_workers=max_workers
            )
        except Exception as e:
            logger.warning(f"CUDA 성능 분석 실패: {e}")
            return self._analyze_cpu_capabilities()
    
    def _analyze_mps_capabilities(self) -> DeviceCapabilities:
        """Apple Silicon MPS 성능 분석"""
        try:
            # 시스템 메모리 정보 (MPS는 통합 메모리 사용)
            memory = psutil.virtual_memory()
            memory_total = memory.total // (1024 * 1024)  # MB
            memory_available = memory.available // (1024 * 1024)  # MB
            
            # Apple Silicon 모델별 추정
            if memory_total > 16000:  # 32GB+ (M1 Ultra, M2 Ultra)
                batch_size = 24
                max_workers = 4
            elif memory_total > 8000:  # 16GB+ (M1 Pro, M2 Pro)
                batch_size = 16
                max_workers = 3
            else:  # 8GB (Base M1, M2)
                batch_size = 12
                max_workers = 2
            
            return DeviceCapabilities(
                device_type='mps',
                device_name='Apple Silicon',
                memory_total=memory_total,
                memory_available=memory_available,
                compute_capability=None,
                gpu_count=1,
                supports_mixed_precision=False,  # MPS는 현재 제한적
                recommended_batch_size=batch_size,
                max_workers=max_workers
            )
        except Exception as e:
            logger.warning(f"MPS 성능 분석 실패: {e}")
            return self._analyze_cpu_capabilities()
    
    def _analyze_cpu_capabilities(self) -> DeviceCapabilities:
        """CPU 성능 분석"""
        try:
            memory = psutil.virtual_memory()
            memory_total = memory.total // (1024 * 1024)  # MB
            memory_available = memory.available // (1024 * 1024)  # MB
            
            cpu_count = psutil.cpu_count(logical=False)  # 물리 코어 수
            
            # CPU 성능에 따른 배치 크기 조정
            if cpu_count >= 8 and memory_available > 8000:
                batch_size = 8
                max_workers = 4
            elif cpu_count >= 4 and memory_available > 4000:
                batch_size = 4
                max_workers = 3
            else:
                batch_size = 2
                max_workers = 2
            
            return DeviceCapabilities(
                device_type='cpu',
                device_name=platform.processor() or 'Unknown CPU',
                memory_total=memory_total,
                memory_available=memory_available,
                compute_capability=None,
                gpu_count=0,
                supports_mixed_precision=False,
                recommended_batch_size=batch_size,
                max_workers=max_workers
            )
        except Exception as e:
            logger.error(f"CPU 성능 분석 실패: {e}")
            # 최소 설정으로 폴백
            return DeviceCapabilities(
                device_type='cpu',
                device_name='Unknown',
                memory_total=4096,
                memory_available=2048,
                recommended_batch_size=2,
                max_workers=1
            )
    
    def _create_performance_config(self) -> PerformanceConfig:
        """성능 최적화 설정 생성"""
        caps = self.capabilities
        
        return PerformanceConfig(
            batch_size=caps.recommended_batch_size,
            num_workers=caps.max_workers,
            pin_memory=(self.device.type == 'cuda'),
            mixed_precision=caps.supports_mixed_precision,
            gradient_accumulation=1,
            memory_limit_mb=int(caps.memory_available * 0.8)  # 80% 사용
        )
    
    def get_torch_device(self) -> torch.device:
        """PyTorch 디바이스 반환 (기존 코드 호환성)"""
        return self.device
    
    def get_model_device(self) -> str:
        """모델 이동용 디바이스 문자열 반환"""
        return str(self.device)
    
    def move_to_device(self, tensor_or_model):
        """텐서 또는 모델을 적절한 디바이스로 이동"""
        try:
            return tensor_or_model.to(self.device)
        except Exception as e:
            logger.warning(f"디바이스 이동 실패: {e}")
            return tensor_or_model
    
    def optimize_model(self, model):
        """모델 최적화 적용"""
        try:
            # 디바이스로 이동
            model = model.to(self.device)
            
            # 컴파일 최적화 (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                try:
                    model = torch.compile(model)
                    logger.info("PyTorch 2.0 컴파일 최적화 적용")
                except Exception as e:
                    logger.debug(f"컴파일 최적화 실패 (정상): {e}")
            
            return model
        except Exception as e:
            logger.warning(f"모델 최적화 실패: {e}")
            return model
    
    def create_dataloader_config(self, dataset_size: int = None) -> Dict[str, Any]:
        """데이터로더 최적화 설정 반환"""
        config = {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'shuffle': True,
            'drop_last': False
        }
        
        # 데이터셋 크기에 따른 조정
        if dataset_size and dataset_size < self.config.batch_size * 10:
            config['batch_size'] = max(1, dataset_size // 10)
            config['num_workers'] = 1
        
        return config
    
    def get_memory_info(self) -> Dict[str, int]:
        """메모리 사용량 정보 반환"""
        info = {
            'total_mb': self.capabilities.memory_total,
            'available_mb': self.capabilities.memory_available,
            'used_mb': self.capabilities.memory_total - self.capabilities.memory_available
        }
        
        if self.device.type == 'cuda':
            try:
                info['gpu_allocated_mb'] = torch.cuda.memory_allocated() // (1024 * 1024)
                info['gpu_reserved_mb'] = torch.cuda.memory_reserved() // (1024 * 1024)
            except Exception:
                pass
        
        return info
    
    def clear_memory(self):
        """메모리 정리"""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                logger.debug("CUDA 메모리 캐시 정리 완료")
        except Exception as e:
            logger.debug(f"메모리 정리 중 오류: {e}")
    
    def benchmark_device(self, iterations: int = 10) -> Dict[str, float]:
        """디바이스 성능 벤치마킹"""
        try:
            import time
            
            # 테스트용 데이터 생성
            test_size = 1024
            x = torch.randn(self.config.batch_size, 3, test_size, test_size)
            x = self.move_to_device(x)
            
            # 워밍업
            for _ in range(3):
                _ = torch.sum(x)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 벤치마킹
            start_time = time.time()
            for _ in range(iterations):
                result = torch.sum(x * x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            return {
                'avg_operation_time_ms': avg_time * 1000,
                'operations_per_second': 1.0 / avg_time,
                'total_test_time_s': total_time
            }
            
        except Exception as e:
            logger.warning(f"벤치마킹 실패: {e}")
            return {
                'avg_operation_time_ms': 0.0,
                'operations_per_second': 0.0,
                'total_test_time_s': 0.0
            }
    
    def get_device_summary(self) -> Dict[str, Any]:
        """디바이스 정보 요약"""
        return {
            'device_type': self.capabilities.device_type,
            'device_name': self.capabilities.device_name,
            'memory_info': self.get_memory_info(),
            'performance_config': {
                'batch_size': self.config.batch_size,
                'num_workers': self.config.num_workers,
                'mixed_precision': self.config.mixed_precision
            },
            'capabilities': {
                'gpu_count': self.capabilities.gpu_count,
                'compute_capability': self.capabilities.compute_capability,
                'mixed_precision_support': self.capabilities.supports_mixed_precision
            }
        }
    
    @classmethod
    def create_compatible_device(cls, legacy_device: str = None) -> 'DeviceManager':
        """
        기존 코드 호환성을 위한 팩토리 메서드
        
        Args:
            legacy_device: 기존에 사용하던 디바이스 문자열 ('cuda', 'cpu' 등)
        
        Returns:
            DeviceManager 인스턴스
        """
        if legacy_device:
            return cls(device=legacy_device)
        else:
            return cls(device='auto')


# 전역 디바이스 매니저 인스턴스 (필요시 사용)
_global_device_manager = None

def get_global_device_manager() -> DeviceManager:
    """전역 디바이스 매니저 인스턴스 반환"""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(device='auto')
    return _global_device_manager

def set_global_device(device: str):
    """전역 디바이스 설정"""
    global _global_device_manager
    _global_device_manager = DeviceManager(device=device)