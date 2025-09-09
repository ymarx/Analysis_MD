"""
디바이스별 최적화 설정

각 하드웨어 환경에 맞춘 성능 최적화 설정을 제공합니다.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceOptimizationConfig:
    """디바이스별 최적화 설정"""
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
    memory_limit_gb: float
    mixed_precision: bool
    compile_model: bool
    gradient_accumulation_steps: int


# CPU 최적화 설정
CPU_CONFIG = DeviceOptimizationConfig(
    batch_size=8,
    num_workers=2,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
    memory_limit_gb=4.0,
    mixed_precision=False,
    compile_model=False,
    gradient_accumulation_steps=1
)

# NVIDIA GPU (CUDA) 최적화 설정
CUDA_CONFIG = DeviceOptimizationConfig(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    memory_limit_gb=8.0,
    mixed_precision=True,
    compile_model=True,
    gradient_accumulation_steps=1
)

# Apple Silicon (MPS) 최적화 설정
MPS_CONFIG = DeviceOptimizationConfig(
    batch_size=16,
    num_workers=3,
    pin_memory=False,
    prefetch_factor=3,
    persistent_workers=True,
    memory_limit_gb=6.0,
    mixed_precision=False,  # MPS 현재 제한적 지원
    compile_model=False,    # MPS에서 torch.compile 안정성 이슈
    gradient_accumulation_steps=2
)

# 고성능 GPU (A100, V100) 최적화 설정
HIGH_END_GPU_CONFIG = DeviceOptimizationConfig(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=8,
    persistent_workers=True,
    memory_limit_gb=32.0,
    mixed_precision=True,
    compile_model=True,
    gradient_accumulation_steps=1
)

# 저사양 GPU (GTX 1060, T4) 최적화 설정
LOW_END_GPU_CONFIG = DeviceOptimizationConfig(
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
    memory_limit_gb=4.0,
    mixed_precision=False,
    compile_model=False,
    gradient_accumulation_steps=2
)

# 디바이스 타입별 설정 매핑
DEVICE_CONFIGS = {
    'cpu': CPU_CONFIG,
    'cuda': CUDA_CONFIG,
    'mps': MPS_CONFIG,
    'high_end_gpu': HIGH_END_GPU_CONFIG,
    'low_end_gpu': LOW_END_GPU_CONFIG
}

# GPU 메모리별 설정 선택 함수
def get_gpu_config_by_memory(memory_gb: float) -> DeviceOptimizationConfig:
    """GPU 메모리 크기에 따른 최적 설정 반환"""
    if memory_gb >= 16.0:
        return HIGH_END_GPU_CONFIG
    elif memory_gb >= 6.0:
        return CUDA_CONFIG
    else:
        return LOW_END_GPU_CONFIG

# 클라우드 환경별 설정 (Runpod 중심)
CLOUD_CONFIGS = {
    'runpod_rtx4090': DeviceOptimizationConfig(
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        memory_limit_gb=20.0,  # RTX 4090 24GB의 83% 사용
        mixed_precision=True,
        compile_model=True,
        gradient_accumulation_steps=1
    ),
    'runpod_rtx3090': DeviceOptimizationConfig(
        batch_size=48,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=6,
        persistent_workers=True,
        memory_limit_gb=20.0,  # RTX 3090 24GB의 83% 사용
        mixed_precision=True,
        compile_model=True,
        gradient_accumulation_steps=1
    ),
    'runpod_a40': DeviceOptimizationConfig(
        batch_size=96,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=12,
        persistent_workers=True,
        memory_limit_gb=40.0,  # A40 48GB의 83% 사용
        mixed_precision=True,
        compile_model=True,
        gradient_accumulation_steps=1
    ),
    'runpod_a100_40gb': DeviceOptimizationConfig(
        batch_size=128,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=16,
        persistent_workers=True,
        memory_limit_gb=32.0,  # A100-40GB의 80% 사용
        mixed_precision=True,
        compile_model=True,
        gradient_accumulation_steps=1
    ),
    'runpod_a100_80gb': DeviceOptimizationConfig(
        batch_size=256,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=24,
        persistent_workers=True,
        memory_limit_gb=64.0,  # A100-80GB의 80% 사용
        mixed_precision=True,
        compile_model=True,
        gradient_accumulation_steps=1
    )
}

def get_optimal_config(device_type: str, memory_gb: float = None, cloud_instance: str = None) -> DeviceOptimizationConfig:
    """최적의 설정 반환"""
    
    # 클라우드 인스턴스 지정된 경우
    if cloud_instance and cloud_instance in CLOUD_CONFIGS:
        return CLOUD_CONFIGS[cloud_instance]
    
    # GPU인 경우 메모리 기반 선택
    if device_type == 'cuda' and memory_gb:
        return get_gpu_config_by_memory(memory_gb)
    
    # 기본 디바이스 설정
    return DEVICE_CONFIGS.get(device_type, CPU_CONFIG)