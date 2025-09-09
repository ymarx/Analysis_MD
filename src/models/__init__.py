"""
딥러닝 모델 모듈

CNN 기반 사이드스캔 소나 기물 탐지 모델들을 제공합니다.
"""

from .cnn_detector import (
    ModelConfig,
    SpatialAttention,
    ChannelAttention,
    CBAM,
    ResNetBackbone,
    MultiHeadClassifier,
    SidescanTargetDetector,
    EnsembleDetector,
    FocalLoss,
    SidescanDataset,
    ModelTrainer
)

__all__ = [
    'ModelConfig',
    'SpatialAttention',
    'ChannelAttention', 
    'CBAM',
    'ResNetBackbone',
    'MultiHeadClassifier',
    'SidescanTargetDetector',
    'EnsembleDetector',
    'FocalLoss',
    'SidescanDataset',
    'ModelTrainer'
]