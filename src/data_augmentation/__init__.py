"""
데이터 증강 모듈

소나 데이터 전용 증강 기법들을 제공합니다.
"""

from .augmentation_engine import (
    AugmentationConfig,
    BaseAugmentation,
    GeometricAugmentation,
    PhotometricAugmentation,
    SonarSpecificAugmentation,
    AdvancedAugmentationEngine,
    AugmentationValidator
)

__all__ = [
    'AugmentationConfig',
    'BaseAugmentation',
    'GeometricAugmentation',
    'PhotometricAugmentation',
    'SonarSpecificAugmentation',
    'AdvancedAugmentationEngine',
    'AugmentationValidator'
]