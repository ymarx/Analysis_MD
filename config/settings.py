"""
사이드스캔 소나 기물 탐지 시스템 설정 파일
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로 설정
DATA_PATHS = {
    'sample_data': PROJECT_ROOT / '[샘플]데이터',
    'datasets': PROJECT_ROOT / 'datasets',
    'processed': PROJECT_ROOT / 'data' / 'processed',
    'augmented': PROJECT_ROOT / 'data' / 'augmented',
    'annotations': PROJECT_ROOT / 'data' / 'annotations'
}

# XTF 파일 설정
XTF_CONFIG = {
    'sample_file': '[샘플]Busan_Eardo_1_Klein3210_500_050_240901073700_001_04.xtf',
    'channels': {
        'port': 0,
        'starboard': 1
    },
    'max_pings_per_load': 1000  # 메모리 효율성을 위한 배치 크기
}

# 이미지 처리 설정
IMAGE_CONFIG = {
    'sample_annotation': '[샘플]BS_mosaic_annotation.bmp',
    'output_format': 'png',
    'dpi': 300,
    'colormap': 'gray'
}

# 좌표 설정
COORDINATE_CONFIG = {
    'location_file': '[위치]부산위치자료-도분초-위경도변환.xlsx',
    'coordinate_system': 'WGS84',
    'utm_zone': 52  # 한국 지역
}

# 특징 추출 설정
FEATURE_CONFIG = {
    'methods': ['hog', 'lbp', 'gabor', 'sfs'],
    'hog': {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'visualize': False
    },
    'lbp': {
        'radius': 1,
        'n_points': 8,
        'method': 'uniform'
    },
    'gabor': {
        'frequency': 0.1,
        'theta_range': [0, 45, 90, 135],
        'sigma_x': 1.0,
        'sigma_y': 1.0
    }
}

# 모델 설정
MODEL_CONFIG = {
    'input_channels': 1,
    'num_classes': 2,  # 기물 vs 배경
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'validation_split': 0.2
}

# 데이터 증강 설정
AUGMENTATION_CONFIG = {
    'target_ratio': 1.0,  # 기물:배경 비율 목표
    'rotation_range': 15,
    'brightness_range': 0.2,
    'noise_level': 0.05,
    'blur_sigma': 0.5
}

# 평가 설정
EVALUATION_CONFIG = {
    'metrics': ['precision', 'recall', 'f1', 'iou', 'map'],
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'confidence_threshold': 0.5
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 150,
    'save_format': 'png',
    'interactive': True
}

# 로깅 설정
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'logs' / 'sidescan_detection.log'
}

# GPU 설정
DEVICE_CONFIG = {
    'use_cuda': True,
    'device_id': 0,
    'num_workers': 4
}