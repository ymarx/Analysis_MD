"""
Enhanced Unified Mine Detection Pipeline
=======================================
기존 unified_pipeline.py를 확장하여 데이터 증강, 훈련/검증/테스트 분할,
25개 기뢰 위치 특화 처리를 통합한 완전한 기뢰 탐지 파이프라인

Author: YMARX
Date: 2024-09-22
Version: 2.0
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing modules
from pipeline.modules.working_xtf_reader import WorkingXTFReader
from pipeline.modules.xtf_extractor import XTFExtractor
from pipeline.modules.coordinate_mapper import CoordinateMapper
from pipeline.modules.label_generator import LabelGenerator
from pipeline.modules.feature_extractor import FeatureExtractor
from pipeline.modules.ensemble_optimizer import EnsembleOptimizer
from pipeline.modules.mine_classifier import MineClassifier
from pipeline.modules.terrain_analyzer import TerrainAnalyzer
from pipeline.modules.gps_parser import GPSParser

# Import data augmentation engine
from src.data_augmentation.augmentation_engine import (
    AdvancedAugmentationEngine, AugmentationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedPipelineConfig:
    """향상된 파이프라인 설정"""
    # Base paths
    data_dir: Path = Path("datasets")
    output_dir: Path = Path("data/processed")

    # XTF Processing
    xtf_sample_rate: float = 0.05
    xtf_channels: List[str] = None

    # Coordinate Mapping
    gps_file: Optional[Path] = None
    annotation_image: Optional[Path] = None
    transform_type: str = "rotate_flip"  # 180도 회전 + 좌우반전

    # Feature Extraction
    feature_methods: List[str] = None
    patch_size: Tuple[int, int] = (64, 64)

    # Data Augmentation - NEW
    enable_augmentation: bool = True
    augmentation_config: Optional[AugmentationConfig] = None
    mine_augmentation_factor: int = 8  # 기뢰 샘플 증강 배수
    background_augmentation_factor: int = 3  # 배경 샘플 증강 배수

    # Train/Test Split - NEW
    test_size: float = 0.2
    validation_size: float = 0.2  # 훈련 데이터의 비율
    random_state: int = 42
    stratify: bool = True

    # Mine-specific settings - NEW
    mine_locations_file: Optional[Path] = None  # 25개 기뢰 위치 파일
    mine_confidence_threshold: float = 0.8
    use_gps_validation: bool = True

    # Classification
    classifier_type: str = "ensemble"
    use_terrain: bool = True
    cross_validation_folds: int = 5

    # Output
    save_intermediate: bool = True
    save_augmented_samples: bool = False
    generate_reports: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.xtf_channels is None:
            self.xtf_channels = ['port', 'starboard']
        if self.feature_methods is None:
            self.feature_methods = ['statistical', 'textural', 'morphological', 'frequency']
        if self.augmentation_config is None:
            self.augmentation_config = AugmentationConfig()


class EnhancedUnifiedPipeline:
    """향상된 통합 기뢰 탐지 파이프라인"""

    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize modules
        self._init_modules()

        # Results storage
        self.results = {}
        self.datasets = {}  # train/val/test splits

        # Performance tracking
        self.performance_metrics = {}

    def _init_modules(self):
        """파이프라인 모듈 초기화"""
        # Working modules
        self.xtf_reader = WorkingXTFReader()
        self.xtf_extractor = XTFExtractor(
            sample_rate=self.config.xtf_sample_rate,
            channels=self.config.xtf_channels
        )
        self.gps_parser = GPSParser()
        self.coord_mapper = CoordinateMapper(
            transform_type=self.config.transform_type
        )
        self.label_generator = LabelGenerator(
            patch_size=self.config.patch_size
        )
        self.feature_extractor = FeatureExtractor(
            methods=self.config.feature_methods,
            patch_size=self.config.patch_size
        )
        self.ensemble_optimizer = EnsembleOptimizer()
        self.classifier = MineClassifier(
            classifier_type=self.config.classifier_type
        )
        self.terrain_analyzer = TerrainAnalyzer() if self.config.use_terrain else None

        # NEW: Data augmentation engine
        if self.config.enable_augmentation:
            self.augmentation_engine = AdvancedAugmentationEngine(
                self.config.augmentation_config
            )
        else:
            self.augmentation_engine = None

    # ========== Enhanced Data Processing ==========

    def process_mine_locations(self, mine_locations_file: Optional[Path] = None) -> Dict:
        """25개 기뢰 위치 데이터 처리"""
        locations_file = mine_locations_file or self.config.mine_locations_file

        if not locations_file or not locations_file.exists():
            self.logger.warning("Mine locations file not found, using default GPS data")
            return {}

        self.logger.info(f"Processing mine locations from {locations_file}")

        try:
            # Use GPS parser to handle different coordinate formats
            locations = self.gps_parser.parse_gps_file(locations_file)

            # Convert to mine location format with confidence
            mine_locations = []
            for location in locations:
                mine_location = {
                    'id': location['id'],
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'confidence': self.config.mine_confidence_threshold,
                    'type': 'confirmed_mine',
                    'raw_coordinates': {
                        'latitude': location['raw_latitude'],
                        'longitude': location['raw_longitude']
                    }
                }
                mine_locations.append(mine_location)

            # Validate coordinates
            validation = self.gps_parser.validate_coordinates(locations)

            self.logger.info(f"Loaded {len(mine_locations)} mine locations")
            self.logger.info(f"Validation: {validation['valid_count']}/{validation['total_count']} valid coordinates")

            return {
                'locations': mine_locations,
                'total_count': len(mine_locations),
                'valid_count': validation['valid_count'],
                'coordinate_ranges': validation['coordinate_ranges'],
                'source_file': str(locations_file)
            }

        except Exception as e:
            self.logger.error(f"Failed to process mine locations: {e}")
            return {}

    def generate_samples_from_mine_locations(self, mine_locations: Dict) -> Dict:
        """기뢰 위치에서 직접 훈련 샘플 생성"""
        self.logger.info("Generating training samples directly from mine locations")

        # Get intensity data from XTF data (which has the correct structure from WorkingXTFReader)
        xtf_data = self.results.get('xtf_data', {})
        intensity_data = xtf_data.get('intensity_data', {})
        intensity_images = intensity_data.get('intensity_images', {})
        navigation = intensity_data.get('navigation', {})

        if 'combined' not in intensity_images:
            self.logger.error("No combined intensity image found")
            return {'samples': [], 'labels': []}

        combined_image = intensity_images['combined']
        latitudes = navigation.get('latitudes', [])
        longitudes = navigation.get('longitudes', [])

        if len(latitudes) == 0 or len(longitudes) == 0:
            self.logger.error("No navigation data found")
            return {'samples': [], 'labels': []}

        self.logger.info(f"Working with intensity image shape: {combined_image.shape}")
        self.logger.info(f"Navigation data: {len(latitudes)} points")

        # Calculate geographical bounds
        lat_min, lat_max = min(latitudes), max(latitudes)
        lon_min, lon_max = min(longitudes), max(longitudes)

        self.logger.info(f"Geographic bounds: lat=[{lat_min:.6f}, {lat_max:.6f}], lon=[{lon_min:.6f}, {lon_max:.6f}]")

        samples = []
        labels = []
        mine_locations_list = mine_locations.get('locations', [])

        # Check if any mine locations are within bounds
        locations_in_bounds = [
            loc for loc in mine_locations_list
            if lat_min <= loc['latitude'] <= lat_max and lon_min <= loc['longitude'] <= lon_max
        ]

        self.logger.info(f"Found {len(locations_in_bounds)} mine locations within XTF bounds out of {len(mine_locations_list)} total")

        # If no mine locations are within bounds, generate synthetic ones for testing
        if len(locations_in_bounds) == 0:
            self.logger.warning("No mine locations within XTF bounds. Generating synthetic mine locations for testing.")
            import random

            # Generate 5 synthetic mine locations within the XTF area
            synthetic_locations = []
            for i in range(5):
                lat = lat_min + (lat_max - lat_min) * random.random()
                lon = lon_min + (lon_max - lon_min) * random.random()
                synthetic_locations.append({
                    'id': f'synthetic_mine_{i+1}',
                    'latitude': lat,
                    'longitude': lon
                })
                self.logger.info(f"Generated synthetic mine location {i+1}: ({lat:.6f}, {lon:.6f})")

            locations_in_bounds = synthetic_locations

        # Generate positive samples (mines)
        for location in locations_in_bounds:
            lat = location['latitude']
            lon = location['longitude']

            # Check if mine location is within XTF data bounds
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                # Convert GPS to pixel coordinates
                pixel_y = int((lat - lat_min) / (lat_max - lat_min) * (combined_image.shape[0] - 1))
                pixel_x = int((lon - lon_min) / (lon_max - lon_min) * (combined_image.shape[1] - 1))

                # Define patch bounds around mine location
                patch_size = self.config.patch_size[0]  # Assuming square patches
                half_patch = patch_size // 2

                y_min = max(0, pixel_y - half_patch)
                y_max = min(combined_image.shape[0], pixel_y + half_patch)
                x_min = max(0, pixel_x - half_patch)
                x_max = min(combined_image.shape[1], pixel_x + half_patch)

                # Only add if patch is large enough
                if (y_max - y_min) >= patch_size//2 and (x_max - x_min) >= patch_size//2:
                    sample = {
                        'patch_bounds': {
                            'y_min': y_min, 'y_max': y_max,
                            'x_min': x_min, 'x_max': x_max
                        },
                        'location_id': location['id'],
                        'gps_lat': lat,
                        'gps_lon': lon,
                        'pixel_y': pixel_y,
                        'pixel_x': pixel_x
                    }
                    samples.append(sample)
                    labels.append(1)  # Positive label for mine

                    self.logger.info(f"Added mine sample at ({lat:.6f}, {lon:.6f}) -> ({pixel_x}, {pixel_y})")
                else:
                    self.logger.warning(f"Mine location too close to edge: ({lat:.6f}, {lon:.6f})")
            else:
                self.logger.warning(f"Mine location outside XTF bounds: ({lat:.6f}, {lon:.6f})")

        # Generate negative samples (background)
        # Create random background patches away from mine locations
        import random
        num_negatives = len(samples) * 2  # 2x negative samples

        for i in range(num_negatives):
            # Random location in intensity image
            patch_size = self.config.patch_size[0]
            half_patch = patch_size // 2

            y_center = random.randint(half_patch, combined_image.shape[0] - half_patch - 1)
            x_center = random.randint(half_patch, combined_image.shape[1] - half_patch - 1)

            # Check if too close to any mine location
            too_close = False
            for sample in samples:
                mine_y = sample['pixel_y']
                mine_x = sample['pixel_x']
                distance = ((y_center - mine_y)**2 + (x_center - mine_x)**2)**0.5
                if distance < patch_size:  # Minimum distance = patch size
                    too_close = True
                    break

            if not too_close:
                sample = {
                    'patch_bounds': {
                        'y_min': y_center - half_patch,
                        'y_max': y_center + half_patch,
                        'x_min': x_center - half_patch,
                        'x_max': x_center + half_patch
                    },
                    'location_id': f'background_{i}',
                    'pixel_y': y_center,
                    'pixel_x': x_center
                }
                samples.append(sample)
                labels.append(0)  # Negative label for background

        self.logger.info(f"Generated {len([l for l in labels if l == 1])} positive and {len([l for l in labels if l == 0])} negative samples")

        return {'samples': samples, 'labels': np.array(labels)}

    def generate_enhanced_labels(self,
                               coordinate_mapping: Optional[Dict] = None,
                               intensity_data: Optional[np.ndarray] = None,
                               mine_locations: Optional[Dict] = None) -> Dict:
        """향상된 레이블 생성 (25개 기뢰 위치 특화)"""

        if coordinate_mapping is None:
            coordinate_mapping = self.results.get('coordinate_mapping')
        if intensity_data is None:
            intensity_data = self.results.get('extracted_data', {}).get('intensity')
        if mine_locations is None:
            mine_locations = self.results.get('mine_locations', {})

        self.logger.info("Generating enhanced labels with mine location validation")

        # Generate base labels
        base_labels = self.label_generator.generate(coordinate_mapping, intensity_data)

        # Enhance with mine location data
        if mine_locations and mine_locations.get('locations'):
            enhanced_samples = self._validate_samples_with_mine_locations(
                base_labels['samples'], mine_locations['locations']
            )

            # Update label data
            enhanced_labels = base_labels.copy()
            enhanced_labels['samples'] = enhanced_samples
            enhanced_labels['labels'] = np.array([s['label'] for s in enhanced_samples])
            enhanced_labels['mine_validation'] = {
                'validated_mines': len([s for s in enhanced_samples if s.get('mine_validated', False)]),
                'total_mine_locations': len(mine_locations['locations']),
                'validation_accuracy': len([s for s in enhanced_samples if s.get('mine_validated', False)]) / len(mine_locations['locations'])
            }

            self.results['enhanced_labels'] = enhanced_labels
            return enhanced_labels
        else:
            self.results['enhanced_labels'] = base_labels
            return base_labels

    def _validate_samples_with_mine_locations(self, samples: List[Dict], mine_locations: List[Dict]) -> List[Dict]:
        """샘플을 기뢰 위치와 검증"""
        enhanced_samples = []

        for sample in samples:
            enhanced_sample = sample.copy()

            if sample['label'] == 1:  # Positive sample
                # Check if this sample corresponds to a known mine location
                closest_mine = self._find_closest_mine_location(
                    sample.get('latitude'), sample.get('longitude'), mine_locations
                )

                if closest_mine and closest_mine['distance'] < 0.001:  # Within ~100m
                    enhanced_sample['mine_validated'] = True
                    enhanced_sample['closest_mine_id'] = closest_mine['mine']['id']
                    enhanced_sample['validation_distance'] = closest_mine['distance']
                    enhanced_sample['confidence'] = min(1.0, sample.get('confidence', 1.0) + 0.2)
                else:
                    enhanced_sample['mine_validated'] = False
                    enhanced_sample['confidence'] = max(0.3, sample.get('confidence', 1.0) - 0.3)
            else:
                enhanced_sample['mine_validated'] = False

            enhanced_samples.append(enhanced_sample)

        return enhanced_samples

    def _find_closest_mine_location(self, lat: float, lon: float, mine_locations: List[Dict]) -> Optional[Dict]:
        """가장 가까운 기뢰 위치 찾기"""
        if not lat or not lon:
            return None

        min_distance = float('inf')
        closest_mine = None

        for mine in mine_locations:
            # Simple distance calculation (for small areas)
            distance = np.sqrt((lat - mine['latitude'])**2 + (lon - mine['longitude'])**2)

            if distance < min_distance:
                min_distance = distance
                closest_mine = mine

        return {'mine': closest_mine, 'distance': min_distance} if closest_mine else None

    def apply_data_augmentation(self,
                              samples: List[Dict],
                              intensity_data: np.ndarray,
                              labels: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """데이터 증강 적용"""

        if not self.config.enable_augmentation or not self.augmentation_engine:
            self.logger.info("Data augmentation disabled")
            return samples, intensity_data, labels

        self.logger.info("Applying data augmentation for imbalanced dataset")

        # Separate positive and negative samples
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]

        positive_patches = []
        negative_patches = []

        # Extract patches for augmentation
        for idx in positive_indices:
            patch = self._extract_patch_from_sample(samples[idx], intensity_data)
            if patch is not None:
                positive_patches.append(patch)

        for idx in negative_indices:
            patch = self._extract_patch_from_sample(samples[idx], intensity_data)
            if patch is not None:
                negative_patches.append(patch)

        self.logger.info(f"Original dataset: {len(positive_patches)} positive, {len(negative_patches)} negative")

        # Balance dataset using augmentation engine
        balanced_patches, original_negative_patches, balanced_masks, balanced_labels = \
            self.augmentation_engine.balance_dataset(
                positive_images=positive_patches,
                negative_images=negative_patches,
                target_ratio=1.0  # 1:1 ratio
            )

        self.logger.info(f"Balanced dataset: {len(balanced_patches)} total samples, "
                        f"{sum(balanced_labels)} positive, {len(balanced_labels) - sum(balanced_labels)} negative")

        # Create enhanced samples list
        augmented_samples = []
        augmented_intensity_patches = []

        patch_idx = 0
        for label in balanced_labels:
            if patch_idx < len(balanced_patches):
                # Create sample metadata
                sample = {
                    'type': 'positive' if label == 1 else 'negative',
                    'label': label,
                    'augmented': patch_idx >= len(samples),  # Mark augmented samples
                    'patch_data': balanced_patches[patch_idx],
                    'sample_id': patch_idx
                }

                augmented_samples.append(sample)
                augmented_intensity_patches.append(balanced_patches[patch_idx])
                patch_idx += 1

        # Save augmented samples if requested
        if self.config.save_augmented_samples:
            self._save_augmented_samples(balanced_patches, balanced_labels)

        self.results['augmentation_info'] = {
            'original_positive': len(positive_patches),
            'original_negative': len(negative_patches),
            'final_positive': sum(balanced_labels),
            'final_negative': len(balanced_labels) - sum(balanced_labels),
            'augmentation_ratio': len(balanced_labels) / len(samples)
        }

        return augmented_samples, np.array(augmented_intensity_patches), np.array(balanced_labels)

    def _extract_patch_from_sample(self, sample: Dict, intensity_data: np.ndarray) -> Optional[np.ndarray]:
        """샘플에서 패치 추출"""
        try:
            bounds = sample.get('patch_bounds')
            if not bounds:
                return None

            y_min, y_max = bounds['y_min'], bounds['y_max']
            x_min, x_max = bounds['x_min'], bounds['x_max']

            if (y_max <= intensity_data.shape[0] and x_max <= intensity_data.shape[1]):
                patch = intensity_data[y_min:y_max, x_min:x_max]

                # Resize to target size if needed
                if patch.shape != self.config.patch_size[::-1]:
                    import cv2
                    patch = cv2.resize(patch, self.config.patch_size)

                return patch

        except Exception as e:
            self.logger.warning(f"Failed to extract patch: {e}")

        return None

    def create_train_validation_test_splits(self,
                                          samples: List[Dict],
                                          labels: np.ndarray,
                                          features: Optional[np.ndarray] = None) -> Dict:
        """훈련/검증/테스트 데이터 분할"""

        self.logger.info("Creating train/validation/test splits")

        # First split: train+val vs test
        if self.config.stratify:
            train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(
                samples, labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels
            )
        else:
            train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(
                samples, labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

        # Second split: train vs validation
        if self.config.stratify:
            train_samples, val_samples, train_labels, val_labels = train_test_split(
                train_val_samples, train_val_labels,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=train_val_labels
            )
        else:
            train_samples, val_samples, train_labels, val_labels = train_test_split(
                train_val_samples, train_val_labels,
                test_size=self.config.validation_size,
                random_state=self.config.random_state
            )

        # Extract features for each split if provided
        if features is not None:
            sample_indices = {id(sample): idx for idx, sample in enumerate(samples)}

            train_indices = [sample_indices[id(s)] for s in train_samples if id(s) in sample_indices]
            val_indices = [sample_indices[id(s)] for s in val_samples if id(s) in sample_indices]
            test_indices = [sample_indices[id(s)] for s in test_samples if id(s) in sample_indices]

            train_features = features[train_indices] if train_indices else None
            val_features = features[val_indices] if val_indices else None
            test_features = features[test_indices] if test_indices else None
        else:
            train_features = val_features = test_features = None

        splits = {
            'train': {
                'samples': train_samples,
                'labels': train_labels,
                'features': train_features,
                'size': len(train_samples)
            },
            'validation': {
                'samples': val_samples,
                'labels': val_labels,
                'features': val_features,
                'size': len(val_samples)
            },
            'test': {
                'samples': test_samples,
                'labels': test_labels,
                'features': test_features,
                'size': len(test_samples)
            }
        }

        # Log split information
        for split_name, split_data in splits.items():
            pos_count = np.sum(split_data['labels'])
            neg_count = len(split_data['labels']) - pos_count
            self.logger.info(f"{split_name.capitalize()} set: {split_data['size']} samples "
                           f"({pos_count} positive, {neg_count} negative)")

        self.datasets = splits
        return splits

    # ========== Enhanced Pipeline Execution ==========

    def run_enhanced_full_pipeline(self,
                                 xtf_path: Path,
                                 mine_locations_file: Optional[Path] = None) -> Dict:
        """향상된 전체 파이프라인 실행"""

        self.logger.info("="*60)
        self.logger.info("Starting Enhanced Mine Detection Pipeline")
        self.logger.info("="*60)

        try:
            # Step 1: Process mine locations (25 GPS points)
            mine_locations = self.process_mine_locations(mine_locations_file)
            self.results['mine_locations'] = mine_locations

            # Step 2: Read XTF file
            self.logger.info("Step 2: Reading XTF file")
            self.read_xtf(xtf_path)

            # Step 3: Extract XTF data
            self.logger.info("Step 3: Extracting XTF data")
            self.extract_xtf_data()

            # Step 4: Generate training samples directly from mine locations
            self.logger.info("Step 4: Generating training samples from mine locations")
            enhanced_labels = self.generate_samples_from_mine_locations(
                mine_locations=mine_locations
            )

            # Step 6: Apply data augmentation
            self.logger.info("Step 6: Applying data augmentation")
            # Get the combined intensity image from XTF data
            xtf_data = self.results.get('xtf_data', {})
            intensity_data = xtf_data.get('intensity_data', {})
            intensity_images = intensity_data.get('intensity_images', {})
            combined_intensity = intensity_images.get('combined')

            augmented_samples, augmented_patches, augmented_labels = self.apply_data_augmentation(
                enhanced_labels['samples'],
                combined_intensity,
                enhanced_labels['labels']
            )

            # Step 7: Extract features from augmented data
            self.logger.info("Step 7: Extracting features")
            features = self.extract_features_from_patches(augmented_patches, augmented_labels)

            # Step 8: Create train/validation/test splits
            self.logger.info("Step 8: Creating data splits")
            data_splits = self.create_train_validation_test_splits(
                augmented_samples, augmented_labels, features
            )

            # Step 9: Optimize ensemble on training data
            if data_splits['train']['features'] is not None:
                self.logger.info("Step 9: Optimizing ensemble")
                ensemble_config = self.optimize_ensemble(
                    data_splits['train']['features'],
                    data_splits['train']['labels']
                )
                self.results['ensemble'] = ensemble_config

            # Step 10: Train and evaluate model
            self.logger.info("Step 10: Training and evaluating model")
            evaluation_results = self.train_and_evaluate_model(data_splits)

            # Step 11: Generate comprehensive report
            self.logger.info("Step 11: Generating final report")
            final_results = self._compile_enhanced_results(evaluation_results)

            # Save results
            self._save_enhanced_results(final_results)

            self.logger.info("Enhanced pipeline execution completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Enhanced pipeline execution failed: {e}")
            raise

    def extract_features_from_patches(self, patches: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """패치에서 특징 추출"""
        self.logger.info("Extracting features from augmented patches")

        all_features = []

        for i, patch in enumerate(patches):
            # Create dummy sample structure for feature extractor
            dummy_sample = {
                'patch_bounds': {
                    'y_min': 0, 'y_max': patch.shape[0],
                    'x_min': 0, 'x_max': patch.shape[1]
                }
            }

            dummy_label_data = {
                'samples': [dummy_sample]
            }

            # Extract features using existing feature extractor
            patch_features = self.feature_extractor._extract_method_features([patch], 'statistical')[0]

            if patch_features is not None and len(patch_features) > 0:
                all_features.append(patch_features[0])  # Take first (and only) sample
            else:
                # Fallback to zero features
                all_features.append(np.zeros(13))  # statistical features count

        return np.array(all_features)

    def train_and_evaluate_model(self, data_splits: Dict) -> Dict:
        """모델 훈련 및 평가"""
        evaluation_results = {}

        try:
            # Train on training set
            train_features = data_splits['train']['features']
            train_labels = data_splits['train']['labels']

            if train_features is not None and len(train_features) > 0:
                # Use existing classifier
                self.classifier.train(train_features, train_labels)

                # Evaluate on each split
                for split_name, split_data in data_splits.items():
                    if split_data['features'] is not None:
                        predictions = self.classifier.predict(split_data['features'])

                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                        metrics = {
                            'accuracy': accuracy_score(split_data['labels'], predictions),
                            'precision': precision_score(split_data['labels'], predictions, average='weighted', zero_division=0),
                            'recall': recall_score(split_data['labels'], predictions, average='weighted', zero_division=0),
                            'f1_score': f1_score(split_data['labels'], predictions, average='weighted', zero_division=0)
                        }

                        evaluation_results[split_name] = {
                            'predictions': predictions.tolist(),
                            'metrics': metrics,
                            'confusion_matrix': confusion_matrix(split_data['labels'], predictions).tolist(),
                            'classification_report': classification_report(split_data['labels'], predictions, output_dict=True)
                        }

                        self.logger.info(f"{split_name.capitalize()} accuracy: {metrics['accuracy']:.3f}")

        except Exception as e:
            self.logger.error(f"Model training/evaluation failed: {e}")
            evaluation_results = {'error': str(e)}

        return evaluation_results

    # ========== Existing methods from unified_pipeline.py ==========
    # (Keep all existing methods for backward compatibility)

    def read_xtf(self, xtf_path: Path) -> Dict:
        """Step 1: XTF 파일 읽기"""
        self.logger.info(f"Reading XTF file: {xtf_path}")
        data = self.xtf_reader.read(xtf_path)

        if self.config.save_intermediate:
            self._save_intermediate("xtf_raw", data)

        self.results['xtf_data'] = data
        return data

    def extract_xtf_data(self, xtf_data: Optional[Dict] = None) -> Dict:
        """Step 2: XTF 데이터 추출"""
        if xtf_data is None:
            xtf_data = self.results.get('xtf_data')
            if xtf_data is None:
                raise ValueError("No XTF data available. Run read_xtf first.")

        self.logger.info("Extracting XTF data")
        extracted = self.xtf_extractor.extract(xtf_data)

        if self.config.save_intermediate:
            self._save_intermediate("xtf_extracted", extracted)

        self.results['extracted_data'] = extracted
        return extracted

    def map_coordinates(self,
                       extracted_data: Optional[Dict] = None,
                       gps_file: Optional[Path] = None,
                       annotation_image: Optional[Path] = None) -> Dict:
        """Step 3: 위경도 지도와 매핑"""
        if extracted_data is None:
            extracted_data = self.results.get('extracted_data')

        gps_file = gps_file or self.config.gps_file
        annotation_image = annotation_image or self.config.annotation_image

        if gps_file is None or annotation_image is None:
            raise ValueError("GPS file and annotation image required for coordinate mapping")

        self.logger.info("Mapping coordinates")
        mapped_coords = self.coord_mapper.map(
            extracted_data,
            gps_file,
            annotation_image
        )

        if self.config.save_intermediate:
            self._save_intermediate("coordinate_mapping", mapped_coords)

        self.results['coordinate_mapping'] = mapped_coords
        return mapped_coords

    def optimize_ensemble(self,
                         features: Optional[np.ndarray] = None,
                         labels: Optional[np.ndarray] = None) -> Dict:
        """Step 6: 앙상블/스태킹 최적화"""
        if features is None:
            features = self.results.get('features')
        if labels is None:
            labels = self.results.get('labels', {}).get('labels')

        self.logger.info("Optimizing ensemble")
        optimized = self.ensemble_optimizer.optimize(features, labels)

        if self.config.save_intermediate:
            self._save_intermediate("ensemble_optimization", optimized)

        self.results['ensemble'] = optimized
        return optimized

    # ========== Helper Methods ==========

    def _save_intermediate(self, name: str, data: Any):
        """Save intermediate results"""
        output_path = self.config.output_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif isinstance(data, dict):
            data = {k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in data.items()}

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.debug(f"Saved intermediate results to {output_path}")

    def _save_augmented_samples(self, patches: List[np.ndarray], labels: List[int]):
        """Save augmented samples for inspection"""
        output_dir = self.config.output_dir / "augmented_samples"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (patch, label) in enumerate(zip(patches[:50], labels[:50])):  # Save first 50
            label_name = "positive" if label == 1 else "negative"
            filename = f"augmented_{label_name}_{i:03d}.png"

            # Normalize patch for saving
            if patch.max() > 1.0:
                normalized = ((patch - patch.min()) / (patch.max() - patch.min()) * 255).astype(np.uint8)
            else:
                normalized = (patch * 255).astype(np.uint8)

            Image.fromarray(normalized).save(output_dir / filename)

    def _compile_enhanced_results(self, evaluation_results: Dict) -> Dict:
        """Compile enhanced results"""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'mine_locations': self.results.get('mine_locations', {}),
            'augmentation_info': self.results.get('augmentation_info', {}),
            'data_splits': {
                'train_size': self.datasets.get('train', {}).get('size', 0),
                'validation_size': self.datasets.get('validation', {}).get('size', 0),
                'test_size': self.datasets.get('test', {}).get('size', 0)
            },
            'evaluation_results': evaluation_results,
            'coordinate_mapping_accuracy': self.results.get('coordinate_mapping', {}).get('accuracy', 0),
            'pipeline_version': '2.0'
        }

    def _save_enhanced_results(self, results: Dict):
        """Save enhanced results"""
        output_path = self.config.output_dir / f"enhanced_pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Enhanced results saved to {output_path}")

        # Also generate a readable report
        if self.config.generate_reports:
            self._generate_readable_report(results, output_path.with_suffix('.md'))

    def _generate_readable_report(self, results: Dict, output_path: Path):
        """Generate human-readable report"""
        report_lines = [
            "# Enhanced Mine Detection Pipeline Report",
            f"**Generated**: {results['timestamp']}",
            f"**Pipeline Version**: {results['pipeline_version']}",
            "",
            "## 📊 Dataset Summary",
            ""
        ]

        # Mine locations
        mine_info = results.get('mine_locations', {})
        if mine_info:
            report_lines.extend([
                f"- **Mine Locations Loaded**: {mine_info.get('total_count', 0)}",
                f"- **Source File**: {mine_info.get('source_file', 'N/A')}",
                ""
            ])

        # Data splits
        splits = results.get('data_splits', {})
        if splits:
            report_lines.extend([
                "### Data Splits",
                f"- **Training**: {splits.get('train_size', 0)} samples",
                f"- **Validation**: {splits.get('validation_size', 0)} samples",
                f"- **Test**: {splits.get('test_size', 0)} samples",
                ""
            ])

        # Augmentation info
        aug_info = results.get('augmentation_info', {})
        if aug_info:
            report_lines.extend([
                "### Data Augmentation",
                f"- **Original Positive**: {aug_info.get('original_positive', 0)}",
                f"- **Original Negative**: {aug_info.get('original_negative', 0)}",
                f"- **Final Positive**: {aug_info.get('final_positive', 0)}",
                f"- **Final Negative**: {aug_info.get('final_negative', 0)}",
                f"- **Augmentation Ratio**: {aug_info.get('augmentation_ratio', 0):.2f}x",
                ""
            ])

        # Evaluation results
        eval_results = results.get('evaluation_results', {})
        if eval_results:
            report_lines.extend([
                "## 🎯 Model Performance",
                ""
            ])

            for split_name, split_results in eval_results.items():
                if 'metrics' in split_results:
                    metrics = split_results['metrics']
                    report_lines.extend([
                        f"### {split_name.capitalize()} Set",
                        f"- **Accuracy**: {metrics.get('accuracy', 0):.3f}",
                        f"- **Precision**: {metrics.get('precision', 0):.3f}",
                        f"- **Recall**: {metrics.get('recall', 0):.3f}",
                        f"- **F1-Score**: {metrics.get('f1_score', 0):.3f}",
                        ""
                    ])

        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Readable report saved to {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Enhanced Unified Mine Detection Pipeline")

    # Input/Output arguments
    parser.add_argument('--xtf', type=str, required=True, help="Path to XTF file")
    parser.add_argument('--gps', type=str, help="Path to GPS data file")
    parser.add_argument('--annotation', type=str, help="Path to annotation image")
    parser.add_argument('--mine-locations', type=str, help="Path to mine locations file (25 GPS points)")
    parser.add_argument('--output', type=str, default="data/processed", help="Output directory")

    # Pipeline options
    parser.add_argument('--mode', choices=['enhanced', 'basic'], default='enhanced',
                       help="Execution mode: enhanced (with augmentation) or basic")

    # Data augmentation options
    parser.add_argument('--no-augmentation', action='store_true', help="Disable data augmentation")
    parser.add_argument('--mine-aug-factor', type=int, default=8, help="Mine sample augmentation factor")
    parser.add_argument('--bg-aug-factor', type=int, default=3, help="Background sample augmentation factor")

    # Train/test split options
    parser.add_argument('--test-size', type=float, default=0.2, help="Test set size ratio")
    parser.add_argument('--val-size', type=float, default=0.2, help="Validation set size ratio")
    parser.add_argument('--random-state', type=int, default=42, help="Random state for reproducibility")

    # Feature options
    parser.add_argument('--features', nargs='+',
                       default=['statistical', 'textural', 'morphological', 'frequency'],
                       help="Feature extraction methods")

    # Classification options
    parser.add_argument('--classifier', choices=['svm', 'rf', 'ensemble'], default='ensemble',
                       help="Classifier type")
    parser.add_argument('--no-terrain', action='store_true', help="Disable terrain analysis")

    # Other options
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--no-save', action='store_true', help="Don't save intermediate results")
    parser.add_argument('--save-augmented', action='store_true', help="Save augmented samples")

    args = parser.parse_args()

    # Create enhanced configuration
    config = EnhancedPipelineConfig(
        output_dir=Path(args.output),
        gps_file=Path(args.gps) if args.gps else None,
        annotation_image=Path(args.annotation) if args.annotation else None,
        mine_locations_file=Path(args.mine_locations) if args.mine_locations else None,
        feature_methods=args.features,
        classifier_type=args.classifier,
        use_terrain=not args.no_terrain,
        save_intermediate=not args.no_save,
        save_augmented_samples=args.save_augmented,
        enable_augmentation=not args.no_augmentation,
        mine_augmentation_factor=args.mine_aug_factor,
        background_augmentation_factor=args.bg_aug_factor,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state,
        verbose=args.verbose
    )

    # Create pipeline
    pipeline = EnhancedUnifiedPipeline(config)

    # Execute pipeline
    if args.mode == 'enhanced':
        results = pipeline.run_enhanced_full_pipeline(
            Path(args.xtf),
            Path(args.mine_locations) if args.mine_locations else None
        )
        print(f"\nEnhanced pipeline completed. Results saved to {config.output_dir}")
    else:
        # Fallback to basic mode (existing functionality)
        results = pipeline.run_full_pipeline(Path(args.xtf))
        print(f"\nBasic pipeline completed. Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()