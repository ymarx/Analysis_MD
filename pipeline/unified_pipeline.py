"""
Unified Mine Detection Pipeline
================================
통합 기뢰 탐지 파이프라인 - 모듈별 실행 및 전체 파이프라인 지원

Author: YMARX
Date: 2024-09-16
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from pipeline.modules.xtf_reader import XTFReader
from pipeline.modules.xtf_extractor import XTFExtractor
from pipeline.modules.coordinate_mapper import CoordinateMapper
from pipeline.modules.label_generator import LabelGenerator
from pipeline.modules.feature_extractor import FeatureExtractor
from pipeline.modules.ensemble_optimizer import EnsembleOptimizer
from pipeline.modules.mine_classifier import MineClassifier
from pipeline.modules.terrain_analyzer import TerrainAnalyzer
from pipeline.modules.gps_parser import GPSParser

# Enhanced imports for data augmentation and train/test splitting
try:
    from src.data_augmentation.augmentation_engine import DataAugmentationEngine
    from sklearn.model_selection import train_test_split
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Enhanced Pipeline configuration with data augmentation and advanced features"""
    # Paths
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

    # Classification
    classifier_type: str = "ensemble"
    use_terrain: bool = True

    # Data Augmentation (Enhanced feature)
    enable_augmentation: bool = True
    mine_augmentation_factor: int = 8
    background_augmentation_factor: int = 3

    # Train/Test Splitting (Enhanced feature)
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Mine Location Processing (Enhanced feature)
    mine_locations_file: Optional[Path] = None

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


class UnifiedPipeline:
    """통합 기뢰 탐지 파이프라인"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize modules
        self._init_modules()

        # Results storage
        self.results = {}

    def _init_modules(self):
        """Initialize pipeline modules with enhanced features"""
        self.xtf_reader = XTFReader()
        self.xtf_extractor = XTFExtractor(
            sample_rate=self.config.xtf_sample_rate,
            channels=self.config.xtf_channels
        )
        self.coord_mapper = CoordinateMapper(
            transform_type=self.config.transform_type
        )
        self.label_generator = LabelGenerator()
        self.feature_extractor = FeatureExtractor(
            methods=self.config.feature_methods,
            patch_size=self.config.patch_size
        )
        self.ensemble_optimizer = EnsembleOptimizer()

        # Enhanced features initialization
        self.gps_parser = GPSParser()

        # Data augmentation engine (if available)
        if ENHANCED_FEATURES_AVAILABLE and self.config.enable_augmentation:
            try:
                self.augmentation_engine = DataAugmentationEngine()
                self.logger.info("Data augmentation engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize augmentation engine: {e}")
                self.augmentation_engine = None
        else:
            self.augmentation_engine = None

        # Storage for enhanced datasets
        self.datasets = {}
        self.classifier = MineClassifier(
            classifier_type=self.config.classifier_type
        )
        self.terrain_analyzer = TerrainAnalyzer() if self.config.use_terrain else None

    # ========== Module-level Operations ==========

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

    def generate_labels(self,
                       coordinate_mapping: Optional[Dict] = None,
                       intensity_data: Optional[np.ndarray] = None) -> Dict:
        """Step 4: 레이블 생성"""
        if coordinate_mapping is None:
            coordinate_mapping = self.results.get('coordinate_mapping')
        if intensity_data is None:
            intensity_data = self.results.get('extracted_data', {}).get('intensity')

        self.logger.info("Generating labels")
        labels = self.label_generator.generate(
            coordinate_mapping,
            intensity_data
        )

        if self.config.save_intermediate:
            self._save_intermediate("labels", labels)

        self.results['labels'] = labels
        return labels

    def extract_features(self,
                        intensity_data: Optional[np.ndarray] = None,
                        labels: Optional[Dict] = None) -> np.ndarray:
        """Step 5: 특징 추출"""
        if intensity_data is None:
            intensity_data = self.results.get('extracted_data', {}).get('intensity')
        if labels is None:
            labels = self.results.get('labels')

        self.logger.info("Extracting features")
        features_result = self.feature_extractor.extract(
            intensity_data,
            labels
        )

        # Extract the numpy array from the feature extractor result
        if isinstance(features_result, dict):
            features = features_result.get('features')
            feature_names = features_result.get('feature_names', [])
            self.results['feature_names'] = feature_names
            self.results['feature_info'] = features_result.get('extraction_info', {})
        else:
            features = features_result

        if self.config.save_intermediate:
            self._save_intermediate("features", features)

        self.results['features'] = features
        return features

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

    def classify(self,
                features: Optional[np.ndarray] = None,
                ensemble_config: Optional[Dict] = None) -> np.ndarray:
        """Step 7: 분류"""
        if features is None:
            features = self.results.get('features')
        if ensemble_config is None:
            ensemble_config = self.results.get('ensemble')

        # Provide default ensemble config if none exists
        if ensemble_config is None:
            self.logger.info("No ensemble config found, using default configuration")
            from sklearn.ensemble import RandomForestClassifier

            ensemble_config = {
                'best_config': {
                    'type': 'individual',
                    'config': ('random_forest', {
                        'model': RandomForestClassifier(n_estimators=100, random_state=42),
                        'params': {'n_estimators': 100, 'random_state': 42}
                    })
                },
                'performance': {'cv_score': 0.0}
            }

        self.logger.info("Classifying objects")
        predictions = self.classifier.classify(
            features,
            ensemble_config
        )

        if self.config.save_intermediate:
            self._save_intermediate("predictions", predictions)

        self.results['predictions'] = predictions
        return predictions

    def analyze_terrain(self,
                       intensity_data: Optional[np.ndarray] = None,
                       predictions: Optional[np.ndarray] = None) -> Dict:
        """Step 8: 해저 지형 고려 분류"""
        if not self.config.use_terrain:
            self.logger.info("Terrain analysis disabled")
            return {}

        if intensity_data is None:
            intensity_data = self.results.get('extracted_data', {}).get('intensity')
        if predictions is None:
            predictions = self.results.get('predictions')

        self.logger.info("Analyzing terrain context")
        terrain_results = self.terrain_analyzer.analyze(
            intensity_data,
            predictions
        )

        if self.config.save_intermediate:
            self._save_intermediate("terrain_analysis", terrain_results)

        self.results['terrain_analysis'] = terrain_results
        return terrain_results

    # ========== Full Pipeline Execution ==========

    def run_full_pipeline(self, xtf_path: Path) -> Dict:
        """전체 파이프라인 실행"""
        self.logger.info("="*50)
        self.logger.info("Starting full pipeline execution")
        self.logger.info("="*50)

        try:
            # Step 1: Read XTF
            self.read_xtf(xtf_path)

            # Step 2: Extract XTF data
            self.extract_xtf_data()

            # Step 3: Map coordinates (if GPS and annotation available)
            if self.config.gps_file and self.config.annotation_image:
                self.map_coordinates()

            # Step 4: Generate labels
            self.generate_labels()

            # Step 5: Extract features
            self.extract_features()

            # Step 6: Optimize ensemble
            if self.results.get('labels', {}).get('labels') is not None:
                self.optimize_ensemble()

            # Step 7: Classify
            self.classify()

            # Step 8: Analyze terrain
            if self.config.use_terrain:
                self.analyze_terrain()

            # Compile final results
            final_results = self._compile_results()

            # Save final results
            self._save_final_results(final_results)

            self.logger.info("Pipeline execution completed successfully")
            return final_results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

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

    def _compile_results(self) -> Dict:
        """Compile all results into final output"""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': {
                'predictions': self.results.get('predictions', []),
                'features': self.results.get('features', {}).get('shape', None) if isinstance(self.results.get('features'), np.ndarray) else None,
                'coordinate_mapping': len(self.results.get('coordinate_mapping', [])) if self.results.get('coordinate_mapping') else 0,
                'terrain_analysis': self.results.get('terrain_analysis', {})
            },
            'performance': self._calculate_performance_metrics()
        }

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics if ground truth available"""
        metrics = {}

        if 'labels' in self.results and 'predictions' in self.results:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            y_true = self.results['labels'].get('labels', [])
            y_pred = self.results.get('predictions', [])

            if len(y_true) == len(y_pred) and len(y_true) > 0:
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }

        return metrics

    def _save_final_results(self, results: Dict):
        """Save final results"""
        output_path = self.config.output_dir / f"pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Final results saved to {output_path}")

    # ========== Enhanced Features Methods ==========

    def process_mine_locations(self, locations_file: Optional[Path] = None) -> Dict:
        """Process mine locations from GPS file (Enhanced feature)"""
        locations_file = locations_file or self.config.mine_locations_file
        if not locations_file:
            self.logger.warning("No mine locations file provided")
            return {'total_count': 0, 'locations': []}

        try:
            self.logger.info(f"Processing mine locations from {locations_file}")
            locations = self.gps_parser.parse_gps_file(locations_file)
            validation = self.gps_parser.validate_coordinates(locations)

            self.logger.info(f"Loaded {len(locations)} mine locations")
            self.logger.info(f"Validation: {validation['valid_count']}/{validation['total_count']} valid coordinates")

            return {
                'total_count': len(locations),
                'valid_count': validation['valid_count'],
                'locations': locations,
                'validation': validation,
                'source_file': str(locations_file)
            }

        except Exception as e:
            self.logger.error(f"Failed to process mine locations: {e}")
            return {'total_count': 0, 'locations': []}

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

                sample = {
                    'patch_bounds': {
                        'y_min': y_min,
                        'y_max': y_max,
                        'x_min': x_min,
                        'x_max': x_max
                    },
                    'location_id': location.get('id', f'mine_{len(samples)}'),
                    'pixel_y': pixel_y,
                    'pixel_x': pixel_x
                }
                samples.append(sample)
                labels.append(1)  # Positive label for mine

                self.logger.debug(f"Mine at ({lat:.6f}, {lon:.6f}) -> pixel ({pixel_x}, {pixel_y})")

        # Generate negative samples (background)
        import random
        num_negative = len(samples) * 3  # 3x more negative samples
        patch_size = self.config.patch_size[0]
        half_patch = patch_size // 2

        for i in range(num_negative):
            # Random location away from mine locations
            attempts = 0
            while attempts < 50:
                y_center = random.randint(half_patch, combined_image.shape[0] - half_patch)
                x_center = random.randint(half_patch, combined_image.shape[1] - half_patch)

                # Check distance from mine locations
                too_close = False
                for sample in samples[:len(locations_in_bounds)]:  # Only check mine samples
                    mine_y = sample['pixel_y']
                    mine_x = sample['pixel_x']
                    distance = ((y_center - mine_y)**2 + (x_center - mine_x)**2)**0.5
                    if distance < patch_size * 2:  # Keep distance from mines
                        too_close = True
                        break

                attempts += 1
                if not too_close:
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

    def run_enhanced_mine_detection_pipeline(self, xtf_path: Path, mine_locations_file: Optional[Path] = None) -> Dict:
        """Enhanced pipeline with mine location processing and data augmentation"""
        self.logger.info("="*60)
        self.logger.info("Starting Enhanced Mine Detection Pipeline")
        self.logger.info("="*60)

        try:
            # Step 1: Process mine locations
            mine_locations = self.process_mine_locations(mine_locations_file)
            self.results['mine_locations'] = mine_locations

            # Step 2: Read XTF file
            self.logger.info("Step 2: Reading XTF file")
            self.read_xtf(xtf_path)

            # Step 3: Extract XTF data (enhanced approach with working reader)
            self.logger.info("Step 3: Extracting XTF data (enhanced)")

            # Get intensity data directly from working XTF reader output
            xtf_data = self.results.get('xtf_data', {})
            intensity_data_raw = xtf_data.get('intensity_data', {})
            intensity_images = intensity_data_raw.get('intensity_images', {})

            if 'combined' in intensity_images:
                # Store intensity data in the expected format for feature extraction
                extracted_data = {
                    'intensity': intensity_images['combined'],
                    'navigation': intensity_data_raw.get('navigation', {}),
                    'metadata': xtf_data.get('summary', {})
                }
                self.results['extracted_data'] = extracted_data
                self.logger.info(f"Enhanced XTF extraction completed: intensity shape {intensity_images['combined'].shape}")
            else:
                # Fallback to regular extraction
                self.extract_xtf_data()

            # Step 4: Generate samples from mine locations
            self.logger.info("Step 4: Generating samples from mine locations")
            label_data = self.generate_samples_from_mine_locations(mine_locations)
            self.results['label_data'] = label_data
            self.results['labels'] = label_data  # Also store as 'labels' for extract_features() compatibility

            # Step 5: Extract features and classify
            self.logger.info("Step 5: Extracting features")
            self.extract_features()

            # Step 6: Optimize ensemble (if configured)
            if getattr(self.config, 'enable_ensemble', False):
                self.logger.info("Step 6: Optimizing ensemble")
                labels_array = label_data.get('labels', [])
                self.optimize_ensemble(labels=labels_array)

            self.logger.info("Step 7: Training and evaluating model")

            # Train the model first
            features = self.results.get('features')
            labels_array = label_data.get('labels', [])
            ensemble_config = self.results.get('ensemble')

            if features is not None and len(labels_array) > 0:
                # Train with default ensemble config if none exists
                if ensemble_config is None:
                    from sklearn.ensemble import RandomForestClassifier
                    ensemble_config = {
                        'best_config': {
                            'type': 'individual',
                            'config': ('random_forest', {
                                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                                'params': {'n_estimators': 100, 'random_state': 42}
                            })
                        },
                        'performance': {'cv_score': 0.0}
                    }

                # Train the model
                training_results = self.classifier.train(features, labels_array, ensemble_config)
                self.results['training_results'] = training_results
                self.logger.info(f"Model training completed with accuracy: {training_results.get('test_accuracy', 'N/A')}")

            # Now classify
            self.classify()

            # Compile results
            results = {
                'mine_locations': mine_locations,
                'pipeline_results': self.results,
                'performance_metrics': self._calculate_performance_metrics()
            }

            if self.config.save_intermediate:
                self._save_final_results(results)

            self.logger.info("Enhanced pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Enhanced pipeline failed: {e}")
            raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Unified Mine Detection Pipeline")

    # Input/Output arguments
    parser.add_argument('--xtf', type=str, help="Path to XTF file")
    parser.add_argument('--gps', type=str, help="Path to GPS data file")
    parser.add_argument('--annotation', type=str, help="Path to annotation image")
    parser.add_argument('--output', type=str, default="data/processed", help="Output directory")

    # Pipeline options
    parser.add_argument('--mode', choices=['full', 'modular', 'enhanced'], default='full',
                       help="Execution mode: full pipeline, modular steps, or enhanced (with mine locations)")
    parser.add_argument('--steps', nargs='+',
                       choices=['read', 'extract', 'map', 'label', 'feature', 'ensemble', 'classify', 'terrain'],
                       help="Specific steps to run in modular mode")

    # Enhanced features
    parser.add_argument('--mine-locations', type=str, help="Path to mine locations GPS file")
    parser.add_argument('--enable-augmentation', action='store_true', help="Enable data augmentation")
    parser.add_argument('--augmentation-factor', type=int, default=5, help="Data augmentation factor")

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

    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig(
        output_dir=Path(args.output),
        gps_file=Path(args.gps) if args.gps else None,
        annotation_image=Path(args.annotation) if args.annotation else None,
        feature_methods=args.features,
        classifier_type=args.classifier,
        use_terrain=not args.no_terrain,
        save_intermediate=not args.no_save,
        verbose=args.verbose,
        # Enhanced features
        mine_locations_file=Path(args.mine_locations) if args.mine_locations else None,
        enable_augmentation=args.enable_augmentation,
        mine_augmentation_factor=args.augmentation_factor
    )

    # Create pipeline
    pipeline = UnifiedPipeline(config)

    # Execute based on mode
    if args.mode == 'full' and args.xtf:
        # Full pipeline execution
        results = pipeline.run_full_pipeline(Path(args.xtf))
        print(f"\nPipeline completed. Results saved to {config.output_dir}")

    elif args.mode == 'enhanced' and args.xtf:
        # Enhanced pipeline execution with mine locations
        results = pipeline.run_enhanced_mine_detection_pipeline(
            Path(args.xtf),
            Path(args.mine_locations) if args.mine_locations else None
        )
        print(f"\nEnhanced pipeline completed. Results saved to {config.output_dir}")
        print(f"Mine locations processed: {results['mine_locations']['total_count']}")

    elif args.mode == 'modular' and args.steps:
        # Modular execution
        print(f"Running steps: {args.steps}")

        for step in args.steps:
            if step == 'read' and args.xtf:
                pipeline.read_xtf(Path(args.xtf))
            elif step == 'extract':
                pipeline.extract_xtf_data()
            elif step == 'map':
                pipeline.map_coordinates()
            elif step == 'label':
                pipeline.generate_labels()
            elif step == 'feature':
                pipeline.extract_features()
            elif step == 'ensemble':
                pipeline.optimize_ensemble()
            elif step == 'classify':
                pipeline.classify()
            elif step == 'terrain':
                pipeline.analyze_terrain()

        print(f"\nModular execution completed. Results saved to {config.output_dir}")

    else:
        print("Please provide --xtf file for full mode or --steps for modular mode")
        parser.print_help()


if __name__ == "__main__":
    main()