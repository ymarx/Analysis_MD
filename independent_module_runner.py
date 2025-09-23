#!/usr/bin/env python3
"""
Independent Module Runner
========================
각 모듈을 독립적으로 실행하고 결과를 체계적으로 저장

Author: YMARX
Date: 2024-09-22
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import pickle

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.unified_pipeline import UnifiedPipeline, PipelineConfig
from pipeline.modules.gps_parser import GPSParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndependentModuleRunner:
    """독립 모듈 실행 및 결과 관리"""

    def __init__(self, output_base_dir: Path = None):
        """
        Args:
            output_base_dir: 결과 저장 기본 디렉토리
        """
        self.output_base_dir = output_base_dir or Path("analysis_results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"session_{self.timestamp}"

        # Create directory structure
        self._setup_directories()

        # Initialize modules
        self.pipeline = None
        self.gps_parser = GPSParser()
        # Note: CoordinateMapper will be initialized when needed

        # Store results
        self.results = {}

    def _setup_directories(self):
        """결과 저장 디렉토리 구조 생성"""
        directories = [
            "01_intensity_extraction",
            "02_coordinate_mapping",
            "03_data_augmentation",
            "04_feature_extraction",
            "05_visualizations",
            "06_reports"
        ]

        for dir_name in directories:
            dir_path = self.session_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created result directories in: {self.session_dir}")

    def run_module_1_intensity_extraction(self, xtf_file: Path):
        """모듈 1: 강도 데이터 추출"""
        logger.info("="*60)
        logger.info("MODULE 1: INTENSITY DATA EXTRACTION")
        logger.info("="*60)

        try:
            # Initialize pipeline for XTF reading
            config = PipelineConfig(
                data_dir=Path("datasets"),
                output_dir=self.session_dir / "01_intensity_extraction",
                save_intermediate=True,
                verbose=True
            )
            self.pipeline = UnifiedPipeline(config)

            # Read XTF file
            logger.info(f"Reading XTF file: {xtf_file}")
            self.pipeline.read_xtf(xtf_file)

            # Get intensity data
            xtf_data = self.pipeline.results.get('xtf_data', {})
            intensity_data = xtf_data.get('intensity_data', {})
            intensity_images = intensity_data.get('intensity_images', {})
            navigation = intensity_data.get('navigation', {})

            if 'combined' in intensity_images:
                combined_image = intensity_images['combined']
                logger.info(f"✅ Extracted intensity data: shape {combined_image.shape}")

                # Save intensity data
                output_file = self.session_dir / "01_intensity_extraction" / "intensity_data.npz"
                np.savez_compressed(
                    output_file,
                    combined=combined_image,
                    port=intensity_images.get('port', []),
                    starboard=intensity_images.get('starboard', []),
                    latitudes=navigation.get('latitudes', []),
                    longitudes=navigation.get('longitudes', [])
                )
                logger.info(f"Saved intensity data to: {output_file}")

                # Create visualization
                self._visualize_intensity(combined_image)

                # Store results
                self.results['intensity'] = {
                    'shape': combined_image.shape,
                    'data': intensity_images,
                    'navigation': navigation,
                    'file': str(output_file)
                }

                return True
            else:
                logger.error("Failed to extract intensity data")
                return False

        except Exception as e:
            logger.error(f"Module 1 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_module_2_coordinate_mapping(self, mine_locations_file: Path):
        """모듈 2: 위경도 데이터 매핑 및 픽셀 레이블 생성"""
        logger.info("="*60)
        logger.info("MODULE 2: COORDINATE MAPPING & PIXEL LABELS")
        logger.info("="*60)

        try:
            # Parse mine locations
            logger.info(f"Parsing mine locations from: {mine_locations_file}")
            mine_locations = self.gps_parser.parse_gps_file(mine_locations_file)
            validation = self.gps_parser.validate_coordinates(mine_locations)

            logger.info(f"Parsed {len(mine_locations)} mine locations")
            logger.info(f"Valid coordinates: {validation['valid_count']}/{validation['total_count']}")

            # Get intensity data bounds
            intensity_data = self.results.get('intensity', {})
            navigation = intensity_data.get('navigation', {})
            latitudes = navigation.get('latitudes', [])
            longitudes = navigation.get('longitudes', [])

            if len(latitudes) == 0 or len(longitudes) == 0:
                logger.error("No navigation data available")
                return False

            # Calculate bounds
            lat_min, lat_max = min(latitudes), max(latitudes)
            lon_min, lon_max = min(longitudes), max(longitudes)

            logger.info(f"XTF data bounds:")
            logger.info(f"  Latitude: [{lat_min:.6f}, {lat_max:.6f}]")
            logger.info(f"  Longitude: [{lon_min:.6f}, {lon_max:.6f}]")

            # Map mine locations to pixels
            combined_image = intensity_data['data']['combined']
            height, width = combined_image.shape

            pixel_labels = []
            mapping_report = []

            for i, location in enumerate(mine_locations):
                lat = location['latitude']
                lon = location['longitude']

                # Check if within bounds
                in_bounds = (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max)

                if in_bounds:
                    # Convert to pixel coordinates
                    pixel_y = int((lat - lat_min) / (lat_max - lat_min) * (height - 1))
                    pixel_x = int((lon - lon_min) / (lon_max - lon_min) * (width - 1))

                    pixel_labels.append({
                        'mine_id': f"mine_{i+1}",
                        'gps_lat': lat,
                        'gps_lon': lon,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'in_bounds': True
                    })

                    status = "✅ MAPPED"
                else:
                    pixel_labels.append({
                        'mine_id': f"mine_{i+1}",
                        'gps_lat': lat,
                        'gps_lon': lon,
                        'pixel_x': -1,
                        'pixel_y': -1,
                        'in_bounds': False
                    })
                    status = "❌ OUT OF BOUNDS"

                mapping_report.append({
                    'Mine ID': f"mine_{i+1}",
                    'GPS Lat': f"{lat:.6f}",
                    'GPS Lon': f"{lon:.6f}",
                    'Status': status,
                    'Pixel X': pixel_labels[-1]['pixel_x'],
                    'Pixel Y': pixel_labels[-1]['pixel_y']
                })

            # Calculate mapping statistics
            in_bounds_count = sum(1 for label in pixel_labels if label['in_bounds'])
            out_bounds_count = len(pixel_labels) - in_bounds_count

            logger.info(f"Mapping results:")
            logger.info(f"  ✅ In bounds: {in_bounds_count}")
            logger.info(f"  ❌ Out of bounds: {out_bounds_count}")

            # Save mapping results
            output_file = self.session_dir / "02_coordinate_mapping" / "pixel_labels.json"
            with open(output_file, 'w') as f:
                json.dump(pixel_labels, f, indent=2)
            logger.info(f"Saved pixel labels to: {output_file}")

            # Save mapping report
            report_df = pd.DataFrame(mapping_report)
            report_file = self.session_dir / "02_coordinate_mapping" / "mapping_report.csv"
            report_df.to_csv(report_file, index=False)
            logger.info(f"Saved mapping report to: {report_file}")

            # Create verification visualization
            self._visualize_coordinate_mapping(
                combined_image, pixel_labels,
                (lat_min, lat_max), (lon_min, lon_max)
            )

            # Store results
            self.results['coordinate_mapping'] = {
                'pixel_labels': pixel_labels,
                'mapping_report': mapping_report,
                'statistics': {
                    'total_mines': len(pixel_labels),
                    'in_bounds': in_bounds_count,
                    'out_of_bounds': out_bounds_count,
                    'mapping_rate': in_bounds_count / len(pixel_labels) if pixel_labels else 0
                },
                'bounds': {
                    'lat_range': [lat_min, lat_max],
                    'lon_range': [lon_min, lon_max]
                }
            }

            return True

        except Exception as e:
            logger.error(f"Module 2 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_module_3_data_augmentation(self):
        """모듈 3: 데이터 증강"""
        logger.info("="*60)
        logger.info("MODULE 3: DATA AUGMENTATION")
        logger.info("="*60)

        try:
            # Get pixel labels
            pixel_labels = self.results.get('coordinate_mapping', {}).get('pixel_labels', [])
            in_bounds_labels = [label for label in pixel_labels if label['in_bounds']]

            if len(in_bounds_labels) == 0:
                logger.warning("No in-bounds mine locations for augmentation")
                # Generate synthetic data for testing
                logger.info("Generating synthetic mine locations for testing")
                intensity_data = self.results.get('intensity', {})
                combined_image = intensity_data['data']['combined']
                height, width = combined_image.shape

                import random
                synthetic_mines = []
                for i in range(5):
                    synthetic_mines.append({
                        'mine_id': f'synthetic_{i+1}',
                        'pixel_x': random.randint(100, width-100),
                        'pixel_y': random.randint(100, height-100),
                        'in_bounds': True
                    })
                in_bounds_labels = synthetic_mines

            # Augment positive samples (mines)
            augmented_samples = []
            patch_size = 64

            for label in in_bounds_labels:
                base_x = label['pixel_x']
                base_y = label['pixel_y']

                # Original sample
                augmented_samples.append({
                    'sample_id': f"{label['mine_id']}_original",
                    'center_x': base_x,
                    'center_y': base_y,
                    'label': 1,  # Mine
                    'augmentation': 'original'
                })

                # Augmentation: slight position variations
                for j in range(3):
                    offset_x = random.randint(-10, 10)
                    offset_y = random.randint(-10, 10)

                    augmented_samples.append({
                        'sample_id': f"{label['mine_id']}_shift_{j}",
                        'center_x': base_x + offset_x,
                        'center_y': base_y + offset_y,
                        'label': 1,
                        'augmentation': 'position_shift'
                    })

            # Generate negative samples (background)
            intensity_data = self.results.get('intensity', {})
            combined_image = intensity_data['data']['combined']
            height, width = combined_image.shape

            num_negative = len(augmented_samples) * 2
            for i in range(num_negative):
                # Random background location
                x = random.randint(patch_size, width - patch_size)
                y = random.randint(patch_size, height - patch_size)

                # Check not too close to mines
                too_close = False
                for mine_sample in augmented_samples:
                    if mine_sample['label'] == 1:
                        dist = ((x - mine_sample['center_x'])**2 +
                               (y - mine_sample['center_y'])**2)**0.5
                        if dist < patch_size * 1.5:
                            too_close = True
                            break

                if not too_close:
                    augmented_samples.append({
                        'sample_id': f"background_{i}",
                        'center_x': x,
                        'center_y': y,
                        'label': 0,  # Background
                        'augmentation': 'background'
                    })

            # Calculate statistics
            positive_count = sum(1 for s in augmented_samples if s['label'] == 1)
            negative_count = sum(1 for s in augmented_samples if s['label'] == 0)

            logger.info(f"Data augmentation results:")
            logger.info(f"  Positive samples (mines): {positive_count}")
            logger.info(f"  Negative samples (background): {negative_count}")
            logger.info(f"  Total samples: {len(augmented_samples)}")
            logger.info(f"  Positive ratio: {positive_count/len(augmented_samples):.2%}")

            # Save augmented data
            output_file = self.session_dir / "03_data_augmentation" / "augmented_samples.json"
            with open(output_file, 'w') as f:
                json.dump(augmented_samples, f, indent=2)
            logger.info(f"Saved augmented samples to: {output_file}")

            # Visualize augmented samples
            self._visualize_augmented_samples(combined_image, augmented_samples)

            # Store results
            self.results['augmentation'] = {
                'samples': augmented_samples,
                'statistics': {
                    'positive_samples': positive_count,
                    'negative_samples': negative_count,
                    'total_samples': len(augmented_samples),
                    'positive_ratio': positive_count / len(augmented_samples)
                }
            }

            return True

        except Exception as e:
            logger.error(f"Module 3 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_module_4_feature_extraction(self):
        """모듈 4: 특징 추출"""
        logger.info("="*60)
        logger.info("MODULE 4: FEATURE EXTRACTION")
        logger.info("="*60)

        try:
            # Get augmented samples
            augmented_samples = self.results.get('augmentation', {}).get('samples', [])

            if len(augmented_samples) == 0:
                logger.error("No augmented samples available")
                return False

            # Get intensity data
            intensity_data = self.results.get('intensity', {})
            combined_image = intensity_data['data']['combined']

            # Extract patches and features
            patch_size = 64
            half_patch = patch_size // 2

            patches = []
            labels = []
            sample_ids = []

            for sample in augmented_samples:
                x = sample['center_x']
                y = sample['center_y']

                # Extract patch
                y_min = max(0, y - half_patch)
                y_max = min(combined_image.shape[0], y + half_patch)
                x_min = max(0, x - half_patch)
                x_max = min(combined_image.shape[1], x + half_patch)

                patch = combined_image[y_min:y_max, x_min:x_max]

                if patch.size > 0:
                    patches.append(patch)
                    labels.append(sample['label'])
                    sample_ids.append(sample['sample_id'])

            logger.info(f"Extracted {len(patches)} patches for feature extraction")

            # Extract features from patches
            features_list = []
            feature_names = []

            for i, patch in enumerate(patches):
                # Statistical features
                stat_features = [
                    np.mean(patch),
                    np.std(patch),
                    np.min(patch),
                    np.max(patch),
                    np.median(patch)
                ]

                if i == 0:
                    feature_names.extend(['mean', 'std', 'min', 'max', 'median'])

                # Texture features (simplified)
                texture_features = [
                    np.var(patch),  # Variance
                    np.mean(np.abs(np.diff(patch, axis=0))),  # Horizontal gradient
                    np.mean(np.abs(np.diff(patch, axis=1)))   # Vertical gradient
                ]

                if i == 0:
                    feature_names.extend(['variance', 'h_gradient', 'v_gradient'])

                features_list.append(stat_features + texture_features)

            features_array = np.array(features_list)
            labels_array = np.array(labels)

            logger.info(f"Extracted features shape: {features_array.shape}")
            logger.info(f"Feature names: {feature_names}")

            # Save features
            output_file = self.session_dir / "04_feature_extraction" / "features.npz"
            np.savez_compressed(
                output_file,
                features=features_array,
                labels=labels_array,
                feature_names=feature_names,
                sample_ids=sample_ids
            )
            logger.info(f"Saved features to: {output_file}")

            # Create feature statistics
            feature_stats = pd.DataFrame({
                'Feature': feature_names,
                'Mean': np.mean(features_array, axis=0),
                'Std': np.std(features_array, axis=0),
                'Min': np.min(features_array, axis=0),
                'Max': np.max(features_array, axis=0)
            })

            stats_file = self.session_dir / "04_feature_extraction" / "feature_statistics.csv"
            feature_stats.to_csv(stats_file, index=False)
            logger.info(f"Saved feature statistics to: {stats_file}")

            # Visualize features
            self._visualize_features(features_array, labels_array, feature_names)

            # Store results
            self.results['features'] = {
                'shape': features_array.shape,
                'feature_names': feature_names,
                'statistics': feature_stats.to_dict(),
                'file': str(output_file)
            }

            return True

        except Exception as e:
            logger.error(f"Module 4 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _visualize_intensity(self, intensity_image):
        """강도 데이터 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original image
        im1 = axes[0].imshow(intensity_image, cmap='viridis', aspect='auto')
        axes[0].set_title('Intensity Data (Original)')
        axes[0].set_xlabel('Range (samples)')
        axes[0].set_ylabel('Ping Number')
        plt.colorbar(im1, ax=axes[0])

        # Log-scaled image
        log_image = np.log1p(intensity_image)
        im2 = axes[1].imshow(log_image, cmap='viridis', aspect='auto')
        axes[1].set_title('Intensity Data (Log Scale)')
        axes[1].set_xlabel('Range (samples)')
        axes[1].set_ylabel('Ping Number')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        output_file = self.session_dir / "05_visualizations" / "intensity_data.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved intensity visualization to: {output_file}")

    def _visualize_coordinate_mapping(self, intensity_image, pixel_labels, lat_bounds, lon_bounds):
        """위경도 매핑 검증 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Intensity with mine locations
        axes[0].imshow(intensity_image, cmap='gray', aspect='auto')

        for label in pixel_labels:
            if label['in_bounds']:
                # Draw mine location
                circle = plt.Circle((label['pixel_x'], label['pixel_y']),
                                   20, color='red', fill=False, linewidth=2)
                axes[0].add_patch(circle)
                axes[0].text(label['pixel_x'], label['pixel_y'] - 30,
                           label['mine_id'], color='yellow', fontsize=8,
                           ha='center', fontweight='bold')

        axes[0].set_title('Mine Locations on Intensity Image')
        axes[0].set_xlabel('Range (pixels)')
        axes[0].set_ylabel('Ping (pixels)')

        # Plot 2: Geographic distribution
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds

        # XTF data bounds
        rect = patches.Rectangle((lon_min, lat_min),
                                lon_max - lon_min,
                                lat_max - lat_min,
                                linewidth=2, edgecolor='blue',
                                facecolor='lightblue', alpha=0.3)
        axes[1].add_patch(rect)

        # Mine locations
        for label in pixel_labels:
            color = 'green' if label['in_bounds'] else 'red'
            marker = 'o' if label['in_bounds'] else 'x'
            axes[1].scatter(label['gps_lon'], label['gps_lat'],
                          c=color, marker=marker, s=50, alpha=0.7)

        axes[1].set_xlim(lon_min - 0.001, lon_max + 0.001)
        axes[1].set_ylim(lat_min - 0.001, lat_max + 0.001)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('Geographic Distribution (Blue box = XTF bounds)')
        axes[1].grid(True, alpha=0.3)

        # Add legend
        axes[1].scatter([], [], c='green', marker='o', label='In Bounds')
        axes[1].scatter([], [], c='red', marker='x', label='Out of Bounds')
        axes[1].legend()

        plt.tight_layout()
        output_file = self.session_dir / "05_visualizations" / "coordinate_mapping.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved coordinate mapping visualization to: {output_file}")

    def _visualize_augmented_samples(self, intensity_image, augmented_samples):
        """증강된 샘플 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Show intensity image
        ax.imshow(intensity_image, cmap='gray', aspect='auto', alpha=0.7)

        # Plot augmented samples
        for sample in augmented_samples:
            if sample['label'] == 1:  # Mine
                color = 'red'
                marker = '^'
                size = 50
            else:  # Background
                color = 'blue'
                marker = 's'
                size = 30

            ax.scatter(sample['center_x'], sample['center_y'],
                      c=color, marker=marker, s=size, alpha=0.6)

        ax.set_title('Augmented Sample Distribution')
        ax.set_xlabel('Range (pixels)')
        ax.set_ylabel('Ping (pixels)')

        # Legend
        ax.scatter([], [], c='red', marker='^', s=50, label='Mine samples')
        ax.scatter([], [], c='blue', marker='s', s=30, label='Background samples')
        ax.legend()

        plt.tight_layout()
        output_file = self.session_dir / "05_visualizations" / "augmented_samples.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved augmented samples visualization to: {output_file}")

    def _visualize_features(self, features, labels, feature_names):
        """특징 분포 시각화"""
        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()

        for i, feature_name in enumerate(feature_names):
            ax = axes[i]

            # Separate by class
            mine_features = features[labels == 1, i]
            bg_features = features[labels == 0, i]

            # Plot histograms
            ax.hist(mine_features, bins=20, alpha=0.5, color='red', label='Mine')
            ax.hist(bg_features, bins=20, alpha=0.5, color='blue', label='Background')

            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.legend()

        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        output_file = self.session_dir / "05_visualizations" / "feature_distributions.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature distributions to: {output_file}")

    def generate_report(self):
        """종합 분석 보고서 생성"""
        logger.info("="*60)
        logger.info("GENERATING ANALYSIS REPORT")
        logger.info("="*60)

        report_lines = []
        report_lines.append("# Independent Module Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Session: {self.timestamp}")
        report_lines.append("")

        # Module 1: Intensity Extraction
        report_lines.append("## Module 1: Intensity Data Extraction")
        if 'intensity' in self.results:
            intensity = self.results['intensity']
            report_lines.append(f"- Data shape: {intensity['shape']}")
            nav = intensity['navigation']
            if 'latitudes' in nav:
                report_lines.append(f"- Navigation points: {len(nav['latitudes'])}")
            report_lines.append(f"- Output file: {intensity['file']}")
        else:
            report_lines.append("- Status: Not executed")
        report_lines.append("")

        # Module 2: Coordinate Mapping
        report_lines.append("## Module 2: Coordinate Mapping & Pixel Labels")
        if 'coordinate_mapping' in self.results:
            mapping = self.results['coordinate_mapping']
            stats = mapping['statistics']
            bounds = mapping['bounds']

            report_lines.append(f"### Mapping Statistics:")
            report_lines.append(f"- Total mines: {stats['total_mines']}")
            report_lines.append(f"- In bounds: {stats['in_bounds']} ({stats['mapping_rate']:.1%})")
            report_lines.append(f"- Out of bounds: {stats['out_of_bounds']}")
            report_lines.append("")

            report_lines.append(f"### XTF Data Bounds:")
            report_lines.append(f"- Latitude range: [{bounds['lat_range'][0]:.6f}, {bounds['lat_range'][1]:.6f}]")
            report_lines.append(f"- Longitude range: [{bounds['lon_range'][0]:.6f}, {bounds['lon_range'][1]:.6f}]")
            report_lines.append("")

            # Detailed mapping table
            report_lines.append("### Detailed Mapping Results:")
            report_lines.append("```")
            report_df = pd.DataFrame(mapping['mapping_report'])
            report_lines.append(report_df.to_string(index=False))
            report_lines.append("```")
        else:
            report_lines.append("- Status: Not executed")
        report_lines.append("")

        # Module 3: Data Augmentation
        report_lines.append("## Module 3: Data Augmentation")
        if 'augmentation' in self.results:
            aug = self.results['augmentation']
            stats = aug['statistics']

            report_lines.append(f"- Positive samples (mines): {stats['positive_samples']}")
            report_lines.append(f"- Negative samples (background): {stats['negative_samples']}")
            report_lines.append(f"- Total samples: {stats['total_samples']}")
            report_lines.append(f"- Class balance: {stats['positive_ratio']:.1%} positive")
        else:
            report_lines.append("- Status: Not executed")
        report_lines.append("")

        # Module 4: Feature Extraction
        report_lines.append("## Module 4: Feature Extraction")
        if 'features' in self.results:
            features = self.results['features']
            report_lines.append(f"- Feature matrix shape: {features['shape']}")
            report_lines.append(f"- Feature names: {', '.join(features['feature_names'])}")
            report_lines.append(f"- Output file: {features['file']}")
        else:
            report_lines.append("- Status: Not executed")
        report_lines.append("")

        # Analysis Summary
        report_lines.append("## Analysis Summary")

        if 'coordinate_mapping' in self.results:
            mapping = self.results['coordinate_mapping']
            if mapping['statistics']['mapping_rate'] < 0.1:
                report_lines.append("⚠️ **WARNING**: Very low mapping rate detected!")
                report_lines.append("   - Most mine locations are outside XTF data bounds")
                report_lines.append("   - This suggests coordinate system mismatch or different survey areas")
                report_lines.append("   - Synthetic data was likely used for testing")
            elif mapping['statistics']['mapping_rate'] < 0.5:
                report_lines.append("⚠️ **CAUTION**: Low mapping rate detected")
                report_lines.append("   - Less than 50% of mine locations within XTF bounds")
                report_lines.append("   - Check coordinate systems and survey overlap")
            else:
                report_lines.append("✅ **GOOD**: Majority of mine locations successfully mapped")

        report_lines.append("")
        report_lines.append("## Output Directory Structure")
        report_lines.append(f"```")
        report_lines.append(f"{self.session_dir}/")
        report_lines.append(f"├── 01_intensity_extraction/")
        report_lines.append(f"│   └── intensity_data.npz")
        report_lines.append(f"├── 02_coordinate_mapping/")
        report_lines.append(f"│   ├── pixel_labels.json")
        report_lines.append(f"│   └── mapping_report.csv")
        report_lines.append(f"├── 03_data_augmentation/")
        report_lines.append(f"│   └── augmented_samples.json")
        report_lines.append(f"├── 04_feature_extraction/")
        report_lines.append(f"│   ├── features.npz")
        report_lines.append(f"│   └── feature_statistics.csv")
        report_lines.append(f"├── 05_visualizations/")
        report_lines.append(f"│   ├── intensity_data.png")
        report_lines.append(f"│   ├── coordinate_mapping.png")
        report_lines.append(f"│   ├── augmented_samples.png")
        report_lines.append(f"│   └── feature_distributions.png")
        report_lines.append(f"└── 06_reports/")
        report_lines.append(f"    └── analysis_report.md")
        report_lines.append(f"```")

        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.session_dir / "06_reports" / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        logger.info(f"Saved analysis report to: {report_file}")

        # Also save as pickle for programmatic access
        results_file = self.session_dir / "06_reports" / "results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Saved results dictionary to: {results_file}")

        return report_content


def main():
    """메인 실행 함수"""
    logger.info("Starting Independent Module Analysis")

    # Initialize runner
    runner = IndependentModuleRunner()

    # Define input files
    xtf_file = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")
    mine_locations_file = Path("datasets/Location_MDGPS.xlsx")

    # Check files exist
    if not xtf_file.exists():
        logger.error(f"XTF file not found: {xtf_file}")
        return 1

    if not mine_locations_file.exists():
        logger.error(f"Mine locations file not found: {mine_locations_file}")
        return 1

    # Run modules
    success = True

    # Module 1: Intensity Extraction
    if not runner.run_module_1_intensity_extraction(xtf_file):
        logger.error("Module 1 failed, continuing with other modules...")
        success = False

    # Module 2: Coordinate Mapping
    if not runner.run_module_2_coordinate_mapping(mine_locations_file):
        logger.error("Module 2 failed, continuing with other modules...")
        success = False

    # Module 3: Data Augmentation
    if not runner.run_module_3_data_augmentation():
        logger.error("Module 3 failed, continuing with other modules...")
        success = False

    # Module 4: Feature Extraction
    if not runner.run_module_4_feature_extraction():
        logger.error("Module 4 failed, continuing with other modules...")
        success = False

    # Generate report
    report = runner.generate_report()

    # Print summary
    logger.info("="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {runner.session_dir}")
    logger.info("="*60)

    print("\nReport Summary:")
    print(report)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())