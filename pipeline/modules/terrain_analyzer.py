"""
Terrain Analyzer Module
=======================
해저 지형 분석 모듈 - 지형적 특성을 고려한 분류 개선
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import ndimage, stats
from skimage import segmentation, measure, filters
import cv2

logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """해저 지형 분석 클래스"""

    def __init__(self,
                 depth_estimation_method: str = "intensity_based",
                 terrain_smoothing: float = 1.0):
        """
        Initialize Terrain Analyzer

        Args:
            depth_estimation_method: 깊이 추정 방법 ("intensity_based", "gradient_based")
            terrain_smoothing: 지형 스무딩 파라미터
        """
        self.depth_estimation_method = depth_estimation_method
        self.terrain_smoothing = terrain_smoothing
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self,
               intensity_data: np.ndarray,
               predictions: np.ndarray,
               label_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        해저 지형 분석 및 분류 개선

        Args:
            intensity_data: 강도 데이터
            predictions: 초기 분류 예측
            label_data: 레이블 데이터 (선택적)

        Returns:
            지형 분석 결과
        """
        self.logger.info("Analyzing seafloor terrain characteristics")

        try:
            # Estimate terrain characteristics
            terrain_features = self._estimate_terrain_features(intensity_data)

            # Analyze local context around predictions
            context_analysis = self._analyze_local_context(
                intensity_data, predictions, label_data
            )

            # Apply terrain-based classification refinement
            refined_predictions = self._refine_predictions_with_terrain(
                predictions, terrain_features, context_analysis
            )

            # Calculate terrain statistics
            terrain_stats = self._calculate_terrain_statistics(
                terrain_features, predictions, refined_predictions
            )

            return {
                'terrain_features': terrain_features,
                'context_analysis': context_analysis,
                'refined_predictions': refined_predictions,
                'terrain_statistics': terrain_stats,
                'analysis_info': {
                    'depth_estimation_method': self.depth_estimation_method,
                    'terrain_smoothing': self.terrain_smoothing,
                    'intensity_shape': intensity_data.shape
                }
            }

        except Exception as e:
            self.logger.error(f"Terrain analysis failed: {e}")
            raise

    def _estimate_terrain_features(self, intensity_data: np.ndarray) -> Dict[str, np.ndarray]:
        """지형 특성 추정"""
        self.logger.debug("Estimating terrain features")

        # Smooth intensity data
        smoothed_intensity = filters.gaussian(
            intensity_data, sigma=self.terrain_smoothing
        )

        # Estimate relative depth/elevation
        if self.depth_estimation_method == "intensity_based":
            # Higher intensity often indicates shallower areas or objects
            relative_depth = 1.0 - (smoothed_intensity - np.min(smoothed_intensity)) / \
                           (np.max(smoothed_intensity) - np.min(smoothed_intensity) + 1e-8)
        else:  # gradient_based
            # Use gradient magnitude as depth indicator
            grad_y, grad_x = np.gradient(smoothed_intensity)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            relative_depth = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)

        # Calculate slope
        grad_y, grad_x = np.gradient(relative_depth)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        # Calculate curvature (second derivatives)
        grad_yy, grad_yx = np.gradient(grad_y)
        grad_xy, grad_xx = np.gradient(grad_x)
        curvature = grad_xx + grad_yy

        # Estimate roughness (local standard deviation)
        roughness = ndimage.generic_filter(
            intensity_data, np.std, size=5
        )

        # Terrain classification
        terrain_type = self._classify_terrain_types(
            relative_depth, slope, curvature, roughness
        )

        # Shadow detection
        shadows = self._detect_shadows(intensity_data, grad_x, grad_y)

        return {
            'relative_depth': relative_depth,
            'slope': slope,
            'curvature': curvature,
            'roughness': roughness,
            'terrain_type': terrain_type,
            'shadows': shadows,
            'gradient_x': grad_x,
            'gradient_y': grad_y
        }

    def _classify_terrain_types(self,
                               depth: np.ndarray,
                               slope: np.ndarray,
                               curvature: np.ndarray,
                               roughness: np.ndarray) -> np.ndarray:
        """지형 타입 분류"""
        terrain_type = np.zeros_like(depth, dtype=int)

        # Define thresholds (these can be tuned based on data characteristics)
        slope_threshold = np.percentile(slope, 70)
        curvature_threshold = np.percentile(np.abs(curvature), 70)
        roughness_threshold = np.percentile(roughness, 70)

        # Classify terrain types
        # 0: Flat seafloor
        # 1: Sloped terrain
        # 2: Rough/rocky terrain
        # 3: Depression/valley
        # 4: Ridge/elevation

        # Flat areas (low slope, low roughness)
        flat_mask = (slope < slope_threshold * 0.5) & (roughness < roughness_threshold * 0.5)
        terrain_type[flat_mask] = 0

        # Sloped areas
        slope_mask = (slope >= slope_threshold * 0.5) & (slope < slope_threshold)
        terrain_type[slope_mask] = 1

        # Rough areas
        rough_mask = roughness >= roughness_threshold
        terrain_type[rough_mask] = 2

        # Depressions (negative curvature)
        depression_mask = curvature < -curvature_threshold
        terrain_type[depression_mask] = 3

        # Ridges (positive curvature)
        ridge_mask = curvature > curvature_threshold
        terrain_type[ridge_mask] = 4

        return terrain_type

    def _detect_shadows(self,
                       intensity_data: np.ndarray,
                       grad_x: np.ndarray,
                       grad_y: np.ndarray) -> np.ndarray:
        """그림자 영역 탐지"""
        # Shadow areas typically have low intensity and are adjacent to high gradient areas
        intensity_threshold = np.percentile(intensity_data, 20)  # Bottom 20%
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_threshold = np.percentile(gradient_magnitude, 80)  # Top 20%

        # Low intensity areas
        low_intensity = intensity_data < intensity_threshold

        # Areas adjacent to high gradients
        high_gradient_dilated = ndimage.binary_dilation(
            gradient_magnitude > gradient_threshold,
            structure=np.ones((5, 5))
        )

        # Shadow mask
        shadows = low_intensity & high_gradient_dilated

        return shadows.astype(int)

    def _analyze_local_context(self,
                              intensity_data: np.ndarray,
                              predictions: np.ndarray,
                              label_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """예측 위치 주변의 지형적 맥락 분석"""
        self.logger.debug("Analyzing local context around predictions")

        if label_data is None or 'patch_locations' not in label_data:
            return {'context_features': [], 'contextual_scores': []}

        patch_locations = label_data['patch_locations']
        context_features = []
        contextual_scores = []

        for i, location in enumerate(patch_locations):
            center_x = location['center_x']
            center_y = location['center_y']
            patch_bounds = location['patch_bounds']

            # Extract local region (larger than patch)
            context_size = 32  # Context window size
            y_min = max(0, center_y - context_size)
            y_max = min(intensity_data.shape[0], center_y + context_size)
            x_min = max(0, center_x - context_size)
            x_max = min(intensity_data.shape[1], center_x + context_size)

            local_region = intensity_data[y_min:y_max, x_min:x_max]

            # Calculate context features
            context_feature = self._calculate_context_features(local_region)
            context_features.append(context_feature)

            # Calculate contextual plausibility score
            contextual_score = self._calculate_contextual_score(
                local_region, predictions[i] if i < len(predictions) else 0
            )
            contextual_scores.append(contextual_score)

        return {
            'context_features': context_features,
            'contextual_scores': contextual_scores
        }

    def _calculate_context_features(self, local_region: np.ndarray) -> Dict[str, float]:
        """지역적 맥락 특성 계산"""
        features = {}

        # Basic statistics
        features['mean_intensity'] = float(np.mean(local_region))
        features['std_intensity'] = float(np.std(local_region))
        features['intensity_range'] = float(np.max(local_region) - np.min(local_region))

        # Gradient characteristics
        grad_y, grad_x = np.gradient(local_region)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['mean_gradient'] = float(np.mean(gradient_magnitude))
        features['max_gradient'] = float(np.max(gradient_magnitude))

        # Local variation
        features['local_variance'] = float(np.var(local_region))

        # Spatial patterns
        # Calculate autocorrelation
        center = local_region.shape[0] // 2
        if center > 0:
            central_value = local_region[center, center]
            distances = np.sqrt(
                (np.arange(local_region.shape[0])[:, None] - center)**2 +
                (np.arange(local_region.shape[1])[None, :] - center)**2
            )

            # Calculate correlation with distance
            near_mask = distances <= 5
            far_mask = (distances > 5) & (distances <= 15)

            if np.any(near_mask) and np.any(far_mask):
                near_values = local_region[near_mask]
                far_values = local_region[far_mask]
                features['spatial_correlation'] = float(
                    np.corrcoef(central_value, np.mean(near_values))[0, 1]
                    if not np.isnan(np.corrcoef(central_value, np.mean(near_values))[0, 1])
                    else 0.0
                )
            else:
                features['spatial_correlation'] = 0.0
        else:
            features['spatial_correlation'] = 0.0

        return features

    def _calculate_contextual_score(self,
                                   local_region: np.ndarray,
                                   prediction: int) -> float:
        """예측의 지형적 타당성 점수 계산"""
        # Calculate various contextual indicators
        mean_intensity = np.mean(local_region)
        std_intensity = np.std(local_region)

        # Object-like characteristics (high contrast, defined boundaries)
        grad_y, grad_x = np.gradient(local_region)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = np.mean(gradient_magnitude)

        # Shadow presence (adjacent low-intensity areas)
        intensity_threshold = mean_intensity - std_intensity
        shadow_presence = np.sum(local_region < intensity_threshold) / local_region.size

        if prediction == 1:  # Positive prediction (mine)
            # For positive predictions, higher scores for:
            # - High contrast (edge strength)
            # - Presence of shadows
            # - Moderate to high intensity variation
            score = (
                min(edge_strength / (np.percentile(gradient_magnitude, 90) + 1e-8), 1.0) * 0.4 +
                min(shadow_presence * 2, 1.0) * 0.3 +
                min(std_intensity / (mean_intensity + 1e-8), 1.0) * 0.3
            )
        else:  # Negative prediction (background)
            # For negative predictions, higher scores for:
            # - Low contrast (uniform background)
            # - No shadows
            # - Low intensity variation
            score = (
                max(1.0 - edge_strength / (np.percentile(gradient_magnitude, 90) + 1e-8), 0.0) * 0.4 +
                max(1.0 - shadow_presence * 2, 0.0) * 0.3 +
                max(1.0 - std_intensity / (mean_intensity + 1e-8), 0.0) * 0.3
            )

        return float(np.clip(score, 0.0, 1.0))

    def _refine_predictions_with_terrain(self,
                                       predictions: np.ndarray,
                                       terrain_features: Dict[str, np.ndarray],
                                       context_analysis: Dict[str, Any]) -> np.ndarray:
        """지형 정보를 활용한 예측 개선"""
        self.logger.debug("Refining predictions with terrain information")

        refined_predictions = predictions.copy()
        contextual_scores = context_analysis.get('contextual_scores', [])

        if not contextual_scores:
            return refined_predictions

        # Apply contextual scoring to refine predictions
        for i, (pred, score) in enumerate(zip(predictions, contextual_scores)):
            if i >= len(refined_predictions):
                break

            # Confidence threshold for refinement
            confidence_threshold = 0.3

            if pred == 1 and score < confidence_threshold:
                # Low contextual score for positive prediction - consider changing to negative
                refined_predictions[i] = 0
            elif pred == 0 and score < confidence_threshold:
                # Low contextual score for negative prediction - consider changing to positive
                # (Be more conservative here)
                if score < 0.1:  # Very low score
                    refined_predictions[i] = 1

        return refined_predictions

    def _calculate_terrain_statistics(self,
                                    terrain_features: Dict[str, np.ndarray],
                                    original_predictions: np.ndarray,
                                    refined_predictions: np.ndarray) -> Dict[str, Any]:
        """지형 분석 통계 계산"""
        stats = {}

        # Terrain feature statistics
        for feature_name, feature_data in terrain_features.items():
            if isinstance(feature_data, np.ndarray):
                stats[f'{feature_name}_stats'] = {
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'percentile_25': float(np.percentile(feature_data, 25)),
                    'percentile_75': float(np.percentile(feature_data, 75))
                }

        # Prediction refinement statistics
        changes = original_predictions != refined_predictions
        stats['prediction_refinement'] = {
            'total_predictions': len(original_predictions),
            'changed_predictions': int(np.sum(changes)),
            'change_percentage': float(np.sum(changes) / len(original_predictions) * 100),
            'positive_to_negative': int(np.sum(
                (original_predictions == 1) & (refined_predictions == 0)
            )),
            'negative_to_positive': int(np.sum(
                (original_predictions == 0) & (refined_predictions == 1)
            ))
        }

        # Terrain type distribution
        terrain_types = terrain_features.get('terrain_type')
        if terrain_types is not None:
            unique_types, counts = np.unique(terrain_types, return_counts=True)
            total_pixels = terrain_types.size

            stats['terrain_distribution'] = {
                'terrain_types': {
                    'flat': int(counts[unique_types == 0][0]) if 0 in unique_types else 0,
                    'sloped': int(counts[unique_types == 1][0]) if 1 in unique_types else 0,
                    'rough': int(counts[unique_types == 2][0]) if 2 in unique_types else 0,
                    'depression': int(counts[unique_types == 3][0]) if 3 in unique_types else 0,
                    'ridge': int(counts[unique_types == 4][0]) if 4 in unique_types else 0
                },
                'percentages': {
                    'flat': float(counts[unique_types == 0][0] / total_pixels * 100) if 0 in unique_types else 0.0,
                    'sloped': float(counts[unique_types == 1][0] / total_pixels * 100) if 1 in unique_types else 0.0,
                    'rough': float(counts[unique_types == 2][0] / total_pixels * 100) if 2 in unique_types else 0.0,
                    'depression': float(counts[unique_types == 3][0] / total_pixels * 100) if 3 in unique_types else 0.0,
                    'ridge': float(counts[unique_types == 4][0] / total_pixels * 100) if 4 in unique_types else 0.0
                }
            }

        return stats

    def visualize_terrain_analysis(self,
                                  terrain_features: Dict[str, np.ndarray],
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """지형 분석 시각화"""
        try:
            import matplotlib.pyplot as plt

            # Create subplots for different terrain features
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            features_to_plot = [
                ('relative_depth', 'Relative Depth'),
                ('slope', 'Slope'),
                ('curvature', 'Curvature'),
                ('roughness', 'Roughness'),
                ('terrain_type', 'Terrain Type'),
                ('shadows', 'Shadows')
            ]

            for i, (feature_name, title) in enumerate(features_to_plot):
                if feature_name in terrain_features and i < len(axes):
                    data = terrain_features[feature_name]
                    im = axes[i].imshow(data, cmap='viridis')
                    axes[i].set_title(title)
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i])

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Terrain visualization saved to {output_path}")

            plt.close()

            return {'visualization_created': True, 'output_path': output_path}

        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
            return {'visualization_created': False, 'reason': 'matplotlib_not_available'}

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return {'visualization_created': False, 'reason': str(e)}