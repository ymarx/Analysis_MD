"""
Feature Extractor Module
========================
특징 추출 모듈 - 다양한 방법으로 특징 추출
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing feature extraction modules
try:
    from src.feature_extraction.basic_features import BasicFeatureExtractor
    from src.feature_extraction.advanced_features import AdvancedFeatureExtractor
except ImportError:
    BasicFeatureExtractor = None
    AdvancedFeatureExtractor = None

# Additional feature extraction libraries
from skimage import feature, filters, morphology, measure
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from scipy import ndimage, stats
import cv2

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """특징 추출 클래스"""

    def __init__(self,
                 methods: List[str] = None,
                 patch_size: Tuple[int, int] = (64, 64)):
        """
        Initialize Feature Extractor

        Args:
            methods: 특징 추출 방법 리스트
            patch_size: 패치 크기
        """
        self.methods = methods or ['statistical', 'textural', 'morphological', 'frequency']
        self.patch_size = patch_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize existing extractors if available
        self.basic_extractor = BasicFeatureExtractor() if BasicFeatureExtractor else None
        self.advanced_extractor = AdvancedFeatureExtractor() if AdvancedFeatureExtractor else None

    def extract(self,
                intensity_data: np.ndarray,
                label_data: Dict[str, Any]) -> np.ndarray:
        """
        특징 추출 실행

        Args:
            intensity_data: 강도 데이터
            label_data: 레이블 데이터

        Returns:
            특징 행렬 (n_samples, n_features)
        """
        self.logger.info(f"Extracting features using methods: {self.methods}")

        try:
            samples = label_data.get('samples', [])
            if not samples:
                raise ValueError("No samples available for feature extraction")

            # Extract patches
            patches = self._extract_patches(intensity_data, samples)

            # Extract features for each method
            all_features = []
            feature_names = []

            for method in self.methods:
                method_features, method_names = self._extract_method_features(
                    patches, method
                )
                if method_features is not None:
                    all_features.append(method_features)
                    feature_names.extend([f"{method}_{name}" for name in method_names])

            # Combine all features
            if all_features:
                combined_features = np.hstack(all_features)
            else:
                raise ValueError("No features extracted")

            self.logger.info(f"Extracted {combined_features.shape[1]} features from {combined_features.shape[0]} samples")

            return {
                'features': combined_features,
                'feature_names': feature_names,
                'extraction_info': {
                    'methods': self.methods,
                    'patch_size': self.patch_size,
                    'n_samples': len(samples),
                    'n_features': combined_features.shape[1]
                }
            }

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise

    def _extract_patches(self,
                        intensity_data: np.ndarray,
                        samples: List[Dict]) -> List[np.ndarray]:
        """강도 데이터에서 패치 추출"""
        patches = []

        for sample in samples:
            bounds = sample['patch_bounds']
            y_min, y_max = bounds['y_min'], bounds['y_max']
            x_min, x_max = bounds['x_min'], bounds['x_max']

            # Extract patch
            if (y_max <= intensity_data.shape[0] and
                x_max <= intensity_data.shape[1]):

                patch = intensity_data[y_min:y_max, x_min:x_max]

                # Resize to target size if needed
                if patch.shape != self.patch_size[::-1]:  # (height, width)
                    patch = cv2.resize(patch, self.patch_size)

                patches.append(patch)
            else:
                # Create empty patch if bounds are invalid
                patches.append(np.zeros(self.patch_size[::-1]))

        return patches

    def _extract_method_features(self,
                               patches: List[np.ndarray],
                               method: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """특정 방법으로 특징 추출"""
        if method == 'statistical':
            return self._extract_statistical_features(patches)
        elif method == 'textural':
            return self._extract_textural_features(patches)
        elif method == 'morphological':
            return self._extract_morphological_features(patches)
        elif method == 'frequency':
            return self._extract_frequency_features(patches)
        elif method == 'basic' and self.basic_extractor:
            return self._extract_with_basic_extractor(patches)
        elif method == 'advanced' and self.advanced_extractor:
            return self._extract_with_advanced_extractor(patches)
        else:
            self.logger.warning(f"Unknown feature extraction method: {method}")
            return None, []

    def _extract_statistical_features(self,
                                    patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """통계적 특징 추출"""
        features = []
        feature_names = [
            'mean', 'std', 'var', 'min', 'max', 'median',
            'skewness', 'kurtosis', 'percentile_25', 'percentile_75',
            'range', 'iqr', 'cv'  # coefficient of variation
        ]

        for patch in patches:
            patch_flat = patch.flatten()

            patch_features = [
                np.mean(patch_flat),
                np.std(patch_flat),
                np.var(patch_flat),
                np.min(patch_flat),
                np.max(patch_flat),
                np.median(patch_flat),
                stats.skew(patch_flat),
                stats.kurtosis(patch_flat),
                np.percentile(patch_flat, 25),
                np.percentile(patch_flat, 75),
                np.max(patch_flat) - np.min(patch_flat),  # range
                np.percentile(patch_flat, 75) - np.percentile(patch_flat, 25),  # IQR
                np.std(patch_flat) / np.mean(patch_flat) if np.mean(patch_flat) != 0 else 0  # CV
            ]

            features.append(patch_features)

        return np.array(features), feature_names

    def _extract_textural_features(self,
                                 patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """텍스처 특징 추출"""
        features = []
        feature_names = [
            'lbp_uniformity', 'lbp_contrast',
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
            'glcm_energy', 'glcm_correlation',
            'gabor_mean', 'gabor_std'
        ]

        for patch in patches:
            # Normalize patch to [0, 255] for GLCM
            patch_norm = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-8) * 255).astype(np.uint8)

            patch_features = []

            # Local Binary Pattern
            try:
                lbp = local_binary_pattern(patch, 8, 1, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-8)
                patch_features.extend([np.sum(hist**2), np.var(hist)])  # uniformity, contrast
            except:
                patch_features.extend([0, 0])

            # Gray Level Co-occurrence Matrix
            try:
                glcm = greycomatrix(patch_norm, [1], [0], levels=256, symmetric=True, normed=True)
                contrast = greycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
                energy = greycoprops(glcm, 'energy')[0, 0]
                correlation = greycoprops(glcm, 'correlation')[0, 0]
                patch_features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
            except:
                patch_features.extend([0, 0, 0, 0, 0])

            # Gabor filter response
            try:
                real, _ = filters.gabor(patch, frequency=0.6)
                patch_features.extend([np.mean(real), np.std(real)])
            except:
                patch_features.extend([0, 0])

            features.append(patch_features)

        return np.array(features), feature_names

    def _extract_morphological_features(self,
                                      patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """형태학적 특징 추출"""
        features = []
        feature_names = [
            'erosion_mean', 'dilation_mean', 'opening_mean', 'closing_mean',
            'gradient_mean', 'tophat_mean', 'blackhat_mean',
            'area_ratio', 'perimeter_ratio', 'solidity', 'extent'
        ]

        for patch in patches:
            # Threshold patch for morphological operations
            threshold = np.mean(patch) + np.std(patch)
            binary_patch = patch > threshold

            patch_features = []

            # Basic morphological operations
            selem = morphology.disk(2)
            try:
                erosion = morphology.erosion(patch, selem)
                dilation = morphology.dilation(patch, selem)
                opening = morphology.opening(patch, selem)
                closing = morphology.closing(patch, selem)
                gradient = morphology.gradient(patch, selem)
                tophat = morphology.white_tophat(patch, selem)
                blackhat = morphology.black_tophat(patch, selem)

                patch_features.extend([
                    np.mean(erosion), np.mean(dilation), np.mean(opening),
                    np.mean(closing), np.mean(gradient), np.mean(tophat),
                    np.mean(blackhat)
                ])
            except:
                patch_features.extend([0] * 7)

            # Region properties
            try:
                labeled = measure.label(binary_patch)
                props = measure.regionprops(labeled)

                if props:
                    prop = props[0]  # Take largest region
                    total_area = binary_patch.size
                    area_ratio = prop.area / total_area
                    perimeter_ratio = prop.perimeter / (2 * np.sqrt(np.pi * prop.area) + 1e-8)
                    solidity = prop.solidity
                    extent = prop.extent
                    patch_features.extend([area_ratio, perimeter_ratio, solidity, extent])
                else:
                    patch_features.extend([0, 0, 0, 0])
            except:
                patch_features.extend([0, 0, 0, 0])

            features.append(patch_features)

        return np.array(features), feature_names

    def _extract_frequency_features(self,
                                  patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """주파수 도메인 특징 추출"""
        features = []
        feature_names = [
            'fft_mean', 'fft_std', 'fft_energy',
            'dominant_freq', 'spectral_centroid', 'spectral_spread',
            'low_freq_energy', 'mid_freq_energy', 'high_freq_energy'
        ]

        for patch in patches:
            patch_features = []

            try:
                # 2D FFT
                fft = np.fft.fft2(patch)
                fft_magnitude = np.abs(fft)
                fft_magnitude_log = np.log(fft_magnitude + 1e-8)

                # Basic FFT statistics
                patch_features.extend([
                    np.mean(fft_magnitude_log),
                    np.std(fft_magnitude_log),
                    np.sum(fft_magnitude**2)
                ])

                # Frequency analysis
                freqs = np.fft.fftfreq(patch.shape[0])
                power_spectrum = np.mean(fft_magnitude, axis=1)

                # Dominant frequency
                dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
                dominant_freq = freqs[dominant_freq_idx]
                patch_features.append(dominant_freq)

                # Spectral centroid and spread
                freq_weights = np.arange(len(power_spectrum))
                total_power = np.sum(power_spectrum)
                if total_power > 0:
                    spectral_centroid = np.sum(freq_weights * power_spectrum) / total_power
                    spectral_spread = np.sqrt(np.sum(((freq_weights - spectral_centroid)**2) * power_spectrum) / total_power)
                else:
                    spectral_centroid = 0
                    spectral_spread = 0

                patch_features.extend([spectral_centroid, spectral_spread])

                # Frequency band energies
                n_freqs = len(power_spectrum)
                low_band = np.sum(power_spectrum[:n_freqs//3])
                mid_band = np.sum(power_spectrum[n_freqs//3:2*n_freqs//3])
                high_band = np.sum(power_spectrum[2*n_freqs//3:])

                patch_features.extend([low_band, mid_band, high_band])

            except:
                patch_features.extend([0] * 9)

            features.append(patch_features)

        return np.array(features), feature_names

    def _extract_with_basic_extractor(self,
                                    patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """기존 BasicFeatureExtractor 사용"""
        features = []

        for patch in patches:
            try:
                patch_features = self.basic_extractor.extract_features(patch)
                features.append(patch_features)
            except:
                # Fallback to zeros if extraction fails
                features.append(np.zeros(10))

        feature_names = [f"basic_{i}" for i in range(len(features[0]) if features else 0)]
        return np.array(features), feature_names

    def _extract_with_advanced_extractor(self,
                                       patches: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """기존 AdvancedFeatureExtractor 사용"""
        features = []

        for patch in patches:
            try:
                patch_features = self.advanced_extractor.extract_features(patch)
                features.append(patch_features)
            except:
                # Fallback to zeros if extraction fails
                features.append(np.zeros(20))

        feature_names = [f"advanced_{i}" for i in range(len(features[0]) if features else 0)]
        return np.array(features), feature_names

    def get_feature_importance(self,
                             features: np.ndarray,
                             labels: np.ndarray,
                             feature_names: List[str]) -> Dict[str, float]:
        """특징 중요도 계산"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import mutual_info_classif

            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            rf_importance = rf.feature_importances_

            # Mutual information
            mi_scores = mutual_info_classif(features, labels)

            # Combine scores
            importance_dict = {}
            for i, name in enumerate(feature_names):
                importance_dict[name] = {
                    'rf_importance': float(rf_importance[i]),
                    'mutual_info': float(mi_scores[i]),
                    'combined': float((rf_importance[i] + mi_scores[i]) / 2)
                }

            return importance_dict

        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {e}")
            return {}

    def reduce_features(self,
                       features: np.ndarray,
                       labels: np.ndarray,
                       n_features: int = 50) -> Tuple[np.ndarray, List[int]]:
        """특징 차원 축소"""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif

            selector = SelectKBest(score_func=f_classif, k=min(n_features, features.shape[1]))
            features_reduced = selector.fit_transform(features, labels)
            selected_indices = selector.get_support(indices=True)

            self.logger.info(f"Reduced features from {features.shape[1]} to {features_reduced.shape[1]}")

            return features_reduced, selected_indices.tolist()

        except Exception as e:
            self.logger.error(f"Feature reduction failed: {e}")
            return features, list(range(features.shape[1]))