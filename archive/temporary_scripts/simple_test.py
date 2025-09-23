#!/usr/bin/env python3
"""
Simple Pipeline Test
==================
기본 기능 테스트
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """기본 import 테스트"""
    print("Testing basic imports...")

    try:
        from pipeline.modules.xtf_reader import XTFReader
        print("✅ XTF Reader import OK")

        from pipeline.modules.xtf_extractor import XTFExtractor
        print("✅ XTF Extractor import OK")

        from pipeline.modules.coordinate_mapper import CoordinateMapper
        print("✅ Coordinate Mapper import OK")

        from pipeline.modules.label_generator import LabelGenerator
        print("✅ Label Generator import OK")

        from pipeline.modules.feature_extractor import FeatureExtractor
        print("✅ Feature Extractor import OK")

        # Skip ensemble optimizer for now due to optuna complexity
        # from pipeline.modules.ensemble_optimizer import EnsembleOptimizer
        # print("✅ Ensemble Optimizer import OK")

        from pipeline.modules.mine_classifier import MineClassifier
        print("✅ Mine Classifier import OK")

        from pipeline.modules.terrain_analyzer import TerrainAnalyzer
        print("✅ Terrain Analyzer import OK")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_data_availability():
    """데이터 파일 존재 확인"""
    print("\nTesting data availability...")

    # Check XTF file
    xtf_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")
    if xtf_path.exists():
        print(f"✅ XTF file found: {xtf_path.name} ({xtf_path.stat().st_size / (1024*1024):.1f}MB)")
    else:
        print(f"❌ XTF file not found: {xtf_path}")
        return False

    # Check GPS file
    gps_path = Path("datasets/Location_MDGPS.xlsx")
    if gps_path.exists():
        print(f"✅ GPS file found: {gps_path.name}")
    else:
        print(f"❌ GPS file not found: {gps_path}")
        return False

    # Check annotation file
    annotation_path = Path("datasets/PH_annotation.png")
    if annotation_path.exists():
        print(f"✅ Annotation file found: {annotation_path.name}")
    else:
        print(f"❌ Annotation file not found: {annotation_path}")
        return False

    return True

def test_xtf_reader():
    """XTF Reader 테스트"""
    print("\nTesting XTF Reader...")

    try:
        from pipeline.modules.xtf_reader import XTFReader

        xtf_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")

        reader = XTFReader()
        start_time = time.time()

        # Simple file info test
        file_info = reader.get_file_info(xtf_path)

        if file_info and 'file_size_mb' in file_info:
            print(f"✅ XTF Reader: File size {file_info['file_size_mb']:.1f}MB")
            print(f"   Execution time: {time.time() - start_time:.2f}s")
            return True
        else:
            print(f"❌ XTF Reader: Failed to get file info")
            return False

    except Exception as e:
        print(f"❌ XTF Reader failed: {e}")
        traceback.print_exc()
        return False

def test_coordinate_mapper():
    """Coordinate Mapper 테스트"""
    print("\nTesting Coordinate Mapper...")

    try:
        from pipeline.modules.coordinate_mapper import CoordinateMapper

        gps_path = Path("datasets/Location_MDGPS.xlsx")
        annotation_path = Path("datasets/PH_annotation.png")

        mapper = CoordinateMapper()
        start_time = time.time()

        # Test GPS data loading
        gps_data = mapper._load_gps_data(gps_path)

        if len(gps_data) > 0:
            print(f"✅ Coordinate Mapper: Loaded {len(gps_data)} GPS points")
            print(f"   Execution time: {time.time() - start_time:.2f}s")
            return True
        else:
            print(f"❌ Coordinate Mapper: No GPS data loaded")
            return False

    except Exception as e:
        print(f"❌ Coordinate Mapper failed: {e}")
        traceback.print_exc()
        return False

def test_label_generator():
    """Label Generator 테스트"""
    print("\nTesting Label Generator...")

    try:
        from pipeline.modules.label_generator import LabelGenerator
        import numpy as np

        # Create dummy data
        dummy_mappings = [
            {'pixel_x': 100, 'pixel_y': 100, 'bbox': {'x': 90, 'y': 90, 'width': 20, 'height': 20}},
            {'pixel_x': 200, 'pixel_y': 200, 'bbox': {'x': 190, 'y': 190, 'width': 20, 'height': 20}}
        ]

        dummy_coordinate_mapping = {
            'mappings': dummy_mappings,
            'image_size': (1024, 3862)
        }

        dummy_intensity = np.random.rand(100, 100)

        generator = LabelGenerator(patch_size=(32, 32))
        start_time = time.time()

        labels = generator.generate(dummy_coordinate_mapping, dummy_intensity)

        if labels and 'samples' in labels:
            print(f"✅ Label Generator: Generated {len(labels['samples'])} samples")
            print(f"   Positive: {labels.get('positive_count', 0)}, Negative: {labels.get('negative_count', 0)}")
            print(f"   Execution time: {time.time() - start_time:.2f}s")
            return True
        else:
            print(f"❌ Label Generator: Failed to generate labels")
            return False

    except Exception as e:
        print(f"❌ Label Generator failed: {e}")
        traceback.print_exc()
        return False

def test_feature_extractor():
    """Feature Extractor 테스트"""
    print("\nTesting Feature Extractor...")

    try:
        from pipeline.modules.feature_extractor import FeatureExtractor
        import numpy as np

        # Create dummy data
        dummy_intensity = np.random.rand(200, 200)
        dummy_samples = [
            {
                'patch_bounds': {'x_min': 10, 'x_max': 42, 'y_min': 10, 'y_max': 42},
                'label': 1
            },
            {
                'patch_bounds': {'x_min': 50, 'x_max': 82, 'y_min': 50, 'y_max': 82},
                'label': 0
            }
        ]

        dummy_label_data = {
            'samples': dummy_samples,
            'labels': np.array([1, 0])
        }

        extractor = FeatureExtractor(
            methods=['statistical'],  # Simple method only
            patch_size=(32, 32)
        )
        start_time = time.time()

        features = extractor.extract(dummy_intensity, dummy_label_data)

        if features and 'features' in features:
            print(f"✅ Feature Extractor: Extracted {features['features'].shape} features")
            print(f"   Feature count: {len(features.get('feature_names', []))}")
            print(f"   Execution time: {time.time() - start_time:.2f}s")
            return True
        else:
            print(f"❌ Feature Extractor: Failed to extract features")
            return False

    except Exception as e:
        print(f"❌ Feature Extractor failed: {e}")
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 Simple Pipeline Test")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_data_availability,
        test_xtf_reader,
        test_coordinate_mapper,
        test_label_generator,
        test_feature_extractor
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {total - passed} tests failed")

    return passed == total

if __name__ == "__main__":
    success = main()