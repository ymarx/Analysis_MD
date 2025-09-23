#!/usr/bin/env python3
"""
Enhanced Pipeline Test Script
============================
실제 데이터를 사용하여 향상된 파이프라인 테스트

Author: YMARX
Date: 2024-09-22
"""

import sys
import logging
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.enhanced_unified_pipeline import EnhancedUnifiedPipeline, EnhancedPipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'enhanced_pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)


def test_enhanced_pipeline():
    """향상된 파이프라인 테스트"""

    logger.info("="*60)
    logger.info("ENHANCED PIPELINE TEST STARTING")
    logger.info("="*60)

    try:
        # Test configuration
        config = EnhancedPipelineConfig(
            # Data paths
            data_dir=Path("datasets"),
            output_dir=Path("data/processed/enhanced_pipeline_test"),

            # GPS and annotation data (from successful coordinate mapping)
            gps_file=Path("datasets/Location_MDGPS.xlsx"),
            annotation_image=Path("datasets/PH_annotation.png"),
            mine_locations_file=Path("datasets/Location_MDGPS.xlsx"),

            # Data augmentation settings
            enable_augmentation=True,
            mine_augmentation_factor=5,  # 5x augmentation for mine samples
            background_augmentation_factor=2,  # 2x for background

            # Train/test splits
            test_size=0.2,  # 20% for testing
            validation_size=0.2,  # 20% of training for validation
            random_state=42,
            stratify=True,

            # Feature extraction
            feature_methods=['statistical', 'textural'],  # Use faster methods for testing
            patch_size=(64, 64),

            # Classification
            classifier_type='ensemble',
            use_terrain=True,

            # Output settings
            save_intermediate=True,
            save_augmented_samples=True,
            generate_reports=True,
            verbose=True
        )

        # Create pipeline
        logger.info("Creating enhanced pipeline...")
        pipeline = EnhancedUnifiedPipeline(config)

        # Test dataset path
        xtf_file = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")

        if not xtf_file.exists():
            logger.error(f"XTF file not found: {xtf_file}")
            return False

        logger.info(f"Testing with XTF file: {xtf_file}")
        logger.info(f"File size: {xtf_file.stat().st_size / (1024*1024):.1f} MB")

        # Run enhanced pipeline
        logger.info("Running enhanced pipeline...")
        results = pipeline.run_enhanced_full_pipeline(
            xtf_path=xtf_file,
            mine_locations_file=config.mine_locations_file
        )

        # Verify results
        logger.info("Verifying results...")
        success = verify_results(results, pipeline)

        if success:
            logger.info("✅ Enhanced pipeline test PASSED")

            # Print summary
            print_test_summary(results, pipeline)

        else:
            logger.error("❌ Enhanced pipeline test FAILED")

        return success

    except Exception as e:
        logger.error(f"Enhanced pipeline test failed with exception: {e}")
        logger.error(traceback.format_exc())
        return False


def verify_results(results: dict, pipeline: EnhancedUnifiedPipeline) -> bool:
    """결과 검증"""

    logger.info("Verifying pipeline results...")

    # Check basic structure
    required_keys = ['mine_locations', 'augmentation_info', 'data_splits', 'evaluation_results']

    for key in required_keys:
        if key not in results:
            logger.error(f"Missing required result key: {key}")
            return False

    # Check mine locations
    mine_locations = results['mine_locations']
    if mine_locations.get('total_count', 0) == 0:
        logger.warning("No mine locations loaded - this may be expected if GPS file is missing")
    else:
        logger.info(f"✅ Loaded {mine_locations['total_count']} mine locations")

    # Check augmentation
    aug_info = results['augmentation_info']
    if aug_info:
        logger.info(f"✅ Data augmentation applied: "
                   f"{aug_info.get('original_positive', 0)} → {aug_info.get('final_positive', 0)} positive, "
                   f"{aug_info.get('original_negative', 0)} → {aug_info.get('final_negative', 0)} negative")
    else:
        logger.warning("No augmentation info found")

    # Check data splits
    splits = results['data_splits']
    total_samples = splits.get('train_size', 0) + splits.get('validation_size', 0) + splits.get('test_size', 0)
    if total_samples == 0:
        logger.error("No data splits created")
        return False
    else:
        logger.info(f"✅ Data splits created: {total_samples} total samples")

    # Check evaluation results
    eval_results = results['evaluation_results']
    if not eval_results or 'error' in eval_results:
        logger.error(f"Model evaluation failed: {eval_results.get('error', 'Unknown error')}")
        return False

    # Check if we have metrics for each split
    for split_name in ['train', 'validation', 'test']:
        if split_name in eval_results and 'metrics' in eval_results[split_name]:
            metrics = eval_results[split_name]['metrics']
            accuracy = metrics.get('accuracy', 0)
            logger.info(f"✅ {split_name.capitalize()} accuracy: {accuracy:.3f}")
        else:
            logger.warning(f"No evaluation metrics for {split_name} split")

    # Check pipeline internal state
    if not hasattr(pipeline, 'datasets') or not pipeline.datasets:
        logger.error("Pipeline datasets not created")
        return False

    logger.info("✅ All verifications passed")
    return True


def print_test_summary(results: dict, pipeline: EnhancedUnifiedPipeline):
    """테스트 결과 요약 출력"""

    print("\n" + "="*60)
    print("ENHANCED PIPELINE TEST SUMMARY")
    print("="*60)

    # Mine locations
    mine_info = results.get('mine_locations', {})
    print(f"📍 Mine Locations: {mine_info.get('total_count', 0)}")

    # Data augmentation
    aug_info = results.get('augmentation_info', {})
    if aug_info:
        print(f"🔄 Data Augmentation:")
        print(f"   Original → Final: {aug_info.get('original_positive', 0)} → {aug_info.get('final_positive', 0)} positive")
        print(f"   Original → Final: {aug_info.get('original_negative', 0)} → {aug_info.get('final_negative', 0)} negative")
        print(f"   Augmentation ratio: {aug_info.get('augmentation_ratio', 0):.2f}x")

    # Data splits
    splits = results.get('data_splits', {})
    print(f"📊 Data Splits:")
    print(f"   Training: {splits.get('train_size', 0)} samples")
    print(f"   Validation: {splits.get('validation_size', 0)} samples")
    print(f"   Test: {splits.get('test_size', 0)} samples")

    # Model performance
    eval_results = results.get('evaluation_results', {})
    if eval_results and 'error' not in eval_results:
        print(f"🎯 Model Performance:")
        for split_name, split_results in eval_results.items():
            if 'metrics' in split_results:
                metrics = split_results['metrics']
                print(f"   {split_name.capitalize()}: "
                     f"Acc={metrics.get('accuracy', 0):.3f}, "
                     f"F1={metrics.get('f1_score', 0):.3f}")

    # Coordinate mapping accuracy
    coord_accuracy = results.get('coordinate_mapping_accuracy', 0)
    if coord_accuracy > 0:
        print(f"🗺️ Coordinate Mapping: {coord_accuracy:.1%} accuracy")

    # Output location
    if hasattr(pipeline, 'config'):
        print(f"💾 Results saved to: {pipeline.config.output_dir}")

    print("="*60)


def test_individual_components():
    """개별 컴포넌트 테스트"""

    logger.info("Testing individual enhanced components...")

    try:
        # Test basic configuration
        config = EnhancedPipelineConfig()
        pipeline = EnhancedUnifiedPipeline(config)

        # Test mine locations processing
        logger.info("Testing mine locations processing...")
        mine_locations_file = Path("datasets/Location_MDGPS.xlsx")
        if mine_locations_file.exists():
            mine_locations = pipeline.process_mine_locations(mine_locations_file)
            logger.info(f"✅ Mine locations test: {mine_locations.get('total_count', 0)} locations loaded")
        else:
            logger.warning("⚠️ Mine locations file not found, skipping test")

        # Test data augmentation engine
        logger.info("Testing data augmentation engine...")
        if pipeline.augmentation_engine:
            # Create dummy patch for testing
            import numpy as np
            dummy_patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            augmented_patch, _ = pipeline.augmentation_engine.augment_single(dummy_patch)
            logger.info(f"✅ Data augmentation test: {dummy_patch.shape} → {augmented_patch.shape}")
        else:
            logger.warning("⚠️ Data augmentation engine not initialized")

        logger.info("✅ Individual component tests completed")
        return True

    except Exception as e:
        logger.error(f"Individual component test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """메인 실행 함수"""

    print("Enhanced Pipeline Test Suite")
    print("===========================")

    # Test individual components first
    component_test_passed = test_individual_components()

    if component_test_passed:
        print("\n✅ Individual component tests passed")

        # Run full pipeline test
        pipeline_test_passed = test_enhanced_pipeline()

        if pipeline_test_passed:
            print("\n🎉 All tests passed! Enhanced pipeline is ready for use.")
            return 0
        else:
            print("\n❌ Full pipeline test failed")
            return 1
    else:
        print("\n❌ Component tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)