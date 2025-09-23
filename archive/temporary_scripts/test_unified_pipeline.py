#!/usr/bin/env python3
"""
Unified Pipeline Test Script
============================
통합된 파이프라인 테스트 - 기존 및 enhanced 기능 검증

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

from pipeline.unified_pipeline import UnifiedPipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'unified_pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)


def test_basic_pipeline():
    """기본 파이프라인 테스트"""

    logger.info("="*60)
    logger.info("BASIC UNIFIED PIPELINE TEST")
    logger.info("="*60)

    try:
        # Basic configuration
        config = PipelineConfig(
            data_dir=Path("datasets"),
            output_dir=Path("data/processed/unified_pipeline_test"),
            verbose=True
        )

        # Create pipeline
        pipeline = UnifiedPipeline(config)

        # Test XTF file
        xtf_file = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")

        if not xtf_file.exists():
            logger.error(f"XTF file not found: {xtf_file}")
            return False

        logger.info(f"Testing with XTF file: {xtf_file}")
        logger.info(f"File size: {xtf_file.stat().st_size / (1024*1024):.1f} MB")

        # Run basic pipeline steps
        logger.info("Running basic pipeline steps...")

        # Step 1: Read XTF
        pipeline.read_xtf(xtf_file)
        logger.info("✅ XTF reading completed")

        # Step 2: Extract data
        pipeline.extract_xtf_data()
        logger.info("✅ XTF data extraction completed")

        # Verify results
        if 'xtf_data' in pipeline.results and 'extracted_data' in pipeline.results:
            logger.info("✅ Basic pipeline test PASSED")
            return True
        else:
            logger.error("❌ Basic pipeline test FAILED - Missing results")
            return False

    except Exception as e:
        logger.error(f"Basic pipeline test failed with exception: {e}")
        logger.error(traceback.format_exc())
        return False


def test_enhanced_pipeline():
    """Enhanced 파이프라인 테스트"""

    logger.info("="*60)
    logger.info("ENHANCED UNIFIED PIPELINE TEST")
    logger.info("="*60)

    try:
        # Enhanced configuration
        config = PipelineConfig(
            data_dir=Path("datasets"),
            output_dir=Path("data/processed/unified_pipeline_enhanced_test"),

            # Enhanced features
            mine_locations_file=Path("datasets/Location_MDGPS.xlsx"),
            enable_augmentation=True,
            mine_augmentation_factor=5,

            # Pipeline settings
            save_intermediate=True,
            generate_reports=True,
            verbose=True
        )

        # Create pipeline
        pipeline = UnifiedPipeline(config)

        # Test XTF file
        xtf_file = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")
        mine_locations_file = Path("datasets/Location_MDGPS.xlsx")

        if not xtf_file.exists():
            logger.error(f"XTF file not found: {xtf_file}")
            return False

        if not mine_locations_file.exists():
            logger.error(f"Mine locations file not found: {mine_locations_file}")
            return False

        logger.info(f"Testing enhanced pipeline with:")
        logger.info(f"  XTF file: {xtf_file}")
        logger.info(f"  Mine locations: {mine_locations_file}")

        # Run enhanced pipeline
        results = pipeline.run_enhanced_mine_detection_pipeline(
            xtf_file, mine_locations_file
        )

        # Verify results
        if results and 'mine_locations' in results:
            mine_count = results['mine_locations']['total_count']
            logger.info(f"✅ Enhanced pipeline test PASSED")
            logger.info(f"   Mine locations processed: {mine_count}")

            return True
        else:
            logger.error("❌ Enhanced pipeline test FAILED - Missing results")
            return False

    except Exception as e:
        logger.error(f"Enhanced pipeline test failed with exception: {e}")
        logger.error(traceback.format_exc())
        return False


def test_module_availability():
    """모듈 가용성 테스트"""

    logger.info("Testing module availability...")

    try:
        from pipeline.modules.gps_parser import GPSParser
        logger.info("✅ GPS Parser available")

        # Test GPS parsing
        parser = GPSParser()
        test_coords = ["36.593398", "129 30.557773 E", "36°35'36.23\""]
        for coord in test_coords:
            result = parser.parse_coordinate_string(coord)
            logger.info(f"   '{coord}' -> {result}")

    except Exception as e:
        logger.warning(f"GPS Parser not available: {e}")

    try:
        from src.data_augmentation.augmentation_engine import DataAugmentationEngine
        engine = DataAugmentationEngine()
        logger.info("✅ Data Augmentation Engine available")
    except Exception as e:
        logger.warning(f"Data Augmentation Engine not available: {e}")


def main():
    """메인 실행 함수"""

    print("Unified Pipeline Test Suite")
    print("===========================")

    # Test module availability
    test_module_availability()

    # Test basic pipeline
    basic_test_passed = test_basic_pipeline()

    if basic_test_passed:
        print("\n✅ Basic pipeline tests passed")

        # Test enhanced pipeline
        enhanced_test_passed = test_enhanced_pipeline()

        if enhanced_test_passed:
            print("\n🎉 All tests passed! Unified pipeline is ready for use.")
            return 0
        else:
            print("\n⚠️  Enhanced pipeline test failed, but basic functionality works")
            return 1
    else:
        print("\n❌ Basic pipeline tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)