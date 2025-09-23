#!/usr/bin/env python3
"""
Pipeline Test Script
==================
통합 파이프라인 테스트 스크립트
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)
logger = logging.getLogger(__name__)

def test_individual_modules():
    """개별 모듈 테스트"""
    logger.info("="*60)
    logger.info("INDIVIDUAL MODULE TESTING")
    logger.info("="*60)

    # Test data paths
    xtf_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")
    gps_path = Path("datasets/Location_MDGPS.xlsx")
    annotation_path = Path("datasets/PH_annotation.png")

    results = {}

    try:
        # Test 1: XTF Reader
        logger.info("Testing XTF Reader...")
        from pipeline.modules.xtf_reader import XTFReader

        start_time = time.time()
        xtf_reader = XTFReader()

        if xtf_path.exists():
            xtf_data = xtf_reader.read(xtf_path)
            results['xtf_reader'] = {
                'status': 'success',
                'execution_time': time.time() - start_time,
                'file_size_mb': xtf_path.stat().st_size / (1024 * 1024),
                'ping_count': xtf_data.get('ping_count', 0),
                'channels': list(xtf_data.get('channels', {}).keys()),
                'metadata': xtf_data.get('metadata', {})
            }
            logger.info(f"✅ XTF Reader: {results['xtf_reader']['ping_count']} pings, {results['xtf_reader']['file_size_mb']:.1f}MB")
        else:
            results['xtf_reader'] = {'status': 'failed', 'error': 'XTF file not found'}
            logger.error(f"❌ XTF file not found: {xtf_path}")
            return results

    except Exception as e:
        results['xtf_reader'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ XTF Reader failed: {e}")
        return results

    try:
        # Test 2: XTF Extractor
        logger.info("Testing XTF Extractor...")
        from pipeline.modules.xtf_extractor import XTFExtractor

        start_time = time.time()
        extractor = XTFExtractor(sample_rate=0.1)  # 10% sample for testing
        extracted_data = extractor.extract(xtf_data)

        results['xtf_extractor'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'intensity_shape': extracted_data.get('intensity').shape if extracted_data.get('intensity') is not None else None,
            'processing_info': extracted_data.get('processing_info', {}),
            'navigation_points': len(extracted_data.get('navigation', {}).get('latitude', []))
        }
        logger.info(f"✅ XTF Extractor: Shape {results['xtf_extractor']['intensity_shape']}, {results['xtf_extractor']['navigation_points']} nav points")

    except Exception as e:
        results['xtf_extractor'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ XTF Extractor failed: {e}")
        return results

    try:
        # Test 3: Coordinate Mapper
        logger.info("Testing Coordinate Mapper...")
        from pipeline.modules.coordinate_mapper import CoordinateMapper

        start_time = time.time()
        mapper = CoordinateMapper()
        mapping_data = mapper.map(extracted_data, gps_path, annotation_path)

        results['coordinate_mapper'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'mapping_count': len(mapping_data.get('mappings', [])),
            'image_size': mapping_data.get('image_size'),
            'mapping_stats': mapping_data.get('mapping_statistics', {})
        }
        logger.info(f"✅ Coordinate Mapper: {results['coordinate_mapper']['mapping_count']} mappings")

    except Exception as e:
        results['coordinate_mapper'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Coordinate Mapper failed: {e}")
        return results

    try:
        # Test 4: Label Generator
        logger.info("Testing Label Generator...")
        from pipeline.modules.label_generator import LabelGenerator

        start_time = time.time()
        label_gen = LabelGenerator(patch_size=(32, 32))  # Smaller patches for testing
        label_data = label_gen.generate(mapping_data, extracted_data.get('intensity'))

        results['label_generator'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'total_samples': len(label_data.get('samples', [])),
            'positive_count': label_data.get('positive_count', 0),
            'negative_count': label_data.get('negative_count', 0),
            'class_distribution': label_data.get('class_distribution', {})
        }
        logger.info(f"✅ Label Generator: {results['label_generator']['positive_count']} positive, {results['label_generator']['negative_count']} negative")

    except Exception as e:
        results['label_generator'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Label Generator failed: {e}")
        return results

    try:
        # Test 5: Feature Extractor
        logger.info("Testing Feature Extractor...")
        from pipeline.modules.feature_extractor import FeatureExtractor

        start_time = time.time()
        feature_extractor = FeatureExtractor(
            methods=['statistical', 'textural'],  # Limited methods for testing
            patch_size=(32, 32)
        )
        feature_data = feature_extractor.extract(extracted_data.get('intensity'), label_data)

        results['feature_extractor'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'features_shape': feature_data['features'].shape,
            'n_features': feature_data['features'].shape[1],
            'feature_names_count': len(feature_data['feature_names']),
            'extraction_info': feature_data.get('extraction_info', {})
        }
        logger.info(f"✅ Feature Extractor: {results['feature_extractor']['features_shape']} features extracted")

    except Exception as e:
        results['feature_extractor'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Feature Extractor failed: {e}")
        return results

    try:
        # Test 6: Ensemble Optimizer (Quick test)
        logger.info("Testing Ensemble Optimizer...")
        from pipeline.modules.ensemble_optimizer import EnsembleOptimizer

        start_time = time.time()
        optimizer = EnsembleOptimizer(max_trials=10)  # Quick test
        ensemble_config = optimizer.optimize(feature_data['features'], label_data['labels'])

        results['ensemble_optimizer'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'best_config_type': ensemble_config['best_config']['type'],
            'best_score': ensemble_config['best_config']['score'],
            'individual_results_count': len(ensemble_config.get('individual_results', {}))
        }
        logger.info(f"✅ Ensemble Optimizer: Best {results['ensemble_optimizer']['best_config_type']} with score {results['ensemble_optimizer']['best_score']:.4f}")

    except Exception as e:
        results['ensemble_optimizer'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Ensemble Optimizer failed: {e}")
        return results

    try:
        # Test 7: Mine Classifier
        logger.info("Testing Mine Classifier...")
        from pipeline.modules.mine_classifier import MineClassifier

        start_time = time.time()
        classifier = MineClassifier()
        predictions = classifier.classify(feature_data['features'], ensemble_config)

        results['mine_classifier'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'predictions_shape': predictions.shape,
            'positive_predictions': int(sum(predictions)),
            'negative_predictions': int(len(predictions) - sum(predictions))
        }
        logger.info(f"✅ Mine Classifier: {results['mine_classifier']['positive_predictions']} positive, {results['mine_classifier']['negative_predictions']} negative predictions")

    except Exception as e:
        results['mine_classifier'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Mine Classifier failed: {e}")
        return results

    try:
        # Test 8: Terrain Analyzer
        logger.info("Testing Terrain Analyzer...")
        from pipeline.modules.terrain_analyzer import TerrainAnalyzer

        start_time = time.time()
        terrain_analyzer = TerrainAnalyzer()
        terrain_results = terrain_analyzer.analyze(extracted_data.get('intensity'), predictions, label_data)

        results['terrain_analyzer'] = {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'refined_predictions_shape': terrain_results['refined_predictions'].shape,
            'terrain_stats': terrain_results.get('terrain_statistics', {}),
            'changes_made': int(sum(predictions != terrain_results['refined_predictions']))
        }
        logger.info(f"✅ Terrain Analyzer: {results['terrain_analyzer']['changes_made']} predictions refined")

    except Exception as e:
        results['terrain_analyzer'] = {'status': 'failed', 'error': str(e)}
        logger.error(f"❌ Terrain Analyzer failed: {e}")

    return results

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    logger.info("="*60)
    logger.info("FULL PIPELINE TESTING")
    logger.info("="*60)

    try:
        from pipeline.unified_pipeline import UnifiedPipeline, PipelineConfig

        # Configure pipeline
        config = PipelineConfig(
            data_dir=Path("datasets"),
            output_dir=Path("test_results"),
            xtf_sample_rate=0.1,  # 10% for testing
            gps_file=Path("datasets/Location_MDGPS.xlsx"),
            annotation_image=Path("datasets/PH_annotation.png"),
            feature_methods=['statistical', 'textural'],  # Limited for testing
            patch_size=(32, 32),
            save_intermediate=True,
            verbose=True
        )

        # Initialize pipeline
        pipeline = UnifiedPipeline(config)

        # XTF file path
        xtf_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")

        # Run full pipeline
        start_time = time.time()
        final_results = pipeline.run_full_pipeline(xtf_path)
        execution_time = time.time() - start_time

        results = {
            'status': 'success',
            'execution_time': execution_time,
            'config': config.__dict__,
            'final_results': final_results,
            'pipeline_steps_completed': len([k for k in pipeline.results.keys() if pipeline.results[k] is not None])
        }

        logger.info(f"✅ Full Pipeline completed in {execution_time:.2f}s")
        logger.info(f"Pipeline steps completed: {results['pipeline_steps_completed']}")

        return results

    except Exception as e:
        logger.error(f"❌ Full Pipeline failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def test_modular_pipeline():
    """모듈별 파이프라인 테스트"""
    logger.info("="*60)
    logger.info("MODULAR PIPELINE TESTING")
    logger.info("="*60)

    try:
        from pipeline.unified_pipeline import UnifiedPipeline, PipelineConfig

        config = PipelineConfig(
            data_dir=Path("datasets"),
            output_dir=Path("test_results"),
            xtf_sample_rate=0.1,
            gps_file=Path("datasets/Location_MDGPS.xlsx"),
            annotation_image=Path("datasets/PH_annotation.png"),
            feature_methods=['statistical'],
            patch_size=(32, 32),
            save_intermediate=True
        )

        pipeline = UnifiedPipeline(config)
        xtf_path = Path("datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf")

        steps = ['read', 'extract', 'map', 'label', 'feature']
        results = {}

        for step in steps:
            try:
                start_time = time.time()

                if step == 'read':
                    result = pipeline.read_xtf(xtf_path)
                elif step == 'extract':
                    result = pipeline.extract_xtf_data()
                elif step == 'map':
                    result = pipeline.map_coordinates()
                elif step == 'label':
                    result = pipeline.generate_labels()
                elif step == 'feature':
                    result = pipeline.extract_features()

                execution_time = time.time() - start_time
                results[step] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'result_keys': list(result.keys()) if isinstance(result, dict) else 'non-dict'
                }

                logger.info(f"✅ Step '{step}' completed in {execution_time:.2f}s")

            except Exception as e:
                results[step] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ Step '{step}' failed: {e}")
                break

        return results

    except Exception as e:
        logger.error(f"❌ Modular Pipeline setup failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def generate_test_report(module_results, full_pipeline_results, modular_results):
    """테스트 결과 리포트 생성"""
    report_path = Path(f"test_report_{datetime.now():%Y%m%d_%H%M%S}.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Pipeline Test Report\n\n")
        f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Individual Modules
        f.write("## Individual Module Test Results\n\n")
        for module, result in module_results.items():
            status = "✅" if result.get('status') == 'success' else "❌"
            f.write(f"### {status} {module.replace('_', ' ').title()}\n")

            if result.get('status') == 'success':
                f.write(f"- **Execution Time**: {result.get('execution_time', 0):.2f}s\n")
                for key, value in result.items():
                    if key not in ['status', 'execution_time']:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            else:
                f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
            f.write("\n")

        # Full Pipeline
        f.write("## Full Pipeline Test Results\n\n")
        if full_pipeline_results.get('status') == 'success':
            f.write("✅ **Status**: SUCCESS\n")
            f.write(f"- **Total Execution Time**: {full_pipeline_results.get('execution_time', 0):.2f}s\n")
            f.write(f"- **Steps Completed**: {full_pipeline_results.get('pipeline_steps_completed', 0)}\n")
        else:
            f.write("❌ **Status**: FAILED\n")
            f.write(f"- **Error**: {full_pipeline_results.get('error', 'Unknown error')}\n")
        f.write("\n")

        # Modular Pipeline
        f.write("## Modular Pipeline Test Results\n\n")
        for step, result in modular_results.items():
            if step != 'status':
                status = "✅" if result.get('status') == 'success' else "❌"
                f.write(f"### {status} Step: {step}\n")
                f.write(f"- **Execution Time**: {result.get('execution_time', 0):.2f}s\n")
                if result.get('status') != 'success':
                    f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
                f.write("\n")

    logger.info(f"📊 Test report generated: {report_path}")
    return report_path

def main():
    """메인 테스트 실행"""
    logger.info("🚀 Starting Pipeline Test Suite")

    # Create test results directory
    test_results_dir = Path("test_results")
    test_results_dir.mkdir(exist_ok=True)

    # Run tests
    logger.info("1️⃣ Running Individual Module Tests...")
    module_results = test_individual_modules()

    logger.info("2️⃣ Running Full Pipeline Test...")
    full_pipeline_results = test_full_pipeline()

    logger.info("3️⃣ Running Modular Pipeline Test...")
    modular_results = test_modular_pipeline()

    # Generate report
    logger.info("📊 Generating Test Report...")
    report_path = generate_test_report(module_results, full_pipeline_results, modular_results)

    # Summary
    successful_modules = sum(1 for r in module_results.values() if r.get('status') == 'success')
    total_modules = len(module_results)

    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Individual Modules: {successful_modules}/{total_modules} successful")
    logger.info(f"Full Pipeline: {'✅ SUCCESS' if full_pipeline_results.get('status') == 'success' else '❌ FAILED'}")
    logger.info(f"Modular Pipeline: {'✅ SUCCESS' if len([r for r in modular_results.values() if isinstance(r, dict) and r.get('status') == 'success']) > 0 else '❌ FAILED'}")
    logger.info(f"Report saved to: {report_path}")

    return {
        'module_results': module_results,
        'full_pipeline_results': full_pipeline_results,
        'modular_results': modular_results,
        'report_path': str(report_path)
    }

if __name__ == "__main__":
    test_results = main()