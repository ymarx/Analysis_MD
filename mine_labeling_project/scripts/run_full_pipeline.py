#!/usr/bin/env python3
"""
Full Mine Labeling Pipeline

Complete workflow:
1. Load verified data (or extract from XTF)
2. Map coordinates and generate labels
3. Validate mapping accuracy
4. Sample mines and background
5. Apply data augmentation
6. Save final dataset

Author: Mine Detection Team
Date: 2025-11-04
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
from mine_labeling.utils import load_config, setup_logging, load_npz_data, create_directories
from mine_labeling.labeling import CoordinateMapper, LabelGenerator
from mine_labeling.validation import MappingValidator
from mine_labeling.sampling import MineSampler, BackgroundSampler
from mine_labeling.augmentation import Augmentor


def load_verified_data(config):
    """Load verified labeled data"""
    logger = setup_logging(config)
    logger.info("="*60)
    logger.info("STEP 1: LOADING VERIFIED DATA")
    logger.info("="*60)

    verified_path = Path(config['paths']['verified_data']) / 'flipped_labeled_intensity_data.npz'

    if not verified_path.exists():
        logger.error(f"Verified data not found: {verified_path}")
        raise FileNotFoundError(f"Verified data not found: {verified_path}")

    logger.info(f"Loading: {verified_path}")
    data = load_npz_data(verified_path)

    logger.info(f"Intensity shape: {data['intensity'].shape}")
    logger.info(f"Labels shape: {data['labels'].shape}")
    logger.info(f"Annotations: {len(data['metadata'])}")

    return data


def validate_data(data, config):
    """Validate data quality and mapping accuracy"""
    logger = setup_logging(config)
    logger.info("\n" + "="*60)
    logger.info("STEP 2: VALIDATING DATA")
    logger.info("="*60)

    validator = MappingValidator.from_config(config)

    # BMP shape (from config)
    bmp_shape = (
        config['coordinate_mapping']['bmp_height'],
        config['coordinate_mapping']['bmp_width']
    )

    # Run all validations
    all_passed = validator.run_all_validations(
        npy_data=data['intensity'],
        bmp_shape=bmp_shape,
        label_mask=data['labels'],
        annotations=data['metadata']
    )

    validator.print_summary()

    if not all_passed:
        logger.error("Validation failed!")
        raise ValueError("Data validation failed. Please check the results above.")

    logger.info("✓ All validations passed")

    return True


def sample_data(data, config):
    """Extract mine and background samples"""
    logger = setup_logging(config)
    logger.info("\n" + "="*60)
    logger.info("STEP 3: SAMPLING DATA")
    logger.info("="*60)

    # Create samplers
    mine_sampler = MineSampler.from_config(config)
    background_sampler = BackgroundSampler.from_config(config)

    # Extract mine samples
    logger.info("\nExtracting mine samples...")
    mine_samples = mine_sampler.extract_all_mines(
        intensity=data['intensity'],
        label_mask=data['labels'],
        annotations=data['metadata']
    )

    mine_stats = mine_sampler.get_sampling_statistics(mine_samples)
    logger.info(f"✓ Extracted {mine_stats['total_samples']} mine samples")
    logger.info(f"  - Padded samples: {mine_stats['padded_samples']}")
    logger.info(f"  - Label coverage: {mine_stats['label_coverage']['mean']:.2%}")

    # Save mine samples
    mine_output_dir = Path(config['paths']['sampled']) / 'mines'
    mine_sampler.save_samples(mine_samples, mine_output_dir)
    logger.info(f"✓ Saved to: {mine_output_dir}")

    # Extract background samples
    logger.info("\nExtracting background samples...")
    background_samples = background_sampler.extract_background_samples(
        intensity=data['intensity'],
        label_mask=data['labels'],
        annotations=data['metadata']
    )

    bg_stats = background_sampler.get_sampling_statistics(background_samples)
    logger.info(f"✓ Extracted {bg_stats['total_samples']} background samples")
    logger.info(f"  - Pure background: {bg_stats['pure_background']}")
    logger.info(f"  - Purity rate: {bg_stats['purity_rate']:.2%}")

    # Save background samples
    bg_output_dir = Path(config['paths']['sampled']) / 'background'
    background_sampler.save_samples(background_samples, bg_output_dir)
    logger.info(f"✓ Saved to: {bg_output_dir}")

    return mine_samples, background_samples


def augment_data(mine_samples, background_samples, config):
    """Apply data augmentation"""
    logger = setup_logging(config)
    logger.info("\n" + "="*60)
    logger.info("STEP 4: DATA AUGMENTATION")
    logger.info("="*60)

    augmentor = Augmentor.from_config(config)

    save_original = config['augmentation'].get('save_original', False)

    # Augment mine samples
    logger.info("\nAugmenting mine samples...")
    augmented_mines = augmentor.process_dataset(mine_samples, save_original=save_original)
    logger.info(f"✓ Generated {len(augmented_mines)} augmented mine samples")

    mine_aug_dir = Path(config['paths']['augmented']) / 'mines'
    separate_folders = config['augmentation'].get('separate_folders', True)
    augmentor.save_augmented_samples(augmented_mines, mine_aug_dir, separate_folders)
    logger.info(f"✓ Saved to: {mine_aug_dir}")

    # Augment background samples
    logger.info("\nAugmenting background samples...")
    augmented_background = augmentor.process_dataset(background_samples, save_original=save_original)
    logger.info(f"✓ Generated {len(augmented_background)} augmented background samples")

    bg_aug_dir = Path(config['paths']['augmented']) / 'background'
    augmentor.save_augmented_samples(augmented_background, bg_aug_dir, separate_folders)
    logger.info(f"✓ Saved to: {bg_aug_dir}")

    return augmented_mines, augmented_background


def create_summary_report(config, mine_samples, background_samples,
                         augmented_mines, augmented_background):
    """Create summary report"""
    logger = setup_logging(config)
    logger.info("\n" + "="*60)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("="*60)

    summary = {
        'project': config['metadata']['project_name'],
        'version': config['metadata']['version'],
        'date': config['metadata']['date_created'],
        'configuration': {
            'patch_size': config['sampling']['patch_size'],
            'augmentation_factor': config['augmentation']['augmentation_factor'],
            'apply_y_flip': config['coordinate_mapping']['apply_y_flip']
        },
        'sampling': {
            'mine_samples': len(mine_samples),
            'background_samples': len(background_samples),
            'total_samples': len(mine_samples) + len(background_samples)
        },
        'augmentation': {
            'augmented_mines': len(augmented_mines),
            'augmented_background': len(augmented_background),
            'total_augmented': len(augmented_mines) + len(augmented_background)
        },
        'final_dataset': {
            'total_samples': len(augmented_mines) + len(augmented_background),
            'mine_ratio': len(augmented_mines) / (len(augmented_mines) + len(augmented_background))
        }
    }

    # Save summary
    summary_path = Path(config['paths']['output_root']) / 'pipeline_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Summary saved: {summary_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Original samples: {summary['sampling']['total_samples']}")
    logger.info(f"  - Mines: {summary['sampling']['mine_samples']}")
    logger.info(f"  - Background: {summary['sampling']['background_samples']}")
    logger.info(f"\nAugmented samples: {summary['augmentation']['total_augmented']}")
    logger.info(f"  - Mines: {summary['augmentation']['augmented_mines']}")
    logger.info(f"  - Background: {summary['augmentation']['augmented_background']}")
    logger.info(f"\nFinal dataset size: {summary['final_dataset']['total_samples']}")
    logger.info(f"Mine ratio: {summary['final_dataset']['mine_ratio']:.2%}")

    return summary


def main(config_path=None):
    """Run full pipeline"""
    # Load configuration
    if config_path:
        from mine_labeling.utils import load_custom_config
        config = load_custom_config(config_path)
    else:
        config = load_config()

    # Setup logging
    logger = setup_logging(config)

    logger.info("="*60)
    logger.info("MINE LABELING PIPELINE - FULL WORKFLOW")
    logger.info("="*60)
    logger.info(f"Project: {config['metadata']['project_name']}")
    logger.info(f"Version: {config['metadata']['version']}")

    # Create output directories
    create_directories(config)

    try:
        # Step 1: Load data
        data = load_verified_data(config)

        # Step 2: Validate data
        validate_data(data, config)

        # Step 3: Sample data
        mine_samples, background_samples = sample_data(data, config)

        # Step 4: Augment data
        augmented_mines, augmented_background = augment_data(
            mine_samples, background_samples, config
        )

        # Step 5: Create summary
        summary = create_summary_report(
            config, mine_samples, background_samples,
            augmented_mines, augmented_background
        )

        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        return summary

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run full mine labeling pipeline')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')

    args = parser.parse_args()

    main(args.config)
