# Configuration Guide

Complete guide for configuring the Mine Labeling Project.

---

## Configuration Files

### 1. Default Configuration

**Location**: `config/default_config.yaml`

Contains all default settings for the project. Used automatically if no custom configuration is specified.

### 2. Example Configuration

**Location**: `config/example_config.yaml`

Contains example configurations for different use cases:
- Quick testing
- Production pipeline
- New data processing
- Minimal configuration

### 3. Custom Configuration

Create your own configuration file by copying and modifying:

```bash
cp config/example_config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
```

---

## Configuration Structure

### Paths Section

```yaml
paths:
  # Input data
  xtf_data: "datasets/original"
  bmp_images: "KleinLabeling"
  xml_annotations: "KleinLabeling"

  # Verified data (pre-processed)
  verified_data: "verified_data/flipped_20251104"

  # Output directories
  output_root: "data/processed"
  extracted: "data/processed/xtf_extracted"
  labeled: "data/processed/labeled"
  sampled: "data/processed/sampled"
  augmented: "data/processed/augmented"
```

**Key Points**:
- `verified_data`: Use this for pre-processed, validated data
- Output directories are created automatically if they don't exist

### Extraction Section

```yaml
extraction:
  sample_rate: 1.0        # Full data (1.0), or sampled (e.g., 0.1)
  ping_range: null        # null = all pings, or [start, end]
  normalize: true         # Normalize intensity values to [0, 1]
  dtype: "float32"        # Data type for intensity arrays
```

**Key Points**:
- `sample_rate: 1.0` extracts all pings (5137 for verified data)
- Always use `normalize: true` for consistent intensity ranges

### Coordinate Mapping Section

```yaml
coordinate_mapping:
  # Source dimensions (BMP)
  bmp_width: 1024
  bmp_height: 5137

  # Target dimensions (NPY)
  npy_width: 6400
  npy_height: 5137

  # Scaling factors
  scale_x: 6.25           # Automatically calculated: 6400 / 1024
  scale_y: 1.0            # 5137 / 5137

  # Y-axis flip (CRITICAL)
  apply_y_flip: true      # TRUE for existing BMP data
                          # FALSE for new XTF→image conversions
```

**CRITICAL**: Y-axis flip setting

- **`apply_y_flip: true`**: Use for existing BMP annotations (already flipped)
- **`apply_y_flip: false`**: Use for new image generation from XTF

**Flip formula**: `Y_npy = (npy_height - 1) - Y_bmp`

### Labeling Section

```yaml
labeling:
  label_dtype: "uint8"
  background_value: 0
  mine_value: 1

  enable_instance_ids: true    # Assign unique ID to each mine
  instance_start_id: 1

  bbox_padding: 0              # Pixels to expand bounding boxes

  # Output formats
  save_npz: true               # Primary format
  save_json: true              # COCO format
  save_xml: false              # PASCAL VOC format
```

**Key Points**:
- Binary labels: 0 = background, 1 = mine
- NPZ format is primary (intensity + labels + metadata)
- JSON format useful for COCO-style training

### Sampling Section

```yaml
sampling:
  mine_samples: 25             # All mines from dataset
  background_samples: 125      # 1:5 ratio (mine:background)

  random_seed: 42              # Reproducibility
  ensure_no_overlap: true      # Samples don't overlap
  min_distance_from_mine: 50   # Pixels away from mines

  patch_size: 128              # Sample patch size
  center_crop: true            # Center objects in patches
```

**Key Points**:
- 1:5 ratio maintains class balance
- `min_distance_from_mine` ensures clean background samples
- `patch_size: 128` good for CNN training

### Augmentation Section

```yaml
augmentation:
  augmentation_factor: 9       # 9 versions per sample
  random_seed: 42

  techniques:
    horizontal_flip:
      enabled: true
      probability: 1.0

    vertical_flip:
      enabled: true
      probability: 1.0

    rotation:
      enabled: true
      limit: 15                # ±15 degrees
      probability: 1.0

    scale:
      enabled: true
      scale_limit: 0.1         # ±10%
      probability: 1.0

    brightness:
      enabled: true
      limit: 0.2               # ±20%
      probability: 1.0

    contrast:
      enabled: true
      limit: 0.2               # ±20%
      probability: 1.0

    gaussian_blur:
      enabled: true
      blur_limit: [3, 7]
      probability: 0.5

    gaussian_noise:
      enabled: true
      var_limit: [10, 50]
      probability: 0.5

    elastic_transform:
      enabled: true
      alpha: 1
      sigma: 50
      probability: 0.3
```

**Key Points**:
- 9 techniques total (excluding original)
- Adjust `probability` to control frequency
- Set `enabled: false` to disable specific techniques

---

## Usage Examples

### Using Default Configuration

```python
from mine_labeling.utils import load_config

# Load default configuration
config = load_config()

print(config['paths']['verified_data'])
# Output: verified_data/flipped_20251104
```

### Using Custom Configuration

```python
from mine_labeling.utils import load_custom_config

# Load and merge with defaults
config = load_custom_config('config/my_config.yaml')

print(config['sampling']['mine_samples'])
# Output: Your custom value or default (25)
```

### Validating Configuration

```python
from mine_labeling.utils import load_config, validate_config

config = load_config('config/my_config.yaml')

try:
    validate_config(config)
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

### Accessing Configuration Values

```python
config = load_config()

# Access nested values
bmp_width = config['coordinate_mapping']['bmp_width']
mine_samples = config['sampling']['mine_samples']
apply_flip = config['coordinate_mapping']['apply_y_flip']

# Access paths
verified_data = config['paths']['verified_data']
output_root = config['paths']['output_root']
```

---

## Common Configuration Scenarios

### Scenario 1: Quick Testing

```yaml
# config/test_config.yaml
paths:
  verified_data: "verified_data/flipped_20251104"
  output_root: "data/test_output"

sampling:
  mine_samples: 10
  background_samples: 50
  patch_size: 64

augmentation:
  augmentation_factor: 3
```

### Scenario 2: Full Production Pipeline

```yaml
# config/production_config.yaml
paths:
  verified_data: "verified_data/flipped_20251104"
  output_root: "data/production"

sampling:
  mine_samples: 25
  background_samples: 125
  patch_size: 128

augmentation:
  augmentation_factor: 9

logging:
  level: "INFO"
  file_enabled: true
```

### Scenario 3: Processing New XTF Data

```yaml
# config/new_data_config.yaml
paths:
  xtf_data: "datasets/new_survey/original"
  bmp_images: "datasets/new_survey/annotations"
  xml_annotations: "datasets/new_survey/annotations"
  output_root: "data/new_survey"

extraction:
  sample_rate: 1.0
  normalize: true

coordinate_mapping:
  apply_y_flip: false  # NEW DATA: no flip needed

labeling:
  validate_labels: true
  save_npz: true
```

---

## Configuration Best Practices

1. **Always Use Defaults as Base**: Start with `default_config.yaml` and override only what you need

2. **Validate Configuration**: Always validate before running pipeline

3. **Use Descriptive Names**: Name custom configs by purpose: `test_config.yaml`, `production_config.yaml`

4. **Document Changes**: Add comments explaining why you changed default values

5. **Version Control**: Keep configuration files in version control

6. **Environment-Specific**: Create separate configs for different environments

---

## Troubleshooting

### Configuration File Not Found

**Error**: `FileNotFoundError: Configuration file not found`

**Solution**:
```python
from pathlib import Path

config_path = Path('config/my_config.yaml')
if not config_path.exists():
    print(f"Config file doesn't exist: {config_path.absolute()}")
```

### Invalid Configuration Values

**Error**: `ValueError: Missing required configuration key`

**Solution**: Check that all required sections exist:
- `paths`
- `extraction`
- `coordinate_mapping`
- `labeling`

### Y-Axis Flip Confusion

**Problem**: Labels don't match visual inspection

**Solution**: Check `apply_y_flip` setting:
- **Existing BMP data**: `apply_y_flip: true`
- **New XTF extraction**: `apply_y_flip: false`

---

## Advanced Configuration

### Custom Augmentation Pipeline

```yaml
augmentation:
  techniques:
    # Enable only geometric transforms
    horizontal_flip:
      enabled: true
    vertical_flip:
      enabled: true
    rotation:
      enabled: true

    # Disable intensity transforms
    brightness:
      enabled: false
    contrast:
      enabled: false
    gaussian_blur:
      enabled: false
    gaussian_noise:
      enabled: false
    elastic_transform:
      enabled: false
```

### Performance Tuning

```yaml
performance:
  chunk_size: 5000      # Reduce for low memory
  max_workers: 8        # Increase for parallel processing

  enable_cache: true
  cache_dir: ".cache"
```

### Logging Configuration

```yaml
logging:
  level: "DEBUG"        # Verbose logging

  console_enabled: true
  file_enabled: true

  rotate_logs: true
  max_bytes: 10485760   # 10MB per log file
  backup_count: 5
```

---

**Last Updated**: 2025-11-04
