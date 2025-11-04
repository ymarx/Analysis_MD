"""
File Utility Functions
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create output directory structure

    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})

    # Output directories
    output_dirs = [
        paths.get('output_root'),
        paths.get('extracted'),
        paths.get('labeled'),
        paths.get('sampled'),
        paths.get('augmented'),
        paths.get('logs'),
        paths.get('checkpoints')
    ]

    for dir_path in output_dirs:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_npz_data(npz_path: str) -> Dict[str, Any]:
    """
    Load labeled NPZ data

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary with intensity, labels, and metadata
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    result = {
        'intensity': data['intensity'],
        'labels': data['labels']
    }

    # Parse metadata
    if 'metadata' in data:
        metadata = data['metadata']

        # Handle different metadata formats
        if isinstance(metadata, np.ndarray):
            if metadata.ndim == 0:
                # Scalar array containing JSON string
                metadata = json.loads(metadata.item())
            else:
                metadata = json.loads(str(metadata))
        elif isinstance(metadata, str):
            metadata = json.loads(metadata)

        result['metadata'] = metadata

    return result


def save_npz_data(output_path: str,
                  intensity: np.ndarray,
                  labels: np.ndarray,
                  metadata: Dict[str, Any]) -> None:
    """
    Save labeled NPZ data

    Args:
        output_path: Output NPZ file path
        intensity: Intensity array (H, W)
        labels: Label mask array (H, W)
        metadata: Metadata dictionary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert metadata to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Save NPZ
    np.savez_compressed(
        output_path,
        intensity=intensity,
        labels=labels,
        metadata=metadata_json
    )


def load_npy_file(npy_path: str) -> np.ndarray:
    """
    Load NPY file

    Args:
        npy_path: Path to NPY file

    Returns:
        Numpy array
    """
    npy_path = Path(npy_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")

    return np.load(npy_path)


def save_npy_file(output_path: str, data: np.ndarray) -> None:
    """
    Save NPY file

    Args:
        output_path: Output NPY file path
        data: Numpy array
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, data)


def load_json_file(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_json_file(output_path: str, data: Dict[str, Any]) -> None:
    """
    Save JSON file

    Args:
        output_path: Output JSON file path
        data: Dictionary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
