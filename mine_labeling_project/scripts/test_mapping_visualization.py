#!/usr/bin/env python3
"""
매핑 검증 시각화

BMP 좌표 → NPY 좌표 매핑이 올바른지 시각적으로 검증
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from mine_labeling.utils import load_npz_data
from mine_labeling.visualization import NpyToBmpConverter


def create_comparison_visualization(npy_data, bmp_image, annotations, output_path):
    """BMP와 NPY 비교 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 1. BMP 이미지 (1024폭)
    axes[0].imshow(bmp_image)
    axes[0].set_title(f'BMP 시각화 ({bmp_image.shape[1]}×{bmp_image.shape[0]})\n'
                     f'{len(annotations)}개 기뢰 (BMP 좌표)', fontsize=14)
    axes[0].set_xlabel('X (BMP 좌표)', fontsize=12)
    axes[0].set_ylabel('Y (ping)', fontsize=12)

    # BMP 바운딩 박스 그리기
    for i, ann in enumerate(annotations):
        bbox = ann['bmp_coords']
        rect = patches.Rectangle(
            (bbox['xmin'], bbox['ymin']),
            bbox['width'], bbox['height'],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0].add_patch(rect)

        # 라벨 추가
        axes[0].text(bbox['xmin'], bbox['ymin'] - 5,
                    f"M{i+1}", color='red', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 2. NPY 강도 데이터 (원본 해상도)
    axes[1].imshow(npy_data, cmap='gray', aspect='auto')
    axes[1].set_title(f'NPY 강도 데이터 ({npy_data.shape[1]}×{npy_data.shape[0]})\n'
                     f'{len(annotations)}개 기뢰 (NPY 좌표)', fontsize=14)
    axes[1].set_xlabel('X (NPY 좌표)', fontsize=12)
    axes[1].set_ylabel('Y (ping)', fontsize=12)

    # NPY 바운딩 박스 그리기
    for i, ann in enumerate(annotations):
        bbox = ann['npy_coords']
        rect = patches.Rectangle(
            (bbox['xmin'], bbox['ymin']),
            bbox['width'], bbox['height'],
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        axes[1].add_patch(rect)

        # 라벨 추가
        axes[1].text(bbox['xmin'], bbox['ymin'] - 5,
                    f"M{i+1}", color='yellow', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 비교 시각화 저장: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("매핑 검증 시각화 테스트")
    print("="*60)

    # 검증된 데이터 사용
    npz_path = project_root / 'verified_data/flipped_20251104/flipped_labeled_intensity_data.npz'

    print(f"\n1. 검증된 데이터 로드: {npz_path.name}")
    data = load_npz_data(npz_path)

    intensity = data['intensity']
    metadata = data['metadata']

    print(f"   - Intensity: {intensity.shape}")
    print(f"   - 기뢰 개수: {len(metadata)}")

    # BMP 변환
    print(f"\n2. NPY → BMP 변환")
    converter = NpyToBmpConverter(target_width=1024, apply_clahe=True)

    output_dir = project_root / 'data/test_visualization'
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_path = output_dir / 'verified_visualization.bmp'
    bmp_image = converter.convert_to_bmp(intensity, bmp_path)

    print(f"   - BMP: {bmp_image.shape}")
    print(f"   - 스케일: {converter.get_scale_factor(intensity.shape[1]):.2f}x")

    # 주석 변환 (NPY → BMP)
    print(f"\n3. 좌표 변환 (NPY → BMP)")
    annotations = []
    for i, ann in enumerate(metadata[:5]):  # 처음 5개만
        npy_bbox = ann['mapped_npy']
        bmp_bbox = converter.npy_to_bmp_coordinates(npy_bbox, intensity.shape[1])

        annotations.append({
            'name': ann['name'],
            'bmp_coords': bmp_bbox,
            'npy_coords': npy_bbox
        })

        print(f"   Mine {i+1}:")
        print(f"     NPY: ({npy_bbox['xmin']}, {npy_bbox['ymin']}) → ({npy_bbox['xmax']}, {npy_bbox['ymax']})")
        print(f"     BMP: ({bmp_bbox['xmin']}, {bmp_bbox['ymin']}) → ({bmp_bbox['xmax']}, {bmp_bbox['ymax']})")

    # 비교 시각화
    print(f"\n4. 비교 시각화 생성")
    viz_path = output_dir / 'mapping_comparison.png'
    create_comparison_visualization(intensity, bmp_image, annotations, viz_path)

    print("\n" + "="*60)
    print("✓ 테스트 완료!")
    print("="*60)
    print(f"\n결과 파일:")
    print(f"  - BMP: {bmp_path}")
    print(f"  - 비교 시각화: {viz_path}")
    print(f"\n매핑이 정확한지 시각적으로 확인하세요.")


if __name__ == '__main__':
    main()
