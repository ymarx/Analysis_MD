#!/usr/bin/env python3
"""
인터랙티브 레이블링 전체 워크플로우

1. NPY → BMP 변환 (시각화용)
2. BMP에서 인터랙티브 바운딩 박스 그리기
3. BMP 좌표 → NPY 좌표 변환
4. NPY에 매핑된 결과 시각화 및 검증

사용법:
    python scripts/interactive_labeling_workflow.py <npy_path>
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import json
from mine_labeling.visualization import NpyToBmpConverter
from mine_labeling.visualization.interactive_labeling import InteractiveBBoxLabeler


def step1_convert_npy_to_bmp(npy_path: Path, output_dir: Path):
    """1단계: NPY → BMP 변환"""
    print("\\n" + "="*60)
    print("1단계: NPY → BMP 변환 (시각화용)")
    print("="*60)

    # Load NPY
    npy_data = np.load(npy_path)
    print(f"NPY 데이터 로드: {npy_data.shape}")

    # Convert to BMP
    converter = NpyToBmpConverter(target_width=1024, apply_clahe=True)

    bmp_path = output_dir / f'{npy_path.stem}_visualization.bmp'
    bmp_image = converter.convert_to_bmp(npy_data, bmp_path)

    print(f"\\n✓ BMP 변환 완료")
    print(f"  원본 NPY: {npy_data.shape}")
    print(f"  변환 BMP: {bmp_image.shape}")
    print(f"  스케일: {converter.get_scale_factor(npy_data.shape[1]):.2f}x")

    return bmp_path, npy_data.shape[1]


def step2_interactive_labeling(bmp_path: Path, output_dir: Path):
    """2단계: 인터랙티브 바운딩 박스 그리기"""
    print("\\n" + "="*60)
    print("2단계: 인터랙티브 바운딩 박스 레이블링")
    print("="*60)
    print(f"BMP 파일: {bmp_path.name}")
    print("\\n인터랙티브 창이 열립니다...")
    print("바운딩 박스를 그리고 's' 키를 눌러 저장하세요.\\n")

    labeler = InteractiveBBoxLabeler(str(bmp_path), str(output_dir))
    labeler.show()

    # Load saved annotations
    json_path = output_dir / f'{bmp_path.stem}.json'

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        bmp_annotations = data['annotations']
        print(f"\\n✓ {len(bmp_annotations)}개 바운딩 박스 저장됨")
        return bmp_annotations
    else:
        print("\\n⚠️  저장된 주석이 없습니다.")
        return []


def step3_convert_coordinates(bmp_annotations, npy_width: int, output_dir: Path):
    """3단계: BMP 좌표 → NPY 좌표 변환"""
    print("\\n" + "="*60)
    print("3단계: 좌표 변환 (BMP → NPY)")
    print("="*60)

    converter = NpyToBmpConverter(target_width=1024)

    mapped_annotations = []
    for i, bmp_bbox in enumerate(bmp_annotations):
        # Convert to NPY coordinates
        npy_bbox = converter.bmp_to_npy_coordinates(bmp_bbox, npy_width)

        mapped = {
            'name': 'mine',
            'bmp_coords': bmp_bbox,
            'npy_coords': npy_bbox
        }

        mapped_annotations.append(mapped)

        print(f"  Mine {i+1}:")
        print(f"    BMP: ({bmp_bbox['xmin']}, {bmp_bbox['ymin']}) → ({bmp_bbox['xmax']}, {bmp_bbox['ymax']})")
        print(f"    NPY: ({npy_bbox['xmin']}, {npy_bbox['ymin']}) → ({npy_bbox['xmax']}, {npy_bbox['ymax']})")

    # Save mapped annotations
    output_path = output_dir / 'mapped_annotations.json'
    with open(output_path, 'w') as f:
        json.dump(mapped_annotations, f, indent=2)

    print(f"\\n✓ 매핑된 주석 저장: {output_path.name}")

    return mapped_annotations


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/interactive_labeling_workflow.py <npy_path>")
        print("\\n예제:")
        print("  python scripts/interactive_labeling_workflow.py data/intensity.npy")
        sys.exit(1)

    npy_path = Path(sys.argv[1])

    if not npy_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {npy_path}")
        sys.exit(1)

    # Output directory
    output_dir = project_root / 'data/interactive_labeling'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("인터랙티브 레이블링 워크플로우")
    print("="*60)
    print(f"입력 NPY: {npy_path}")
    print(f"출력 폴더: {output_dir}")

    # Step 1: Convert NPY to BMP
    bmp_path, npy_width = step1_convert_npy_to_bmp(npy_path, output_dir)

    # Step 2: Interactive labeling
    bmp_annotations = step2_interactive_labeling(bmp_path, output_dir)

    if not bmp_annotations:
        print("\\n레이블링이 완료되지 않았습니다.")
        return

    # Step 3: Convert coordinates
    mapped_annotations = step3_convert_coordinates(bmp_annotations, npy_width, output_dir)

    print("\\n" + "="*60)
    print("✓ 워크플로우 완료!")
    print("="*60)
    print(f"결과 파일:")
    print(f"  - BMP: {bmp_path.name}")
    print(f"  - 주석: {bmp_path.stem}.xml, {bmp_path.stem}.json")
    print(f"  - 매핑: mapped_annotations.json")
    print(f"\\n총 {len(mapped_annotations)}개 기뢰 레이블링 완료")


if __name__ == '__main__':
    main()
