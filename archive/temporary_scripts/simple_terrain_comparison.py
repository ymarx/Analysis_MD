#!/usr/bin/env python3
"""
간단한 BMP와 PH_annotation 이미지 비교 스크립트 (빠른 분석용)

목적: 이미지 크기와 기본적인 시각적 특성 비교로 빠른 판단
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_image_comparison():
    """빠른 이미지 비교"""

    print("="*60)
    print("BMP vs PH_annotation 빠른 비교 분석")
    print("="*60)

    # 파일 경로들
    bmp_files = [
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_IMG_00.BMP",
        "datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/original/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04_IMG_00.BMP",
        "datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012181900_001_04_IMG_00.BMP"
    ]

    annotation_files = [
        "datasets/PH_annotation.bmp",
        "datasets/PH_annotation.png"
    ]

    results = []

    for bmp_path in bmp_files:
        if not os.path.exists(bmp_path):
            continue

        print(f"\n📊 분석: {os.path.basename(bmp_path)}")

        try:
            # BMP 이미지 기본 정보
            bmp_img = cv2.imread(bmp_path)
            if bmp_img is None:
                print(f"   ❌ 이미지 로드 실패")
                continue

            bmp_h, bmp_w = bmp_img.shape[:2]
            bmp_gray = cv2.cvtColor(bmp_img, cv2.COLOR_BGR2GRAY)
            bmp_mean = np.mean(bmp_gray)
            bmp_std = np.std(bmp_gray)

            print(f"   📐 크기: {bmp_w} x {bmp_h}")
            print(f"   🔆 평균 밝기: {bmp_mean:.1f}")
            print(f"   📊 밝기 편차: {bmp_std:.1f}")

            # Annotation 이미지들과 비교
            for ann_path in annotation_files:
                if not os.path.exists(ann_path):
                    continue

                print(f"\n   vs {os.path.basename(ann_path)}")

                ann_img = cv2.imread(ann_path)
                if ann_img is None:
                    print(f"      ❌ Annotation 로드 실패")
                    continue

                ann_h, ann_w = ann_img.shape[:2]
                ann_gray = cv2.cvtColor(ann_img, cv2.COLOR_BGR2GRAY)
                ann_mean = np.mean(ann_gray)
                ann_std = np.std(ann_gray)

                print(f"      📐 Annotation 크기: {ann_w} x {ann_h}")
                print(f"      🔆 Annotation 평균 밝기: {ann_mean:.1f}")

                # 크기 비율 비교
                width_ratio = min(bmp_w, ann_w) / max(bmp_w, ann_w)
                height_ratio = min(bmp_h, ann_h) / max(bmp_h, ann_h)

                # 밝기 유사도
                brightness_diff = abs(bmp_mean - ann_mean) / 255.0
                brightness_similarity = 1.0 - brightness_diff

                print(f"      📏 크기 유사도: W={width_ratio:.3f}, H={height_ratio:.3f}")
                print(f"      💡 밝기 유사도: {brightness_similarity:.3f}")

                # 간단한 템플릿 매칭 (축소 이미지로)
                try:
                    # 이미지를 작은 크기로 축소하여 빠른 비교
                    target_size = (256, 256)
                    bmp_small = cv2.resize(bmp_gray, target_size)
                    ann_small = cv2.resize(ann_gray, target_size)

                    # 정규화 상호 상관
                    correlation = cv2.matchTemplate(bmp_small, ann_small, cv2.TM_CCOEFF_NORMED)[0, 0]
                    print(f"      🎯 상관계수: {correlation:.3f}")

                    # 종합 유사도 (간단한 가중 평균)
                    overall = (width_ratio * 0.2 + height_ratio * 0.2 +
                             brightness_similarity * 0.3 + (correlation + 1) / 2 * 0.3)
                    print(f"      ⭐ 종합 유사도: {overall:.3f}")

                    if overall > 0.7:
                        similarity_level = "높음 🟢"
                    elif overall > 0.5:
                        similarity_level = "보통 🟡"
                    else:
                        similarity_level = "낮음 🔴"

                    print(f"      📈 유사도 평가: {similarity_level}")

                    results.append({
                        'bmp_file': os.path.basename(bmp_path),
                        'annotation_file': os.path.basename(ann_path),
                        'bmp_size': (bmp_w, bmp_h),
                        'ann_size': (ann_w, ann_h),
                        'size_similarity': (width_ratio + height_ratio) / 2,
                        'brightness_similarity': brightness_similarity,
                        'correlation': correlation,
                        'overall_similarity': overall,
                        'assessment': similarity_level
                    })

                except Exception as e:
                    print(f"      ❌ 상관계수 계산 실패: {e}")

        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")

    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 종합 분석 결과")
    print(f"{'='*60}")

    if results:
        # 최고 유사도
        best_result = max(results, key=lambda x: x['overall_similarity'])
        print(f"\n🏆 최고 유사도:")
        print(f"   파일: {best_result['bmp_file']} vs {best_result['annotation_file']}")
        print(f"   유사도: {best_result['overall_similarity']:.3f}")
        print(f"   평가: {best_result['assessment']}")

        # 평균 유사도
        avg_similarity = np.mean([r['overall_similarity'] for r in results])
        print(f"\n📈 평균 유사도: {avg_similarity:.3f}")

        # 유사도 분포
        high_count = len([r for r in results if r['overall_similarity'] > 0.7])
        medium_count = len([r for r in results if 0.5 <= r['overall_similarity'] <= 0.7])
        low_count = len([r for r in results if r['overall_similarity'] < 0.5])

        print(f"\n📊 유사도 분포:")
        print(f"   높음 (>0.7): {high_count}개")
        print(f"   보통 (0.5-0.7): {medium_count}개")
        print(f"   낮음 (<0.5): {low_count}개")

        # 결론
        print(f"\n💡 결론:")
        if best_result['overall_similarity'] > 0.7:
            print("   ✅ 일부 이미지에서 높은 유사도 발견")
            print("   ✅ 동일하거나 유사한 지형일 가능성 높음")
            print("   ✅ 좌표 차이에도 불구하고 지형적 연관성 존재")
        elif best_result['overall_similarity'] > 0.5:
            print("   ⚠️ 중간 정도의 유사도")
            print("   ⚠️ 부분적으로 유사한 특징 존재")
            print("   ⚠️ 인접 지역이거나 유사한 환경일 가능성")
        else:
            print("   ❌ 낮은 유사도")
            print("   ❌ 서로 다른 지형일 가능성 높음")
            print("   ❌ 좌표 차이와 지형 차이가 일치")

    else:
        print("❌ 비교 가능한 결과가 없습니다.")

    # 간단한 보고서 저장
    output_dir = Path("analysis_results/quick_terrain_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "QUICK_TERRAIN_COMPARISON_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""# 빠른 지형 비교 분석 보고서
**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석자**: YMARX

## 분석 결과

""")
        if results:
            best = best_result
            f.write(f"""### 최고 유사도
- **파일**: {best['bmp_file']} vs {best['annotation_file']}
- **유사도**: {best['overall_similarity']:.3f}
- **평가**: {best['assessment']}

### 전체 통계
- **평균 유사도**: {avg_similarity:.3f}
- **높은 유사도**: {high_count}개
- **중간 유사도**: {medium_count}개
- **낮은 유사도**: {low_count}개

### 결론
""")
            if best['overall_similarity'] > 0.7:
                f.write("**높은 유사도**: 동일하거나 유사한 지형일 가능성이 높습니다.\n")
            elif best['overall_similarity'] > 0.5:
                f.write("**중간 유사도**: 부분적으로 유사한 특징이 존재합니다.\n")
            else:
                f.write("**낮은 유사도**: 서로 다른 지형일 가능성이 높습니다.\n")

    print(f"\n📁 보고서 저장: {report_file}")

if __name__ == "__main__":
    quick_image_comparison()