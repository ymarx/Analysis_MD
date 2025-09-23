#!/usr/bin/env python3
"""
분산된 분석 결과들 통합 정리

목적: 여러 곳에 흩어진 분석 결과들을 체계적으로 정리하고 통합
"""

import os
import shutil
from pathlib import Path
import json

def organize_scattered_results():
    """분산된 분석 결과들 정리"""

    print("="*70)
    print("분산된 분석 결과 통합 정리")
    print("="*70)

    # 분석 결과가 있을 수 있는 위치들
    result_locations = [
        'outputs',
        'logs',
        'config',
        'scripts',
        'archive/analysis_results_backup'
    ]

    print(f"\n🔍 분석 결과 탐지:")

    found_results = {}

    for location in result_locations:
        if os.path.exists(location):
            print(f"\n   📁 {location}/:")

            location_results = []
            for root, dirs, files in os.walk(location):
                for file in files:
                    if any(ext in file for ext in ['.png', '.jpg', '.md', '.json', '.csv', '.txt', '.npy']):
                        rel_path = os.path.relpath(os.path.join(root, file), location)
                        location_results.append(rel_path)
                        print(f"      📄 {rel_path}")

            found_results[location] = location_results

    return found_results

def categorize_results(found_results):
    """결과 파일들을 카테고리별로 분류"""

    print(f"\n📊 결과 파일 분류:")

    categories = {
        'coordinate_analysis': [],
        'terrain_analysis': [],
        'ship_movement': [],
        'data_validation': [],
        'reports': [],
        'visualizations': [],
        'raw_data': []
    }

    for location, files in found_results.items():
        for file in files:
            file_lower = file.lower()

            # 카테고리 분류
            if any(keyword in file_lower for keyword in ['coordinate', 'location', 'position', 'gps']):
                categories['coordinate_analysis'].append((location, file))
            elif any(keyword in file_lower for keyword in ['terrain', 'similarity', 'image', 'bmp']):
                categories['terrain_analysis'].append((location, file))
            elif any(keyword in file_lower for keyword in ['ship', 'movement', 'direction', 'path']):
                categories['ship_movement'].append((location, file))
            elif any(keyword in file_lower for keyword in ['validation', 'verification', 'check', 'test']):
                categories['data_validation'].append((location, file))
            elif file_lower.endswith(('.md', '.txt', '.pdf')):
                categories['reports'].append((location, file))
            elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                categories['visualizations'].append((location, file))
            elif file_lower.endswith(('.npy', '.csv', '.json', '.pkl')):
                categories['raw_data'].append((location, file))

    # 분류 결과 출력
    for category, items in categories.items():
        if items:
            print(f"\n   📂 {category}: {len(items)}개")
            for location, file in items[:3]:  # 처음 3개만 출력
                print(f"      - {location}/{file}")
            if len(items) > 3:
                print(f"      ... 및 {len(items)-3}개 더")

    return categories

def move_results_to_organized_structure(categories):
    """분류된 결과들을 정리된 구조로 이동"""

    print(f"\n📦 결과 파일 이동:")

    moved_count = 0

    for category, items in categories.items():
        if not items:
            continue

        # 대상 디렉토리 생성
        target_dir = f"analysis_results/{category}"
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n   📂 {category}:")

        for location, file in items:
            src_path = os.path.join(location, file)

            if os.path.exists(src_path):
                # 파일명 중복 방지
                base_name = os.path.basename(file)
                counter = 1
                dest_path = os.path.join(target_dir, base_name)

                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_name)
                    dest_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                    counter += 1

                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"      ✅ {src_path} → {dest_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"      ❌ 이동 실패: {src_path} - {e}")

    print(f"\n   총 이동된 파일: {moved_count}개")

def create_result_index():
    """정리된 결과들의 인덱스 생성"""

    print(f"\n📄 결과 인덱스 생성:")

    index_data = {}

    # analysis_results 디렉토리 내 파일들 인덱싱
    if os.path.exists('analysis_results'):
        for category in os.listdir('analysis_results'):
            category_path = os.path.join('analysis_results', category)
            if os.path.isdir(category_path):
                files = []
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if os.path.isfile(file_path):
                        file_info = {
                            'name': file,
                            'path': file_path,
                            'size': os.path.getsize(file_path),
                            'modified': os.path.getmtime(file_path)
                        }
                        files.append(file_info)

                index_data[category] = {
                    'count': len(files),
                    'files': files
                }

    # JSON 인덱스 파일 생성
    with open('analysis_results/RESULTS_INDEX.json', 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # 마크다운 인덱스 생성
    markdown_content = """# 분석 결과 인덱스

생성일시: {}

## 📊 카테고리별 결과 요약

""".format(os.popen('date').read().strip())

    for category, info in index_data.items():
        markdown_content += f"### {category.replace('_', ' ').title()}\n\n"
        markdown_content += f"- **파일 수**: {info['count']}개\n"
        markdown_content += f"- **파일 목록**:\n"

        for file_info in info['files']:
            size_mb = file_info['size'] / (1024 * 1024)
            markdown_content += f"  - `{file_info['name']}` ({size_mb:.2f}MB)\n"

        markdown_content += "\n"

    with open('analysis_results/RESULTS_INDEX.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"   ✅ analysis_results/RESULTS_INDEX.json")
    print(f"   ✅ analysis_results/RESULTS_INDEX.md")

def identify_duplicate_analysis():
    """중복된 분석 결과 식별"""

    print(f"\n🔍 중복 분석 결과 식별:")

    # 유사한 이름의 파일들 찾기
    all_files = []

    if os.path.exists('analysis_results'):
        for root, dirs, files in os.walk('analysis_results'):
            for file in files:
                all_files.append(file)

    # 패턴별 그룹핑
    patterns = {}
    for file in all_files:
        # 숫자나 날짜 제거하여 기본 패턴 추출
        import re
        base_pattern = re.sub(r'[0-9_-]+', '', file.lower())
        base_pattern = re.sub(r'\.(png|jpg|md|json|csv|txt|npy)$', '', base_pattern)

        if base_pattern not in patterns:
            patterns[base_pattern] = []
        patterns[base_pattern].append(file)

    # 중복 가능성이 있는 패턴들 출력
    duplicates_found = False
    for pattern, files in patterns.items():
        if len(files) > 1:
            duplicates_found = True
            print(f"   🔄 유사 패턴 '{pattern}': {len(files)}개")
            for file in files:
                print(f"      - {file}")

    if not duplicates_found:
        print(f"   ✅ 중복 패턴 없음")

def check_module_status():
    """핵심 모듈들의 현재 상태 확인"""

    print(f"\n🔧 핵심 모듈 상태 확인:")

    # 중요한 모듈들
    critical_modules = [
        ('src/data_processing/xtf_reader.py', 'XTF 파일 읽기'),
        ('src/data_processing/xtf_intensity_extractor.py', '강도 데이터 추출'),
        ('src/data_processing/preprocessor.py', '전처리'),
        ('pipeline/modules/xtf_reader.py', '파이프라인 XTF 리더'),
        ('pipeline/modules/xtf_extractor.py', '파이프라인 추출기'),
        ('pipeline/unified_pipeline.py', '통합 파이프라인')
    ]

    working_modules = []
    broken_modules = []

    for module_path, description in critical_modules:
        if os.path.exists(module_path):
            try:
                # 간단한 import 테스트
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 구문 검사
                compile(content, module_path, 'exec')

                # 기본적인 클래스/함수 존재 확인
                has_class = 'class ' in content
                has_function = 'def ' in content

                status = "✅ 정상"
                if has_class or has_function:
                    working_modules.append((module_path, description))
                else:
                    status = "⚠️ 빈 파일"

                print(f"   {status} {module_path} - {description}")

            except Exception as e:
                broken_modules.append((module_path, description, str(e)))
                print(f"   ❌ 오류 {module_path} - {description}: {e}")
        else:
            broken_modules.append((module_path, description, "파일 없음"))
            print(f"   ❌ 없음 {module_path} - {description}")

    print(f"\n   📊 모듈 상태:")
    print(f"      ✅ 정상: {len(working_modules)}개")
    print(f"      ❌ 문제: {len(broken_modules)}개")

    return working_modules, broken_modules

def main():
    """메인 실행 함수"""

    print("📊 분산된 분석 결과 통합 정리 시작")

    # 1. 분산된 결과 탐지
    found_results = organize_scattered_results()

    # 2. 결과 분류
    categories = categorize_results(found_results)

    # 3. 정리된 구조로 이동
    move_results_to_organized_structure(categories)

    # 4. 결과 인덱스 생성
    create_result_index()

    # 5. 중복 분석 식별
    identify_duplicate_analysis()

    # 6. 모듈 상태 확인
    working_modules, broken_modules = check_module_status()

    print(f"\n{'='*70}")
    print("✅ 분석 결과 통합 정리 완료")
    print(f"{'='*70}")

    print(f"\n📋 정리 결과:")
    print(f"   📂 정리된 카테고리: {len([c for c in categories.values() if c])}개")
    print(f"   📄 총 파일: {sum(len(files) for files in categories.values())}개")
    print(f"   ✅ 정상 모듈: {len(working_modules)}개")
    print(f"   ❌ 문제 모듈: {len(broken_modules)}개")

    if broken_modules:
        print(f"\n⚠️ 수정 필요한 모듈들:")
        for path, desc, error in broken_modules:
            print(f"   - {path}: {error}")

if __name__ == "__main__":
    main()