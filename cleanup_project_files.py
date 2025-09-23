#!/usr/bin/env python3
"""
프로젝트 파일 정리 스크립트

목적: 임시 검증 스크립트, 테스트 결과, 분산된 처리 결과들을 체계적으로 정리
"""

import os
import shutil
from pathlib import Path
import re
from datetime import datetime

def analyze_project_structure():
    """현재 프로젝트 구조 분석"""

    print("="*70)
    print("프로젝트 파일 구조 분석")
    print("="*70)

    # 루트 레벨 Python 파일들 분석
    root_py_files = []
    for file in os.listdir('.'):
        if file.endswith('.py'):
            root_py_files.append(file)

    print(f"\n📁 루트 레벨 Python 파일: {len(root_py_files)}개")

    # 임시/테스트 파일 분류
    temporary_patterns = [
        'test_', 'verify_', 'investigate_', 'analyze_', 'compare_',
        'fix_', 'check_', 'simple_', 'direct_', 'correct_',
        'recalculate_', 'parse_', 'final_'
    ]

    temporary_files = []
    core_files = []

    for file in root_py_files:
        is_temp = any(file.startswith(pattern) for pattern in temporary_patterns)
        if is_temp:
            temporary_files.append(file)
        else:
            core_files.append(file)

    print(f"\n🧹 임시/검증 파일: {len(temporary_files)}개")
    for file in sorted(temporary_files):
        print(f"   - {file}")

    print(f"\n⚙️ 핵심 모듈 파일: {len(core_files)}개")
    for file in sorted(core_files):
        print(f"   - {file}")

    # 디렉토리 구조 분석
    print(f"\n📂 디렉토리 구조:")
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item) and not item.startswith('.'):
            print(f"   📁 {item}/")

            # 하위 파일 개수
            try:
                sub_items = os.listdir(item)
                py_count = len([f for f in sub_items if f.endswith('.py')])
                total_count = len(sub_items)
                print(f"      - 총 {total_count}개 파일 (Python: {py_count}개)")
            except:
                print(f"      - 접근 불가")

    return temporary_files, core_files

def create_cleanup_directories():
    """정리용 디렉토리 생성"""

    print(f"\n📁 정리용 디렉토리 생성:")

    directories = {
        'archive': 'archive',
        'temp_scripts': 'archive/temporary_scripts',
        'test_results': 'archive/test_results',
        'analysis_backup': 'archive/analysis_results_backup',
        'deprecated': 'archive/deprecated_modules'
    }

    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        print(f"   ✅ {path}")

    return directories

def move_temporary_files(temporary_files, directories):
    """임시 파일들을 archive로 이동"""

    print(f"\n🧹 임시 파일 정리:")

    moved_count = 0

    for file in temporary_files:
        if os.path.exists(file):
            dest_path = os.path.join(directories['temp_scripts'], file)
            shutil.move(file, dest_path)
            print(f"   📦 {file} → archive/temporary_scripts/")
            moved_count += 1

    print(f"\n   이동 완료: {moved_count}개 파일")

def organize_analysis_results(directories):
    """analysis_results 정리"""

    print(f"\n📊 분석 결과 정리:")

    if not os.path.exists('analysis_results'):
        print("   analysis_results 디렉토리 없음")
        return

    # 기존 analysis_results 백업
    backup_path = directories['analysis_backup']
    if os.path.exists('analysis_results'):
        for item in os.listdir('analysis_results'):
            src = os.path.join('analysis_results', item)
            dst = os.path.join(backup_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"   📦 analysis_results → {backup_path}")

    # 새로운 analysis_results 구조 생성
    new_structure = {
        'coordinate_analysis': 'analysis_results/coordinate_analysis',
        'terrain_analysis': 'analysis_results/terrain_analysis',
        'ship_movement': 'analysis_results/ship_movement',
        'data_validation': 'analysis_results/data_validation',
        'reports': 'analysis_results/reports'
    }

    for name, path in new_structure.items():
        os.makedirs(path, exist_ok=True)
        print(f"   ✅ {path}")

def identify_core_modules():
    """핵심 모듈 현황 파악"""

    print(f"\n⚙️ 핵심 모듈 현황:")

    # src 디렉토리 확인
    if os.path.exists('src'):
        print(f"   📁 src/ 디렉토리:")
        for root, dirs, files in os.walk('src'):
            level = root.replace('src', '').count(os.sep)
            indent = '   ' + '  ' * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = '   ' + '  ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    print(f"{subindent}📄 {file}")

    # pipeline 디렉토리 확인
    if os.path.exists('pipeline'):
        print(f"\n   📁 pipeline/ 디렉토리:")
        for root, dirs, files in os.walk('pipeline'):
            level = root.replace('pipeline', '').count(os.sep)
            indent = '   ' + '  ' * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = '   ' + '  ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    print(f"{subindent}📄 {file}")

def check_module_functionality():
    """모듈 작동 가능성 체크"""

    print(f"\n🔧 모듈 작동성 검사:")

    critical_modules = [
        'src/data_processing/xtf_reader.py',
        'pipeline/modules/xtf_reader.py',
        'pipeline/modules/xtf_extractor.py'
    ]

    for module_path in critical_modules:
        if os.path.exists(module_path):
            print(f"   ✅ {module_path} - 존재")

            # 간단한 구문 검사
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    compile(content, module_path, 'exec')
                print(f"      ✅ 구문 오류 없음")
            except SyntaxError as e:
                print(f"      ❌ 구문 오류: {e}")
            except Exception as e:
                print(f"      ⚠️ 검사 실패: {e}")
        else:
            print(f"   ❌ {module_path} - 없음")

def create_cleanup_report(temporary_files, core_files):
    """정리 보고서 생성"""

    print(f"\n📄 정리 보고서 생성:")

    report_content = f"""# 프로젝트 파일 정리 보고서

**정리 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 정리 요약

### 이동된 임시 파일들 ({len(temporary_files)}개)
"""

    for file in sorted(temporary_files):
        report_content += f"- `{file}` → `archive/temporary_scripts/`\n"

    report_content += f"""

### 보존된 핵심 모듈들 ({len(core_files)}개)
"""

    for file in sorted(core_files):
        report_content += f"- `{file}`\n"

    report_content += """

## 📁 새로운 디렉토리 구조

```
├── archive/                          # 정리된 파일들
│   ├── temporary_scripts/            # 임시 검증 스크립트들
│   ├── test_results/                 # 테스트 결과들
│   ├── analysis_results_backup/      # 기존 분석 결과 백업
│   └── deprecated_modules/           # 사용하지 않는 모듈들
├── analysis_results/                 # 정리된 분석 결과
│   ├── coordinate_analysis/          # 좌표 분석 결과
│   ├── terrain_analysis/            # 지형 분석 결과
│   ├── ship_movement/               # 선박 이동 분석
│   ├── data_validation/             # 데이터 검증 결과
│   └── reports/                     # 종합 보고서들
├── src/                             # 핵심 소스 코드
├── pipeline/                        # 처리 파이프라인
└── datasets/                        # 데이터셋
```

## 🔧 다음 단계

1. **모듈 통합**: src와 pipeline의 중복 모듈들 정리
2. **의존성 정리**: 각 모듈의 import 관계 정리
3. **테스트 추가**: 핵심 모듈들의 단위 테스트 작성
4. **문서화**: 정리된 모듈들의 API 문서 작성

## ⚠️ 주의사항

정리된 파일들은 `archive/` 디렉토리에 보관되어 있으며, 필요시 복원 가능합니다.
"""

    with open('PROJECT_CLEANUP_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"   📄 PROJECT_CLEANUP_REPORT.md 생성")

def main():
    """메인 실행 함수"""

    print("🧹 프로젝트 파일 정리 시작")

    # 1. 현재 구조 분석
    temporary_files, core_files = analyze_project_structure()

    # 2. 정리용 디렉토리 생성
    directories = create_cleanup_directories()

    # 3. 임시 파일들 이동
    move_temporary_files(temporary_files, directories)

    # 4. analysis_results 정리
    organize_analysis_results(directories)

    # 5. 핵심 모듈 현황 파악
    identify_core_modules()

    # 6. 모듈 작동성 검사
    check_module_functionality()

    # 7. 정리 보고서 생성
    create_cleanup_report(temporary_files, core_files)

    print(f"\n{'='*70}")
    print("✅ 프로젝트 파일 정리 완료")
    print(f"{'='*70}")

    print(f"\n📋 정리 결과:")
    print(f"   📦 임시 파일 이동: {len(temporary_files)}개")
    print(f"   ⚙️ 핵심 모듈 보존: {len(core_files)}개")
    print(f"   📁 정리된 디렉토리 구조")
    print(f"   📄 정리 보고서: PROJECT_CLEANUP_REPORT.md")

    print(f"\n🔧 다음 단계:")
    print(f"   1. 모듈 중복성 제거 및 통합")
    print(f"   2. 의존성 관계 정리")
    print(f"   3. 작동 가능한 최신 형태로 업데이트")

if __name__ == "__main__":
    main()