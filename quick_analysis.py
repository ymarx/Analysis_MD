#!/usr/bin/env python3
"""
간단한 샘플 데이터 분석 스크립트

패키지 의존성 문제를 최소화하여 기본적인 분석을 수행합니다.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def analyze_project_structure():
    """프로젝트 구조 분석"""
    print("🔍 프로젝트 구조 분석")
    print("=" * 50)
    
    # 샘플 데이터 확인
    sample_dir = project_root / '[샘플]데이터'
    if sample_dir.exists():
        print(f"✅ 샘플 데이터 디렉토리 존재: {sample_dir}")
        sample_files = list(sample_dir.glob('*'))
        for file_path in sample_files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   - {file_path.name}: {size_mb:.1f} MB")
    else:
        print("❌ 샘플 데이터 디렉토리 없음")
    
    # 데이터셋 확인
    dataset_dir = project_root / 'datasets'
    if dataset_dir.exists():
        print(f"\\n✅ 데이터셋 디렉토리 존재: {dataset_dir}")
        dataset_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"   - 데이터셋 수: {len(dataset_folders)}")
        for dataset in dataset_folders[:3]:  # 처음 3개만 표시
            print(f"   - {dataset.name}")
    else:
        print("\\n❌ 데이터셋 디렉토리 없음")
    
    # 소스 코드 확인
    src_dir = project_root / 'src'
    if src_dir.exists():
        print(f"\\n✅ 소스 코드 디렉토리 존재: {src_dir}")
        py_files = list(src_dir.rglob('*.py'))
        print(f"   - Python 파일 수: {len(py_files)}")
        
        # 주요 모듈 확인
        key_modules = [
            'src/data_processing/xtf_reader.py',
            'src/data_processing/coordinate_mapper.py', 
            'src/data_processing/preprocessor.py'
        ]
        
        for module in key_modules:
            module_path = project_root / module
            if module_path.exists():
                print(f"   ✅ {module}")
            else:
                print(f"   ❌ {module}")
    else:
        print("\\n❌ 소스 코드 디렉토리 없음")

def check_data_requirements():
    """데이터 요구사항 체크"""
    print("\\n🔎 데이터 요구사항 체크")
    print("=" * 50)
    
    requirements = {
        'XTF 파일': {
            'pattern': '*.xtf',
            'min_size_mb': 10,
            'description': '사이드스캔 소나 데이터'
        },
        'BMP 이미지': {
            'pattern': '*.bmp', 
            'min_size_mb': 1,
            'description': '어노테이션된 이미지'
        },
        'Excel 위치 파일': {
            'pattern': '*.xlsx',
            'min_size_mb': 0.01,
            'description': '기물 위치 좌표'
        }
    }
    
    sample_dir = project_root / '[샘플]데이터'
    if not sample_dir.exists():
        print("❌ 샘플 데이터 디렉토리가 없습니다.")
        return False
    
    all_found = True
    
    for req_name, req_info in requirements.items():
        files = list(sample_dir.glob(req_info['pattern']))
        
        if files:
            largest_file = max(files, key=lambda f: f.stat().st_size)
            size_mb = largest_file.stat().st_size / (1024 * 1024)
            
            if size_mb >= req_info['min_size_mb']:
                print(f"✅ {req_name}: {largest_file.name} ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  {req_name}: {largest_file.name} ({size_mb:.1f} MB) - 크기가 작을 수 있음")
        else:
            print(f"❌ {req_name}: 파일을 찾을 수 없음 ({req_info['pattern']})")
            all_found = False
    
    return all_found

def estimate_system_readiness():
    """시스템 준비도 평가"""
    print("\\n📊 시스템 준비도 평가")
    print("=" * 50)
    
    readiness_factors = {
        'project_structure': 0,
        'sample_data': 0,
        'source_code': 0,
        'documentation': 0
    }
    
    # 프로젝트 구조 점수
    required_dirs = ['src', 'config', 'notebooks', 'docs']
    existing_dirs = sum(1 for d in required_dirs if (project_root / d).exists())
    readiness_factors['project_structure'] = (existing_dirs / len(required_dirs)) * 100
    
    # 샘플 데이터 점수
    sample_dir = project_root / '[샘플]데이터'
    if sample_dir.exists():
        required_patterns = ['*.xtf', '*.bmp', '*.xlsx']
        existing_patterns = sum(1 for pattern in required_patterns 
                              if list(sample_dir.glob(pattern)))
        readiness_factors['sample_data'] = (existing_patterns / len(required_patterns)) * 100
    
    # 소스 코드 점수
    src_dir = project_root / 'src'
    if src_dir.exists():
        key_modules = [
            'src/data_processing/xtf_reader.py',
            'src/data_processing/coordinate_mapper.py',
            'src/data_processing/preprocessor.py'
        ]
        existing_modules = sum(1 for module in key_modules 
                             if (project_root / module).exists())
        readiness_factors['source_code'] = (existing_modules / len(key_modules)) * 100
    
    # 문서화 점수
    docs_dir = project_root / 'docs'
    if docs_dir.exists():
        doc_files = list(docs_dir.glob('*.md'))
        readiness_factors['documentation'] = min(len(doc_files) * 25, 100)
    
    # 결과 출력
    for factor, score in readiness_factors.items():
        status = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
        print(f"{status} {factor.replace('_', ' ').title()}: {score:.0f}%")
    
    # 전체 점수
    overall_score = sum(readiness_factors.values()) / len(readiness_factors)
    print(f"\\n🎯 전체 준비도: {overall_score:.0f}%")
    
    # 권장사항
    print("\\n💡 권장사항:")
    if overall_score >= 80:
        print("   ✨ Phase 2 진행 준비 완료!")
        print("   → 고급 특징 추출 및 딥러닝 모델 개발 시작 가능")
        plan_type = "고급 딥러닝 중심"
    elif overall_score >= 60:
        print("   📈 Phase 2 진행 가능 (일부 개선 권장)")
        print("   → 하이브리드 접근법으로 점진적 개발")
        plan_type = "하이브리드 접근법"
    elif overall_score >= 40:
        print("   🔧 기초 시스템 개선 필요")
        print("   → 데이터 품질 및 기본 시스템 안정화 우선")
        plan_type = "기초 안정화"
    else:
        print("   ⚠️  Phase 1 재점검 필요")
        print("   → 기본 구성 요소 점검 및 재구축")
        plan_type = "기본 구성 재검토"
    
    return overall_score, plan_type

def generate_next_steps(readiness_score, plan_type):
    """다음 단계 권장사항 생성"""
    print("\\n📋 다음 단계 권장사항")
    print("=" * 50)
    
    if readiness_score >= 80:
        steps = [
            "1. 샘플 데이터로 전체 파이프라인 테스트",
            "2. HOG, LBP, Gabor 특징 추출기 구현",
            "3. CNN 기반 딥러닝 모델 설계",
            "4. 데이터 증강 시스템 구축",
            "5. 실시간 처리 최적화"
        ]
    elif readiness_score >= 60:
        steps = [
            "1. 기존 모듈 안정성 테스트",
            "2. 전통적 특징 추출 방법 우선 구현",
            "3. SVM/Random Forest 분류기 학습",
            "4. 단순 CNN 모델 실험", 
            "5. 점진적 성능 향상"
        ]
    elif readiness_score >= 40:
        steps = [
            "1. 데이터 품질 개선",
            "2. 좌표 매핑 시스템 디버깅",
            "3. 전처리 파이프라인 최적화",
            "4. 기본 탐지 알고리즘 구현",
            "5. 시스템 안정성 확보"
        ]
    else:
        steps = [
            "1. 프로젝트 구조 재정비",
            "2. 필수 데이터 파일 확보",
            "3. 기본 모듈 재구현",
            "4. 단위 테스트 작성",
            "5. 문서화 완성"
        ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\\n🗓️  추정 소요 기간: {get_estimated_timeline(plan_type)}")
    print(f"🎯 목표 성능: {get_target_performance(plan_type)}")

def get_estimated_timeline(plan_type):
    """계획 유형별 예상 소요 시간"""
    timelines = {
        "고급 딥러닝 중심": "4-6주",
        "하이브리드 접근법": "6-8주", 
        "기초 안정화": "8-12주",
        "기본 구성 재검토": "12-16주"
    }
    return timelines.get(plan_type, "미정")

def get_target_performance(plan_type):
    """계획 유형별 목표 성능"""
    performances = {
        "고급 딥러닝 중심": "정확도 90% 이상",
        "하이브리드 접근법": "정확도 80-85%",
        "기초 안정화": "정확도 70-75%", 
        "기본 구성 재검토": "기본 동작 확인"
    }
    return performances.get(plan_type, "미정")

def save_analysis_report(readiness_score, plan_type):
    """분석 결과 저장"""
    report = {
        'analysis_date': datetime.now().isoformat(),
        'readiness_score': readiness_score,
        'recommended_plan': plan_type,
        'estimated_timeline': get_estimated_timeline(plan_type),
        'target_performance': get_target_performance(plan_type),
        'project_status': {
            'sample_data_available': (project_root / '[샘플]데이터').exists(),
            'source_code_complete': (project_root / 'src').exists(),
            'documentation_ready': (project_root / 'docs').exists(),
        }
    }
    
    # 결과 디렉토리 생성
    results_dir = project_root / 'data' / 'processed'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    with open(results_dir / 'quick_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 분석 결과 저장: {results_dir / 'quick_analysis_report.json'}")

def main():
    """메인 분석 함수"""
    print("🚀 사이드스캔 소나 프로젝트 빠른 분석")
    print("=" * 60)
    print(f"📁 프로젝트 위치: {project_root}")
    print(f"🕒 분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 단계별 분석 수행
    analyze_project_structure()
    check_data_requirements()
    readiness_score, plan_type = estimate_system_readiness()
    generate_next_steps(readiness_score, plan_type)
    save_analysis_report(readiness_score, plan_type)
    
    print("\\n" + "=" * 60)
    print("✅ 빠른 분석 완료!")
    print("\\n다음 명령어로 상세 분석을 실행할 수 있습니다:")
    print("python main.py --mode sample")
    print("=" * 60)

if __name__ == "__main__":
    main()