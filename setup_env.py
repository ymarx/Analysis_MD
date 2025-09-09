#!/usr/bin/env python
"""
🔧 기뢰탐지 시스템 환경 설정 스크립트 (크로스플랫폼)
작성일: 2025-09-09

이 스크립트는 Python 가상환경을 생성하고 필요한 패키지를 설치합니다.
Windows, macOS, Linux에서 모두 동작합니다.

사용법:
    python setup_env.py
    python setup_env.py --env-name my_custom_env
    python setup_env.py --no-venv  # 현재 환경에 설치
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def run_command(command, check=True, shell=False):
    """안전한 명령어 실행"""
    try:
        if isinstance(command, str):
            command = command.split() if not shell else command
        
        result = subprocess.run(
            command, 
            check=check, 
            capture_output=True, 
            text=True,
            shell=shell
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

def detect_python_version():
    """Python 버전 감지"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro} 감지됨")
    
    if version < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    elif version >= (3, 12):
        print("⚠️  Python 3.12+는 일부 패키지와 호환성 문제가 있을 수 있습니다.")
    
    return True

def detect_system():
    """운영체제 감지"""
    system = platform.system().lower()
    arch = platform.machine()
    print(f"🖥️  시스템: {system} ({arch})")
    
    return system, arch

def check_conda():
    """Conda 설치 확인"""
    success, stdout, stderr = run_command("conda --version")
    return success

def check_virtualenv():
    """virtualenv 설치 확인"""
    success, stdout, stderr = run_command([sys.executable, "-m", "venv", "--help"])
    return success

def create_virtual_environment(env_name, use_conda=False):
    """가상환경 생성"""
    print(f"🔨 가상환경 '{env_name}' 생성 중...")
    
    if use_conda:
        success, stdout, stderr = run_command(f"conda create -n {env_name} python=3.9 -y", shell=True)
        if success:
            print(f"✅ Conda 환경 '{env_name}' 생성 완료")
            return True, "conda"
        else:
            print(f"❌ Conda 환경 생성 실패: {stderr}")
            return False, None
    else:
        # Python venv 사용
        success, stdout, stderr = run_command([sys.executable, "-m", "venv", env_name])
        if success:
            print(f"✅ Python venv 환경 '{env_name}' 생성 완료")
            return True, "venv"
        else:
            print(f"❌ Python venv 환경 생성 실패: {stderr}")
            return False, None

def get_activation_command(env_name, env_type, system):
    """환경별 활성화 명령어 반환"""
    if env_type == "conda":
        return f"conda activate {env_name}"
    elif env_type == "venv":
        if system == "windows":
            return f"{env_name}\\Scripts\\activate"
        else:
            return f"source {env_name}/bin/activate"
    return None

def install_packages(env_name, env_type, system):
    """패키지 설치"""
    print("📦 필수 패키지 설치 중...")
    
    # 활성화 및 패키지 설치 명령어 구성
    if env_type == "conda":
        # Conda 환경에서 설치
        commands = [
            f"conda activate {env_name}",
            "pip install --upgrade pip",
            'pip install "numpy>=1.21,<2.0"',
            'pip install "scipy>=1.7,<1.14"',
            'pip install "matplotlib>=3.5"',
            'pip install "pandas>=1.3,<3.0"',
            'pip install "scikit-learn>=1.0,<2.0"',
            'pip install "scikit-image>=0.18,<0.25"',
            'pip install "opencv-python>=4.5,<5.0"',
            'pip install "pyxtf>=1.4,<2.0"',
            'pip install "pyproj>=3.0,<4.0"',
            'pip install "tqdm>=4.60"',
            'pip install "pillow>=8.0"'
        ]
        
        # conda에서 명령어들을 하나로 합쳐서 실행
        full_command = " && ".join(commands)
        success, stdout, stderr = run_command(full_command, shell=True)
        
    elif env_type == "venv":
        # Python venv 환경에서 설치
        if system == "windows":
            pip_path = f"{env_name}\\Scripts\\pip"
        else:
            pip_path = f"{env_name}/bin/pip"
        
        packages = [
            "numpy>=1.21,<2.0",
            "scipy>=1.7,<1.14", 
            "matplotlib>=3.5",
            "pandas>=1.3,<3.0",
            "scikit-learn>=1.0,<2.0",
            "scikit-image>=0.18,<0.25",
            "opencv-python>=4.5,<5.0",
            "pyxtf>=1.4,<2.0",
            "pyproj>=3.0,<4.0",
            "tqdm>=4.60",
            "pillow>=8.0"
        ]
        
        # pip 업그레이드
        success, stdout, stderr = run_command([pip_path, "install", "--upgrade", "pip"])
        if not success:
            print(f"⚠️  pip 업그레이드 실패: {stderr}")
        
        # 각 패키지 설치
        failed_packages = []
        for package in packages:
            success, stdout, stderr = run_command([pip_path, "install", package])
            if success:
                print(f"✅ {package} 설치 완료")
            else:
                print(f"❌ {package} 설치 실패: {stderr}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n⚠️  실패한 패키지들: {failed_packages}")
            return False
        
        success = True
    
    if success:
        print("✅ 모든 패키지 설치 완료!")
        return True
    else:
        print(f"❌ 패키지 설치 중 오류 발생: {stderr}")
        return False

def test_installation(env_name, env_type, system):
    """설치된 패키지 테스트"""
    print("🧪 설치 확인 테스트 중...")
    
    if env_type == "conda":
        python_path = f"conda run -n {env_name} python"
    elif env_type == "venv":
        if system == "windows":
            python_path = f"{env_name}\\Scripts\\python"
        else:
            python_path = f"{env_name}/bin/python"
    else:
        python_path = sys.executable
    
    test_script = '''
import sys
print(f"Python: {sys.version}")

modules_to_test = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"), 
    ("matplotlib", "Matplotlib"),
    ("pandas", "Pandas"),
    ("sklearn", "Scikit-learn"),
    ("skimage", "Scikit-image"),
    ("cv2", "OpenCV"),
    ("pyxtf", "PyXTF"),
    ("pyproj", "PyProj"),
    ("tqdm", "TQDM")
]

failed = []
for module, name in modules_to_test:
    try:
        __import__(module)
        version = eval(f"{module}.__version__")
        print(f"✅ {name}: {version}")
    except ImportError:
        print(f"❌ {name}: 설치되지 않음")
        failed.append(name)
    except AttributeError:
        print(f"⚠️  {name}: 버전 정보 없음 (설치됨)")
    except Exception as e:
        print(f"❌ {name}: 오류 - {str(e)[:50]}")
        failed.append(name)

if failed:
    print(f"\\n❌ 실패한 모듈들: {failed}")
    sys.exit(1)
else:
    print("\\n🎉 모든 의존성 설치 및 테스트 완료!")
'''
    
    if env_type == "conda":
        success, stdout, stderr = run_command(f'{python_path} -c "{test_script}"', shell=True)
    else:
        success, stdout, stderr = run_command([python_path, "-c", test_script])
    
    if success:
        print(stdout)
        return True
    else:
        print(f"❌ 테스트 실패: {stderr}")
        return False

def print_usage_instructions(env_name, env_type, system):
    """사용법 안내"""
    activation_cmd = get_activation_command(env_name, env_type, system)
    
    print("\n" + "="*60)
    print("🎉 설치 완료! 다음 단계를 따라하세요:")
    print("="*60)
    print(f"1. 가상환경 활성화:")
    print(f"   {activation_cmd}")
    print()
    print("2. 프로젝트 디렉토리로 이동:")
    print(f"   cd {Path.cwd()}")
    print()
    print("3. 테스트 실행:")
    print("   python test_pipeline_modules.py --mode quick")
    print()
    print("4. 메인 파이프라인 실행:")
    print("   python main.py")
    print()
    print("💡 팁:")
    print("- 매번 터미널을 열 때마다 1단계 명령어로 환경을 활성화하세요")
    print("- 환경 비활성화: deactivate")
    print("="*60)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="기뢰탐지 시스템 환경 설정")
    parser.add_argument("--env-name", default="mine_detection_env", 
                      help="가상환경 이름 (기본: mine_detection_env)")
    parser.add_argument("--no-venv", action="store_true",
                      help="가상환경 생성 없이 현재 환경에 설치")
    parser.add_argument("--use-conda", action="store_true",
                      help="conda 사용 (venv 대신)")
    
    args = parser.parse_args()
    
    print("🎯 기뢰탐지 시스템 환경 설정 시작")
    print("="*50)
    
    # Python 버전 확인
    if not detect_python_version():
        sys.exit(1)
    
    # 시스템 정보
    system, arch = detect_system()
    
    if args.no_venv:
        print("⚠️  현재 환경에 직접 설치합니다.")
        env_name = None
        env_type = None
    else:
        # 가상환경 도구 확인
        has_conda = check_conda()
        has_venv = check_virtualenv()
        
        print(f"📋 사용 가능한 도구:")
        print(f"   Conda: {'✅' if has_conda else '❌'}")
        print(f"   Python venv: {'✅' if has_venv else '❌'}")
        
        if args.use_conda and not has_conda:
            print("❌ Conda가 설치되지 않았습니다.")
            sys.exit(1)
        
        if not has_conda and not has_venv:
            print("❌ 가상환경 도구가 없습니다. Conda 또는 Python 3.3+가 필요합니다.")
            sys.exit(1)
        
        # 가상환경 생성
        use_conda = args.use_conda or (has_conda and not has_venv)
        success, env_type = create_virtual_environment(args.env_name, use_conda)
        
        if not success:
            sys.exit(1)
        
        env_name = args.env_name
    
    # 패키지 설치
    if args.no_venv:
        # 현재 환경에 설치
        print("📦 현재 환경에 패키지 설치 중...")
        packages = [
            "numpy>=1.21,<2.0",
            "scipy>=1.7,<1.14",
            "matplotlib>=3.5", 
            "pandas>=1.3,<3.0",
            "scikit-learn>=1.0,<2.0",
            "scikit-image>=0.18,<0.25",
            "opencv-python>=4.5,<5.0",
            "pyxtf>=1.4,<2.0",
            "pyproj>=3.0,<4.0",
            "tqdm>=4.60",
            "pillow>=8.0"
        ]
        
        for package in packages:
            success, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", package])
            if success:
                print(f"✅ {package} 설치 완료")
            else:
                print(f"❌ {package} 설치 실패: {stderr}")
        
        install_success = True
    else:
        install_success = install_packages(env_name, env_type, system)
    
    if not install_success:
        print("❌ 패키지 설치에 실패했습니다.")
        sys.exit(1)
    
    # 설치 테스트
    test_success = test_installation(env_name, env_type, system)
    
    if not test_success:
        print("❌ 설치 테스트에 실패했습니다.")
        sys.exit(1)
    
    # 사용법 안내
    if not args.no_venv:
        print_usage_instructions(env_name, env_type, system)
    else:
        print("\n🎉 설치 완료! 바로 사용할 수 있습니다.")
        print("python test_pipeline_modules.py --mode quick")

if __name__ == "__main__":
    main()