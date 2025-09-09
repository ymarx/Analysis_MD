#!/bin/bash
# 🐧 Linux/macOS용 기뢰탐지 시스템 설치 스크립트  
# 작성일: 2025-09-09

set -e

echo "🎯 기뢰탐지 시스템 설치 시작 (Linux/macOS)"
echo "=============================================="

# 운영체제 감지
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac
echo "🖥️  운영체제: $MACHINE"

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python이 설치되지 않았습니다."
        echo "설치 방법:"
        if [[ "$MACHINE" == "Mac" ]]; then
            echo "  brew install python"
        else
            echo "  sudo apt-get update && sudo apt-get install python3 python3-pip python3-venv"
        fi
        exit 1
    else
        PYTHON_CMD=python
    fi
else
    PYTHON_CMD=python3
fi

echo "✅ Python 확인됨"
$PYTHON_CMD --version

# 가상환경 생성
echo "🔨 가상환경 생성 중..."
ENV_NAME="mine_detection_env"

# 기존 환경 제거
if [ -d "$ENV_NAME" ]; then
    echo "⚠️  기존 환경 제거 중..."
    rm -rf "$ENV_NAME"
fi

$PYTHON_CMD -m venv $ENV_NAME

# 가상환경 활성화
echo "🔄 가상환경 활성화 중..."
source $ENV_NAME/bin/activate

# pip 업그레이드
echo "📦 pip 업그레이드 중..."
pip install --upgrade pip

# 시스템별 추가 의존성 설치
if [[ "$MACHINE" == "Linux" ]]; then
    echo "🐧 Linux 시스템 패키지 확인 중..."
    # OpenCV를 위한 시스템 패키지가 필요할 수 있음
    echo "⚠️  OpenCV 설치에 실패하면 다음 명령어를 실행하세요:"
    echo "  sudo apt-get install python3-opencv libopencv-dev"
elif [[ "$MACHINE" == "Mac" ]]; then
    echo "🍎 macOS 시스템 설정 확인 중..."
    # Homebrew 확인
    if command -v brew &> /dev/null; then
        echo "✅ Homebrew 감지됨"
    else
        echo "⚠️  Homebrew가 없습니다. 필요시 설치:"
        echo "  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    fi
fi

# 패키지 설치
echo "📦 필수 패키지 설치 중..."

packages=(
    "numpy>=1.21,<2.0"
    "scipy>=1.7,<1.14"
    "matplotlib>=3.5"
    "pandas>=1.3,<3.0"
    "scikit-learn>=1.0,<2.0"
    "scikit-image>=0.18,<0.25"
    "opencv-python>=4.5,<5.0"
    "pyxtf>=1.4,<2.0"
    "pyproj>=3.0,<4.0"
    "tqdm>=4.60"
    "pillow>=8.0"
)

failed_packages=()
for package in "${packages[@]}"; do
    echo "설치 중: $package"
    if pip install "$package"; then
        echo "✅ $package 설치 완료"
    else
        echo "❌ $package 설치 실패"
        failed_packages+=("$package")
    fi
done

# 실패한 패키지 확인
if [ ${#failed_packages[@]} -ne 0 ]; then
    echo "⚠️  실패한 패키지들:"
    for pkg in "${failed_packages[@]}"; do
        echo "  - $pkg"
    done
    echo "이러한 패키지들을 수동으로 설치해야 할 수 있습니다."
fi

# 설치 테스트
echo "🧪 설치 테스트 중..."
python -c "
import sys
print(f'Python: {sys.version}')

modules_to_test = [
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'), 
    ('matplotlib', 'Matplotlib'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('skimage', 'Scikit-image'),
    ('cv2', 'OpenCV'),
    ('pyxtf', 'PyXTF'),
    ('pyproj', 'PyProj'),
    ('tqdm', 'TQDM')
]

failed = []
for module, name in modules_to_test:
    try:
        __import__(module)
        try:
            version = eval(f'{module}.__version__')
            print(f'✅ {name}: {version}')
        except:
            print(f'⚠️  {name}: 설치됨 (버전 정보 없음)')
    except ImportError:
        print(f'❌ {name}: 설치되지 않음')
        failed.append(name)
    except Exception as e:
        print(f'❌ {name}: 오류 - {str(e)[:50]}')
        failed.append(name)

if failed:
    print(f'\n❌ 실패한 모듈들: {failed}')
    sys.exit(1)
else:
    print('\n🎉 모든 의존성 설치 및 테스트 완료!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "🎉 설치 성공! 사용법:"
    echo "=============================================="
    echo "1. 환경 활성화: source $ENV_NAME/bin/activate"
    echo "2. 프로젝트로 이동: cd $(pwd)"
    echo "3. 테스트 실행: python test_pipeline_modules.py --mode quick"
    echo "4. 실행: python main.py"
    echo ""
    echo "💡 팁:"
    echo "- 매번 새 터미널에서 '1단계' 명령어로 환경을 활성화하세요"
    echo "- 환경 비활성화: deactivate"
    echo "=============================================="
else
    echo "❌ 설치 테스트 실패. 수동으로 문제를 해결해야 합니다."
    exit 1
fi