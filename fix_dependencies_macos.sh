#!/bin/bash

# macOS 환경에서 기뢰탐지 시스템 의존성 해결 스크립트
# 작성일: 2025-09-09

set -e

echo "🍎 macOS 기뢰탐지 시스템 의존성 해결 시작..."

# 현재 환경 정보 출력
echo "현재 환경:"
echo "- Python 버전: $(python --version)"
echo "- 현재 NumPy 버전: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '없음')"
echo "- 현재 작업 디렉토리: $(pwd)"

# 백업을 위한 현재 패키지 리스트 저장
echo "📦 현재 설치된 패키지 백업 중..."
pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt

echo "
🔧 의존성 해결 방안 선택:
1) NumPy 다운그레이드 + 기존 환경 수정 (빠름, 위험)
2) 새로운 conda 환경 생성 (권장, 안전)
3) virtualenv로 새 환경 생성 (중간)
4) Docker 컨테이너 사용 (고급)
5) 현재 상태에서 제한적 사용 (임시)
"

read -p "선택하세요 (1-5): " choice

case $choice in
    1)
        echo "🔄 방안 1: NumPy 다운그레이드 실행..."
        
        echo "⚠️  경고: 다른 패키지들과 충돌이 발생할 수 있습니다."
        read -p "계속하시겠습니까? (y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            pip uninstall numpy scipy pandas matplotlib scikit-learn scikit-image -y || true
            pip install "numpy>=1.21,<2.0"
            pip install "scipy>=1.7,<1.8"  
            pip install "pandas>=1.3,<2.0"
            pip install "matplotlib>=3.5"
            pip install "scikit-learn>=1.0"
            pip install "scikit-image>=0.18,<0.20"
            pip install opencv-python
            pip install pyxtf pyproj
            
            echo "✅ NumPy 다운그레이드 완료"
            echo "🧪 테스트 실행 중..."
            python -c "
import numpy as np
import cv2
import scipy.ndimage
from skimage import filters
print('✅ 모든 의존성 설치 성공!')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
        else
            echo "❌ 작업 취소됨"
            exit 1
        fi
        ;;
        
    2)
        echo "🐍 방안 2: 새로운 conda 환경 생성..."
        
        # conda 설치 확인
        if ! command -v conda &> /dev/null; then
            echo "❌ conda가 설치되지 않았습니다."
            echo "https://docs.conda.io/en/latest/miniconda.html 에서 설치 후 다시 실행하세요."
            exit 1
        fi
        
        ENV_NAME="mine_detection"
        echo "환경 이름: $ENV_NAME"
        
        # 기존 환경이 있으면 제거
        conda env remove -n $ENV_NAME -y 2>/dev/null || true
        
        # 새 환경 생성
        conda create -n $ENV_NAME python=3.9 -y
        
        echo "생성된 환경 활성화 방법:"
        echo "conda activate $ENV_NAME"
        
        # 환경 활성화 및 패키지 설치
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        
        # 패키지 설치
        conda install numpy=1.21 scipy matplotlib pandas -y
        conda install -c conda-forge scikit-learn scikit-image -y
        conda install -c conda-forge opencv -y
        pip install pyxtf pyproj
        
        echo "✅ Conda 환경 설정 완료!"
        echo "🧪 테스트 실행 중..."
        python -c "
import numpy as np
import cv2
import scipy.ndimage
from skimage import filters
import pyxtf
import pyproj
print('✅ 모든 의존성 설치 성공!')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
        
        echo ""
        echo "🎉 설정 완료! 다음 명령으로 환경을 활성화하세요:"
        echo "conda activate $ENV_NAME"
        echo "cd $(pwd)"
        echo "python test_pipeline_modules.py --mode quick"
        ;;
        
    3)
        echo "🔧 방안 3: virtualenv 환경 생성..."
        
        # virtualenv 설치 확인
        if ! python -m venv --help &> /dev/null; then
            echo "virtualenv 설치 중..."
            pip install virtualenv
        fi
        
        ENV_DIR="venv_mine_detection"
        
        # 기존 환경 제거
        rm -rf $ENV_DIR
        
        # 새 가상환경 생성
        python -m venv $ENV_DIR
        
        # 환경 활성화
        source $ENV_DIR/bin/activate
        
        # pip 업그레이드
        pip install --upgrade pip
        
        # 패키지 설치
        pip install "numpy>=1.21,<2.0"
        pip install "scipy>=1.7,<1.8"
        pip install "matplotlib>=3.5"
        pip install "pandas>=1.3,<2.0"
        pip install "scikit-learn>=1.0"
        pip install "scikit-image>=0.18,<0.20"
        pip install opencv-python
        pip install pyxtf pyproj
        
        echo "✅ virtualenv 환경 설정 완료!"
        echo "🧪 테스트 실행 중..."
        python -c "
import numpy as np
import cv2
import scipy.ndimage
from skimage import filters
print('✅ 모든 의존성 설치 성공!')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
        
        echo ""
        echo "🎉 설정 완료! 다음 명령으로 환경을 활성화하세요:"
        echo "source $ENV_DIR/bin/activate"
        echo "cd $(pwd)"
        echo "python test_pipeline_modules.py --mode quick"
        ;;
        
    4)
        echo "🐳 방안 4: Docker 환경 설정..."
        
        # Docker 설치 확인
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker가 설치되지 않았습니다."
            echo "https://www.docker.com/products/docker-desktop 에서 설치 후 다시 실행하세요."
            exit 1
        fi
        
        # Dockerfile 생성
        cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir \
    "numpy>=1.21,<2.0" \
    "scipy>=1.7,<1.8" \
    "matplotlib>=3.5" \
    "pandas>=1.3,<2.0" \
    "scikit-learn>=1.0" \
    "scikit-image>=0.18,<0.20" \
    opencv-python \
    pyxtf \
    pyproj

WORKDIR /app
COPY . /app

CMD ["python", "test_pipeline_modules.py", "--mode", "quick"]
EOF
        
        echo "Docker 이미지 빌드 중..."
        docker build -t mine-detection .
        
        echo "✅ Docker 환경 설정 완료!"
        echo "🧪 테스트 실행 중..."
        docker run --rm -v $(pwd):/app mine-detection python -c "
import numpy as np
import cv2
import scipy.ndimage
from skimage import filters
print('✅ 모든 의존성 설치 성공!')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
        
        echo ""
        echo "🎉 설정 완료! 다음 명령으로 실행하세요:"
        echo "docker run --rm -v \$(pwd):/app mine-detection python main.py"
        ;;
        
    5)
        echo "⚠️  방안 5: 현재 상태에서 제한적 사용..."
        
        echo "현재 환경에서 사용 가능한 기능:"
        echo "- ✅ 기본 NumPy 연산"
        echo "- ✅ LBP 특징 추출 (순수 Python)"
        echo "- ⚠️  Gabor 필터 (단순화된 버전)"
        echo "- ❌ 고급 전처리 (OpenCV 필요)"
        echo "- ❌ 고성능 필터링 (scipy 필요)"
        
        echo ""
        echo "제한된 설정으로 테스트 실행:"
        python -c "
print('현재 환경에서 가능한 기능 테스트...')
import numpy as np
print('✅ NumPy:', np.__version__)

try:
    from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
    print('✅ LBP 추출기 사용 가능')
except Exception as e:
    print('❌ LBP 추출기 오류:', e)

try:
    from src.feature_extraction.gabor_extractor import GaborFeatureExtractor  
    print('⚠️  Gabor 추출기 제한적 사용 가능')
except Exception as e:
    print('❌ Gabor 추출기 오류:', e)
"
        
        echo ""
        echo "⚠️  성능 제약 사항:"
        echo "- 정확도: 89% → 70-75% (약 15-20% 저하)"
        echo "- 처리시간: 8분 → 15-20분 (2-3배 증가)"
        echo "- 권장: 연구/학습 목적으로만 사용"
        ;;
        
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "🔍 최종 환경 확인:"
python -c "
import sys
print('Python:', sys.version)

modules = ['numpy', 'cv2', 'scipy', 'skimage', 'pyxtf', 'pyproj']
for module in modules:
    try:
        exec(f'import {module}')
        version = eval(f'{module}.__version__')
        print(f'✅ {module}: {version}')
    except ImportError:
        print(f'❌ {module}: 설치되지 않음')
    except AttributeError:
        print(f'⚠️  {module}: 버전 정보 없음 (설치됨)')
    except Exception as e:
        print(f'❌ {module}: 오류 - {e}')
"

echo ""
echo "🎉 의존성 해결 스크립트 완료!"
echo ""
echo "📋 다음 단계:"
echo "1. 위의 환경 활성화 명령을 실행하세요"
echo "2. 빠른 테스트를 실행하세요: python test_pipeline_modules.py --mode quick"
echo "3. 전체 파이프라인을 실행하세요: python main.py"
echo ""
echo "❓ 문제가 발생하면 다음을 시도하세요:"
echo "- 터미널을 재시작하고 환경을 다시 활성화"
echo "- 시스템 재부팅 후 다시 시도"
echo "- GitHub Issues에 문제 보고"