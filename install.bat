@echo off
REM 🪟 Windows용 기뢰탐지 시스템 설치 스크립트
REM 작성일: 2025-09-09

echo 🎯 Windows 기뢰탐지 시스템 설치 시작
echo ====================================

REM Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다.
    echo https://python.org에서 Python 3.8+ 설치 후 다시 시도하세요.
    pause
    exit /b 1
)

echo ✅ Python 설치 확인됨
python --version

REM 가상환경 생성
echo 🔨 가상환경 생성 중...
python -m venv mine_detection_env
if errorlevel 1 (
    echo ❌ 가상환경 생성 실패
    pause
    exit /b 1
)

REM 가상환경 활성화 및 패키지 설치
echo 📦 패키지 설치 중...
call mine_detection_env\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install "numpy>=1.21,<2.0"
python -m pip install "scipy>=1.7,<1.14"
python -m pip install "matplotlib>=3.5"
python -m pip install "pandas>=1.3,<3.0"
python -m pip install "scikit-learn>=1.0,<2.0"
python -m pip install "scikit-image>=0.18,<0.25"
python -m pip install "opencv-python>=4.5,<5.0"
python -m pip install "pyxtf>=1.4,<2.0"
python -m pip install "pyproj>=3.0,<4.0"
python -m pip install "tqdm>=4.60"
python -m pip install "pillow>=8.0"

if errorlevel 1 (
    echo ❌ 패키지 설치 중 오류 발생
    pause
    exit /b 1
)

echo ✅ 설치 완료!

REM 테스트 실행
echo 🧪 설치 테스트 중...
python -c "
import numpy, scipy, matplotlib, pandas, sklearn, skimage, cv2, pyxtf, pyproj, tqdm
print('🎉 모든 의존성 설치 완료!')
print('NumPy:', numpy.__version__)
print('OpenCV:', cv2.__version__)
"

if errorlevel 1 (
    echo ❌ 테스트 실패
    pause
    exit /b 1
)

echo.
echo ====================================
echo 🎉 설치 성공! 사용법:
echo ====================================
echo 1. 환경 활성화: mine_detection_env\Scripts\activate.bat
echo 2. 테스트 실행: python test_pipeline_modules.py --mode quick
echo 3. 실행: python main.py
echo ====================================
pause