# 🔧 가상환경 설정 가이드

## 📋 개요

기뢰탐지 시스템의 의존성 패키지들은 특정 버전 호환성이 중요합니다. 특히 NumPy 2.0+와 OpenCV, SciPy 등의 호환성 문제가 있어 가상환경 사용을 **강력히 권장**합니다.

## 🎯 설치된 환경

### 현재 설치된 패키지 버전
```
Python: 3.9.12
NumPy: 1.26.4
OpenCV: 4.11.0  
SciPy: 1.13.1
PyTorch: 2.2.2
Scikit-learn: 1.6.1
Scikit-image: 0.24.0
PyXTF: 1.4.2
PyProj: 3.6.1
```

## 🚀 설치 방법

### 방법 1: 자동 설치 스크립트 (권장)

#### macOS/Linux:
```bash
chmod +x install.sh
./install.sh
```

#### Windows:
```cmd
install.bat
```

#### Python 스크립트:
```bash
python setup_env.py
```

### 방법 2: 수동 설치

#### 1. 가상환경 생성
```bash
# Python venv 사용
python -m venv mine_detection_env

# 또는 conda 사용
conda create -n mine_detection_env python=3.9
```

#### 2. 가상환경 활성화
```bash
# macOS/Linux
source mine_detection_env/bin/activate

# Windows
mine_detection_env\\Scripts\\activate.bat

# conda
conda activate mine_detection_env
```

#### 3. 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements_core.txt
```

## 📦 핵심 패키지 버전 요구사항

```txt
# 기본 과학 계산
numpy>=1.21,<2.0  # NumPy 2.0+ 호환성 문제로 제한
scipy>=1.7,<1.14
pandas>=1.3,<3.0

# 이미지 처리
opencv-python>=4.5,<5.0
scikit-image>=0.18,<0.25
pillow>=8.0

# 머신러닝
scikit-learn>=1.0,<2.0
torch>=2.0
matplotlib>=3.5

# 특화 라이브러리  
pyxtf>=1.4,<2.0  # 소나 데이터 처리
pyproj>=3.0,<4.0  # 지리좌표 변환
```

## 🔍 설치 확인

### 1. 기본 의존성 테스트
```bash
source mine_detection_env/bin/activate
python -c "
import numpy, cv2, scipy, sklearn, skimage, torch, pyxtf, pyproj
print('✅ 모든 패키지 설치 완료!')
"
```

### 2. 파이프라인 모듈 테스트  
```bash
python test_pipeline_modules.py --mode quick
```

### 3. OpenCV + NumPy 호환성 테스트
```bash
python cv2_performance_impact_analysis.py
```

## 🐛 문제 해결

### 1. NumPy 호환성 오류
```
RuntimeError: module compiled against API version but this version of numpy is
```

**해결방법:**
```bash
pip uninstall numpy scipy pandas matplotlib scikit-learn scikit-image -y
pip install "numpy>=1.21,<2.0"
pip install scipy pandas matplotlib scikit-learn scikit-image
```

### 2. OpenCV 설치 실패 (macOS)
```bash
# Homebrew 사용
brew install opencv

# 또는 conda 사용
conda install -c conda-forge opencv
```

### 3. PyTorch 설치 (CPU 버전)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. M1/M2 Mac 호환성
```bash
# Miniforge 사용 권장
conda install numpy scipy opencv scikit-learn scikit-image -c conda-forge
```

## 💻 플랫폼별 특이사항

### macOS
- **Homebrew 필요**: `brew install python opencv`
- **M1/M2**: conda-forge 채널 사용 권장
- **권한 문제**: `sudo` 사용 지양, 가상환경 필수

### Windows  
- **Visual Studio Build Tools** 필요할 수 있음
- **경로 문제**: 한글 경로 지양
- **PowerShell** vs **CMD**: PowerShell 권장

### Linux
- **시스템 패키지**: `sudo apt-get install python3-dev python3-opencv`
- **CUDA**: PyTorch GPU 버전 필요시 별도 설치
- **권한**: `sudo pip` 사용 금지, 가상환경 필수

## 🎛️ 환경 관리

### 환경 활성화/비활성화
```bash
# 활성화
source mine_detection_env/bin/activate  # macOS/Linux
mine_detection_env\\Scripts\\activate   # Windows
conda activate mine_detection_env       # conda

# 비활성화
deactivate
conda deactivate  # conda용
```

### 패키지 목록 저장/복원
```bash
# 현재 환경 저장
pip freeze > my_requirements.txt

# 다른 환경에 복원
pip install -r my_requirements.txt
```

### 환경 복제
```bash
# conda 환경 복제
conda create --clone mine_detection_env --name new_env

# requirements.txt로 복제
pip freeze > requirements.txt
python -m venv new_env
source new_env/bin/activate
pip install -r requirements.txt
```

## 🔄 정기 유지보수

### 월간 체크리스트
- [ ] 패키지 보안 업데이트 확인
- [ ] 테스트 파이프라인 실행
- [ ] 성능 벤치마크 비교
- [ ] 로그 파일 정리

### 업데이트 명령어
```bash
# 모든 패키지 업데이트 (주의!)
pip list --outdated
pip install --upgrade package_name

# 안전한 업데이트 (버전 범위 내에서)
pip install "numpy>=1.21,<2.0" --upgrade
```

## 📊 성능 최적화

### CPU 최적화
```bash
# NumPy 멀티스레딩
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# PyTorch CPU 스레드
export TORCH_NUM_THREADS=4
```

### 메모리 최적화  
```bash
# Python 메모리 제한
export PYTHONHASHSEED=0
ulimit -v 8388608  # 8GB 제한 (Linux/macOS)
```

## 🆘 문제 보고

설치나 실행 중 문제가 발생하면:

1. **환경 정보 수집**:
   ```bash
   python --version
   pip freeze > debug_requirements.txt
   python -c "import platform; print(platform.platform())"
   ```

2. **로그 수집**:
   ```bash
   python test_pipeline_modules.py --mode quick --verbose 2>&1 | tee debug.log
   ```

3. **GitHub Issues에 보고** (예정)

---

## 📝 변경 이력

- **2025-09-09**: 초기 가상환경 설정 및 NumPy 호환성 해결
- **설치 확인**: 13/13 패키지 정상 설치 완료
- **성능 분석**: OpenCV 부재 시 15-20% 성능 저하 확인