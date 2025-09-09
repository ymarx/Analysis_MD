# 🔧 설치 가이드 - Multi-Environment Support

**업데이트**: 2025-09-09  
**지원 환경**: 로컬 CPU/GPU, Runpod 클라우드

---

## 📋 목차

1. [빠른 시작](#-빠른-시작)
2. [로컬 환경 설치](#-로컬-환경-설치)
3. [GPU 환경 설치](#️-gpu-환경-설치)
4. [Runpod 클라우드 배포](#-runpod-클라우드-배포)
5. [문제 해결](#-문제-해결)

---

## 🚀 빠른 시작

### 자동 환경 감지 설치 (권장)

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/mine-detection.git
cd mine-detection

# 2. 자동 설치 스크립트 실행
chmod +x scripts/install.sh
./scripts/install.sh --auto

# 3. 환경 활성화
source mine_detection_env/bin/activate  # Linux/macOS
# 또는 mine_detection_env\Scripts\activate  # Windows

# 4. 테스트 실행
python main.py --device auto
```

이 명령어는 시스템을 자동 감지하여 최적의 환경을 설정합니다:
- GPU 감지 → GPU 최적화 설치
- GPU 없음 → CPU 최적화 설치
- 의존성 자동 해결

---

## 💻 로컬 환경 설치

### 1. 시스템 요구사항

**최소 요구사항**:
- Python 3.9 이상
- 8GB RAM
- 10GB 디스크 공간

**권장 요구사항**:
- Python 3.10
- 16GB RAM
- NVIDIA GPU (GTX 1060 이상) 또는 Apple Silicon M1/M2
- 50GB 디스크 공간

### 2. 플랫폼별 설치

#### macOS (Intel/Apple Silicon)

```bash
# Homebrew 설치 (없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 시스템 의존성 설치
brew install python@3.10 git cmake

# 프로젝트 클론
git clone https://github.com/your-repo/mine-detection.git
cd mine-detection

# 가상환경 생성
python3.10 -m venv mine_detection_env
source mine_detection_env/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements_core.txt

# Apple Silicon MPS 지원 확인
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Ubuntu/Debian Linux

```bash
# 시스템 의존성 설치
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
                    git build-essential cmake \
                    libopencv-dev libglib2.0-0 \
                    libsm6 libxext6 libxrender-dev

# NVIDIA GPU 드라이버 (GPU 사용시)
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# 프로젝트 설정
git clone https://github.com/your-repo/mine-detection.git
cd mine-detection

python3.10 -m venv mine_detection_env
source mine_detection_env/bin/activate

pip install --upgrade pip
pip install -r requirements_core.txt
```

#### Windows

```powershell
# Python 3.10 설치 (python.org에서 다운로드)
# Git 설치 (git-scm.com에서 다운로드)

# PowerShell 관리자 권한으로 실행
git clone https://github.com/your-repo/mine-detection.git
cd mine-detection

# 가상환경 생성
python -m venv mine_detection_env
mine_detection_env\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements_core.txt

# Visual C++ Build Tools (오류 발생시)
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 3. 설치 검증

```bash
# 환경 활성화
source mine_detection_env/bin/activate  # Linux/macOS
# mine_detection_env\Scripts\activate   # Windows

# 시스템 정보 확인
python scripts/check_environment.py

# 간단한 테스트
python main.py --help
python -c "from src.utils.device_manager import DeviceManager; dm = DeviceManager(); print(dm.get_device_summary())"
```

---

## ⚡️ GPU 환경 설치

### 1. NVIDIA GPU (CUDA)

#### CUDA 도구 설치

```bash
# Ubuntu에서 CUDA 11.8 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-11-8

# 환경 변수 설정
echo 'export PATH="/usr/local/cuda-11.8/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### PyTorch GPU 버전 설치

```bash
# 환경 활성화
source mine_detection_env/bin/activate

# GPU 전용 요구사항 설치
pip install -r requirements_gpu.txt

# PyTorch GPU 설치 확인
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### 2. Apple Silicon (MPS)

Apple Silicon Mac에서는 추가 설정 없이 MPS가 자동 지원됩니다:

```bash
# MPS 지원 확인
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    x = torch.randn(1, 3, 224, 224).to('mps')
    print(f'MPS test successful: {x.device}')
"
```

### 3. GPU 성능 테스트

```bash
# 성능 벤치마크 실행
python scripts/benchmark_performance.py --save

# 결과 확인
cat benchmarks/report_*.txt
```

---

## ☁️ Runpod 클라우드 배포

### 1. Runpod 계정 설정

1. [Runpod.io](https://runpod.io) 회원가입
2. API 키 생성: Settings → API Keys
3. 환경 변수 설정:

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

### 2. 자동 배포

```bash
# Runpod 자동 배포 스크립트
python scripts/deploy_runpod.py \
    --action deploy \
    --gpu-type "RTX 4090" \
    --bid-price 0.5 \
    --name "mine-detection-gpu"

# 배포 상태 확인
python scripts/deploy_runpod.py --action list
```

### 3. 수동 배포

#### Pod 생성

```bash
# 1. 사용 가능한 GPU 확인
python scripts/deploy_runpod.py --action list

# 2. Pod 생성
python scripts/deploy_runpod.py \
    --action create \
    --gpu-type "RTX 3090" \
    --bid-price 0.4

# 3. 프로젝트 업로드 및 환경 설정
python scripts/deploy_runpod.py \
    --action setup \
    --pod-id YOUR_POD_ID
```

#### Docker 이미지 사용

```bash
# 1. Docker 이미지 빌드 (선택사항)
docker build -f docker/Dockerfile.runpod -t mine-detection:runpod .

# 2. Runpod에서 커스텀 이미지 사용
# Template에서 Docker Image를 mine-detection:runpod로 설정
```

### 4. 클라우드 환경 접근

배포 완료 후 접근 정보:

```
Jupyter Lab: http://[POD-IP]:[PORT]/lab
SSH: ssh -p [SSH-PORT] root@[POD-IP]
TensorBoard: http://[POD-IP]:[TB-PORT]
```

### 5. 데이터 동기화

```bash
# 로컬 → Runpod 업로드
rsync -avz -e "ssh -p [SSH-PORT]" \
    ./data/ root@[POD-IP]:/workspace/Analysis_MD/data/

# Runpod → 로컬 다운로드
rsync -avz -e "ssh -p [SSH-PORT]" \
    root@[POD-IP]:/workspace/Analysis_MD/output/ ./output/
```

---

## 🔧 환경별 실행 방법

### 로컬 CPU 실행

```bash
# 기존 방식 (변경 없음)
python main.py

# 명시적 CPU 사용
python main.py --device cpu
```

### 로컬 GPU 실행

```bash
# 자동 GPU 감지 (권장)
python main.py --device auto

# 명시적 CUDA 사용
python main.py --device cuda

# 명시적 MPS 사용 (Apple Silicon)
python main.py --device mps
```

### 클라우드 실행

```bash
# Runpod에서 자동 최적화 실행
python main.py --device auto --cloud runpod

# 성능 벤치마크
python scripts/benchmark_performance.py --device auto
```

---

## 🆘 문제 해결

### 일반적인 문제

#### 1. NumPy 버전 충돌

```bash
# 오류: NumPy 2.0 호환성 문제
# 해결: NumPy 1.26.4로 다운그레이드
pip install numpy==1.26.4
```

#### 2. OpenCV 설치 실패

```bash
# macOS에서 OpenCV 설치 실패
brew install opencv
pip install opencv-python==4.8.1.78

# Ubuntu에서 OpenCV 의존성 문제
sudo apt install -y libopencv-dev python3-opencv
pip install --force-reinstall opencv-python
```

#### 3. CUDA 메모리 부족

```bash
# GPU 메모리 부족 시 배치 크기 감소
python main.py --device cuda --batch-size 8

# 메모리 사용량 모니터링
nvidia-smi -l 1
```

### GPU 관련 문제

#### 1. CUDA 인식 불가

```bash
# CUDA 설치 확인
nvcc --version
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# CUDA 버전 불일치 시 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Apple Silicon MPS 문제

```bash
# MPS 오류 시 CPU 폴백
python main.py --device cpu

# MPS 메모리 정리
python -c "
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print('MPS cache cleared')
"
```

### Runpod 관련 문제

#### 1. SSH 연결 실패

```bash
# SSH 키 확인
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# 방화벽 설정 확인
# Runpod 콘솔에서 포트 22 열려있는지 확인
```

#### 2. 프로젝트 업로드 실패

```bash
# 수동 업로드
scp -P [SSH-PORT] -r ./mine-detection root@[POD-IP]:/workspace/

# 압축하여 업로드
tar -czf mine-detection.tar.gz ./mine-detection
scp -P [SSH-PORT] mine-detection.tar.gz root@[POD-IP]:/workspace/
```

### 성능 최적화

#### 1. 메모리 사용량 최적화

```python
# config/environments/local.yaml에서 배치 크기 조정
data_processing:
  batch_size: 4  # 메모리 부족시 감소
  num_workers: 1
```

#### 2. GPU 사용률 최대화

```python
# Mixed Precision 활성화 (NVIDIA GPU에서)
# config/environments/gpu.yaml
model:
  mixed_precision: true
  compile_optimization: true
```

---

## 📞 지원 및 도움말

### 문제 신고

1. GitHub Issues: [링크]
2. 로그 파일 첨부: `logs/system.log`
3. 환경 정보: `python scripts/check_environment.py --full`

### 커뮤니티

- Discord: [서버 링크]
- 문서 위키: [위키 링크]
- FAQ: [FAQ 링크]

---

**성공적인 설치를 위한 체크리스트**:

- [ ] Python 3.9+ 설치 확인
- [ ] 가상환경 생성 및 활성화
- [ ] 의존성 패키지 설치 완료
- [ ] GPU 드라이버 설치 (GPU 사용시)
- [ ] 환경 테스트 통과
- [ ] 첫 실행 성공

모든 환경에서 `python main.py --device auto`가 정상 실행되면 설치가 완료된 것입니다! 🎉