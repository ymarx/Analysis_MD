#!/bin/bash
# Runpod 컨테이너 시작 스크립트

set -e

echo "=== Mine Detection System - Runpod Startup ==="
echo "시작 시간: $(date)"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.6+PTX"  # RTX 3090/4090 지원

# 작업 디렉토리로 이동
cd /workspace/Analysis_MD

# GPU 정보 확인
echo "=== GPU 정보 ==="
nvidia-smi
echo ""

# Python 환경 확인
echo "=== Python 환경 ==="
python --version
pip --version
echo ""

# CUDA 확인
echo "=== CUDA 확인 ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# 필요 디렉토리 생성
echo "=== 디렉토리 설정 ==="
mkdir -p data/{raw,processed,samples}
mkdir -p output/{results,reports,visualizations}
mkdir -p checkpoints
mkdir -p logs
echo "디렉토리 생성 완료"

# 권한 설정
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Jupyter Lab 설정
echo "=== Jupyter Lab 설정 ==="
if [ ! -f ~/.jupyter/jupyter_lab_config.py ]; then
    jupyter lab --generate-config
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py
fi

# TensorBoard 준비
mkdir -p logs/tensorboard

# 환경 테스트
echo "=== 환경 테스트 ==="
python -c "
import sys
import torch
import numpy as np
import cv2
from src.utils.device_manager import DeviceManager

print('라이브러리 버전:')
print(f'  - Python: {sys.version.split()[0]}')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - NumPy: {np.__version__}')
print(f'  - OpenCV: {cv2.__version__}')

# 디바이스 관리자 테스트
dm = DeviceManager()
print(f'\\n디바이스 정보:')
print(f'  - 선택된 디바이스: {dm.device}')
print(f'  - 메모리: {dm.capabilities.memory_available}MB 사용 가능')
print(f'  - 권장 배치 크기: {dm.config.batch_size}')
print(f'  - Mixed Precision: {dm.config.mixed_precision}')

# 간단한 GPU 테스트
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print(f'  - GPU 연산 테스트: 성공 (결과 형태: {y.shape})')

print('\\n환경 설정 완료!')
"

# 서비스 시작
echo "=== 서비스 시작 ==="

# TensorBoard 백그라운드 실행
nohup tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006 > logs/tensorboard.log 2>&1 &
echo "TensorBoard 시작됨 (포트 6006)"

# Jupyter Lab 백그라운드 실행
nohup jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser > logs/jupyter.log 2>&1 &
echo "Jupyter Lab 시작됨 (포트 8888)"

echo ""
echo "=== 시작 완료 ==="
echo "Jupyter Lab: http://localhost:8888"
echo "TensorBoard: http://localhost:6006"
echo ""
echo "사용 가능한 명령어:"
echo "  - python main.py --device auto                    # 자동 분석 실행"
echo "  - python scripts/benchmark_performance.py         # 성능 벤치마크"
echo "  - python -m src.training.train_detector           # 모델 훈련"
echo ""
echo "로그 확인:"
echo "  - tail -f logs/jupyter.log                        # Jupyter 로그"
echo "  - tail -f logs/tensorboard.log                    # TensorBoard 로그"
echo ""

# 프로세스를 유지하기 위해 대기
wait