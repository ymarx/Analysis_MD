# 📖 사용법 가이드 - Multi-Environment

**업데이트**: 2025-09-09  
**지원 환경**: CPU, GPU, Runpod 클라우드

---

## 📋 목차

1. [기본 사용법](#-기본-사용법)
2. [환경별 실행 방법](#-환경별-실행-방법)
3. [모듈별 분석](#-모듈별-분석)
4. [성능 최적화](#-성능-최적화)
5. [클라우드 활용](#️-클라우드-활용)
6. [고급 기능](#-고급-기능)

---

## 🚀 기본 사용법

### 빠른 시작

```bash
# 환경 활성화
source mine_detection_env/bin/activate

# 자동 환경 감지 실행 (권장)
python main.py --device auto

# 도움말 확인
python main.py --help
```

### 주요 명령어 옵션

```bash
# 디바이스 선택
--device auto          # 자동 감지 (권장)
--device cpu           # CPU 강제 사용
--device cuda          # NVIDIA GPU 사용
--device mps           # Apple Silicon 사용

# 입출력 설정
--input data/sample.xtf              # 입력 파일 지정
--output output/results              # 출력 디렉토리
--config config/custom_settings.yaml # 설정 파일 지정

# 실행 모드
--mode analysis        # 전체 분석 (기본값)
--mode training        # 모델 훈련
--mode inference       # 추론만 실행
--mode benchmark       # 성능 측정

# 성능 옵션
--batch-size 32        # 배치 크기 지정
--num-workers 4        # 데이터 로더 워커 수
--mixed-precision      # Mixed Precision 사용
```

---

## 🖥️ 환경별 실행 방법

### 1. 로컬 CPU 환경

#### 기본 실행 (기존 방식)
```bash
# 변경 없음 - 기존 코드 100% 호환
python main.py
```

#### 최적화된 CPU 실행
```bash
# CPU 전용 최적화 설정 사용
python main.py --device cpu --config config/environments/local.yaml

# 메모리 제한 설정
python main.py --device cpu --batch-size 4 --num-workers 1
```

### 2. 로컬 GPU 환경

#### NVIDIA GPU (CUDA)
```bash
# 자동 CUDA 감지 및 최적화
python main.py --device auto

# 명시적 CUDA 사용
python main.py --device cuda --config config/environments/gpu.yaml

# Mixed Precision 활성화
python main.py --device cuda --mixed-precision

# 메모리 사용량 모니터링
watch -n 1 nvidia-smi
```

#### Apple Silicon (MPS)
```bash
# MPS 자동 감지
python main.py --device auto

# MPS 직접 지정
python main.py --device mps --batch-size 16

# MPS 성능 측정
python scripts/benchmark_performance.py --device mps
```

### 3. 하이브리드 실행

```bash
# GPU가 있으면 GPU, 없으면 CPU
python main.py --device auto --fallback-cpu

# GPU 메모리 부족시 CPU 폴백
python main.py --device auto --memory-fallback
```

---

## 🔬 모듈별 분석

### 1. XTF 데이터 처리

```bash
# XTF 파일 정보 확인
python -m src.data_processing.xtf_processor --info data/sample.xtf

# 배치 처리
python -m src.data_processing.xtf_processor \
    --input data/*.xtf \
    --output data/processed \
    --device auto
```

### 2. 좌표 변환 및 매핑

```bash
# 좌표 시스템 변환
python -m src.data_processing.coordinate_mapper \
    --input data/processed \
    --output output/coordinates \
    --utm-zone 52N
```

### 3. 특징 추출

```bash
# 전통적 특징 추출
python -m src.feature_extraction.traditional_features \
    --input data/processed \
    --methods hog,lbp,gabor \
    --device auto

# 딥러닝 특징 추출
python -m src.feature_extraction.deep_features \
    --input data/processed \
    --model resnet50 \
    --device auto
```

### 4. CNN 탐지 모델

```bash
# 모델 훈련
python -m src.models.cnn_detector \
    --mode train \
    --data data/training \
    --device auto \
    --epochs 100

# 모델 평가
python -m src.models.cnn_detector \
    --mode evaluate \
    --model checkpoints/best_model.pth \
    --data data/test \
    --device auto

# 추론
python -m src.models.cnn_detector \
    --mode inference \
    --model checkpoints/best_model.pth \
    --input data/new_data.xtf \
    --device auto
```

### 5. 종합 평가

```bash
# 다중 지표 평가
python -m src.evaluation.comprehensive_evaluator \
    --predictions output/predictions \
    --ground-truth data/annotations \
    --metrics all \
    --output output/evaluation
```

---

## ⚡ 성능 최적화

### 1. 자동 성능 튜닝

```bash
# 시스템 자동 분석 및 최적화
python scripts/optimize_performance.py --device auto

# 성능 프로파일링
python scripts/benchmark_performance.py --full --save
```

### 2. 메모리 최적화

```bash
# 메모리 사용량 모니터링
python main.py --device auto --profile-memory

# 배치 크기 자동 조정
python main.py --device auto --auto-batch-size

# 그래디언트 누적 사용
python main.py --device auto --gradient-accumulation 4
```

### 3. GPU 활용 최대화

```bash
# 컴파일 최적화 (PyTorch 2.0+)
python main.py --device cuda --compile-model

# 멀티 GPU 사용
python main.py --device cuda --multi-gpu

# Tensor Core 활용 (A100, V100)
python main.py --device cuda --use-tensor-cores
```

### 4. 데이터 로딩 최적화

```bash
# 워커 수 자동 조정
python main.py --device auto --auto-workers

# Pin Memory 사용
python main.py --device cuda --pin-memory

# 데이터 캐싱
python main.py --device auto --cache-data
```

---

## ☁️ 클라우드 활용

### 1. Runpod 배포 및 실행

#### 자동 배포
```bash
# 전체 워크플로우 자동화
export RUNPOD_API_KEY="your-api-key"
python scripts/deploy_runpod.py --action deploy --gpu-type "RTX 4090"
```

#### 원격 실행
```bash
# SSH를 통한 원격 실행
ssh -p [PORT] root@[IP] "cd /workspace/Analysis_MD && python main.py --device auto"

# Jupyter Lab을 통한 대화형 실행
# 브라우저에서 http://[IP]:[PORT]/lab 접속
```

### 2. 대용량 데이터 처리

```bash
# 클라우드에서 대규모 배치 처리
python main.py \
    --device auto \
    --input /workspace/data/large_dataset \
    --output /workspace/output \
    --batch-processing \
    --parallel-jobs 4
```

### 3. 분산 처리

```bash
# 멀티 Pod 분산 처리
python scripts/distributed_processing.py \
    --pods pod1,pod2,pod3 \
    --data-split equal \
    --device auto
```

---

## 🎯 고급 기능

### 1. 커스텀 설정

#### 환경별 설정 파일
```yaml
# config/environments/custom.yaml
environment:
  name: "custom"
  device: "auto"

model:
  architecture: "resnet50"
  mixed_precision: true
  
data_processing:
  batch_size: 64
  augmentation: true
```

```bash
# 커스텀 설정 사용
python main.py --config config/environments/custom.yaml
```

### 2. 파이프라인 조합

```bash
# 전처리 + 훈련 + 평가 파이프라인
python scripts/full_pipeline.py \
    --input data/raw \
    --stages preprocess,train,evaluate \
    --device auto

# 추론 전용 파이프라인
python scripts/inference_pipeline.py \
    --model checkpoints/best_model.pth \
    --input data/new/*.xtf \
    --device auto
```

### 3. 실험 관리

```bash
# TensorBoard 로깅
python main.py --device auto --tensorboard --experiment-name exp_001

# Weights & Biases 연동
python main.py --device auto --wandb --project mine-detection

# 실험 비교
python scripts/compare_experiments.py --experiments exp_001,exp_002,exp_003
```

### 4. 모델 서빙

```bash
# REST API 서버 시작
python scripts/serve_model.py \
    --model checkpoints/best_model.pth \
    --device auto \
    --port 8080

# gRPC 서버
python scripts/grpc_server.py \
    --model checkpoints/best_model.pth \
    --device auto
```

---

## 📊 실행 예시

### 예시 1: 로컬에서 빠른 프로토타이핑

```bash
# 1. 환경 확인
python scripts/check_environment.py

# 2. 샘플 데이터로 테스트
python main.py \
    --input data/samples/sample.xtf \
    --output output/test \
    --device auto \
    --batch-size 8

# 3. 결과 확인
python scripts/visualize_results.py --input output/test
```

### 예시 2: GPU에서 대규모 훈련

```bash
# 1. GPU 성능 확인
python scripts/benchmark_performance.py --device cuda

# 2. 훈련 실행
python main.py \
    --mode training \
    --data data/training_large \
    --device cuda \
    --batch-size 32 \
    --mixed-precision \
    --tensorboard

# 3. 모델 평가
python main.py \
    --mode evaluate \
    --model checkpoints/latest.pth \
    --data data/test \
    --device cuda
```

### 예시 3: Runpod 클라우드 활용

```bash
# 1. 클라우드 배포
export RUNPOD_API_KEY="your-key"
python scripts/deploy_runpod.py \
    --action deploy \
    --gpu-type "A100-40GB" \
    --name "production-training"

# 2. 원격 실행 (SSH)
ssh -p [PORT] root@[IP] << 'EOF'
cd /workspace/Analysis_MD
python main.py \
    --device auto \
    --data /workspace/data/production \
    --output /workspace/output \
    --mixed-precision \
    --compile-model
EOF

# 3. 결과 다운로드
rsync -avz -e "ssh -p [PORT]" \
    root@[IP]:/workspace/output/ ./output/
```

---

## 🐛 디버깅 및 모니터링

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f logs/system.log

# 특정 모듈 로그
grep "CNN" logs/system.log

# 에러 로그만
grep "ERROR" logs/system.log
```

### 성능 모니터링

```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 시스템 리소스 모니터링
htop

# 메모리 사용량 추적
python scripts/memory_profiler.py --device auto
```

### 디버그 모드 실행

```bash
# 상세 디버그 정보
python main.py --device auto --debug --verbose

# 중간 결과 저장
python main.py --device auto --save-intermediate

# 프로파일링 활성화
python main.py --device auto --profile
```

---

## 💡 팁과 모범 사례

### 1. 효율적인 개발 워크플로우

```bash
# 개발용 빠른 테스트
python main.py --device auto --debug --input data/samples --batch-size 2

# 실험용 중간 규모 테스트
python main.py --device auto --input data/validation --save-checkpoints

# 프로덕션용 전체 실행
python main.py --device auto --input data/production --mixed-precision
```

### 2. 리소스 관리

```bash
# 메모리 절약 모드
python main.py --device auto --memory-efficient --gradient-accumulation 8

# 빠른 실행 모드
python main.py --device auto --fast-mode --compile-model

# 안정성 우선 모드
python main.py --device cpu --safe-mode
```

### 3. 결과 관리

```bash
# 버전 관리
python main.py --device auto --experiment-tag v1.2 --output output/v1.2

# 자동 백업
python main.py --device auto --auto-backup --backup-frequency 10

# 결과 비교
python scripts/compare_results.py --runs output/v1.1,output/v1.2
```

---

이 가이드를 통해 다양한 환경에서 Mine Detection 시스템을 효과적으로 활용할 수 있습니다. 
추가 질문이나 문제가 있으시면 GitHub Issues를 통해 문의해 주세요! 🚀