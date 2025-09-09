# 🚀 GPU/클라우드 배포 전략 계획서

**작성일**: 2025-09-09  
**목표**: 로컬 기능 손상 없이 GPU 및 클라우드 환경 지원  

---

## 🎯 핵심 원칙

### 1. **기존 코드 보존** 
- 현재 CPU 기반 로컬 실행 기능 100% 유지
- 하위 호환성 보장 (기존 스크립트 그대로 작동)
- 옵션 방식으로 GPU/클라우드 기능 추가

### 2. **점진적 확장**
- 자동 디바이스 감지 및 최적 환경 선택
- 환경별 설정 자동 조정
- 폴백 메커니즘 (GPU 없으면 CPU로 자동 전환)

### 3. **확장성**
- 로컬 → GPU → 분산 → 클라우드 단계적 지원
- 모듈별 독립적 GPU 활용
- 클라우드 서비스별 최적화

---

## 📊 구현 단계

### Phase 1: 디바이스 관리 시스템 (1-2일)

#### 1.1 자동 디바이스 감지
```python
# src/utils/device_manager.py
class DeviceManager:
    def __init__(self):
        self.device = self.detect_optimal_device()
        self.capabilities = self.analyze_capabilities()
    
    def detect_optimal_device(self):
        # CUDA → MPS → CPU 순서로 감지
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():  # Apple Silicon
            return torch.device('mps') 
        else:
            return torch.device('cpu')
```

#### 1.2 설정 자동 조정
```python
# config/device_config.py
DEVICE_CONFIGS = {
    'cpu': {
        'batch_size': 8,
        'num_workers': 2,
        'memory_limit': '4GB'
    },
    'cuda': {
        'batch_size': 32,
        'num_workers': 4,
        'memory_limit': '8GB'
    },
    'mps': {
        'batch_size': 16,
        'num_workers': 3,
        'memory_limit': '6GB'
    }
}
```

### Phase 2: GPU 최적화 모듈 (2-3일)

#### 2.1 CNN 모델 GPU 가속
```python
# src/models/gpu_detector.py
class GPUOptimizedDetector(SidescanTargetDetector):
    def __init__(self, config=None, device='auto'):
        super().__init__(config)
        self.device_manager = DeviceManager()
        self.device = device if device != 'auto' else self.device_manager.device
        self.to(self.device)
        
        # GPU별 최적화
        if self.device.type == 'cuda':
            self.enable_cuda_optimizations()
        elif self.device.type == 'mps':
            self.enable_mps_optimizations()
```

#### 2.2 배치 처리 최적화
```python
# src/training/gpu_pipeline.py
class GPUTrainingPipeline:
    def __init__(self, device_manager):
        self.device = device_manager.device
        self.batch_size = device_manager.get_optimal_batch_size()
        self.dataloader = self.create_optimized_dataloader()
    
    def create_optimized_dataloader(self):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.get_num_workers(),
            pin_memory=self.device.type == 'cuda'
        )
```

### Phase 3: 클라우드 배포 지원 (3-4일)

#### 3.1 Docker 컨테이너화
```dockerfile
# docker/Dockerfile.gpu
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 기본 환경 설정
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.9 python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements_gpu.txt .
RUN pip install -r requirements_gpu.txt

# 애플리케이션 복사
COPY . /app
WORKDIR /app

# 실행 권한 설정
RUN chmod +x scripts/run_gpu.sh

CMD ["python", "main.py", "--device", "auto"]
```

#### 3.2 클라우드 서비스별 배포
```python
# cloud/aws_deployment.py
class AWSGPUDeployment:
    def __init__(self):
        self.instance_types = {
            'p3.2xlarge': {'gpu': 'V100', 'memory': '61GB'},
            'g4dn.xlarge': {'gpu': 'T4', 'memory': '16GB'},
            'p4d.24xlarge': {'gpu': 'A100', 'memory': '1152GB'}
        }
    
    def deploy_model(self, model_path, instance_type='g4dn.xlarge'):
        # EC2 인스턴스 생성 및 모델 배포
        pass
```

### Phase 4: 분산 처리 지원 (선택사항)

#### 4.1 멀티 GPU 지원
```python
# src/training/distributed.py
class DistributedTrainer:
    def __init__(self, model, device_manager):
        if device_manager.gpu_count > 1:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
```

---

## 🔧 구현 상세

### 1. 디바이스 관리 시스템

**파일**: `src/utils/device_manager.py`

**핵심 기능**:
- 자동 하드웨어 감지 (CUDA, MPS, CPU)
- 메모리 사용량 모니터링
- 배치 크기 자동 조정
- 성능 벤치마킹

**기존 코드와의 통합**:
```python
# 기존 코드 (변경 없음)
pipeline = MineDetectionPipeline(config)

# 새로운 GPU 지원 (옵션)
pipeline = MineDetectionPipeline(config, device='auto')
```

### 2. 설정 계층화

**파일 구조**:
```
config/
├── settings.py           # 기본 설정 (변경 없음)
├── device_configs.py     # 디바이스별 설정 (신규)
├── cloud_configs.py      # 클라우드 설정 (신규)
└── environments/         # 환경별 설정 (신규)
    ├── local.yaml
    ├── gpu.yaml
    ├── aws.yaml
    └── azure.yaml
```

### 3. 백워드 호환성 보장

**원칙**:
- 모든 기존 스크립트는 그대로 작동
- 새 기능은 opt-in 방식
- 환경 감지 후 자동 최적화

**예시**:
```bash
# 기존 방식 (변경 없음)
python main.py

# 새로운 옵션 (GPU 강제 사용)
python main.py --device cuda

# 자동 최적화 (권장)
python main.py --device auto
```

---

## 🌐 클라우드 플랫폼별 지원

### 1. Runpod (주요 지원 플랫폼)

**지원 GPU**:
- RTX 3090 (24GB VRAM) - $0.44/시간
- RTX 4090 (24GB VRAM) - $0.69/시간
- A40 (48GB VRAM) - $1.29/시간
- A100-40GB - $1.89/시간
- A100-80GB - $2.99/시간

**주요 기능**:
- Spot 인스턴스로 최대 70% 비용 절약
- SSH, Jupyter Lab, TensorBoard 통합 환경
- 자동 데이터 동기화 및 프로젝트 업로드
- GPU 사용률 및 비용 실시간 모니터링

**배포 방법**:
```bash
# Runpod 자동 배포
export RUNPOD_API_KEY="your-api-key"
python scripts/deploy_runpod.py --gpu-type "RTX 4090" --bid-price 0.5
```

### 2. 로컬 GPU 환경

**지원 GPU**:
- NVIDIA CUDA (GTX 1060 이상)
- Apple Silicon M1/M2 (MPS)
- Intel GPU (실험적 지원)

**최적화 기능**:
- 자동 메모리 관리
- Mixed Precision 훈련
- 동적 배치 크기 조정

### 3. 기타 클라우드 플랫폼

**확장 가능한 플랫폼**:
- Paperspace Gradient
- Lambda Labs
- Vast.ai
- Google Colab Pro
- 개인 GPU 서버

---

## 📦 패키징 전략

### 1. 의존성 관리

**CPU 전용** (현재):
```txt
# requirements.txt (변경 없음)
numpy>=1.21,<2.0
opencv-python>=4.5,<5.0
# ... 기존 패키지들
```

**GPU 지원**:
```txt
# requirements_gpu.txt (신규)
torch>=2.0.0+cu118
torchvision>=0.15.0+cu118
# ... GPU 최적화 패키지들
```

**클라우드 최적화**:
```txt
# requirements_cloud.txt (신규)  
boto3>=1.26.0  # AWS
google-cloud-storage>=2.7.0  # GCP
azure-storage-blob>=12.14.0  # Azure
```

### 2. Docker 이미지 계층화

**기본 이미지**:
```dockerfile
# Dockerfile (CPU, 변경 없음)
FROM python:3.9-slim
```

**GPU 이미지**:
```dockerfile
# Dockerfile.gpu (신규)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
```

**클라우드 이미지**:
```dockerfile
# Dockerfile.cloud (신규)
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
# 클라우드 SDK 포함
```

---

## 🧪 테스트 전략

### 1. 환경별 테스트

**로컬 CPU** (기존):
```bash
python test_pipeline_modules.py --device cpu
```

**로컬 GPU**:
```bash
python test_pipeline_modules.py --device cuda
```

**클라우드**:
```bash
python test_pipeline_modules.py --cloud aws --device auto
```

### 2. 성능 벤치마킹

**지표**:
- 훈련 시간 (epoch당)
- 추론 속도 (이미지당)
- 메모리 사용량
- 비용 효율성

**자동 벤치마킹**:
```python
# benchmarks/performance_test.py
class MultiEnvironmentBenchmark:
    def run_all_tests(self):
        results = {}
        for device in ['cpu', 'cuda', 'mps']:
            if self.is_available(device):
                results[device] = self.benchmark_device(device)
        return results
```

---

## 🔒 보안 및 모니터링

### 1. 보안

**클라우드 보안**:
- IAM 역할 기반 액세스
- VPC 내 배포
- 암호화된 스토리지

**API 키 관리**:
```python
# src/utils/secrets_manager.py
class SecretsManager:
    def get_cloud_credentials(self, provider):
        # 환경변수 또는 클라우드 키 관리 서비스 사용
        pass
```

### 2. 모니터링

**성능 모니터링**:
- GPU 사용률
- 메모리 사용량
- 배치 처리 시간
- 오류율

**로깅**:
```python
# src/utils/cloud_logger.py
class CloudLogger:
    def __init__(self, provider='local'):
        if provider == 'aws':
            self.setup_cloudwatch()
        elif provider == 'gcp':
            self.setup_cloud_logging()
```

---

## 📈 마이그레이션 로드맵

### 단계 1: 준비 (1일)
- [ ] 현재 코드 백업
- [ ] 디바이스 관리 시스템 구현
- [ ] 기본 GPU 지원 추가

### 단계 2: GPU 최적화 (2일)  
- [ ] CNN 모델 GPU 가속
- [ ] 배치 처리 최적화
- [ ] 메모리 관리 개선

### 단계 3: 클라우드 준비 (2일)
- [ ] Docker 컨테이너화
- [ ] 클라우드 설정 추가
- [ ] 배포 스크립트 작성

### 단계 4: 테스트 및 문서화 (1일)
- [ ] 다중 환경 테스트
- [ ] 성능 벤치마킹
- [ ] 문서 업데이트

---

## 💰 비용 분석

### 로컬 vs 클라우드 비교

| 환경 | 초기 비용 | 운영 비용 | 성능 | 확장성 | 설정 시간 |
|------|-----------|-----------|------|--------|----------|
| **로컬 CPU** | $0 | 전력비 | 기준 (1x) | 제한적 | 0분 |
| **로컬 GPU** | $500-2000 | 전력비 | 5-15배 | 제한적 | 30분 |
| **Runpod RTX 4090** | $0 | $0.35-0.69/시간 | 15-25배 | 무제한 | 5분 |
| **Runpod A100** | $0 | $1.5-3/시간 | 20-50배 | 무제한 | 5분 |

### 권장 사용 시나리오

**로컬 CPU**: 개발, 테스트, 소규모 분석 (< 1000 이미지)  
**로컬 GPU**: 중규모 연구, 프로토타이핑 (< 10,000 이미지)  
**Runpod RTX 4090**: 대규모 훈련, 실험 (< 100,000 이미지)  
**Runpod A100**: 프로덕션 배포, 실시간 서비스 (무제한)

### 비용 효율성 분석

**1시간 훈련 작업 기준**:
- 로컬 CPU (8코어): 24시간 소요 → 전력비 $2-5
- 로컬 GPU (RTX 3080): 2시간 소요 → 전력비 $0.3
- Runpod RTX 4090: 1시간 소요 → $0.69
- Runpod A100-80GB: 30분 소요 → $1.5

**결론**: 중규모 이상 작업에서는 Runpod이 가장 비용 효율적

---

이 계획을 통해 현재 로컬 기능을 완전히 보존하면서 GPU 및 클라우드 환경으로 확장할 수 있습니다. 모든 변경사항은 opt-in 방식으로 구현되어 기존 사용자에게 영향을 주지 않습니다.