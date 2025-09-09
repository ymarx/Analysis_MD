# 🚀 사이드스캔 소나 기뢰탐지 시스템 사용 가이드

**버전**: v2.0  
**업데이트**: 2025-09-09  
**난이도**: 초급 ~ 고급  

---

## 🎯 빠른 시작 가이드

### 1단계: 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd /path/to/Analysis_MD

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 필수 패키지 설치
pip install numpy scipy scikit-learn scikit-image
pip install pyxtf pyproj matplotlib
```

### 2단계: 간단한 테스트 실행
```bash
# 빠른 모듈 테스트
python test_pipeline_modules.py --mode quick

# 결과 확인
# ✅ imports: PASS
# ✅ intensity_extraction: PASS  
# ✅ feature_extraction: PASS
```

### 3단계: 전체 파이프라인 실행
```python
from src.main_pipeline import MineDetectionPipeline, PipelineConfig

# 기본 설정으로 실행
config = PipelineConfig(
    output_dir="data/results/my_analysis",
    use_synthetic_data=True,
    feature_extractors=['lbp', 'gabor']
)

pipeline = MineDetectionPipeline(config)
results = pipeline.run_full_pipeline()

print("🎉 분석 완료!")
```

---

## 📁 프로젝트 구조

```
Analysis_MD/
├── 📂 src/                      # 핵심 소스 코드
│   ├── 📂 data_processing/      # 데이터 처리
│   ├── 📂 feature_extraction/   # 특징 추출
│   ├── 📂 models/              # 분류 모델
│   ├── 📂 evaluation/          # 성능 평가
│   ├── 📂 data_simulation/     # 모의데이터
│   ├── 📂 data_augmentation/   # 데이터 증강
│   ├── 📂 training/            # 모델 훈련
│   ├── 📂 utils/               # 유틸리티
│   ├── 📂 interactive/         # 대화형 도구
│   └── 📄 main_pipeline.py     # 메인 파이프라인
├── 📂 data/                     # 데이터 저장소
│   ├── 📂 processed/           # 전처리된 데이터
│   └── 📂 results/             # 분석 결과
├── 📂 datasets/                 # 실제 데이터셋
│   ├── 📂 Pohang_Eardo_1_*/    # 포항 이어도 데이터
│   └── 📂 */original/          # 원본 데이터
│       └── 📂 */simulation/    # 모의 데이터
├── 📂 [샘플]데이터/             # 샘플 데이터
├── 📂 docs/                     # 문서
├── 📂 config/                   # 설정 파일
├── 📂 notebooks/               # Jupyter 노트북
├── 📂 outputs/                  # 출력 결과
│   ├── 📂 figures/             # 그래프/차트
│   └── 📂 models/              # 저장된 모델
├── 📂 logs/                     # 로그 파일
├── 📂 tests/                    # 테스트 코드
├── 📄 main.py                   # 메인 실행 파일
├── 📄 test_*.py                 # 각종 테스트 스크립트
└── 📄 requirements.txt          # 패키지 의존성
```

---

## 🛠️ 사용법별 가이드

### 🔰 초급 사용자: GUI 스타일 실행

#### 메인 스크립트 실행
```bash
# 기본 분석 실행 (대화형 메뉴 포함)
python main.py

# 또는 간단한 분석 실행
python quick_analysis.py

# 샘플 데이터 분석 실행
python sample_analysis.py
```

#### 설정 파일 사용
```json
// config/user_settings.json
{
    "input_xtf_path": "datasets/Pohang_Eardo_1_*/simulation/xtf_input/*.xtf",
    "output_dir": "data/results/analysis_output",
    "feature_extractors": ["lbp", "gabor"],
    "use_synthetic_data": true,
    "enable_visualization": true,
    "patch_size": 64
}
```

```bash
# 설정 파일로 실행 (설정 파일이 있을 경우)
python main.py --config config/user_settings.json

# 또는 기본 설정으로 실행
python main.py
```

### 🔧 중급 사용자: 모듈별 실행

#### 1. XTF 데이터 처리만 실행
```python
from src.data_processing.xtf_intensity_extractor import XTFIntensityExtractor

# XTF 강도 데이터 추출
extractor = XTFIntensityExtractor()
intensity_data = extractor.extract_intensity_data(
    xtf_path="datasets/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04/simulation/xtf_input/sample.xtf",
    output_dir="data/results/intensity"
)

print(f"추출된 Ping 수: {intensity_data['metadata'].ping_count}")
print(f"이미지 크기: {intensity_data['intensity_images']['combined'].shape}")
```

#### 2. 특징 추출만 실행
```python
from src.feature_extraction.lbp_extractor import ComprehensiveLBPExtractor
from src.feature_extraction.gabor_extractor import GaborFeatureExtractor
import numpy as np

# 샘플 이미지 준비
image = np.random.rand(128, 128)  # 더미 이미지 (실제로는 XTF에서 추출)

# LBP 특징 추출
lbp_extractor = ComprehensiveLBPExtractor()
lbp_features = lbp_extractor.extract_comprehensive_features(image)
print(f"LBP 특징 차원: {len(lbp_features)}")

# Gabor 특징 추출
gabor_extractor = GaborFeatureExtractor()
gabor_features = gabor_extractor.extract_comprehensive_features(image)
print(f"Gabor 특징 차원: {len(gabor_features)}")
```

#### 3. 모의데이터만 생성
```python
from src.data_simulation.scenario_generator import ScenarioBasedGenerator

# 시나리오별 모의데이터 생성
generator = ScenarioBasedGenerator()

scenarios = ['A_deep_ocean', 'B_shallow_coastal', 'C_medium_depth']
for scenario in scenarios:
    synthetic_data = generator.generate_scenario_data(scenario, num_samples=50)
    print(f"{scenario}: {len(synthetic_data)} 샘플 생성")
```

### ⚡ 고급 사용자: 커스터마이징

#### 1. 새로운 특징 추출기 구현
```python
from src.feature_extraction.base_extractor import BaseFeatureExtractor
import numpy as np

class CustomTextureExtractor(BaseFeatureExtractor):
    def __init__(self, param1=1.0, param2=2.0):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def extract_features(self, image):
        """사용자 정의 텍스처 특징 추출"""
        # 여기에 커스텀 알고리즘 구현
        features = self.my_custom_algorithm(image)
        return features
    
    def my_custom_algorithm(self, image):
        # 예시: 통계적 특징
        features = [
            np.mean(image),
            np.std(image),
            np.skew(image.flatten()),
            np.kurtosis(image.flatten())
        ]
        return np.array(features)

# 사용법
custom_extractor = CustomTextureExtractor(param1=1.5)
custom_features = custom_extractor.extract_features(image)
```

#### 2. 파이프라인 커스터마이징
```python
from src.main_pipeline import MineDetectionPipeline, PipelineConfig

class CustomPipeline(MineDetectionPipeline):
    def __init__(self, config):
        super().__init__(config)
        # 커스텀 컴포넌트 추가
        self.custom_extractor = CustomTextureExtractor()
    
    def custom_step_extra_processing(self):
        """추가 처리 단계"""
        print("커스텀 처리 단계 실행 중...")
        # 커스텀 로직
        pass
    
    def run_custom_pipeline(self):
        """커스터마이징된 파이프라인"""
        # 기본 단계들
        self.step1_extract_intensity_data()
        self.step2_preprocess_and_map()
        
        # 커스텀 단계
        self.custom_step_extra_processing()
        
        # 나머지 단계들
        self.step4_extract_and_validate_features()
        self.step5_evaluate_feature_performance()

# 사용법
custom_config = PipelineConfig(
    output_dir="data/results/custom_analysis",
    feature_extractors=['lbp', 'gabor', 'custom']
)

custom_pipeline = CustomPipeline(custom_config)
custom_pipeline.feature_extractors['custom'] = custom_pipeline.custom_extractor
results = custom_pipeline.run_custom_pipeline()
```

---

## 📊 실행 모드별 가이드

### 1. 전체 파이프라인 모드
```bash
# 전체 7단계 순차 실행 (메인 스크립트 사용)
python main.py

# 또는 직접 파이프라인 모듈 실행
python -c "
from src.main_pipeline import *
config = PipelineConfig()
pipeline = MineDetectionPipeline(config)
pipeline.run_full_pipeline()
"
```

**실행 시간**: 약 10-30분 (데이터 크기에 따라)  
**결과물**: 
- 강도 이미지
- 추출된 특징
- 훈련된 모델
- 성능 평가 리포트
- 시각화 차트

### 2. 개별 단계 모드
```bash
# 1단계만 실행: XTF 데이터 추출
python test_pipeline_modules.py --mode step --step 1

# 4단계만 실행: 특징 추출
python test_pipeline_modules.py --mode step --step 4

# 7단계만 실행: 실-모의 데이터 비교
python test_pipeline_modules.py --mode step --step 7
```

### 3. 빠른 테스트 모드
```bash
# 핵심 기능만 빠르게 테스트
python test_pipeline_modules.py --mode quick

# 예상 실행 시간: 2-5분
# 결과: 각 모듈의 기본 동작 검증
```

### 4. 배치 처리 모드
```python
# 여러 XTF 파일 일괄 처리
import os
from pathlib import Path

def batch_process_xtf_files(input_dir, output_dir):
    xtf_files = list(Path(input_dir).glob("*.xtf"))
    
    for i, xtf_file in enumerate(xtf_files):
        print(f"처리 중 ({i+1}/{len(xtf_files)}): {xtf_file.name}")
        
        config = PipelineConfig(
            input_xtf_path=str(xtf_file),
            output_dir=f"{output_dir}/{xtf_file.stem}",
            enable_visualization=False  # 속도 향상
        )
        
        pipeline = MineDetectionPipeline(config)
        results = pipeline.run_full_pipeline()
        
        print(f"✅ 완료: {xtf_file.name}")

# 사용법
batch_process_xtf_files("datasets/", "data/results/batch_analysis/")
```

---

## 🎨 결과 분석 및 시각화

### 결과 구조 이해
```
data/results/pipeline_output/
├── 01_intensity_data/          # 추출된 강도 데이터
│   ├── combined_intensity.npy
│   ├── port_intensity.npy
│   └── navigation_data.npz
├── 02_preprocessed/            # 전처리된 데이터
├── 03_features/               # 추출된 특징
│   ├── lbp_features.npy
│   └── gabor_features.npy
├── 04_models/                 # 훈련된 모델
├── 05_evaluation/             # 성능 평가 결과
│   ├── performance_metrics.json
│   └── confusion_matrix.png
├── 06_comparison/             # 실-모의 데이터 비교
└── 07_visualization/          # 시각화 결과
    ├── intensity_images/
    ├── feature_importance/
    └── performance_charts/
```

### 결과 해석 가이드

#### 1. 성능 지표 해석
```python
import json

# 성능 결과 로드
with open('data/results/pipeline_output/05_evaluation/performance_metrics.json', 'r') as f:
    performance = json.load(f)

# 해석
for extractor, metrics in performance.items():
    print(f"\n📊 {extractor.upper()} 성능:")
    print(f"  정확도: {metrics.get('accuracy', 0):.1%}")
    print(f"  정밀도: {metrics.get('precision', 0):.1%}")
    print(f"  재현율: {metrics.get('recall', 0):.1%}")
    print(f"  F1점수: {metrics.get('f1_score', 0):.1%}")
    
    # 해석 가이드
    accuracy = metrics.get('accuracy', 0)
    if accuracy > 0.9:
        print("  ✅ 우수한 성능")
    elif accuracy > 0.8:
        print("  ✅ 양호한 성능")
    elif accuracy > 0.7:
        print("  ⚠️  개선 필요")
    else:
        print("  ❌ 성능 불량 - 재검토 필요")
```

#### 2. 특징 중요도 분석
```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_feature_importance(feature_file):
    features = np.load(feature_file)
    
    # 특징 통계
    print(f"특징 차원: {features.shape[1]}")
    print(f"샘플 수: {features.shape[0]}")
    print(f"평균 특징값: {np.mean(features):.4f}")
    print(f"특징 범위: [{np.min(features):.4f}, {np.max(features):.4f}]")
    
    # 특징 분포 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.hist(features.flatten(), bins=50, alpha=0.7)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    plt.subplot(132)
    feature_means = np.mean(features, axis=0)
    plt.plot(feature_means)
    plt.title('Feature Means')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    
    plt.subplot(133)
    feature_stds = np.std(features, axis=0)
    plt.plot(feature_stds)
    plt.title('Feature Standard Deviations')
    plt.xlabel('Feature Index')
    plt.ylabel('Std Value')
    
    plt.tight_layout()
    plt.show()

# 사용법
analyze_feature_importance('data/results/pipeline_output/03_features/lbp_features.npy')
```

#### 3. 실-모의 데이터 비교 결과 해석
```python
def interpret_comparison_results(comparison_file):
    with open(comparison_file, 'r') as f:
        comparison = json.load(f)
    
    print("🔍 실-모의 데이터 비교 분석:")
    
    # 특징 분포 유사도
    feature_dist = comparison.get('feature_distributions', {})
    kl_div = feature_dist.get('kl_divergence', 0)
    similarity = feature_dist.get('distribution_similarity', 0)
    
    print(f"\n📈 분포 유사도:")
    print(f"  KL Divergence: {kl_div:.4f} (낮을수록 유사)")
    print(f"  유사도 점수: {similarity:.3f} (높을수록 유사)")
    
    if similarity > 0.8:
        print("  ✅ 매우 유사한 분포 - 모의데이터 품질 우수")
    elif similarity > 0.6:
        print("  ✅ 양호한 유사도 - 모의데이터 활용 가능")
    else:
        print("  ⚠️  유사도 부족 - 모의데이터 개선 필요")
    
    # 교차 도메인 성능
    cross_perf = comparison.get('cross_domain_performance', {})
    print(f"\n🔄 교차 도메인 성능:")
    for test_type, result in cross_perf.items():
        accuracy = result.get('accuracy', 0)
        print(f"  {test_type}: {accuracy:.1%}")

# 사용법  
interpret_comparison_results('data/results/pipeline_output/06_comparison/comparison_results.json')
```

---

## ⚠️ 문제 해결 가이드

### 자주 발생하는 오류

#### 1. 모듈 import 오류
```
ModuleNotFoundError: No module named 'src.xxx'
```
**해결책**:
```bash
# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 스크립트에 추가
import sys
sys.path.append('/path/to/Analysis_MD')
```

#### 2. 메모리 부족 오류
```
MemoryError: Unable to allocate xxx GB
```
**해결책**:
```python
# 배치 크기 줄이기
config = PipelineConfig(
    patch_size=32,  # 64에서 32로 줄임
    use_synthetic_data=False,  # 모의데이터 비활성화
)

# 또는 처리 단위 줄이기
def process_in_chunks(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        yield process_chunk(chunk)
```

#### 3. OpenCV 설치 오류
```
ImportError: No module named 'cv2'
```
**해결책**:
```bash
# OpenCV 설치 시도 (여러 옵션)
pip install opencv-python
# 또는
pip install opencv-python-headless
# 또는 conda
conda install opencv

# 설치 실패 시 시스템은 대안 구현 자동 사용
```

#### 4. XTF 파일 처리 오류
```
FileNotFoundError: XTF file not found
```
**해결책**:
```python
# 더미 데이터로 테스트
config = PipelineConfig(
    input_xtf_path=None,  # None으로 설정하면 더미 데이터 사용
    use_synthetic_data=True
)

# 또는 사용 가능한 데이터셋 확인
import os
from pathlib import Path
dataset_dirs = [d for d in os.listdir('datasets/') if d.startswith('Pohang_')]
print(f"사용 가능한 데이터셋: {dataset_dirs}")

# 샘플 데이터 확인
sample_files = list(Path('[샘플]데이터/').glob('*'))
print(f"샘플 파일들: {[f.name for f in sample_files]}")
```

### 성능 최적화 팁

#### 1. 처리 속도 향상
```python
# 멀티프로세싱 사용
from multiprocessing import Pool
import numpy as np

def extract_features_parallel(images):
    with Pool(processes=4) as pool:
        results = pool.map(extract_single_feature, images)
    return results

# 특징 캐싱
import pickle
import os

def cached_feature_extraction(image, cache_dir='cache/features/'):
    os.makedirs(cache_dir, exist_ok=True)
    
    # 이미지 해시로 캐시 키 생성
    image_hash = str(hash(image.tobytes()))
    cache_file = os.path.join(cache_dir, f'features_{image_hash}.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 특징 추출
    features = extractor.extract_features(image)
    
    # 캐시 저장
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    
    return features
```

#### 2. 메모리 효율성
```python
# 제너레이터 사용으로 메모리 절약
def data_generator(file_paths):
    for file_path in file_paths:
        data = load_data(file_path)
        yield process_data(data)
        del data  # 명시적 메모리 해제

# 메모리 사용량 모니터링
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"현재 메모리 사용량: {memory_usage:.1f} MB")

# 주기적으로 호출
monitor_memory()
```

---

## 🚀 고급 활용 사례

### 1. 실시간 기뢰 탐지 시스템
```python
import time
from threading import Thread
import queue

class RealTimeDetector:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
    def start_real_time_processing(self):
        self.is_running = True
        
        # 데이터 수신 스레드
        data_thread = Thread(target=self.data_receiver)
        data_thread.start()
        
        # 처리 스레드
        processing_thread = Thread(target=self.data_processor)
        processing_thread.start()
        
        print("🚀 실시간 처리 시작")
        
    def data_receiver(self):
        # 소나 데이터 수신 시뮬레이션
        while self.is_running:
            # 실제로는 소나 장비에서 데이터 수신
            simulated_data = generate_test_data()
            self.data_queue.put(simulated_data)
            time.sleep(0.1)  # 100ms 간격
    
    def data_processor(self):
        while self.is_running:
            if not self.data_queue.empty():
                data = self.data_queue.get()
                
                # 빠른 특징 추출 및 분류
                features = self.fast_feature_extraction(data)
                prediction = self.quick_classify(features)
                
                self.result_queue.put({
                    'timestamp': time.time(),
                    'prediction': prediction,
                    'confidence': 0.85  # 임시값
                })
                
                if prediction == 'mine':
                    print("⚠️  기뢰 탐지!")

# 사용법
detector = RealTimeDetector()
detector.start_real_time_processing()
```

### 2. 자동 보고서 생성
```python
from datetime import datetime
import matplotlib.pyplot as plt
import json

class AutoReportGenerator:
    def __init__(self, analysis_results_dir):
        self.results_dir = analysis_results_dir
        
    def generate_comprehensive_report(self):
        report_html = f"""
        <html>
        <head>
            <title>기뢰탐지 시스템 분석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background: #f8f9fa; border-radius: 5px; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌊 기뢰탐지 시스템 분석 보고서</h1>
                <p>생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            {self.generate_performance_section()}
            {self.generate_feature_analysis_section()}
            {self.generate_recommendations_section()}
        </body>
        </html>
        """
        
        # 보고서 저장
        report_path = f"{self.results_dir}/auto_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"📄 자동 보고서 생성 완료: {report_path}")
        return report_path
    
    def generate_performance_section(self):
        # 성능 데이터 로드 및 HTML 생성
        perf_data = self.load_performance_data()
        
        html = "<div class='section'><h2>📊 성능 분석</h2>"
        for extractor, metrics in perf_data.items():
            accuracy = metrics.get('accuracy', 0)
            status_class = 'good' if accuracy > 0.8 else 'warning' if accuracy > 0.7 else 'error'
            
            html += f"""
            <div class='metric {status_class}'>
                <strong>{extractor.upper()}</strong><br>
                정확도: {accuracy:.1%}
            </div>
            """
        html += "</div>"
        return html

# 사용법
report_gen = AutoReportGenerator('data/results/pipeline_output/')
report_gen.generate_comprehensive_report()
```

### 3. 웹 기반 분석 인터페이스
```python
# Flask 웹 인터페이스 예시
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('analysis_interface.html')

@app.route('/upload_xtf', methods=['POST'])
def upload_xtf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename.endswith('.xtf'):
        # XTF 파일 저장 및 처리
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # 백그라운드에서 분석 시작
        analysis_id = start_background_analysis(filepath)
        
        return jsonify({
            'status': 'success',
            'analysis_id': analysis_id,
            'message': '분석이 시작되었습니다.'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analysis_status/<analysis_id>')
def get_analysis_status(analysis_id):
    status = check_analysis_status(analysis_id)
    return jsonify(status)

# 실행
# python web_interface.py
# http://localhost:5000 접속
```

---

## 📝 체크리스트

### 시작 전 확인사항
- [ ] Python 3.9+ 설치됨
- [ ] 필수 패키지 설치됨 (`pip install -r requirements.txt`)
- [ ] PYTHONPATH 설정됨
- [ ] 데이터 디렉토리 존재 (`data/` 폴더)
- [ ] 충분한 디스크 공간 (최소 2GB)

### 실행 중 확인사항
- [ ] 메모리 사용량 모니터링 (8GB 이하 권장)
- [ ] 로그 파일 생성 확인
- [ ] 중간 결과 파일 생성 확인
- [ ] 오류 메시지 없음

### 완료 후 확인사항
- [ ] 모든 결과 파일 생성됨
- [ ] 성능 지표가 합리적 범위 (50% 이상)
- [ ] 시각화 차트 생성됨
- [ ] 최종 보고서 완성됨

---

## 📞 지원 및 문의

### 문제 해결 순서
1. **로그 파일 확인**: `data/results/pipeline_output/logs/`
2. **테스트 실행**: `python test_pipeline_modules.py --mode quick`
3. **설정 확인**: 파라미터가 올바른지 점검
4. **문서 참조**: `docs/analysis_methodology.md`
5. **GitHub 이슈**: [Issues 페이지](https://github.com/your-repo/issues)

### 자주 묻는 질문 (FAQ)

**Q: XTF 파일 없이 테스트할 수 있나요?**  
A: 네, `input_xtf_path=None`으로 설정하면 더미 데이터로 테스트할 수 있습니다.

**Q: 메모리가 부족합니다.**  
A: `patch_size`를 32로 줄이고 `use_synthetic_data=False`로 설정해보세요.

**Q: OpenCV 설치가 안됩니다.**  
A: 시스템이 자동으로 대안 구현을 사용하므로 그대로 진행하셔도 됩니다.

**Q: 결과 해석이 어렵습니다.**  
A: `docs/analysis_methodology.md`의 "성능 평가 체계" 섹션을 참조하세요.

**Q: 커스텀 특징을 추가하고 싶습니다.**  
A: 고급 사용자 가이드의 "새로운 특징 추출기 구현" 부분을 참조하세요.

---

**버전 히스토리**:
- v2.0 (2025-09-09): 전체 파이프라인 구현 및 문서화
- v1.5 (2025-09-08): 모의데이터 및 비교 분석 추가
- v1.0 (2025-09-07): 기본 특징 추출 및 분류 구현

**라이선스**: Research & Educational Use Only  
**개발팀**: ECMiner 기뢰탐지시스템 개발팀