# 설치 가이드

## 시스템 요구사항

### 필수 요구사항

- **Python**: 3.8 이상 (3.9 권장)
- **메모리**: 8GB 이상 RAM
- **디스크**: 10GB 이상 여유 공간
- **운영체제**: macOS, Linux, Windows

### 선택적 요구사항

- **GPU**: 데이터 증강 시 속도 향상 (선택사항)
- **conda**: 가상환경 관리 권장

---

## 설치 방법

### 방법 1: pip 설치 (권장)

```bash
# 1. 프로젝트 디렉토리로 이동
cd mine_labeling_project

# 2. 가상환경 생성 (선택사항, 권장)
python -m venv venv

# 3. 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. 의존성 설치
pip install -r requirements.txt

# 5. 설치 확인
python -c "import numpy, PIL, matplotlib, yaml; print('✓ 설치 완료!')"
```

### 방법 2: conda 설치

```bash
# 1. conda 환경 생성
conda create -n mine_labeling python=3.9

# 2. 환경 활성화
conda activate mine_labeling

# 3. 기본 패키지 설치
conda install numpy scipy matplotlib pillow pandas pyyaml

# 4. 나머지 패키지 설치
pip install -r requirements.txt

# 5. 설치 확인
python -c "import numpy, PIL, matplotlib; print('✓ 설치 완료!')"
```

---

## 의존성 패키지 상세

### Core Scientific Computing

```bash
numpy>=1.21.0       # 다차원 배열 처리
scipy>=1.7.0        # 과학 계산
```

### Image Processing

```bash
Pillow>=9.0.0       # 이미지 로드/저장
opencv-python>=4.5.0  # 이미지 처리
scikit-image>=0.19.0  # 이미지 분석
```

### XTF Processing

```bash
pyxtf>=1.0.0        # XTF 파일 파싱
```

**설치 시 주의사항**:
- `pyxtf`가 설치되지 않으면 XTF 추출 기능을 사용할 수 없습니다
- 설치 실패 시: `pip install pyxtf --no-cache-dir`

### Visualization

```bash
matplotlib>=3.4.0   # 시각화
seaborn>=0.11.0     # 통계 시각화
```

### Configuration & Utilities

```bash
PyYAML>=6.0         # YAML 설정 파일
click>=8.0.0        # CLI 도구
tqdm>=4.62.0        # 진행바
lxml>=4.6.0         # XML 파싱
```

### Data Augmentation

```bash
albumentations>=1.1.0  # 이미지 증강
imgaug>=0.4.0         # 추가 증강 기법
```

---

## 설치 확인

### 1. Python 버전 확인

```bash
python --version
# 출력 예: Python 3.9.7
```

### 2. 패키지 설치 확인

```bash
python -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import PIL
print(f'Pillow: {PIL.__version__}')

import matplotlib
print(f'Matplotlib: {matplotlib.__version__}')

import yaml
print(f'PyYAML: {yaml.__version__}')

try:
    import pyxtf
    print(f'pyxtf: available')
except ImportError:
    print('pyxtf: NOT INSTALLED (XTF 추출 불가)')

print('\n✓ 모든 핵심 패키지 설치 완료!')
"
```

### 3. 프로젝트 모듈 확인

```bash
cd mine_labeling_project

python -c "
import sys
sys.path.insert(0, 'src')

from mine_labeling.extractors.xtf_intensity_extractor import XTFIntensityExtractor
print('✓ XTF Extractor 로드 성공')

print('\n✓ 프로젝트 모듈 설치 완료!')
"
```

---

## 문제 해결

### pyxtf 설치 실패

**증상**: `pip install pyxtf` 실패

**해결 방법**:
```bash
# 방법 1: 캐시 없이 설치
pip install pyxtf --no-cache-dir

# 방법 2: 소스에서 설치
pip install git+https://github.com/oysstu/pyxtf.git

# 방법 3: conda 사용
conda install -c conda-forge pyxtf
```

### OpenCV 설치 문제

**증상**: `import cv2` 실패

**해결 방법**:
```bash
# OpenCV headless 버전 설치
pip uninstall opencv-python
pip install opencv-python-headless
```

### 메모리 부족 오류

**증상**: "MemoryError" 발생

**해결 방법**:
```python
# config/default_config.yaml에서 조정
extraction:
  chunk_size: 1000  # 기본값 5000에서 줄임

labeling:
  batch_size: 10    # 기본값 50에서 줄임
```

### macOS에서 matplotlib 오류

**증상**: "RuntimeError: Python is not installed as a framework"

**해결 방법**:
```bash
# matplotlib backend 변경
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc

# 또는 코드에서
import matplotlib
matplotlib.use('TkAgg')
```

---

## 개발 환경 설정 (선택사항)

### 코드 포매팅 도구

```bash
pip install black flake8 pytest
```

### VS Code 설정

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

---

## 다음 단계

설치가 완료되었으면 [USAGE.md](USAGE.md)를 참조하여 프로젝트를 사용하세요.

---

## 지원

설치 중 문제가 발생하면:

1. [문제 해결](#문제-해결) 섹션 확인
2. `requirements.txt` 버전 확인
3. Python 버전 확인 (3.8+)
4. 프로젝트 관리자에게 문의

---

**최종 업데이트**: 2025-11-04
