# Mine Labeling Project 세션 기록

## 세션 개요

**날짜**: 2025-11-04
**프로젝트**: Mine Labeling Project - 사이드스캔 소나 기뢰 탐지 레이블링 도구
**목적**: NPY 강도 데이터를 BMP로 변환하고 인터랙티브 레이블링 도구 구현

## 주요 구현 내용

### 1. JSON → XML 변환기
- **파일**: `scripts/convert_json_to_xml.py`
- **기능**: `flipped_mapped_annotations.json`을 PASCAL VOC XML 형식으로 변환
- **출력**: `flipped_npy_annotations.xml` (NPY 좌표 6400×5137 기준)
- **상태**: ✅ 완료

### 2. NPY → BMP 변환기
- **파일**: `src/mine_labeling/visualization/npy_to_bmp_converter.py`
- **핵심 기능**:
  - 방향 보존 (`origin='upper'`, 플립 없음)
  - CLAHE 대비 향상 (clip_limit=2.0)
  - 가로 해상도 축소: 6400→1024 (Klein 6.25배), 6832→1024 (Edgetech 6.67배)
  - 세로 해상도 유지 (1:1)
  - 양방향 좌표 변환: `bmp_to_npy_coordinates()`, `npy_to_bmp_coordinates()`
- **상태**: ✅ 완료 및 테스트 통과

### 3. 인터랙티브 레이블링 도구
- **파일**: `src/mine_labeling/visualization/interactive_labeling.py`
- **기능**:
  - Matplotlib 기반 GUI
  - 마우스 이벤트: 왼쪽 드래그 (그리기), 오른쪽 클릭 (삭제)
  - 키보드 단축키: 's' (저장), 'q' (종료), 'Esc' (취소)
  - XML (PASCAL VOC) + JSON 형식으로 저장
- **상태**: ✅ 코드 완료 (수동 테스트 필요)

### 4. 통합 워크플로우
- **파일**: `scripts/interactive_labeling_workflow.py`
- **3단계 워크플로우**:
  1. NPY → BMP 변환
  2. 인터랙티브 레이블링 (수동)
  3. BMP 좌표 → NPY 좌표 변환
- **상태**: ✅ 완료

### 5. 매핑 검증 시각화
- **파일**: `scripts/test_mapping_visualization.py`
- **기능**: BMP와 NPY 이미지를 나란히 표시하며 바운딩 박스 매핑 검증
- **상태**: ✅ 완료 및 검증 완료

## 테스트 결과

### Test 1: XTF 강도 데이터 추출
- **입력**: `Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04.xtf`
- **출력**: Combined intensity (7974, 6832) float32
- **저장 위치**: `data/test_extraction/`
- **상태**: ✅ PASS

### Test 2: NPY → BMP 변환
- **입력**: (7974, 6832) NPY
- **출력**: (7974, 1024, 3) BMP
- **스케일 팩터**: 6.67배
- **좌표 정확도**: ±1 픽셀 오차
- **저장 위치**: `data/test_bmp/edgetech_visualization.bmp`
- **상태**: ✅ PASS

### Test 3: 좌표 매핑 검증
- **검증 데이터**: Klein 3900 (5137, 6400) - 5개 기뢰
- **스케일 팩터**: 6.25배
- **시각화**: `data/test_visualization/mapping_comparison.png`
- **상태**: ✅ PASS (모든 좌표 정확)

## 기술적 주요 결정사항

### 방향 보존
- **문제**: 이전 flipped 데이터는 Y축 플립됨, 새 데이터는 원본 방향 유지 필요
- **해결**: `matplotlib.imshow(..., origin='upper')` 사용, Y축 변환 없음
- **검증**: 시각적 비교로 방향 정확성 확인

### 좌표 변환 공식
```python
# BMP → NPY
X_npy = X_bmp × scale_factor  # scale_factor = NPY_width / BMP_width
Y_npy = Y_bmp  # 1:1 매핑

# Klein 3900: 6400 / 1024 = 6.25배
# Edgetech 4205: 6832 / 1024 = 6.67배
```

### CLAHE 대비 향상
- **파라미터**: `clip_limit=2.0`, `tile_grid_size=(8,8)`
- **목적**: 육안으로 기뢰 식별 가능하도록 대비 향상
- **결과**: BMP 파일에서 기뢰 윤곽 명확히 보임

## 해결된 오류

### Error 1: Module Import AttributeError
- **증상**: `AttributeError: module 'mine_labeling.sampling' has no attribute 'MineSampler'`
- **원인**: `__init__.py`에 `__all__`만 있고 실제 import 없음
- **해결**: `sampling/__init__.py`와 `augmentation/__init__.py`에 명시적 import 추가

### Error 2: Type Hint NameError
- **증상**: `NameError: name 'A' is not defined` (augmentor.py:65)
- **원인**: albumentations 체크 전에 `List[A.Compose]` 타입 힌트 사용
- **해결**: 반환 타입을 `List`로 변경

### Error 3: XTFIntensityExtractor Arguments
- **증상**: `TypeError: __init__() got an unexpected keyword argument 'xtf_file_path'`
- **원인**: 잘못된 메서드 시그니처 사용
- **해결**: `extract_intensity_data(xtf_path=...)`로 수정

## 생성된 문서

1. **빠른시작 가이드** (`docs/빠른시작_가이드.md`)
   - 전체 워크플로우 한글 설명
   - 설치 방법, 사용 예제, PyTorch 데이터셋 통합

2. **테스트 결과 보고서** (`docs/테스트_결과_보고서.md`)
   - 3개 자동화 테스트 결과
   - 성능 메트릭, 파일 인벤토리
   - 다음 단계 권장사항

## 프로젝트 구조

```
mine_labeling_project/
├── src/
│   └── mine_labeling/
│       ├── visualization/
│       │   ├── npy_to_bmp_converter.py      # NPY↔BMP 변환기
│       │   └── interactive_labeling.py      # 인터랙티브 GUI 도구
│       ├── sampling/                        # 기뢰/배경 샘플링
│       └── augmentation/                    # 데이터 증강
├── scripts/
│   ├── convert_json_to_xml.py              # JSON→XML 변환기
│   ├── interactive_labeling_workflow.py     # 통합 워크플로우
│   ├── test_npy_to_bmp.py                  # BMP 변환 테스트
│   └── test_mapping_visualization.py        # 매핑 검증
├── data/
│   ├── flipped_20251104/                    # 기존 레이블 데이터 (25개 기뢰)
│   ├── test_extraction/                     # XTF 추출 결과
│   ├── test_bmp/                           # BMP 변환 결과
│   └── test_visualization/                  # 검증 시각화
└── docs/
    ├── 빠른시작_가이드.md
    └── 테스트_결과_보고서.md
```

## 다음 세션 작업사항

### 필수 (수동 실행 필요)
1. **인터랙티브 레이블링 도구 테스트**
   ```bash
   python scripts/interactive_labeling_workflow.py \
     data/test_extraction/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_combined_intensity.npy
   ```
   - GUI 환경에서 바운딩 박스 그리기 테스트
   - 저장된 XML/JSON 파일 검증

2. **전체 파이프라인 엔드투엔드 테스트**
   - 새로운 XTF 파일로 전체 워크플로우 실행
   - 사용자가 직접 바운딩 박스 그리고 좌표 변환 확인

### 선택적
1. **Albumentations 설치**
   ```bash
   pip install albumentations
   ```
   - 완전한 데이터 증강 기능 활성화

2. **PyTorch 데이터셋 통합 테스트**
   - `빠른시작_가이드.md`의 PyTorch 예제 코드 실행
   - 실제 모델 학습 파이프라인 검증

## 중요 참고사항

### 데이터 형식
- **XTF 원본**: Klein 3900 (900kHz), Edgetech 4205 (800kHz)
- **NPY 해상도**: Klein (H×6400), Edgetech (H×6832)
- **BMP 해상도**: (H×1024) - 가로만 축소
- **좌표 시스템**: 왼쪽 상단 (0,0), 오른쪽 하단 (width, height)

### 플립 이슈 주의
- **기존 flipped 데이터**: Y축 플립됨 (역사적 데이터)
- **새 데이터 처리**: 플립 없음, 원본 방향 보존
- **변환 시 확인사항**: `origin='upper'`, Y좌표 변환 없음

### 좌표 변환 정확도
- **X축**: 스케일 팩터 적용 (6.25배 또는 6.67배)
- **Y축**: 1:1 매핑 (변환 없음)
- **허용 오차**: ±1-6 픽셀 (정수 변환으로 인한 불가피한 오차)
- **검증 방법**: 시각적 비교 및 라운드트립 테스트

## 사용 예제

### 기본 워크플로우
```python
from mine_labeling.visualization import NpyToBmpConverter, InteractiveBBoxLabeler
import numpy as np

# 1. NPY 데이터 로드
npy_data = np.load('intensity_data.npy')

# 2. BMP 변환
converter = NpyToBmpConverter(target_width=1024, apply_clahe=True)
bmp_image = converter.convert_to_bmp(npy_data, 'output.bmp')

# 3. 인터랙티브 레이블링 (GUI)
labeler = InteractiveBBoxLabeler('output.bmp', 'annotations/')
labeler.show()

# 4. 좌표 변환
bmp_annotations = labeler.get_annotations()
for bbox in bmp_annotations:
    npy_coords = converter.bmp_to_npy_coordinates(bbox, npy_data.shape[1])
    print(f"BMP: {bbox} → NPY: {npy_coords}")
```

### PyTorch 데이터셋 통합
```python
from mine_labeling.sampling import MineSampler, BackgroundSampler
from mine_labeling.augmentation import Augmentor

# 샘플 추출 및 증강 (빠른시작_가이드.md 참조)
mine_sampler = MineSampler(intensity_data, annotations)
background_sampler = BackgroundSampler(intensity_data, annotations)
augmentor = Augmentor()

# PyTorch Dataset 클래스 구현
```

## 참고 문서

1. **빠른시작 가이드**: `mine_labeling_project/docs/빠른시작_가이드.md`
2. **테스트 결과**: `mine_labeling_project/docs/테스트_결과_보고서.md`
3. **기존 XML 예제**: `data/flipped_20251104/Pohang_Eardo_1_Klein3900_900_050_20241011171100_001_04.xml`
4. **기존 JSON 예제**: `data/flipped_20251104/flipped_mapped_annotations.json`

## 프로젝트 상태

- **코드 완성도**: ✅ 100% (모든 모듈 구현 완료)
- **자동화 테스트**: ✅ 3/3 통과
- **수동 테스트**: ⏳ 대기 중 (GUI 환경 필요)
- **문서화**: ✅ 완료 (한글)
- **독립 실행 가능**: 확인 필요 (다음 작업)
