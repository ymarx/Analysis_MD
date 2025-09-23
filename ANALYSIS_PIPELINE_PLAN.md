# 분석 파이프라인 현대화 계획

**계획일시**: 2025-09-23 18:00:00
**목적**: 기존 작동하는 코드 보전하면서 9단계 분석 파이프라인 최신화

## 🎯 **전체 파이프라인 개요**

### 분석 단계별 현황
```
1. XTF 메타데이터/강도 데이터 추출      ✅ 구현완료 (검증 필요)
2. 위경도-픽셀 매핑 및 레이블링         ⚠️ GPS 데이터 대기 중
3. 데이터 증강 (회전, blur 등)          ✅ 구현완료 (검증 필요)
4. 특징 추출 (여러 기법)               ✅ 구현완료 (검증 필요)
5. 기뢰 분류 (여러 기법)               ✅ 구현완료 (검증 필요)
6. Original → Simulation 적용 테스트    📋 계획 수립 필요
7. 혼합 데이터 훈련 검토              📋 계획 수립 필요
8. 지형 영향 분석                    📋 계획 수립 필요
9. 지형이 기물 특징/분류에 미치는 영향   📋 계획 수립 필요
```

## 📊 **현재 모듈 현황 분석**

### ✅ 확인된 작동 모듈
1. **XTF Reader** (`src/data_processing/xtf_reader.py`)
   - 메타데이터 추출: ✅ 검증 완료
   - 강도 데이터 추출: ✅ 200 pings 처리 성공
   - 좌표 수정 기능: ✅ 자동 보정 작동

2. **XTF Intensity Extractor** (`src/data_processing/xtf_intensity_extractor.py`)
   - Intensity matrix: ✅ (200, 6832) 형태 출력
   - Port/Starboard 채널: ✅ 분리 처리

### 🔍 검증 필요 모듈
1. **Data Augmentation** (`src/data_augmentation/augmentation_engine.py`)
2. **Feature Extraction** (`src/feature_extraction/`)
3. **Classification** (`src/models/`)
4. **Coordinate Mapper** (`src/data_processing/coordinate_mapper.py`)

## 🛠️ **단계별 실행 계획**

### Phase 1: 기존 모듈 검증 (1-5단계)

#### 1단계: XTF 데이터 추출 검증 ✅
**목표**: 메타데이터와 강도 데이터 추출 기능 완전 검증
- **현황**: 이미 검증 완료 (200 pings, matrix (200, 6832))
- **작업**: 추가 XTF 파일로 안정성 테스트

#### 2단계: 위경도-픽셀 매핑 ⚠️ GPS 데이터 대기
**목표**: 기물 위치 레이블링 시스템 준비
- **현황**: Coordinate Mapper 모듈 존재하나 GPS 데이터 불일치
- **작업**:
  - 모듈 구조 검증 및 최신화
  - GPS 데이터 수령 시 즉시 적용 가능하도록 준비
  - 테스트용 더미 데이터로 로직 검증

#### 3단계: 데이터 증강 검증
**목표**: 회전, blur 등 25개 기물 데이터 증강 시스템 검증
- **작업**:
  - Augmentation Engine 작동 테스트
  - 회전, blur, 노이즈 추가 등 기능 검증
  - 증강 데이터 품질 확인

#### 4단계: 특징 추출 모듈 검증
**목표**: 여러 기법의 특징 추출 및 앙상블 시스템 검증
- **모듈들**:
  - HOG, LBP, Gabor, SFS extractors
  - Feature Ensemble 시스템
- **작업**: 각 추출기 개별 테스트 → 앙상블 통합 테스트

#### 5단계: 기뢰 분류 모듈 검증
**목표**: CNN 등 분류 모델 작동 확인
- **모듈**: CNN Detector, Mine Classifier
- **작업**: 모델 로딩, 훈련 파이프라인 검증

### Phase 2: 통합 분석 시스템 구축 (6-9단계)

#### 6단계: Original → Simulation 적용 계획
**목표**: Original 데이터 훈련 모델을 Simulation 데이터에 적용
- **검증 항목**:
  - 모델 일반화 성능
  - Simulation 데이터 활용 정합성
  - 성능 지표 비교 분석

#### 7단계: 혼합 데이터 훈련 검토
**목표**: Original + Simulation 혼합 데이터 훈련 효과 분석
- **검증 항목**:
  - 혼합 데이터 훈련 시 성능 변화
  - 데이터 불균형 해결 방안
  - 최적 혼합 비율 도출

#### 8단계: 지형 영향 분석
**목표**: 여러 지형 vs 단일 지형 영향 분석
- **분석 항목**:
  - Original: 여러 지형 기물 데이터
  - Simulation: 단일 지형 기물 데이터
  - 지형별 특징 추출 차이 분석

#### 9단계: 지형-기물 특징 상관관계 분석
**목표**: 지형이 기물 특징과 분류에 미치는 영향 정량화
- **분석 방법**:
  - 지형별 특징 벡터 클러스터링
  - 분류 성능 지형별 비교
  - 지형 불변 특징 도출

## 🔧 **즉시 실행 작업**

### 우선순위 1: 모듈 상태 진단
```python
# 검증할 모듈들
modules_to_check = [
    'src/data_augmentation/augmentation_engine.py',
    'src/feature_extraction/feature_ensemble.py',
    'src/feature_extraction/hog_extractor.py',
    'src/feature_extraction/lbp_extractor.py',
    'src/feature_extraction/gabor_extractor.py',
    'src/feature_extraction/sfs_extractor.py',
    'src/models/cnn_detector.py',
    'pipeline/modules/mine_classifier.py'
]
```

### 우선순위 2: GPS 데이터 대기 중 준비 작업
- Coordinate Mapper 모듈 최신화
- 더미 GPS 데이터로 매핑 로직 검증
- 레이블링 시스템 구조 점검

### 우선순위 3: 통합 파이프라인 구축
- Main Pipeline과 Unified Pipeline 연동
- 단계별 데이터 흐름 검증
- 에러 처리 및 로깅 시스템 강화

## ⚠️ **제약사항 및 고려사항**

### GPS 데이터 이슈
- **현재 상황**: Location_MDGPS와 Original XTF 좌표 불일치 (55km 차이)
- **대응 방안**:
  - 정확한 GPS 데이터 수령 시까지 2단계 보류
  - 모듈 구조는 미리 준비하여 즉시 적용 가능하도록 함
  - 테스트용 더미 데이터로 로직 검증

### 기존 코드 보전
- **원칙**: 작동 확인된 코드는 절대 수정하지 않음
- **방법**: 검증 → 필요시 래퍼 추가 → 점진적 개선
- **백업**: 모든 변경 전 현재 상태 보관

### 데이터 제약
- **기물 수**: 25개 (적은 양)
- **해결책**: 데이터 증강으로 충분한 훈련 데이터 확보
- **검증**: 증강 데이터 품질과 실제 성능 개선 효과 측정

## 📋 **실행 체크리스트**

### 즉시 실행 (오늘)
- [ ] 전체 모듈 import 및 초기화 테스트
- [ ] 1단계: XTF 추가 파일 처리 테스트
- [ ] 3단계: Data Augmentation 기능 검증
- [ ] 4단계: Feature Extraction 모듈별 테스트

### 단기 (1-2일)
- [ ] 5단계: Classification 모델 검증
- [ ] 2단계: Coordinate Mapper 구조 최신화 (GPS 대기)
- [ ] 통합 파이프라인 기본 구조 구축

### 중기 (GPS 데이터 수령 후)
- [ ] 2단계: 실제 레이블링 시스템 적용
- [ ] 6-7단계: Original/Simulation 통합 분석
- [ ] 8-9단계: 지형 영향 분석

---

**이 계획은 기존 작동 코드를 보전하면서 체계적으로 파이프라인을 현대화하는 로드맵입니다.**