# .npy 기반 기뢰 라벨링 계획서

## 현황 분석

### 파일 구조
```
BMP 원본 (XTF 추출):        1024 × 7974  (W×H)
Annotation (기뢰 표시):      1024 × 3862  (BMP 하단 3862행, 180° 회전)
npy intensity:              6832 × 200   (W×H, 다차원 배열)
기존 라벨 (mine_labels.npy): 200 × 6832  (H×W)
```

### 좌표 관계
- **Annotation**: BMP 하단 3862행을 180° 회전한 것
- **npy**: BMP 전체(7974행)를 200 ping으로 압축
- **해상도 차이**:
  - 너비: npy/Ann = 6832/1024 = 6.67배
  - 높이: npy/Ann = 200/3862 = 0.052배 (약 1/19)

---

## 작업 계획

### Phase 1: 좌표 변환 함수 정밀 검증 ✓

**목표**: Annotation → npy 좌표 변환의 정확성 검증

**작업**:
1. ✓ 이미지 회전/반전 관계 확인 (완료: BMP 하단, 180° 회전)
2. 좌표 변환 수식 정립
3. 샘플 포인트로 변환 정확도 검증

**변환 수식**:
```python
# Annotation → BMP 좌표
ann_x, ann_y → (1024×3862 좌표계)

# 1. 180° 역회전
bmp_x = 1024 - ann_x
bmp_y = 3862 - ann_y

# 2. BMP 하단 위치로 이동
bmp_y_full = 7974 - 3862 + bmp_y = 4112 + bmp_y

# 3. BMP → npy 변환
npy_y = int(bmp_y_full * 200 / 7974)  # ping index (0-199)
npy_x = int(bmp_x * 6832 / 1024)       # sample index (0-6831)
```

---

### Phase 2: npy 배열 라벨링 적용

**목표**: 검증된 좌표로 npy 배열에 정확한 라벨 생성

**작업**:
1. annotation_bboxes_manual.json의 25개 기뢰 좌표 로드
2. 각 좌표를 npy 좌표계로 변환
3. Bounding box 크기 결정 (현재 20×20 검토 필요)
4. 새로운 라벨 배열 생성 (200×6832)
5. GPS 좌표 매핑 재확인

**출력**:
- `mine_labels_verified.npy`: 검증된 라벨 배열
- `mine_positions_npy.csv`: npy 좌표계의 기뢰 위치

---

### Phase 3: 역변환 검증

**목표**: npy 라벨을 원본 이미지 좌표로 역변환하여 정확도 검증

**작업**:
1. npy 라벨 → BMP 좌표 역변환
2. BMP 좌표 → Annotation 좌표 역변환
3. 원본 annotation과 비교
4. 오차 분석 (픽셀 단위)

**역변환 수식**:
```python
# npy → BMP
bmp_y_full = int(npy_y * 7974 / 200)
bmp_x_full = int(npy_x * 1024 / 6832)

# BMP → Annotation (하단 3862행만)
if 4112 <= bmp_y_full < 7974:  # 하단 영역
    bmp_y_crop = bmp_y_full - 4112
    # 180° 역회전
    ann_x = 1024 - bmp_x_full
    ann_y = 3862 - bmp_y_crop
else:
    # 범위 밖
    return None
```

**검증 기준**:
- 픽셀 오차 < 5픽셀: 우수
- 픽셀 오차 < 10픽셀: 양호
- 픽셀 오차 >= 10픽셀: 재검토 필요

---

### Phase 4: 시각화 및 검증

**목표**: 라벨링 결과를 원본 이미지에 오버레이하여 육안 검증

**작업**:
1. npy 라벨을 BMP 크기로 업스케일링
2. BMP 이미지에 라벨 오버레이
3. Annotation 원본과 side-by-side 비교
4. 각 기뢰별 위치 정확도 시각화

**출력 이미지**:
- `npy_label_verification.png`: npy 라벨 + intensity 오버레이
- `bmp_label_overlay.png`: BMP + 업스케일 라벨
- `annotation_comparison.png`: 원본 vs 재구성 비교
- `position_accuracy_heatmap.png`: 25개 기뢰 위치 오차 히트맵

---

### Phase 5: 최종 검증 리포트

**목표**: 라벨링 정확도 정량적 평가 및 문서화

**평가 항목**:
1. **위치 정확도**:
   - 평균 픽셀 오차
   - 최대 픽셀 오차
   - 오차 분포 (히스토그램)

2. **라벨 커버리지**:
   - 라벨된 픽셀 수
   - 전체 대비 비율
   - 기뢰당 평균 픽셀 수

3. **좌표 변환 정확도**:
   - Forward 변환 오차
   - Backward 변환 오차
   - Round-trip 오차

4. **GPS 정확도**:
   - npy 좌표 → GPS 매핑 검증
   - 투하 좌표와 비교
   - 상관계수 재확인

**출력**:
- `NPY_LABELING_VERIFICATION_REPORT.md`: 종합 검증 리포트
- `labeling_accuracy_stats.csv`: 정량적 통계
- `labeling_quality_score.json`: 품질 점수

---

## 성공 기준

### 필수 조건
- ✓ 25개 기뢰 모두 npy 좌표로 변환
- ✓ 역변환 시 원본 annotation과 평균 오차 < 10픽셀
- ✓ 육안 검증으로 위치 일치 확인

### 우수 조건
- 평균 오차 < 5픽셀
- 모든 기뢰 GPS 좌표 정확히 매핑
- 자동화 스크립트로 재현 가능

---

## 잠재적 문제 및 대응

### 문제 1: 좌표 변환 오차 누적
**원인**: 여러 단계 변환 과정에서 반올림 오차
**대응**:
- 부동소수점 연산 사용
- 최종 단계에서만 정수 변환
- 역변환으로 오차 검증

### 문제 2: Bounding box 크기 부적절
**원인**: npy 해상도에서 20×20이 너무 크거나 작음
**대응**:
- npy 해상도 기준으로 bbox 크기 재계산
- Annotation에서 평균 bbox 크기 측정
- 해상도 비율 적용

### 문제 3: Annotation이 BMP 일부만 포함
**원인**: Annotation은 하단 3862행만, 나머지 4112행은 없음
**대응**:
- 범위 체크 추가
- BMP 전체 대상 기뢰가 있는지 확인
- 필요시 전체 BMP 재작업

---

## 다음 단계

승인 후 다음 순서로 진행:
1. Phase 2: 좌표 변환 함수 작성 및 테스트
2. Phase 3: npy 라벨 생성
3. Phase 4: 역변환 검증
4. Phase 5: 시각화
5. Phase 6: 최종 리포트

**예상 소요 시간**: 약 1-2시간
**출력 파일 수**: 약 10개 (npy, csv, png, md)
