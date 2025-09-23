# 이미지 비교 분석 보고서
**생성일시**: 2025-09-22 15:23:14
**분석자**: YMARX

## 🎯 **분석 목적**
PH_annotation.bmp와 XTF 이미지가 같은 기뢰 위치를 나타내는지 확인
180도 회전, 좌우 반전 등 다양한 변환을 적용하여 형상 유사성 검증

## 📁 **분석 대상 이미지**

### Annotation Image
- **파일 경로**: `datasets/PH_annotation.bmp`
- **이미지 크기**: (3862, 1024, 3)
- **파일 크기**: 11.3 MB

### Xtf Image
- **파일 경로**: `datasets/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04/original/Pohang_Eardo_1_Edgetech4205_800_050_20241012110900_001_04_IMG_00.BMP`
- **이미지 크기**: (7974, 1024, 3)
- **파일 크기**: 23.4 MB

## 🔄 **변환별 유사도 분석**

| 변환 타입 | MSE ↓ | NCC ↑ | Hist Corr ↑ | SSIM ↑ | 종합 점수 |
|-----------|-------|-------|-------------|---------|----------|
| 🥇 Flip Horizontal | 0.0235 | 0.330 | 0.890 | 0.443 | 2.639 |
|  Rotate 180 | 0.0237 | 0.325 | 0.890 | 0.443 | 2.635 |
|  Original | 0.0239 | 0.317 | 0.890 | 0.443 | 2.625 |
|  Flip Vertical | 0.0241 | 0.312 | 0.890 | 0.444 | 2.621 |
|  Rotate Flip | 0.0241 | 0.312 | 0.890 | 0.444 | 2.621 |
|  Rotate 90 | 0.0327 | 0.001 | 0.980 | 0.237 | 2.186 |
|  Rotate 270 | 0.0328 | -0.003 | 0.980 | 0.239 | 2.184 |

## 📊 **분석 결과**

### ✅ **최적 변환**: `Flip Horizontal`

**유사도 지표**:
- MSE (Mean Squared Error): 0.0235 (낮을수록 유사)
- NCC (Normalized Cross-Correlation): 0.330 (높을수록 유사)
- Histogram Correlation: 0.890 (높을수록 유사)
- SSIM (Structural Similarity): 0.443 (높을수록 유사)
- **종합 점수**: 2.639

**형상 특징 비교**:
- 윤곽선 개수: Annotation(8185) vs Transformed(104864)
- 총 면적: Annotation(2440334) vs Transformed(1661026)
- 평균 종횡비: Annotation(0.54) vs Transformed(0.55)

## 🎯 **결론**

### ❌ **두 이미지는 다른 위치를 나타내는 것으로 판단됩니다**

**근거**:
- 최고 유사도에서도 임계값 미달
- 최적 변환: Flip Horizontal
- 최고 NCC: 0.330 (임계값: 0.3)
- 최고 Hist Corr: 0.890 (임계값: 0.5)

**의미**:
- XTF 이미지와 annotation 이미지가 서로 다른 영역
- 이전 좌표 분석과 일치: 지리적으로 다른 지역의 데이터
- 좌표 변환만으로는 해결할 수 없는 근본적인 데이터 불일치

## 🛠️ **기술적 세부사항**

**적용된 변환**:
- Original (변환 없음)
- 180도 회전
- 좌우 반전
- 상하 반전
- 180도 회전 + 좌우 반전
- 90도 회전
- 270도 회전

**유사도 측정 방법**:
- MSE (Mean Squared Error): 픽셀 간 차이의 제곱 평균
- NCC (Normalized Cross-Correlation): 정규화된 교차 상관계수
- Histogram Correlation: 히스토그램 간 상관계수
- SSIM (Structural Similarity Index): 구조적 유사성 지수