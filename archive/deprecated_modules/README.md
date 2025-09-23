# Deprecated Files

이 폴더에는 더 이상 사용하지 않는 파일들이 포함되어 있습니다.

## 새로운 통합 파이프라인 사용

기존의 개별 파일들 대신 새로운 통합 파이프라인을 사용하세요:

```bash
# 전체 파이프라인 실행
python pipeline/unified_pipeline.py --xtf path/to/file.xtf --gps path/to/gps.xlsx --annotation path/to/annotation.png

# 모듈별 실행
python pipeline/unified_pipeline.py --mode modular --steps read extract map label feature classify
```

## 추천 사항

- **활성 파일**: `real_data_pipeline.py`, `process_edgetech_complete.py`는 계속 사용 가능
- **통합 파이프라인**: 새로운 작업에는 `pipeline/unified_pipeline.py` 사용
- **기존 파일**: 필요시 참조용으로만 사용