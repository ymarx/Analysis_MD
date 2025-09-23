#!/usr/bin/env python3
"""
부산 위치자료 Excel 파일 구조 분석

목적: 복잡한 Excel 구조를 파악하고 올바른 데이터 추출
"""

import pandas as pd
import numpy as np

def analyze_busan_excel():
    """부산 Excel 파일 구조 분석"""

    print("="*60)
    print("부산 위치자료 Excel 파일 구조 분석")
    print("="*60)

    busan_file = "[샘플]데이터/[위치]부산위치자료-도분초-위경도변환.xlsx"

    try:
        # 모든 시트 정보 확인
        excel_file = pd.ExcelFile(busan_file)
        print(f"📋 시트 목록: {excel_file.sheet_names}")

        for sheet_name in excel_file.sheet_names:
            print(f"\n{'='*40}")
            print(f"시트: {sheet_name}")
            print(f"{'='*40}")

            # 시트 읽기
            df = pd.read_excel(busan_file, sheet_name=sheet_name)

            print(f"크기: {df.shape}")
            print(f"컬럼: {list(df.columns)}")

            # 비어있지 않은 데이터 찾기
            non_empty_rows = []
            for i, row in df.iterrows():
                if not row.isna().all():
                    non_empty_rows.append(i)

            print(f"비어있지 않은 행: {len(non_empty_rows)}개")

            if non_empty_rows:
                print(f"데이터 시작 행: {non_empty_rows[0]}")
                print(f"데이터 종료 행: {non_empty_rows[-1]}")

                # 실제 데이터가 있는 부분만 출력
                start_row = non_empty_rows[0]
                end_row = min(start_row + 10, non_empty_rows[-1] + 1)

                print(f"\n데이터 샘플 (행 {start_row}-{end_row-1}):")
                sample_data = df.iloc[start_row:end_row]

                for i, (idx, row) in enumerate(sample_data.iterrows()):
                    print(f"행 {idx}: {[str(x) if not pd.isna(x) else 'NaN' for x in row.values[:5]]}")  # 처음 5개 컬럼만

                # 숫자 데이터가 있는 컬럼 찾기
                print(f"\n숫자 데이터 컬럼 분석:")
                for col in df.columns:
                    numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        print(f"   {col}: {len(numeric_data)}개 숫자 ({numeric_data.min():.3f} ~ {numeric_data.max():.3f})")

                # 좌표로 보이는 데이터 찾기
                print(f"\n좌표 후보 데이터:")
                for col in df.columns:
                    numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        min_val = numeric_data.min()
                        max_val = numeric_data.max()

                        # 위도 범위 (30-40도)
                        if 30 <= min_val <= 40 and 30 <= max_val <= 40:
                            print(f"   위도 후보 - {col}: {min_val:.6f} ~ {max_val:.6f}")

                        # 경도 범위 (120-140도)
                        elif 120 <= min_val <= 140 and 120 <= max_val <= 140:
                            print(f"   경도 후보 - {col}: {min_val:.6f} ~ {max_val:.6f}")

        # 특정 행부터 다시 읽기 시도
        print(f"\n{'='*60}")
        print("데이터 재추출 시도")
        print(f"{'='*60}")

        # 3행부터 읽기 (헤더가 3행에 있는 것 같음)
        try:
            df_from_3 = pd.read_excel(busan_file, header=2)
            print(f"\n3행부터 읽기:")
            print(f"크기: {df_from_3.shape}")
            print(f"컬럼: {list(df_from_3.columns)}")

            # 상위 5행 출력
            print(f"\n상위 5행:")
            print(df_from_3.head().to_string())

            # 좌표 컬럼 찾기
            lat_col = None
            lon_col = None

            for col in df_from_3.columns:
                if '위도' in str(col):
                    lat_col = col
                elif '경도' in str(col):
                    lon_col = col

            print(f"\n좌표 컬럼:")
            print(f"위도: {lat_col}")
            print(f"경도: {lon_col}")

            if lat_col and lon_col:
                # 유효한 좌표 데이터 추출
                valid_data = df_from_3.dropna(subset=[lat_col, lon_col])
                print(f"\n유효한 좌표 데이터: {len(valid_data)}개")

                if len(valid_data) > 0:
                    # 좌표 범위
                    lat_values = pd.to_numeric(valid_data[lat_col], errors='coerce').dropna()
                    lon_values = pd.to_numeric(valid_data[lon_col], errors='coerce').dropna()

                    print(f"위도 범위: {lat_values.min():.6f} ~ {lat_values.max():.6f}")
                    print(f"경도 범위: {lon_values.min():.6f} ~ {lon_values.max():.6f}")

                    # 샘플 데이터
                    print(f"\n좌표 샘플:")
                    for i, (_, row) in enumerate(valid_data.head().iterrows()):
                        print(f"   {i+1}: {row[lat_col]}, {row[lon_col]}")

        except Exception as e:
            print(f"3행부터 읽기 실패: {e}")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_busan_excel()