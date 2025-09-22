"""
Context7 MCP Test
- Testing up-to-date documentation retrieval
- Verifying code suggestions are current
"""

import pandas as pd
import numpy as np

# Test 1: Modern pandas DataFrame operations
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['a', 'b', 'c', 'd']
})

# Using pandas 2.x style operations
print("Original DataFrame:")
print(df)

# Modern method chaining with pipe
result = (df
    .assign(D=lambda x: x['A'] * x['B'])
    .query('D > 10')
    .sort_values('D', ascending=False)
)

print("\nProcessed DataFrame:")
print(result)

# Test 2: Using modern string methods
df['C_upper'] = df['C'].str.upper()
print("\nWith uppercase column:")
print(df)

print("\n✅ Context7 테스트 완료 - 최신 pandas API 정상 작동")