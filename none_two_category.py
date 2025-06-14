# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:27:46 2025

@author: User
"""

import pandas as pd
import ast

# CSV 파일 로드
df = pd.read_csv("sentence_category_labels.csv", encoding="utf-8-sig")

# categories를 실제 리스트로 변환
df['categories_list'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# None 또는 카테고리 2개 이상 포함된 문장만 추출
df_filtered = df[df['categories_list'].apply(lambda cats: cats == ["None"])]

# 저장
df_filtered.to_csv("None_sentences.csv", index=False, encoding="utf-8-sig")

# 결과 출력
print(f"추출된 문장 수: {len(df_filtered)}개")
