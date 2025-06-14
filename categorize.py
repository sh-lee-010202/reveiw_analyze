# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:30:37 2025

@author: User
"""

import pandas as pd
import ast
import json

# 1) 카테고리별 키워드 로드 (JSON 또는 CSV 사용 가능)
# 여기서는 JSON 이용 예시
with open("category_seed_json/expanded_category_dict.json", "r", encoding="utf-8") as f:
    category_dict = json.load(f)

# 2) split_results CSV 로드
df_split = pd.read_csv("split_results.csv", encoding="utf-8-sig")

# 3) 첫 번째 열이 문자열로 저장된 리스트 형태라면 ast.literal_eval로 변환
# 컬럼 이름에 따라 'sentences'로 수정하세요
sent_col = df_split.columns[0]
df_split['sentences_list'] = df_split[sent_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 4) 문장별 카테고리 라벨링
records = []
for idx, row in df_split.iterrows():
    sentences = row['sentences_list']
    for sent_idx, sent in enumerate(sentences):
        # 각 카테고리 키워드 매칭
        labels = []
        for cat, keywords in category_dict.items():
            for kw in keywords:
                if kw in sent:
                    labels.append(cat)
                    break
        # 중복 없이, 순서 유지
        unique_labels = list(dict.fromkeys(labels))
        records.append({
            "row_id": idx,
            "sent_idx": sent_idx,
            "sentence": sent,
            "categories": unique_labels if unique_labels else ["None"]
        })

# 5) 결과 DataFrame 생성 및 표시
df_labeled = pd.DataFrame(records)
df_labeled.head(20)
df_labeled.to_csv("sentence_category_labels.csv", index=False, encoding="utf-8-sig")