# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:34:45 2025

@author: User
"""

import pandas as pd

file_path = "./review_raw.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 3번째 열 추출 및 리스트 변환
keyword = df.iloc[:, 3].tolist()

unique_keywords = list(set(keyword))

# 4. 결과를 CSV로 저장
seed_df = pd.DataFrame({
    "keyword": unique_keywords
})
seed_df.to_csv("unique_keywords.csv", index=False, encoding="utf-8-sig")

print(f"총 {len(unique_keywords)}개의 유니크 키워드를 unique_keywords.csv에 저장했습니다.")
