# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:15:24 2025

@author: User
"""

import pandas as pd
from konlpy.tag import Mecab

# 1) 이전에 저장된 유니크 키워드 로드
df_keywords = pd.read_csv("unique_keywords.csv", encoding="utf-8-sig")
keywords = df_keywords['keyword'].dropna().astype(str).tolist()

# 2) MeCab 형태소 분석기 초기화 (경로는 환경에 맞게 수정)
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# 3) 제외할 품사 태그 리스트 (JOSA: 조사, Eomi: 어미, Punctuation 등)
exclude_tags = {
    'JKS','JKC','JKG','JKO','JKB','JKV','JKQ',  # 조사
    'EP','EF','EC','ETN','ETM',                # 어미/연결 어미/명사형 전성 어미/관형형 전성 어미
    'SF','SP','SS','SE','SO',                  # 구두점
    'SL','SH','SW'                             # 외국어, 한자, 기호
}

# 4) 키워드에서 주요 형태소(체언·용언·형용사 등)만 추출
tokens = set()
for kw in keywords:
    for word, tag in mecab.pos(kw):
        if tag not in exclude_tags and len(word) > 1:
            tokens.add(word)

# 5) 정렬 후 CSV 저장
unique_tokens = sorted(tokens)
pd.DataFrame({'token': unique_tokens}) \
  .to_csv("unique_filtered_tokens.csv", index=False, encoding="utf-8-sig")

print(f"총 {len(unique_tokens)}개의 토큰을 unique_filtered_tokens.csv에 저장했습니다.")

