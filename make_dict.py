# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 20:47:49 2025

@author: User
"""

import json
from gensim.models.fasttext import load_facebook_model
import pandas as pd
import re
from konlpy.tag import Mecab

# Windows: MeCab 설치 경로를 정확히 지정하세요
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
# Facebook-FastText 형식 그대로 로드
fast_model = load_facebook_model("cc.ko.300.bin")

# 실제 키 벡터는 .wv 에 들어 있습니다
ft = fast_model.wv
# 2) 카테고리별 시드 리스트 로드
with open("category_seed_json/taste_list.json",   encoding="utf-8") as f: taste_seeds   = json.load(f)
with open("category_seed_json/service_list.json", encoding="utf-8") as f: service_seeds = json.load(f)
with open("category_seed_json/atmos_list.json",   encoding="utf-8") as f: atmos_seeds   = json.load(f)
with open("category_seed_json/price_list.json",   encoding="utf-8") as f: price_seeds   = json.load(f)

# 3) 유사어 확장 함수
def expand_seeds(seeds, topn=20, sim_threshold=0.65):
    expanded = list(seeds)
    for word in seeds:
        if word in ft:
            for sim_word, score in ft.most_similar(word, topn=topn):
                # 1) 유사도 기준
                if score < sim_threshold:
                    continue
                # 2) 한글만
                if not re.fullmatch(r'[가-힣]+', sim_word):
                    continue
                # 3) 길이 제한
                if len(sim_word) < 2 or len(sim_word) > 6:
                    continue
                # 4) POS 필터링
                tag = mecab.pos(sim_word)[0][1]
                if tag.startswith(('J','E','S','X')):  # 조사/어미/구두점/기호
                    continue
                expanded.append(sim_word)
    return expanded

# 4) 각 카테고리별 사전 생성
category_dict = {
    "taste":   expand_seeds(taste_seeds,   topn=5),
    "service": expand_seeds(service_seeds, topn=5),
    "atmos":   expand_seeds(atmos_seeds,   topn=5),
    "price":   expand_seeds(price_seeds,   topn=5),
}

# 5) DataFrame으로 저장 (중복·순서 유지)
rows = []
for cat, words in category_dict.items():
    for w in words:
        rows.append({"category": cat, "keyword": w})
df = pd.DataFrame(rows)

# 6) CSV와 JSON으로 출력
df.to_csv("category_seed_json/expanded_category_dict.csv", index=False, encoding="utf-8-sig")
with open("category_seed_json/expanded_category_dict.json", "w", encoding="utf-8") as f:
    json.dump(category_dict, f, ensure_ascii=False, indent=2)

