# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:08:04 2025

@author: User
"""

import re
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Mecab

# 1) Mecab 형태소 분석기 로드
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# 2) CSV 불러와서 사업자명별 리뷰 합치기
df = pd.read_csv("filtered_100attrs.csv", encoding="utf-8")
grouped = (
    df
    .groupby("사업자명")["리뷰본문"]
    .apply(lambda texts: " ".join(texts.dropna().astype(str)))
    .reset_index()
)

# 3) 리뷰 전처리: 특수문자 제거
def clean_text(text):
    text = re.sub(r"[^가-힣0-9A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# 4) 조사 제거용 정규표현식 (끝에 하나 이상 붙은 조사 전부 제거)
JOSA_PATTERN = re.compile(
    r"(?P<noun>.+?)(이|가|은|는|을|를|과|와|랑|으로|로|에게|께|에서|부터|까지|만|도|만큼|처럼|보다|밖에|조차|마저|까지)+$"
)

def normalize_noun(token: str) -> str:
    """
    - Mecab이 뽑은 토큰(token)을 다시 분석한 뒤,
    - 명사(NN*) 부분만 이어붙이고,
    - 끝에 붙어 있는 조사(josa)를 모두 제거.
    """
    parts = mecab.pos(token)
    # 1) NN* 품사만 골라 합치기
    noun = "".join([w for w, p in parts if p.startswith("NN")])
    # 2) 맨 끝에 붙은 조사 전부 제거
    m = JOSA_PATTERN.match(noun)
    return m.group("noun") if m else noun

def get_noun_candidates_mecab(text: str) -> list[str]:
    """
    텍스트 전체에서 MeCab이 뽑은 NN* 토큰을 후보로 모은 뒤,
    normalize_noun()을 통해 조사를 제거 → 
    두 글자 이상, 숫자 아닌 순수 명사만 리턴
    """
    tokens = mecab.pos(text)
    raw_nouns = [w for w, p in tokens if p.startswith("NN") and len(w) > 1]
    normed = [normalize_noun(w) for w in raw_nouns]
    # 2글자 이상, 숫자 아닌 것만
    return [w for w in set(normed) if len(w) > 1 and not w.isdigit()]

# 5) SBERT + KeyBERT 로드
sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kw_model = KeyBERT(model=sbert)

# 6) 명사 후보 기반 키워드 추출 함수
def extract_noun_keywords_with_mecab(doc: str, top_n: int = 3) -> list[str]:
    cleaned = clean_text(doc)
    candidates = get_noun_candidates_mecab(cleaned)
    if not candidates:
        return []
    kws = kw_model.extract_keywords(
        cleaned,
        candidates=candidates,        # Mecab 정제 명사 후보만
        keyphrase_ngram_range=(1, 1),  # unigram
        stop_words=None,
        use_mmr=True,
        diversity=0.7,
        top_n=top_n
    )
    return [kw for kw, score in kws]

# 7) 사업자명별 대표 명사 키워드 뽑기
grouped["대표명사키워드"] = grouped["리뷰본문"].apply(
    lambda txt: extract_noun_keywords_with_mecab(txt, top_n=3)
)

# 8) 결과 확인 및 저장
print(grouped[["사업자명", "대표명사키워드"]].head(20))
grouped.to_csv("business_noun_keywords_mecab.csv", index=False, encoding="utf-8-sig")


