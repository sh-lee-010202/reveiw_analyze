# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:20:41 2025

@author: User
"""

import pandas as pd
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# 1) 전체 문장·카테고리 라벨 파일 로드
df_all = pd.read_csv("sentence_category_labels.csv", encoding="utf-8-sig")
df_all["categories_list"] = df_all["categories"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# 2) 이미 감성 라벨링된 파일 로드
df_exist = pd.read_csv("sentence_sentiment_labels.csv", encoding="utf-8-sig")
df_exist["categories_list"] = df_exist["categories"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# 3) 고유 키 생성(row_id, sent_idx 결합)
df_all["_key"]   = list(zip(df_all["row_id"], df_all["sent_idx"]))
df_exist["_key"] = list(zip(df_exist["row_id"], df_exist["sent_idx"]))

# 4) 누락된 행만 추출
missing = df_all[~df_all["_key"].isin(df_exist["_key"])].copy()

print(f"{len(missing)}개의 문장이 누락되어 추가로 처리합니다.")

# 5) 감성 분석 모델 로드
model_dir = "./sentiment/pretrained_model/klue/klue_finetuned_model"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer= AutoTokenizer.from_pretrained(model_dir)
model    = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device).eval()

# 6) 누락 문장에 대해 추론
sentiments = []
for _, row in tqdm(missing.iterrows(), total=len(missing)):
    cats = row["categories_list"]
    sent = row["sentence"]
    # None 카테고리는 상관없음(2)
    if cats == ["None"]:
        sentiments.append(2)
    else:
        inputs = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred   = logits.softmax(dim=-1).argmax(dim=-1).item()  # 0=부정,1=긍정
        sentiments.append(pred)

missing["sentiment"] = sentiments

# 7) 기존 DataFrame에 누락 부분 합치기
df_complete = pd.concat([df_exist, missing], ignore_index=True)

# 8) 불필요 키 삭제
df_complete.drop(columns=["_key"], inplace=True)

# 9) 완전 처리된 파일 저장
df_complete.to_csv("sentence_sentiment_labels_complete.csv", index=False, encoding="utf-8-sig")

print("✅ 모든 문장(약 1.8M개)에 감성 라벨을 부여하여 저장했습니다.")
