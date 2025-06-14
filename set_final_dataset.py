# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:03:29 2025

@author: User
"""


import pandas as pd
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# 1) 문장별 카테고리 라벨링 로드
df = pd.read_csv("sentence_category_labels.csv", encoding="utf-8-sig")
# categories 문자열을 리스트로 변환
df["categories_list"] = df["categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# 2) 감성 분석 모델 로드 (klue fine-tuned)
model_dir = "./sentiment/pretrained_model/klue/klue_finetuned_model"  # 실제 저장 경로로 바꾸세요
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3) 각 문장에 대해 부정/긍정/상관없음 라벨링
sentiments = []
for sentence, cats in tqdm(zip(df["sentence"], df["categories_list"]), total=len(df)):
    # 카테고리가 None인 경우 무조건 상관없음(2)
    if cats == ["None"]:
        sentiments.append(2)
        continue

    # 그 외 문장: 모델 추론
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.softmax(outputs.logits, dim=-1).argmax(dim=-1).item()

    # pred: 0=부정, 1=긍정
    sentiments.append(pred)

# 4) 결과를 DataFrame에 추가
df["sentiment"] = sentiments
# sentiment: 0=부정, 1=긍정, 2=상관없음

# 5) CSV로 저장
df.to_csv("sentence_sentiment_labels.csv", index=False, encoding="utf-8-sig")

print("sentence_sentiment_labels.csv에 문장별 감성 라벨링 결과를 저장했습니다.")
