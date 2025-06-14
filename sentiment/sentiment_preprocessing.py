# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:14:09 2025

@author: User
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1) TSV 불러오기 (탭 '\t' 구분자)
df = pd.read_csv("kr3.tsv", sep="\t")

# 2) Rating==2를 0으로 매핑 (부정/중립을 모두 부정으로)
df["Rating"] = df["Rating"].replace(2, 0)

# 3) Review, Rating 결측 제거
df = df.dropna(subset=["Review", "Rating"])

# 4) 클래스별 개수 확인
count_0 = len(df[df["Rating"] == 0])
count_1 = len(df[df["Rating"] == 1])
print(f"원본 클래스 분포 -> 0: {count_0}, 1: {count_1}")

# 5) 언더샘플링으로 다수 클래스를 소수 클래스 수에 맞추기
if count_0 > count_1:
    df_0 = df[df["Rating"] == 0].sample(n=count_1, random_state=42)
    df_1 = df[df["Rating"] == 1]
elif count_1 > count_0:
    df_1 = df[df["Rating"] == 1].sample(n=count_0, random_state=42)
    df_0 = df[df["Rating"] == 0]
else:
    df_0 = df[df["Rating"] == 0]
    df_1 = df[df["Rating"] == 1]

# 6) 균형 데이터셋 합치고 섞기
df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"균형 맞춘 클래스 분포 ->\n{df_balanced['Rating'].value_counts()}")

# 7) Train/Test Split (80/20), stratify 적용
train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["Rating"],
    random_state=42
)

# 8) 컬럼명 변경 (Trainer가 기대하는 'labels' 컬럼으로)
train_df = train_df.rename(columns={"Rating": "labels"})
test_df  = test_df.rename(columns={"Rating": "labels"})

# 9) 필요한 컬럼만 골라서 저장
train_df[["Review", "labels"]].to_csv("train_reviews.csv", index=False, encoding="utf-8-sig")
test_df[["Review", "labels"]].to_csv("test_reviews.csv", index=False, encoding="utf-8-sig")

print(f"훈련 데이터: {len(train_df)}개 (0:{sum(train_df['labels']==0)}, 1:{sum(train_df['labels']==1)})")
print(f"테스트 데이터: {len(test_df)}개 (0:{sum(test_df['labels']==0)}, 1:{sum(test_df['labels']==1)})")
