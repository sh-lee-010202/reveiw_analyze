# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:04:10 2025

@author: User
"""
"""
import pandas as pd
import ast

# 1) 문장별 라벨링 결과 로드
df = pd.read_csv("sentence_sentiment_labels_complete.csv", encoding="utf-8-sig")

# 'categories_list'가 문자열로 저장되어 있으면 리스트로 변환
if df['categories_list'].dtype == object:
    df['categories_list'] = df['categories_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

# 2) 카테고리 매핑 (영어 키 → 한글 컬럼명)
mapping = {
    "taste":   "맛",
    "price":   "가격",
    "service": "서비스",
    "atmos":   "분위기"
}

# 3) row_id 단위로 집계
rows = []
for row_id, group in df.groupby("row_id"):
    out = {"row_id": row_id}
    for eng, kor in mapping.items():
        # 해당 카테고리에 속한 문장들의 sentiment 값 수집
        cat_sents = group.loc[
            group["categories_list"].apply(lambda cats: eng in cats),
            "sentiment"
        ]
        # 부정(0) 우선, 그다음 긍정(1), 없으면 상관없음(2)
        if (cat_sents == 0).any():
            out[kor] = 0
        elif (cat_sents == 1).any():
            out[kor] = 1
        else:
            out[kor] = 2
    rows.append(out)

final_df = pd.DataFrame(rows)

# 4) CSV로 저장
final_df.to_csv("final_review_labels.csv", index=False, encoding="utf-8-sig")

print(final_df.head())
print(f"✅ 최종 리뷰 단위 라벨링 ({len(final_df)}개) 저장 완료")
"""
import pandas as pd

# 1) 최종 라벨링 결과 로드
final_df = pd.read_csv("final_review_labels.csv", encoding="utf-8-sig")

# 2) 원본 리뷰 로드
reviews_df = pd.read_csv("review_raw.csv", encoding="utf-8-sig")

# 리뷰 본문만 Series로 추출 (순서대로)
review_texts = reviews_df["리뷰본문"]

# 3) 길이 검증
n_reviews = len(review_texts)
n_final   = len(final_df)
if n_reviews != n_final:
    print(f"리뷰 본문 수: {n_reviews}, final_df 행 수: {n_final}")
else:
    # 4) '리뷰본문' 컬럼을 final_df에 추가
    final_df["리뷰본문"] = review_texts.values
    
    # 5) 저장
    final_df.to_csv("final_review_labels_with_text.csv", index=False, encoding="utf-8-sig")
    print("✅ final_review_labels_with_text.csv에 '리뷰본문'을 추가하여 저장했습니다.")
