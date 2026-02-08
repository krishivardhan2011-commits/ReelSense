import numpy as np


# -----------------------------
# Precision@K
# -----------------------------
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / k


# -----------------------------
# Recall@K
# -----------------------------
def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum([1 for item in recommended_k if item in relevant_set])
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0


# -----------------------------
# NDCG@K
# -----------------------------
def ndcg_at_k(recommended, relevant, k=10):
    dcg = 0.0
    idcg = 0.0

    for i in range(k):
        if i < len(recommended) and recommended[i] in relevant:
            dcg += 1 / np.log2(i + 2)

        if i < len(relevant):
            idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0


# -----------------------------
# Diversity (Genre based)
# -----------------------------
def diversity_score(recommended_movies_df):
    genres = recommended_movies_df["genres"].tolist()
    unique_genres = set("|".join(genres).split("|"))
    return len(unique_genres) / (len(genres) * 2)


# -----------------------------
# Catalog Coverage
# -----------------------------
def catalog_coverage(recommended_all_users, total_movies):
    recommended_unique = set()

    for rec_list in recommended_all_users:
        recommended_unique.update(rec_list)

    return len(recommended_unique) / total_movies
