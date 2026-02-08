import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_cf_model(ratings):
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    similarity = cosine_similarity(matrix)
    return similarity, matrix


def recommend_cf(user_id, similarity, matrix, k=10):
    user_index = user_id - 1
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]

    similar_users = [i[0] for i in sim_scores]
    rec_movies = matrix.iloc[similar_users].mean().sort_values(ascending=False)

    top_movies = rec_movies.head(k).index.tolist()
    return top_movies
