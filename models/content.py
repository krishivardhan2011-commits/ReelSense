import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_model(movies, tags):
    # Merge tags with movies
    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    data = movies.merge(tags_grouped, on="movieId", how="left")
    data["tag"] = data["tag"].fillna("")
    data["content"] = data["genres"] + " " + data["tag"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["content"])

    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity, data


def recommend_content(movie_title, similarity, data, k=10):
    idx = data[data["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]
    movie_indices = [i[0] for i in scores]
    return data.iloc[movie_indices][["title", "genres"]]
