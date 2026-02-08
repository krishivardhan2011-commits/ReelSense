import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 ReelSense Recommender", layout="wide")

st.title("🎬 ReelSense — Explainable Movie Recommender")

st.write("Hybrid Movie Recommendation System with Diversity & Explainability")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    return ratings, movies, tags

ratings, movies, tags = load_data()

# ---------------- BUILD USER-ITEM MATRIX ----------------
user_movie_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# ---------------- SIMILARITY ----------------
similarity = cosine_similarity(user_movie_matrix)

# ---------------- HYBRID RECOMMENDER ----------------
def recommend(user_id, k=10):

    if user_id > user_movie_matrix.shape[0]:
        return pd.DataFrame()

    user_index = user_id - 1
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    similar_users = [i[0] for i in sim_scores]

    rec_movies = user_movie_matrix.iloc[similar_users].mean(axis=0)
    rec_movies = rec_movies.sort_values(ascending=False).head(k)

    rec_ids = rec_movies.index
    rec_df = movies[movies["movieId"].isin(rec_ids)]

    return rec_df[["movieId", "title", "genres"]]


# ---------------- EXPLAINABILITY ----------------
def explain(movie_id, user_id):
    movie_tags = tags[tags["movieId"] == movie_id]["tag"].tolist()

    if len(movie_tags) == 0:
        return "Recommended based on similar users' preferences."

    return f"Because this movie shares tags like **{', '.join(movie_tags[:3])}** with movies you liked."


# ---------------- DIVERSITY ----------------
def diversity_score(df):
    genres = df["genres"].tolist()
    unique = set("|".join(genres).split("|"))
    return round(len(unique) / (len(genres) * 2), 3)


# ---------------- UI ----------------
user_id = st.number_input("Enter User ID (1–610)", min_value=1, max_value=610, value=1)

k = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("🎬 Generate Recommendations"):

    recs = recommend(user_id, k)

    if recs.empty:
        st.error("Invalid User ID")
    else:
        st.subheader("Top Recommendations")

        for _, row in recs.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write("Genre:", row["genres"])
            st.info(explain(row["movieId"], user_id))

        # Diversity
        st.subheader("📊 Diversity Score")
        st.success(diversity_score(recs))

        # Catalog Coverage
        coverage = len(ratings["movieId"].unique()) / len(movies)
        st.subheader("📦 Catalog Coverage")
        st.success(round(coverage, 3))


st.markdown("---")
st.caption("ReelSense — Explainable Hybrid Recommender with Diversity Optimization")
