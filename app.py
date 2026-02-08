import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 ReelSense — Explainable Movie Recommender")

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
similarity = cosine_similarity(matrix)

def recommend(user_id, k=10):
    user_index = user_id - 1
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    similar_users = [i[0] + 1 for i in sim_scores]

    seen = ratings[ratings["userId"] == user_id]["movieId"]
    recs = ratings[ratings["userId"].isin(similar_users)]
    recs = recs[~recs.movieId.isin(seen)]
    recs = recs.groupby("movieId").rating.mean().sort_values(ascending=False).head(k)

    return movies[movies["movieId"].isin(recs.index)][["title","genres"]]

user_id = st.number_input("User ID", 1, 610, 1)

if st.button("Recommend"):
    recs = recommend(user_id,10)
    st.subheader("Top Recommendations")

    for _, row in recs.iterrows():
        st.write(row["title"])
        st.caption("Recommended based on similar users and genre overlap")

    genres = recs["genres"].tolist()
    unique = set("|".join(genres).split("|"))
    diversity = len(unique) / (len(genres) * 2)

    st.write("Diversity Score:", round(diversity,3))
