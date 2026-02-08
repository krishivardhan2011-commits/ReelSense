def hybrid_recommend(user_id, ratings, movies, cf_similarity, cf_matrix,
                     content_similarity, content_data, k=10):

    # Collaborative recommendations
    from models.collaborative import recommend_cf
    cf_movies = recommend_cf(user_id, cf_similarity, cf_matrix, k)

    # Get user's last watched movie for content similarity
    user_movies = ratings[ratings["userId"] == user_id].sort_values("timestamp", ascending=False)
    if len(user_movies) == 0:
        return []

    last_movie_id = user_movies.iloc[0]["movieId"]
    movie_title = movies[movies["movieId"] == last_movie_id]["title"].values[0]

    from models.content import recommend_content
    content_recs = recommend_content(movie_title, content_similarity, content_data, k)

    # Combine both
    hybrid_titles = list(content_recs["title"])
    return hybrid_titles[:k]
