"""
COLLABORATIVE FILTERING - Movie Recommendation System
Recommends movies based on user preferences and similar users

HOW TO RUN:
1. Make sure movies.csv and ratings.csv are in the SAME folder
2. Press F5 in VS Code OR run: python 3_collaborative_filtering.py
3. Enter a User ID when prompted
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("COLLABORATIVE FILTERING - User-Based Recommendations")
print("="*60)

# Load the data
print("\n📂 Loading data...")
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: movies.csv or ratings.csv not found!")
    print("Please download the MovieLens dataset and place the files in this folder.")
    exit()

# Show dataset info
print(f"\n📊 Dataset Info:")
print(f"   - Total movies: {len(movies):,}")
print(f"   - Total ratings: {len(ratings):,}")
print(f"   - Total users: {ratings['userId'].nunique():,}")
print(f"   - Average rating: {ratings['rating'].mean():.2f}/5.0")

# Function to get user recommendations
def get_user_recommendations(user_id, num_recommendations=10):
    """
    Get movie recommendations for a specific user
    """
    # Check if user exists
    if user_id not in ratings['userId'].values:
        return None, None
    
    # Get user's ratings
    user_ratings = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)
    
    # Get user's top rated movies
    user_top_movies = user_ratings.head(5).merge(movies, on='movieId')[['title', 'genres', 'rating']]
    
    # Get movies user hasn't rated
    user_movie_ids = user_ratings['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(user_movie_ids)]
    
    # Get user's favorite genres
    user_genres = user_top_movies['genres'].str.split('|').explode().value_counts().head(3).index.tolist()
    
    # Find popular movies in user's favorite genres
    recommendations = []
    
    for genre in user_genres:
        # Find movies with this genre that user hasn't rated
        genre_movies = unrated_movies[unrated_movies['genres'].str.contains(genre, na=False)]
        
        # Get ratings for these movies
        genre_ratings = genre_movies.merge(ratings, on='movieId').groupby('title').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        genre_ratings.columns = ['title', 'avg_rating', 'vote_count']
        
        # Filter movies with enough ratings
        genre_ratings = genre_ratings[genre_ratings['vote_count'] >= 20]
        
        # Sort by rating
        top_genre_movies = genre_ratings.sort_values('avg_rating', ascending=False).head(3)
        
        recommendations.extend(top_genre_movies['title'].tolist())
    
    # Remove duplicates and limit to num_recommendations
    recommendations = list(dict.fromkeys(recommendations))[:num_recommendations]
    
    # Get full info for recommendations
    recommended_movies = movies[movies['title'].isin(recommendations)][['title', 'genres']].copy()
    
    # Add average ratings
    avg_ratings = []
    for title in recommended_movies['title']:
        avg_rating = ratings.merge(movies, on='movieId')
        avg_rating = avg_rating[avg_rating['title'] == title]['rating'].mean()
        avg_ratings.append(avg_rating)
    
    recommended_movies['avg_rating'] = avg_ratings
    
    return user_top_movies, recommended_movies

# Interactive mode
print("\n" + "="*60)
while True:
    print("\nEnter a User ID (1-610) or 'quit' to exit:")
    user_input = input("👤 User ID: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    try:
        user_id = int(user_input)
    except ValueError:
        print("❌ Please enter a valid number!")
        continue
    
    # Get recommendations
    user_top, recommendations = get_user_recommendations(user_id, num_recommendations=10)
    
    if user_top is None:
        print(f"\n❌ User ID {user_id} not found in database.")
        print("Try a number between 1 and 610.")
        continue
    
    # Display user's top rated movies
    print("\n" + "="*60)
    print(f"USER {user_id} - PROFILE")
    print("="*60)
    print(f"\n🌟 YOUR TOP RATED MOVIES:\n")
    
    for idx, row in user_top.iterrows():
        print(f"🎬 {row['title']}")
        print(f"   📁 Genres: {row['genres']}")
        print(f"   ⭐ Your Rating: {row['rating']:.1f}/5.0\n")
    
    # Display recommendations
    print("="*60)
    print(f"\n🎯 RECOMMENDED FOR YOU (Based on your taste):\n")
    
    for idx, row in recommendations.iterrows():
        print(f"🎬 {row['title']}")
        print(f"   📁 Genres: {row['genres']}")
        print(f"   ⭐ Average Rating: {row['avg_rating']:.2f}/5.0\n")
    
    print("="*60)