"""
DEMOGRAPHIC FILTERING - Movie Recommendation System
Shows the most popular/highest-rated movies to all users

HOW TO RUN:
1. Make sure movies.csv and ratings.csv are in the SAME folder
2. Press F5 in VS Code OR run: python 1_demographic_filtering.py
"""

import pandas as pd

print("="*60)
print("DEMOGRAPHIC FILTERING - Most Popular Movies")
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

# Merge movies and ratings
print("\n🔄 Processing data...")
data = movies.merge(ratings, on='movieId')

# Calculate average rating and vote count for each movie
movie_stats = data.groupby('title').agg({
    'rating': ['mean', 'count']
}).reset_index()

movie_stats.columns = ['title', 'avg_rating', 'vote_count']

# Filter movies with at least 50 votes (to avoid obscure movies)
min_votes = 50
popular_movies = movie_stats[movie_stats['vote_count'] >= min_votes]

# Sort by average rating
top_movies = popular_movies.sort_values('avg_rating', ascending=False).head(10)

# Display results
print("\n" + "="*60)
print(f"TOP 10 MOVIES (with at least {min_votes} votes)")
print("="*60)

for idx, row in top_movies.iterrows():
    print(f"\n🎬 {row['title']}")
    print(f"   ⭐ Rating: {row['avg_rating']:.2f}/5.0")
    print(f"   👥 Number of votes: {int(row['vote_count'])}")

print("\n" + "="*60)
print("✅ DONE! These are the most popular movies in the database.")
print("="*60)

# Optional: Save results to file
top_movies.to_csv('demographic_recommendations.csv', index=False)
print("\n💾 Results saved to 'demographic_recommendations.csv'")