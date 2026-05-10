"""
CONTENT-BASED FILTERING - Movie Recommendation System
Recommends movies similar to a given movie based on genre

HOW TO RUN:
1. Make sure movies.csv is in the SAME folder
2. Press F5 in VS Code OR run: python 2_content_based_filtering.py
3. Type a movie name when prompted
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("CONTENT-BASED FILTERING - Find Similar Movies")
print("="*60)

# Load the data
print("\n📂 Loading data...")
try:
    movies = pd.read_csv('movies.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: movies.csv not found!")
    print("Please download the MovieLens dataset and place the file in this folder.")
    exit()

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

print("\n🔄 Building similarity matrix...")
# Create TF-IDF matrix from genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("✅ Similarity matrix created!")

# Function to get recommendations
def get_recommendations(movie_title, num_recommendations=10):
    """
    Get movie recommendations based on content similarity
    """
    # Check if movie exists
    if movie_title not in movies['title'].values:
        return None
    
    # Find movie index
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return movie titles and similarity scores
    recommendations = movies.iloc[movie_indices][['title', 'genres']].copy()
    recommendations['similarity'] = [score[1] for score in sim_scores]
    
    return recommendations

# Show some example movies
print("\n" + "="*60)
print("EXAMPLE MOVIES IN DATABASE:")
print("="*60)
sample_movies = movies.sample(20)['title'].values
for i, movie in enumerate(sample_movies[:10], 1):
    print(f"{i}. {movie}")

# Interactive mode
print("\n" + "="*60)
while True:
    print("\nEnter a movie name (or 'quit' to exit):")
    user_input = input("🎬 Movie: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    # Get recommendations
    recommendations = get_recommendations(user_input, num_recommendations=10)
    
    if recommendations is None:
        print(f"\n❌ Movie '{user_input}' not found in database.")
        print("Try one of the example movies above or check spelling.")
        continue
    
    # Display results
    print("\n" + "="*60)
    print(f"✅ YOU SELECTED: {user_input}")
    selected_genre = movies[movies['title'] == user_input]['genres'].values[0]
    print(f"📁 Genre: {selected_genre}")
    print("="*60)
    print(f"\n🎥 TOP 10 SIMILAR MOVIES:\n")
    
    for idx, row in recommendations.iterrows():
        similarity_percent = row['similarity'] * 100
        print(f"🎬 {row['title']}")
        print(f"   📁 Genres: {row['genres']}")
        print(f"   🔗 Similarity: {similarity_percent:.1f}%\n")
    
    print("="*60)