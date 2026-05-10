"""
COMPLETE MOVIE RECOMMENDATION SYSTEM - Streamlit Web App
Fixed version that handles CSV parsing errors

HOW TO RUN:
1. Make sure movies.csv and ratings.csv are in the SAME folder
2. In VS Code terminal, run: streamlit run streamlit_app.py
3. Browser will open automatically with the web app
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load data with error handling
@st.cache_data
def load_data():
    try:
        # Try reading with error handling
        movies = pd.read_csv('movies.csv', on_bad_lines='skip', encoding='utf-8')
        ratings = pd.read_csv('ratings.csv', on_bad_lines='skip', encoding='utf-8')
        
        # Fill missing values
        movies['genres'] = movies['genres'].fillna('')
        
        # Clean data - remove any rows with missing critical info
        movies = movies.dropna(subset=['movieId', 'title'])
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        
        return movies, ratings
        
    except FileNotFoundError:
        st.error("""
        ❌ **Data files not found!**
        
        Please download the MovieLens dataset:
        1. Go to: https://grouplens.org/datasets/movielens/latest/
        2. Download 'ml-latest-small.zip'
        3. Extract and copy 'movies.csv' and 'ratings.csv' to this folder
        """)
        st.stop()
        
    except Exception as e:
        st.error(f"""
        ❌ **Error loading data:** {str(e)}
        
        **Try this:**
        1. Download the correct dataset from: https://grouplens.org/datasets/movielens/latest/
        2. Use the 'ml-latest-small.zip' file (1 MB)
        3. Extract and replace your current CSV files
        """)
        st.stop()

# Try to load data
try:
    movies, ratings = load_data()
    data_loaded = True
except:
    data_loaded = False
    st.error("Could not load data. Please check the error message above.")

if data_loaded:
    # Title
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Choose Method")
    method = st.sidebar.radio(
        "Recommendation Type:",
        ["🏠 Home", "📊 Popular Movies", "🎯 Similar Movies", "👥 User Recommendations"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Dataset Info:**
    - Movies: {len(movies):,}
    - Ratings: {len(ratings):,}
    - Users: {ratings['userId'].nunique():,}
    """)

    # ============ HOME PAGE ============
    if method == "🏠 Home":
        st.header("Welcome! 👋")
        
        st.write("""
        This app uses 3 different ways to recommend movies:
        
        1. **📊 Popular Movies** - Shows highest-rated movies
        2. **🎯 Similar Movies** - Find movies like the ones you love
        3. **👥 User Recommendations** - Based on your viewing history
        
        Choose a method from the sidebar to start!
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Fast & Simple**\n\nSee what everyone loves!")
        
        with col2:
            st.success("**Personalized**\n\nFind similar content!")
        
        with col3:
            st.warning("**Smart**\n\nLearn from your taste!")

    # ============ DEMOGRAPHIC FILTERING ============
    elif method == "📊 Popular Movies":
        st.header("📊 Most Popular Movies")
        
        min_votes = st.slider("Minimum number of votes:", 10, 200, 50)
        
        # Calculate stats
        data = movies.merge(ratings, on='movieId')
        movie_stats = data.groupby('title').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['title', 'avg_rating', 'vote_count']
        
        # Filter and sort
        popular = movie_stats[movie_stats['vote_count'] >= min_votes]
        
        if len(popular) == 0:
            st.warning(f"No movies found with at least {min_votes} votes. Try lowering the slider.")
        else:
            top_movies = popular.sort_values('avg_rating', ascending=False).head(10)
            
            st.success(f"🏆 Top 10 Movies (with at least {min_votes} votes)")
            
            for idx, row in top_movies.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### 🎬 {row['title']}")
                    with col2:
                        st.metric("⭐ Rating", f"{row['avg_rating']:.2f}/5.0")
                    st.caption(f"👥 {int(row['vote_count'])} votes")
                    st.markdown("---")

    # ============ CONTENT-BASED FILTERING ============
    elif method == "🎯 Similar Movies":
        st.header("🎯 Find Similar Movies")
        
        # Create similarity matrix
        @st.cache_data
        def create_similarity():
            try:
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(movies['genres'])
                return cosine_similarity(tfidf_matrix, tfidf_matrix)
            except Exception as e:
                st.error(f"Error creating similarity matrix: {str(e)}")
                return None
        
        cosine_sim = create_similarity()
        
        if cosine_sim is not None:
            # Movie selection
            movie_list = sorted(movies['title'].unique())
            
            if len(movie_list) == 0:
                st.error("No movies available in the dataset.")
            else:
                selected_movie = st.selectbox(
                    "🎬 Select a movie you like:",
                    movie_list
                )
                
                num_recs = st.slider("Number of recommendations:", 5, 20, 10)
                
                if st.button("🔍 Get Recommendations", type="primary"):
                    try:
                        # Get recommendations
                        idx = movies[movies['title'] == selected_movie].index[0]
                        sim_scores = list(enumerate(cosine_sim[idx]))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                        sim_scores = sim_scores[1:num_recs+1]
                        movie_indices = [i[0] for i in sim_scores]
                        
                        recommendations = movies.iloc[movie_indices][['title', 'genres']].copy()
                        
                        # Display
                        st.success(f"✅ You selected: **{selected_movie}**")
                        selected_genre = movies[movies['title'] == selected_movie]['genres'].values[0]
                        st.info(f"📁 Genres: {selected_genre}")
                        
                        st.markdown("---")
                        st.subheader(f"🎥 Top {num_recs} Similar Movies:")
                        
                        for i, (idx_val, row) in enumerate(recommendations.iterrows(), 1):
                            similarity = sim_scores[i-1][1]
                            with st.container():
                                st.markdown(f"### {i}. {row['title']}")
                                st.caption(f"📁 {row['genres']}")
                                st.progress(similarity, text=f"Similarity: {similarity*100:.1f}%")
                                st.markdown("---")
                                
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")

    # ============ COLLABORATIVE FILTERING ============
    elif method == "👥 User Recommendations":
        st.header("👥 User-Based Recommendations")
        
        max_user_id = int(ratings['userId'].max())
        
        user_id = st.number_input(
            "👤 Enter User ID:",
            min_value=1,
            max_value=max_user_id,
            value=1
        )
        
        if st.button("🎬 Get My Recommendations", type="primary"):
            # Get user ratings
            user_ratings = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False)
            
            if len(user_ratings) == 0:
                st.warning(f"User {user_id} hasn't rated any movies yet! Try a different User ID.")
            else:
                try:
                    # User's top movies
                    user_top = user_ratings.head(5).merge(movies, on='movieId')[['title', 'genres', 'rating']]
                    
                    st.success("🌟 Your Top Rated Movies:")
                    for idx, row in user_top.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{row['title']}**")
                                st.caption(f"📁 {row['genres']}")
                            with col2:
                                st.metric("⭐", f"{row['rating']:.1f}/5.0")
                    
                    st.markdown("---")
                    
                    # Get recommendations
                    user_movie_ids = user_ratings['movieId'].values
                    unrated_movies = movies[~movies['movieId'].isin(user_movie_ids)]
                    
                    user_genres = user_top['genres'].str.split('|').explode().value_counts().head(3).index.tolist()
                    
                    recommendations = []
                    for genre in user_genres:
                        genre_movies = unrated_movies[unrated_movies['genres'].str.contains(genre, na=False)]
                        genre_ratings = genre_movies.merge(ratings, on='movieId').groupby('title')['rating'].agg(['mean', 'count'])
                        genre_ratings = genre_ratings[genre_ratings['count'] >= 20].sort_values('mean', ascending=False).head(3)
                        recommendations.extend(genre_ratings.index.tolist())
                    
                    recommendations = list(dict.fromkeys(recommendations))[:10]
                    
                    if len(recommendations) == 0:
                        st.info("No recommendations found. Try a different user or check if there's enough data.")
                    else:
                        st.success("🎯 Recommended for You:")
                        for movie_title in recommendations:
                            movie_info = movies[movies['title'] == movie_title].iloc[0]
                            avg_rating = ratings.merge(movies, on='movieId')
                            avg_rating = avg_rating[avg_rating['title'] == movie_title]['rating'].mean()
                            
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{movie_title}**")
                                    st.caption(f"📁 {movie_info['genres']}")
                                with col2:
                                    st.metric("⭐", f"{avg_rating:.2f}/5.0")
                                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Made with using Streamlit | Movie Recommendation System </div>",
        unsafe_allow_html=True
    )
