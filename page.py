import os
import streamlit as st
import httpx
from typing import List, Dict, Optional
import asyncio
import pandas as pd
import random
import sqlite3
from datetime import datetime
import webbrowser
from urllib.parse import urlencode

# --------------------
# Configuration
# --------------------

# Load secrets - Ensure these are set in Streamlit Cloud or your .streamlit/secrets.toml
GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = st.secrets.get("REDIRECT_URI") # e.g., "http://localhost:8501" for local dev

# Basic check if secrets are loaded
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not REDIRECT_URI:
    st.error("OAuth secrets not found! Please configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and REDIRECT_URI in Streamlit secrets.")
    st.stop()

# --- SQLite Database Setup ---
def init_db():
    conn = sqlite3.connect('movie_recommender.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, name TEXT, created_at TIMESTAMP)''')
    
    # Create watch_history table
    c.execute('''CREATE TABLE IF NOT EXISTS watch_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_email TEXT,
                  movie_id TEXT,
                  movie_name TEXT,
                  year INTEGER,
                  genre TEXT,
                  description TEXT,
                  imdb_rating REAL,
                  added_at TIMESTAMP,
                  FOREIGN KEY (user_email) REFERENCES users(email))''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# --- Initialize User-Specific Session State ---
if "user" not in st.session_state:
    st.session_state.user = None
if "token" not in st.session_state:
    st.session_state.token = None
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []
if "seen_movies" not in st.session_state:
    st.session_state.seen_movies = {}
if 'added_movie_ids' not in st.session_state:
    st.session_state.added_movie_ids = set()

# --------------------
# SQLite Helper Functions
# --------------------

def get_user_watch_history(user_email: str) -> List[Dict]:
    """Gets the user's watch history from SQLite."""
    conn = sqlite3.connect('movie_recommender.db')
    c = conn.cursor()
    c.execute('''SELECT movie_id, movie_name, year, genre, description, imdb_rating 
                 FROM watch_history WHERE user_email = ?''', (user_email,))
    movies = []
    for row in c.fetchall():
        movies.append({
            'id': row[0],
            'movie_name': row[1],
            'year': row[2],
            'genre': row[3],
            'description': row[4],
            'imdbRating': row[5]
        })
    conn.close()
    return movies

def add_movie_to_watch_history(user_email: str, movie: Dict):
    """Adds a movie to the user's watch history in SQLite."""
    conn = sqlite3.connect('movie_recommender.db')
    c = conn.cursor()
    c.execute('''INSERT INTO watch_history 
                 (user_email, movie_id, movie_name, year, genre, description, imdb_rating, added_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_email, movie['id'], movie['movie_name'], movie['year'],
               movie['genre'], movie['description'], movie['imdbRating'], datetime.now()))
    conn.commit()
    conn.close()

def remove_movie_from_watch_history(user_email: str, movie_id: str):
    """Removes a movie from the user's watch history in SQLite."""
    conn = sqlite3.connect('movie_recommender.db')
    c = conn.cursor()
    c.execute('''DELETE FROM watch_history 
                 WHERE user_email = ? AND movie_id = ?''', (user_email, movie_id))
    conn.commit()
    conn.close()

def create_or_get_user(user_email: str, user_name: str):
    """Creates a new user or gets existing user from SQLite."""
    conn = sqlite3.connect('movie_recommender.db')
    c = conn.cursor()
    c.execute('''INSERT OR IGNORE INTO users (email, name, created_at)
                 VALUES (?, ?, ?)''', (user_email, user_name, datetime.now()))
    conn.commit()
    conn.close()

# --------------------
# OAuth Functions
# --------------------

def get_google_auth_url():
    """Generate Google OAuth URL."""
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

async def get_token(code: str) -> Optional[Dict]:
    """Exchange authorization code for access token."""
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(token_url, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error getting token: {e}")
            return None

async def get_user_info(access_token: str) -> Optional[Dict]:
    """Fetches user info from Google using the access token."""
    USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            user_dict = response.json()
            return {"email": user_dict.get("email"), "name": user_dict.get("name")}
        except Exception as e:
            st.error(f"An error occurred while fetching user info: {e}")
            return None

def handle_login() -> Optional[Dict]:
    """Handles Google OAuth login and user data loading."""
    if st.session_state.user:
        return st.session_state.user

    # Check for authorization code in URL
    query_params = st.experimental_get_query_params()
    if 'code' in query_params:
        code = query_params['code'][0]
        token = asyncio.run(get_token(code))
        if token:
            st.session_state.token = token
            user_info = asyncio.run(get_user_info(token['access_token']))
            if user_info and user_info.get("email"):
                st.session_state.user = user_info
                user_email = user_info["email"]
                user_name = user_info.get("name", "")

                # Create or get user in SQLite
                create_or_get_user(user_email, user_name)

                # Load watch history
                st.session_state.watch_history = get_user_watch_history(user_email)
                st.session_state.seen_movies = {
                    movie['id']: True for movie in st.session_state.watch_history
                }

                st.rerun()
            else:
                st.error("Could not fetch user information or email from Google. Please try again.")
                st.session_state.token = None
                st.session_state.user = None
        else:
            st.error("Failed to get access token. Please try again.")
            st.session_state.token = None
            st.session_state.user = None
    else:
        # Show login button
        if st.button("Continue with Google"):
            auth_url = get_google_auth_url()
            webbrowser.open(auth_url)
            st.info("Please complete the Google login in your browser and return to this page.")

    return st.session_state.user

def handle_logout():
    """Clears session state for logout."""
    if st.sidebar.button("Logout"):
        for key in ["user", "token", "watch_history", "seen_movies", "added_movie_ids"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --------------------
# Recommendation Logic (Placeholder)
# --------------------

def get_recommendations(mood_answers: List[str], user_email: str) -> List[Dict]:
    """Placeholder function to simulate fetching recommendations."""
    st.info(f"Simulating recommendation fetch for {user_email} based on answers: {mood_answers}")
    
    import hashlib
    recommendations = []
    example_descriptions = [
        "A perfect pick for a relaxed evening.", "An inspiring journey.",
        "A thrilling ride.", "A heartwarming story.", "Something light and fun.",
        "A thought-provoking film.", "An epic adventure.", "A captivating mystery.",
        "A refreshing comedy.", "A classic feel-good movie."
    ]
    
    answers_hash = hashlib.md5(str(mood_answers).encode()).hexdigest()
    random.seed(answers_hash)

    for i in range(10):
        rec_id = f"rec_{answers_hash[:4]}_{i}"
        desc_index = random.randint(0, len(example_descriptions) - 1)
        base_rating = random.uniform(5.0, 9.0)
        recommendations.append({
            "id": rec_id,
            "title": f"Recommended Movie {i+1} ({answers_hash[i:i+3]})",
            "description": example_descriptions[desc_index],
            "imdbRating": round(base_rating, 1),
            "year": random.randint(1990, 2024),
            "genre": random.choice(["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"])
        })

    return recommendations

# --------------------
# Streamlit Page Layout
# --------------------

def main():
    st.set_page_config(page_title="Movie Mood Recommender", layout="wide")
    st.title("🎬 Movie Mood Recommender")

    # --- Authentication Flow ---
    user = handle_login()

    if not user:
        st.warning("Please log in using Google to continue.")
        st.stop()

    # --- Logged In View ---
    st.sidebar.header(f"Welcome, {user.get('name', user.get('email'))}!")
    st.sidebar.write(f"Logged in as: {user.get('email')}")
    handle_logout()

    st.markdown("---")

    # --- Watch History Display Section ---
    with st.expander("View/Manage Your Watch History"):
        if not st.session_state.watch_history:
            st.info("Your watch history is empty. Add movies from recommendations to build your history.")
        else:
            st.subheader("Your Watched Movies")
            num_history_cols = 4
            history_cols = st.columns(num_history_cols)
            
            for idx, movie in enumerate(st.session_state.watch_history):
                col_index = idx % num_history_cols
                with history_cols[col_index]:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; height: 250px; overflow-y: auto;">
                            <p style="font-weight: bold;">{movie.get('movie_name', 'N/A')} ({movie.get('year', 'N/A')})</p>
                            <p style="font-size: small;"><strong>Genre:</strong> {movie.get('genre', 'N/A')}</p>
                            <p style="font-size: small;"><strong>Rating:</strong> {movie.get('imdbRating', 'N/A')} ⭐</p>
                            <details style="font-size: small;"><summary>Description</summary>{movie.get('description', 'N/A')}</details>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("Remove", key=f"remove_{movie['id']}_{idx}"):
                        try:
                            remove_movie_from_watch_history(user['email'], movie['id'])
                            st.session_state.watch_history.pop(idx)
                            if 'seen_movies' in st.session_state:
                                st.session_state.seen_movies.pop(movie['id'], None)
                            st.success(f"Removed '{movie.get('movie_name', 'Movie')}' from history.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove movie: {e}")

    st.markdown("---")

    # --- Mood Questionnaire and Recommendations Section ---
    st.subheader("Get Movie Recommendations Based on Your Mood")
    st.caption("Answer these questions to help us find movies matching your mood.")

    mood_form = st.form(key="mood_form")
    with mood_form:
        mood_questions = [
            "Energy level? (e.g., 'High', 'Relaxed', 'Tired')",
            "Story type? (e.g., 'Inspiring', 'Thriller', 'Heartwarming')",
            "Focus level? (e.g., 'Focused', 'Distracted', 'Light')",
            "Preferred pace? (e.g., 'Fast', 'Slow', 'Medium')",
            "Genre preference/avoidance? (e.g., 'Love sci-fi', 'No horror', 'Any')"
        ]
        mood_answers: List[str] = []

        for i, question in enumerate(mood_questions):
            ans = st.text_input(question, key=f"q{i+1}")
            mood_answers.append(ans.strip() if ans else "")

        submit_button = st.form_submit_button(label="Find Movies!")

    if submit_button:
        st.session_state.added_movie_ids = set()

        if not any(mood_answers):
            st.warning("Please answer at least one question to get recommendations.")
        else:
            with st.spinner("Generating recommendations based on your mood..."):
                movies = get_recommendations(mood_answers, user['email'])
            
            st.session_state.recommendations = movies
            st.session_state.show_recommendations = True

    if st.session_state.get('show_recommendations', False):
        movies = st.session_state.recommendations
        st.markdown("---")
        if movies:
            st.subheader("Here are some movies tailored to your mood:")

            history_ids = {m.get('id') for m in st.session_state.watch_history if m.get('id')}
            unwatched_movies = [m for m in movies if m.get('id') not in history_ids]
            filtered_movies = [m for m in unwatched_movies if m.get('id') not in st.session_state.added_movie_ids]

            if not filtered_movies:
                st.info("No new recommendations match your criteria, or you've already seen/added them all from this batch!")
            else:
                num_rec_cols = 3
                rec_cols = st.columns(num_rec_cols)

                # Initialize selected movies in session state if not exists
                if 'selected_movies' not in st.session_state:
                    st.session_state.selected_movies = set()

                # Track current selections
                current_selections = set()

                for idx, movie in enumerate(filtered_movies):
                    with rec_cols[idx % num_rec_cols]:
                        movie_identifier = movie.get('id')
                        if not movie_identifier:
                            st.warning(f"Recommendation {idx+1} missing 'id'. Skipping.")
                            continue

                        st.markdown(f"**{movie.get('title', 'N/A')}** ({movie.get('year', 'N/A')})")
                        if movie.get('description'):
                            st.write(movie['description'])
                        if movie.get('imdbRating'):
                            st.caption(f"IMDb Rating: {movie.get('imdbRating')} ⭐ ({movie.get('genre', 'N/A')})")

                        # Initialize checkbox state if not exists
                        checkbox_key = f"add_{movie_identifier}_{idx}"  # Make the key unique
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = False

                        # Check if movie is already in watch history
                        is_in_history = any(m['id'] == movie_identifier for m in st.session_state.watch_history)
                        
                        # Create checkbox
                        checkbox_value = st.checkbox(
                            "✓ In Watch History" if is_in_history else "Add to Watch History",
                            key=checkbox_key,
                            value=st.session_state[checkbox_key]
                        )

                        if checkbox_value:
                            current_selections.add(movie_identifier)
                        else:
                            current_selections.discard(movie_identifier)

                # Review Button
                review_selections = st.button("Review Selections and Add to Watch History")

                if review_selections:
                    added_count = 0
                    failed_movies = []
                    for idx, movie in enumerate(filtered_movies):
                        movie_identifier = movie.get('id')
                        checkbox_key = f"add_{movie_identifier}_{idx}"
                        if st.session_state.get(checkbox_key, False):  # Use .get() to avoid KeyError
                            movie_to_add = {
                                'id': movie_identifier,
                                'movie_name': movie.get('title'),
                                'year': movie.get('year'),
                                'genre': movie.get('genre'),
                                'description': movie.get('description'),
                                'imdbRating': movie.get('imdbRating'),
                            }
                            try:
                                add_movie_to_watch_history(user['email'], movie_to_add)
                                st.session_state.watch_history.append(movie_to_add)
                                if 'seen_movies' in st.session_state:
                                    st.session_state.seen_movies[movie_identifier] = True
                                added_count += 1
                                st.session_state[checkbox_key] = False  # Uncheck after adding
                            except Exception as e:
                                failed_movies.append(movie.get('title', 'Unknown Movie'))
                                st.error(f"Failed to add movie '{movie.get('title')}': {e}")

                    if added_count > 0:
                        st.success(f"✅ Successfully added {added_count} movie(s) to your watch history!")
                        if failed_movies:
                            st.warning(f"⚠️ Failed to add the following movies: {', '.join(failed_movies)}")
                        st.rerun()  # Refresh to update watch history and checkboxes

        else:
            st.error("Sorry, we couldn't find any recommendations at this time. Please try again later.")

if __name__ == "__main__":
    main()