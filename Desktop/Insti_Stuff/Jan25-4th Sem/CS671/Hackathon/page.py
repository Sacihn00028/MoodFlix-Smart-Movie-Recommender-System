import os
import streamlit as st
import httpx
from typing import List, Dict, Optional
from streamlit_oauth import OAuth2Component
# from httpx_oauth.clients.google import GoogleOAuth2 # You only need this for authorize_button setup via OAuth2Component
import asyncio # Required by streamlit_oauth
import pandas as pd # Import pandas for potential CSV handling
import random # Import random for selecting random movies
import firebase_admin
from firebase_admin import credentials, firestore

# --------------------
# Configuration
# --------------------

GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = st.secrets.get("REDIRECT_URI")

# Basic check if secrets are loaded
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not REDIRECT_URI:
    st.error("OAuth secrets not found! Please configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and REDIRECT_URI in Streamlit secrets.")
    st.stop()

# Initialize OAuth2Component only once using session state
if "oauth_component" not in st.session_state:
    st.session_state.oauth_component = OAuth2Component(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token", # Optional: Needed if you want refresh tokens
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke"  # Optional: Needed for explicit logout/revoke
    )

# Initialize watch history and seen movies in session state
if "watch_history" not in st.session_state:
    st.session_state.watch_history = [] # Will store a list of movie dictionaries
if "seen_movies" not in st.session_state:
    st.session_state.seen_movies = {} # Use a dictionary to store seen status by movie ID or title

# --------------------
# Authentication & User Handling
# --------------------

async def get_user_info(access_token: str) -> Optional[Dict]:
    """
    Helper function to get user info from Google using the access token
    by calling the userinfo endpoint manually using httpx.
    This replaces the old google_client.get_userinfo(token) method.
    """
    USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            user_dict = response.json()

            # Expected keys: sub, name, given_name, family_name, picture, email, email_verified, locale
            # We mainly care about email and name
            return {"email": user_dict.get("email"), "name": user_dict.get("name")}

        except httpx.HTTPStatusError as e:
            st.error(f"Error fetching user info: HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            st.error(f"Error fetching user info: Request error occurred: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching user info: {e}")
            return None

def handle_login() -> Optional[Dict]:
    """
    Handles the Google OAuth login flow.
    Returns user info dictionary if logged in, otherwise None.
    Stores token and user info in session state.
    """
    # Check if user info is already in session state
    if "user" in st.session_state and st.session_state.user:
        return st.session_state.user

    # Check if token info is in session state (means redirect from Google happened)
    if "token" not in st.session_state:
        st.session_state.token = None

    # Create the authorization button
    result = st.session_state.oauth_component.authorize_button(
        name="Continue with Google",
        icon="https://www.google.com/favicon.ico",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile", # Standard scopes for basic info
        key="google_login", # Unique key for the button
        use_container_width=True,
        extras_params={"prompt": "consent", "access_type": "offline"} # Optional: Force consent screen, get refresh token
    )

    # If we have a result (token response from redirect)
    if result and "token" in result:
        st.session_state.token = result.get("token") # Contains access_token, id_token, etc.
        # Fetch user info using the access token
        access_token = st.session_state.token.get('access_token')
        if access_token:
            # Call the async get_user_info function using asyncio.run
            user_info = asyncio.run(get_user_info(access_token))
            if user_info:
                st.session_state.user = user_info # Store user info
                st.rerun() # Rerun the script to reflect the logged-in state immediately
            else:
                st.error("Could not fetch user information from Google. Please try again.")
                st.session_state.token = None # Clear invalid token or token that failed to get user info
                st.session_state.user = None
        else:
             st.error("Access token not found in the response.")
             st.session_state.token = None
             st.session_state.user = None


    # If still not logged in after checking/processing button result
    if not st.session_state.get("user"):
        return None

    return st.session_state.user

def handle_logout():
    """ Clears session state for logout """
    if st.button("Logout"):
        # Optionally revoke token if you set up revoke_token_endpoint
        # if st.session_state.get("token") and st.session_state.oauth_component.revoke_token_endpoint:
        #    asyncio.run(st.session_state.oauth_component.revoke_token(st.session_state.token["access_token"])) # Or refresh_token
        st.session_state.token = None
        st.session_state.user = None
        st.session_state.watch_history = [] # Clear watch history on logout
        st.session_state.seen_movies = {} # Clear seen movies on logout
        st.rerun() # Rerun to go back to login state

# Initialize Firebase Admin SDK once
if "firebase_app" not in st.session_state:
    # Path to your service account key JSON
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    st.session_state.db = firestore.client()
    st.session_state.firebase_app = True

# --------------------
# Firestore Helper Functions
# --------------------

def get_user_doc_ref(user_email: str):
    return st.session_state.db.collection("users").document(user_email)


def get_watch_history_from_db(user_email: str) -> List[Dict]:
    doc = get_user_doc_ref(user_email).get()
    return doc.to_dict().get("watch_history", []) if doc.exists else []


def add_movie_to_watch_history(user_email: str, movie: Dict):
    ref = get_user_doc_ref(user_email)
    ref.set({"watch_history": firestore.ArrayUnion([movie])}, merge=True)


def get_seen_status(user_email: str) -> Dict:
    doc = get_user_doc_ref(user_email).get()
    return doc.to_dict().get("seen_status", {}) if doc.exists else {}


def update_seen_status(user_email: str, movie_id: str, status: bool):
    ref = get_user_doc_ref(user_email)
    ref.set({f"seen_status.{movie_id}": status}, merge=True)


# --------------------
# Recommendation Logic (Stub)
# --------------------

def get_recommendations(mood_answers: List[str], user_email: str) -> List[Dict]:
    """
    Placeholder function to simulate calling the backend recommendation engine.
    In a real scenario, this would make an HTTP request to your backend API,
    sending the mood answers and potentially the user email for personalization.
    """
    st.info(f"Simulating recommendation fetch for {user_email} based on answers: {mood_answers}")
    # TODO: Replace with actual backend API call
    # Example using requests library (install it: pip install requests):
    # try:
    #     response = requests.post(
    #         "YOUR_BACKEND_API_ENDPOINT",
    #         json={"answers": mood_answers, "user_email": user_email}
    #     )
    #     response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    #     recommendations = response.json() # Assuming backend returns JSON list of movie dicts
    #     return recommendations
    # except requests.exceptions.RequestException as e:
    #     st.error(f"Error contacting recommendation backend: {e}")
    #     return []

    # --- Placeholder Data ---
    import hashlib
    # Use hashlib for a slightly more robust hash than built-in hash()
    hash_object = hashlib.md5(str(mood_answers).encode())
    base_rating = int(hash_object.hexdigest(), 16) % 40 / 10.0 + 5.0 # Simple hash-based rating variation
    recommendations = []
    example_descriptions = [
        "A perfect pick for a relaxed evening.",
        "An inspiring journey to lift your spirits.",
        "A thrilling ride to keep you on the edge of your seat.",
        "A heartwarming story for a cozy day.",
        "Something light and fun to watch.",
        "A thought-provoking film to engage your mind.",
        "An epic adventure that matches your energy.",
        "A captivating mystery.",
        "A refreshing comedy.",
        "A classic feel-good movie."
    ]
    for i in range(1, 11):
        rec_mood_index = (hash(tuple(mood_answers)) + i) % len(mood_answers) # Simple variation based on input moods
        desc_index = (hash(tuple(mood_answers)) + i) % len(example_descriptions)
        recommendations.append({
            "id": f"rec_{i}", # Use a unique ID for the movie
            "title": f"Recommended Movie {i}",
            "description": f"{example_descriptions[desc_index]} (Mood hint: '{mood_answers[rec_mood_index]}' related).",
            "imdbRating": round(min(10.0, max(1.0, base_rating + (i - 5) * 0.2)), 1) # Vary ratings slightly
        })
    # --- End Placeholder Data ---

    return recommendations

# --- Watch History Logic ---
def load_watch_history():
    """
    Loads random movies as watch history from the CSV file.
    Stores movie details (id, name, year, genre, overview, director, cast).
    """
    if not st.session_state.watch_history:
        try:
            # Load the CSV file
            df = pd.read_csv('movies.csv')

            # Ensure required columns exist
            required_cols = ['movie_id', 'movie_name', 'year', 'genre', 'overview', 'director', 'cast']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"Missing required columns in movies.csv: {', '.join(missing)}")
                # Fallback or stop execution as needed
                return

            # Select a random sample (e.g., 10-15 movies for history)
            num_history_movies = random.randint(10, 15)
            if len(df) < num_history_movies:
                 num_history_movies = len(df) # Ensure we don't sample more rows than available

            history_movies_df = df.sample(n=num_history_movies).reset_index(drop=True)

            # Convert the sampled dataframe to a list of dictionaries
            st.session_state.watch_history = history_movies_df.to_dict('records')

        except FileNotFoundError:
            st.error("movies.csv not found. Please make sure it's in the same directory.")
            # Optionally load placeholder data here if CSV is critical

        except Exception as e:
            st.error(f"An error occurred while loading watch history: {e}")
            # Optionally load placeholder data here if CSV is critical


# --------------------
# Streamlit Page Layout
# --------------------

def main():
    st.set_page_config(page_title="Movie Mood Recommender", layout="wide")
    st.title("🎬 Movie Mood Recommender")

    # --- Authentication ---
    user = handle_login()

    if not user:
        st.warning("Please log in using Google to continue.")
        # The authorize_button is shown by handle_login when user is None
        st.stop() # Stop execution if not logged in

    # --- Logged In View ---
    st.sidebar.header(f"Welcome, {user.get('name', user.get('email'))}!")
    st.sidebar.write(f"Logged in as: {user.get('email')}")
    with st.sidebar:
        handle_logout()

    st.markdown("---") # Separator

    # --- Watch History Toggle Menu ---
    with st.expander("View Your Watch History"):
        load_watch_history() # Load history when the user expands the section
        if st.session_state.watch_history:
            # Display history in columns with card-like formatting
            history_cols = st.columns(4) # Adjust number of columns for watch history display
            for idx, movie in enumerate(st.session_state.watch_history):
                with history_cols[idx % len(history_cols)]:
                    # Use markdown with a border for a simple card effect
                    # Displaying multiple attributes from the movie dictionary
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <p style="font-weight: bold;">{movie.get('movie_name', 'N/A')} ({movie.get('year', 'N/A')})</p>
                            <p style="font-size: small;"><strong>Genre:</strong> {movie.get('genre', 'N/A')}</p>
                            <p style="font-size: small;"><strong>Director:</strong> {movie.get('director', 'N/A')}</p>
                            <p style="font-size: small;"><strong>Cast:</strong> {movie.get('cast', 'N/A')}</p>
                            <details><summary>Overview</summary>{movie.get('overview', 'N/A')}</details>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Your watch history is currently empty (or CSV not loaded).")


    st.markdown("---") # Separator

    # --- Mood Questionnaire ---
    st.subheader("How are you feeling today?")
    st.caption("Answer these questions to help us find movies matching your mood.")

    # Use a form to collect all answers before processing
    with st.form(key="mood_form"):
        mood_questions = [
            "Describe your current energy level (e.g., 'High energy', 'Relaxed', 'Tired'):",
            "What kind of story are you looking for? (e.g., 'Something inspiring', 'A thrilling adventure', 'A heartwarming tale'):",
            "How much focus do you have right now? (e.g., 'Fully focused', 'Easily distracted', 'Want something light'):",
            "What's your preferred movie pace? (e.g., 'Fast-paced', 'Slow burn', 'Medium tempo'):",
            "Any specific genre you're leaning towards or avoiding? (e.g., 'Love sci-fi', 'No horror please', 'Open to anything'):"
        ]
        mood_answers: List[str] = []

        for i, question in enumerate(mood_questions):
            # Using text_input as per your original code, but consider st.selectbox or st.radio for better structure if applicable
            ans = st.text_input(question, key=f"q{i+1}")
            mood_answers.append(ans.strip() if ans else "") # Store trimmed answer, or empty string

        submit_button = st.form_submit_button(label="Find Movies!")

    # --- Recommendations Display ---
    if submit_button:
        # Basic validation: Check if at least some answers were provided
        if not any(mood_answers):
             st.warning("Please answer at least one question to get recommendations.")
        else:
            with st.spinner("Generating recommendations based on your mood..."):
                # Pass the user's email for potential backend personalization
                # Note: The placeholder get_recommendations still uses its own data structure.
                # You would ideally modify your backend/recommendation function
                # to return data using the same structure as your CSV attributes if possible.
                movies = get_recommendations(mood_answers, user['email'])

            st.markdown("---") # Separator
            if movies:
                st.subheader("Here are some movies tailored to your mood:")
                # Display movies in columns with checkboxes
                cols = st.columns(3) # Adjust number of columns as needed
                for idx, movie in enumerate(movies):
                    with cols[idx % len(cols)]: # Cycle through columns
                        # Use a unique key for each checkbox, linked to the movie ID or title
                        # Assuming the recommendation dictionary has an 'id' or 'title'
                        movie_identifier = movie.get('id', movie.get('title', f"rec_{idx}"))
                        checkbox_key = f"seen_{movie_identifier}"
                        # Set the initial state of the checkbox from session state
                        initial_seen_status = st.session_state.seen_movies.get(checkbox_key, False)
                        seen = st.checkbox(f"Seen: {movie.get('title', 'N/A')}", key=checkbox_key, value=initial_seen_status)

                        # Update session state when checkbox is toggled
                        if seen != initial_seen_status:
                            st.session_state.seen_movies[checkbox_key] = seen
                            # Optionally, you could add logic here to send this update to your backend

                        if movie.get('description'):
                             st.write(movie['description'])
                        if movie.get('imdbRating'):
                            st.caption(f"IMDb Rating: {movie['imdbRating']} ⭐")
                        # You could add more details like posters, links, etc. here
                        st.markdown("---") # Separator between movies in a column
            else:
                st.error("Sorry, we couldn't find any recommendations at this time. Please try again later.")

if __name__ == "__main__":
    # This block simply runs the main Streamlit function.
    # The handling of async calls (like fetching user info) occurs within
    # the handle_login function using asyncio.run(), which is a common way
    # to execute a single async task from a synchronous Streamlit context.
    main()