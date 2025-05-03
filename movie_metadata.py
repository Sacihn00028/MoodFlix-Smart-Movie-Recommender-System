import pandas as pd
import json
import os
import time
import numpy as np
from transformers import pipeline
import requests
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MovieEmotionKnowledgeBase:
    """
    Creates a knowledge base mapping movies to emotions by analyzing plot summaries.
    This can be used to recommend movies based on detected emotions from voice.
    """
    
    def __init__(self, use_api=False, api_key=None):
        """
        Initialize the movie-emotion knowledge base generator
        
        Args:
            use_api (bool): Whether to use an external LLM API for plot analysis
            api_key (str): API key if using external API
        """
        self.use_api = use_api
        self.api_key = api_key
        
        print("Loading emotion analysis models...")
        # Emotion analysis model - RoBERTa fine-tuned on emotion datasets
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        
        # Movie genres to emotion mapping - predefined mapping based on research
        self.genre_emotion_map = {
            "Action": ["excitement", "anger", "anticipation"],
            "Adventure": ["excitement", "anticipation", "joy"],
            "Animation": ["joy", "surprise", "amusement"],
            "Comedy": ["joy", "amusement", "surprise"],
            "Crime": ["fear", "anger", "anticipation"],
            "Documentary": ["interest", "surprise", "sadness"],
            "Drama": ["sadness", "anger", "anxiety"],
            "Family": ["joy", "love", "optimism"],
            "Fantasy": ["surprise", "joy", "anticipation"],
            "History": ["interest", "sadness", "pride"],
            "Horror": ["fear", "tension", "disgust"],
            "Music": ["joy", "nostalgia", "love"],
            "Mystery": ["anticipation", "surprise", "tension"],
            "Romance": ["love", "joy", "optimism"],
            "Science Fiction": ["surprise", "anticipation", "awe"],
            "Thriller": ["tension", "fear", "anticipation"],
            "War": ["sadness", "anger", "fear"],
            "Western": ["anticipation", "tension", "pride"]
        }
        
        # Create standardized emotion categories
        self.emotion_categories = {
            "Primary": ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
            "Secondary": ["love", "optimism", "tension", "anticipation", "amusement", 
                         "awe", "nostalgia", "interest", "pride", "anxiety"]
        }
        
        # Map from model output to standardized emotions
        self.model_emotion_mapping = {
            "anger": "anger",
            "joy": "joy", 
            "optimism": "optimism",
            "sadness": "sadness",
            "fear": "fear",
            "disgust": "disgust",
            "surprise": "surprise",
            "anticipation": "anticipation",
            "trust": "optimism",
            "neutral": None
        }
        
        # Initialize movie database (will be loaded or created)
        self.movie_db = {}
        
        print("Movie-Emotion Knowledge Base generator initialized")
    
    def load_movie_dataset(self, file_path=None):
        """
        Load movie dataset from CSV or JSON file
        
        Args:
            file_path (str): Path to movie dataset file
            
        Returns:
            pd.DataFrame: DataFrame containing movie data
        """
        if file_path is None:
            # Create a simple sample dataset if no file provided
            sample_movies = [
                {"title": "The Shawshank Redemption", "year": 1994, "genres": ["Drama"], 
                 "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
                {"title": "The Godfather", "year": 1972, "genres": ["Crime", "Drama"], 
                 "plot": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son."},
                {"title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Drama"], 
                 "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
                {"title": "Finding Nemo", "year": 2003, "genres": ["Animation", "Adventure", "Comedy"], 
                 "plot": "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home."}
            ]
            movies_df = pd.DataFrame(sample_movies)
            print(f"Created sample dataset with {len(movies_df)} movies")
            return movies_df
        
        # Load dataset from file
        if file_path.endswith('.csv'):
            movies_df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            movies_df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSON file.")
        
        print(f"Loaded {len(movies_df)} movies from {file_path}")
        return movies_df
            
    def fetch_plot_summary(self, title, year=None):
        """
        Fetch plot summary for a movie using external API or dataset
        
        Args:
            title (str): Movie title
            year (int): Release year
            
        Returns:
            str: Plot summary
        """
        # For demo purposes, return a placeholder
        # In a real implementation, you would use OMDB API, TMDB API, or similar
        if self.use_api and self.api_key:
            try:
                # Example with OMDB API
                year_param = f"&y={year}" if year else ""
                url = f"http://www.omdbapi.com/?t={title.replace(' ', '+')}{year_param}&plot=full&apikey={self.api_key}"
                response = requests.get(url)
                data = response.json()
                
                if data.get('Response') == 'True':
                    return data.get('Plot', 'No plot available')
                else:
                    print(f"Error fetching plot for {title}: {data.get('Error')}")
                    return None
            except Exception as e:
                print(f"API error: {e}")
                return None
        else:
            # Return placeholder in demo mode
            return f"This is a placeholder plot summary for {title}."
    
    def analyze_plot_emotions(self, plot_text):
        """
        Analyze emotions present in movie plot
        
        Args:
            plot_text (str): Movie plot summary
            
        Returns:
            dict: Emotions detected with scores
        """
        if not plot_text or len(plot_text) < 10:
            return {"dominant_emotion": "unknown", "emotions": {}}
        
        # Use RoBERTa model for emotion analysis
        inputs = self.emotion_tokenizer(plot_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
        emotions = ["anger", "joy", "optimism", "sadness"]
        
        emotion_scores = {emotion: float(score) for emotion, score in zip(emotions, scores)}
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Map to standardized emotion
        mapped_emotion = self.model_emotion_mapping.get(dominant_emotion[0], dominant_emotion[0])
        
        return {
            "dominant_emotion": mapped_emotion,
            "emotions": emotion_scores,
            "original_emotions": emotion_scores
        }
    
    def enhance_with_genre_emotions(self, movie_data, emotion_analysis):
        """
        Enhance emotion analysis using movie genres
        
        Args:
            movie_data (dict): Movie information including genres
            emotion_analysis (dict): Emotion analysis from plot
            
        Returns:
            dict: Enhanced emotion analysis
        """
        if 'genres' not in movie_data or not movie_data['genres']:
            return emotion_analysis
        
        # Get emotions associated with genres
        genre_emotions = []
        for genre in movie_data['genres']:
            if genre in self.genre_emotion_map:
                genre_emotions.extend(self.genre_emotion_map[genre])
        
        # Count frequency of each emotion from genres
        emotion_counts = {}
        for emotion in genre_emotions:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        # Normalize counts
        total = sum(emotion_counts.values())
        if total > 0:
            for emotion in emotion_counts:
                emotion_counts[emotion] = emotion_counts[emotion] / total
        
        # Combine with plot emotions (70% plot, 30% genre)
        combined_emotions = emotion_analysis["emotions"].copy()
        
        # Add genre emotions that don't exist in plot analysis
        for emotion, score in emotion_counts.items():
            if emotion in self.model_emotion_mapping.values():
                # Find corresponding key in model emotions
                model_emotion = next((k for k, v in self.model_emotion_mapping.items() 
                                     if v == emotion and k in combined_emotions), None)
                
                if model_emotion:
                    # Blend with existing emotion
                    combined_emotions[model_emotion] = 0.7 * combined_emotions[model_emotion] + 0.3 * score
                else:
                    # Add as separate entry if we can match it
                    reverse_map = {v: k for k, v in self.model_emotion_mapping.items() if v is not None}
                    if emotion in reverse_map:
                        combined_emotions[reverse_map[emotion]] = 0.3 * score
        
        # Recalculate dominant emotion
        dominant_emotion = max(combined_emotions.items(), key=lambda x: x[1])
        mapped_emotion = self.model_emotion_mapping.get(dominant_emotion[0], dominant_emotion[0])
        
        return {
            "dominant_emotion": mapped_emotion,
            "emotions": combined_emotions,
            "original_emotions": emotion_analysis["original_emotions"],
            "genre_emotions": emotion_counts
        }
    
    def generate_knowledge_base(self, movies_df, output_file="movie_emotion_kb.json"):
        """
        Generate knowledge base mapping movies to emotions
        
        Args:
            movies_df (pd.DataFrame): DataFrame containing movies
            output_file (str): Path to save knowledge base
            
        Returns:
            dict: Knowledge base mapping
        """
        knowledge_base = {
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_movies": len(movies_df),
                "emotions": self.emotion_categories
            },
            "movies": {},
            "emotion_to_movies": {emotion: [] for emotion in 
                               self.emotion_categories["Primary"] + self.emotion_categories["Secondary"]}
        }
        
        # Process each movie
        for _, movie in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Analyzing movies"):
            movie_id = f"{movie['title']}_{movie.get('year', '')}"
            
            # Get plot summary
            if 'plot' in movie and movie['plot']:
                plot = movie['plot']
            else:
                plot = self.fetch_plot_summary(movie['title'], movie.get('year'))
                
            if not plot:
                continue
                
            # Analyze emotions in plot
            emotion_analysis = self.analyze_plot_emotions(plot)
            
            # Enhance with genre information
            enhanced_analysis = self.enhance_with_genre_emotions(movie, emotion_analysis)
            
            # Store in knowledge base
            movie_entry = {
                "title": movie['title'],
                "year": movie.get('year', None),
                "genres": movie.get('genres', []),
                "dominant_emotion": enhanced_analysis["dominant_emotion"],
                "emotion_scores": enhanced_analysis["emotions"],
                "genre_emotions": enhanced_analysis.get("genre_emotions", {})
            }
            
            knowledge_base["movies"][movie_id] = movie_entry
            
            # Add to emotion-to-movies mapping
            dominant = enhanced_analysis["dominant_emotion"]
            if dominant and dominant in knowledge_base["emotion_to_movies"]:
                knowledge_base["emotion_to_movies"][dominant].append(movie_id)
        
        # Save knowledge base
        with open(output_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
            
        print(f"Knowledge base saved to {output_file}")
        return knowledge_base
    
    def get_recommendations(self, detected_emotion, kb_file=None, num_recommendations=5):
        """
        Get movie recommendations based on detected emotion
        
        Args:
            detected_emotion (str): Detected emotion
            kb_file (str): Knowledge base file
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: Recommended movies
        """
        # Load knowledge base if needed
        if kb_file and os.path.exists(kb_file):
            with open(kb_file, 'r') as f:
                kb = json.load(f)
        else:
            # Generate simple KB with sample data
            movies_df = self.load_movie_dataset()
            kb = self.generate_knowledge_base(movies_df)
        
        # Map to standardized emotion if needed
        mapped_emotion = detected_emotion
        for e_map in self.model_emotion_mapping.items():
            if detected_emotion == e_map[0]:
                mapped_emotion = e_map[1] if e_map[1] else detected_emotion
                break
        
        # Get recommendations
        recommendations = []
        
        # Direct match with dominant emotion
        if mapped_emotion in kb["emotion_to_movies"]:
            direct_matches = kb["emotion_to_movies"][mapped_emotion]
            for movie_id in direct_matches[:num_recommendations]:
                if movie_id in kb["movies"]:
                    recommendations.append(kb["movies"][movie_id])
        
        # If not enough, add recommendations from related emotions
        if len(recommendations) < num_recommendations:
            # Get related emotions
            related = {
                "joy": ["optimism", "love", "amusement"],
                "sadness": ["nostalgia", "anxiety"],
                "anger": ["tension", "disgust"],
                "fear": ["anxiety", "tension"],
                "surprise": ["awe", "amusement"],
                "disgust": ["anger"],
                "love": ["joy", "optimism"],
                "optimism": ["joy", "pride"],
                "tension": ["anticipation", "fear"],
                "anticipation": ["excitement", "tension"],
                "amusement": ["joy", "surprise"],
                "awe": ["surprise", "anticipation"],
                "nostalgia": ["joy", "sadness"],
                "interest": ["anticipation"],
                "pride": ["joy", "optimism"],
                "anxiety": ["fear", "tension"]
            }
            
            related_emotions = related.get(mapped_emotion, [])
            
            for rel_emotion in related_emotions:
                if rel_emotion in kb["emotion_to_movies"]:
                    for movie_id in kb["emotion_to_movies"][rel_emotion]:
                        if movie_id in kb["movies"] and len(recommendations) < num_recommendations:
                            if not any(r["title"] == kb["movies"][movie_id]["title"] for r in recommendations):
                                recommendations.append(kb["movies"][movie_id])
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Create movie-emotion knowledge base
    movie_kb = MovieEmotionKnowledgeBase(use_api=False)
    
    # Load or create dataset
    movies_df = movie_kb.load_movie_dataset()
    
    # Generate knowledge base
    kb = movie_kb.generate_knowledge_base(movies_df)
    
    # Get recommendations for an emotion
    recommendations = movie_kb.get_recommendations("anger")
    
    print("\n=== Movie Recommendations for 'anger' ===")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['title']} ({movie.get('year', 'N/A')}) - {movie['dominant_emotion']}")