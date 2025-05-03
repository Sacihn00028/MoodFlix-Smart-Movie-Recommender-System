import streamlit as st
import json
import os
from typing import List, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
# Your imports (adjust paths if needed)
from KG.KG_pipeline import run_KG_Fetch
from CStrings.iterative import iterative_cstring_gen
from KnowledgeBase.structure_data import Get_Knowledge_Base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from MoodHandling.mood_handling_text import infer_user_mood
# Initialize vector store once
vs = Get_Knowledge_Base("E")

# Pydantic models
class MovieRecommendation(BaseModel):
    title: str = Field(..., description="Movie title")
    year: Union[str, int]
    genre: str
    director: str
    reason: str = Field(..., description="Why this movie is recommended")

class FinalMovieList(BaseModel):
    recommendations: List[MovieRecommendation]

def build_context_string(movies: List[dict]) -> str:
    context = ""
    for i, movie in enumerate(movies, 1):
        context += f"""
Movie {i}:
Title: {movie.get("Title", "N/A")}
Year: {movie.get("Year", "N/A")}
Genre: {movie.get("Genre", "N/A")}
Director: {movie.get("Director", "N/A")}
Cast: {movie.get("Cast", "N/A")}
Metascore: {movie.get("Metascore", "N/A")}
Description: {movie.get("Full Description", "N/A")}
"""
    return context

prompt_template = PromptTemplate.from_template(
    """
You are a movie assistant. Given the following movie candidates, select the best {k} movie recommendations for the user.

Provide your response in the following JSON format:
```json
{{
  "recommendations": [
    {{
      "title": "...",
      "year": ...,
      "genre": "...",
      "director": "...",
      "reason": "..."
    }}
  ]
}}
Movies:
{context}
"""
)

def get_top_k_movies_llm(combined_movies: List[dict], k: int = 5) -> dict:
    context_str = build_context_string(combined_movies)
    prompt = prompt_template.invoke({
        "context": context_str,
        "k": k,
    })

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
    response = llm.invoke(prompt)

    try:
        start = response.content.find("{")
        end = response.content.rfind("}") + 1
        json_content = response.content[start:end]
        parsed = FinalMovieList.model_validate_json(json_content)
        return parsed.model_dump()
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": response.content,
        }

def main():
    st.title("Movie Recommendation Assistant")
    mood = st.text_input("Describe What you are Feeling", value="")
    rating = st.slider("Rating of the movie you would like to watch", 1, 10, 5)
    year = st.text_input("Year", value="2020")
    minutes = st.slider("Minutes", 20, 300, 120)
    genre = st.text_input("Genre you would like to watch", value="")
    if st.button("Get Recommendations"):
        user_data = {
            "mood": mood,
            "runtime": str(minutes),
            "age": year,
            "genre": genre,
        }
        user_data["mood"] = infer_user_mood(user_data)
        # st.write(f'Inferred Moods: {user_data["mood"]}')
        recommendations_from_KG = run_KG_Fetch(user_data, 3)
        similarity_recommendations_from_KG = []
        for recond in recommendations_from_KG:
            stri = json.dumps(recond)
            similar_docs = vs.similarity_search(stri, k=1)
            if similar_docs:
                for doc in similar_docs:
                    metadata = doc.metadata
                    movie_data = {
                        "Title": metadata.get("movie_name", "N/A"),
                        "Year": metadata.get("year", "N/A"),
                        "Genre": metadata.get("genre", "N/A"),
                        "Director": metadata.get("director", "N/A"),
                        "Cast": metadata.get("cast", "N/A"),
                        "Metascore": metadata.get("metascore", "N/A"),
                        "Full Description": doc.page_content,
                    }
                    similarity_recommendations_from_KG.append(movie_data)
        resp = iterative_cstring_gen(user_data, 2, 2)
        results = []
        for iterres in resp:
            similar_docs = vs.similarity_search(iterres.prompt, k=1)
            if similar_docs:
                for doc in similar_docs:
                    metadata = doc.metadata
                    movie_data = {
                        "Title": metadata.get("movie_name", "N/A"),
                        "Year": metadata.get("year", "N/A"),
                        "Genre": metadata.get("genre", "N/A"),
                        "Director": metadata.get("director", "N/A"),
                        "Cast": metadata.get("cast", "N/A"),
                        "Metascore": metadata.get("metascore", "N/A"),
                        "Full Description": doc.page_content,
                    }
                    results.append(movie_data)
        combined_movies = similarity_recommendations_from_KG + results
        os.environ["GOOGLE_API_KEY"] = "AIzaSyApbJXSZW8wV7MBXlBv0W0MA3KGHI_fDp4"
        top_movies = get_top_k_movies_llm(combined_movies, k=5)
        if "error" in top_movies:
            st.error(f"Error: {top_movies['error']}")
            st.text(top_movies["raw_response"])
        else:
            st.subheader("Top Movie Recommendations")
            for rec in top_movies.get("recommendations", []):
                st.markdown(f"### {rec.get('title', 'N/A')} ({rec.get('year', 'N/A')})")
                st.markdown(f"**Genre:** {rec.get('genre', 'N/A')}")
                st.markdown(f"**Director:** {rec.get('director', 'N/A')}")
                st.markdown(f"**Reason:** {rec.get('reason', 'N/A')}")
                st.markdown("---")
if __name__ == "__main__":
    main()
