import streamlit as st
from CStrings.iterative import iterative_cstring_gen 
from langchain.schema import Document
from KnowledgeBase.knowledge_base import Get_knowledge_Base_Lang
from langchain_groq import ChatGroq
from typing import List, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
vs = Get_knowledge_Base_Lang()
st.title("Movie Recommender")
st.markdown("I will guess your Movie based of what you fill below (all are optional to fill)")

# Input fields
mood = st.text_input("Describe what you are feeling or what you want to feel?", "")
runtime = st.text_input("How much time can you spare for the content?", "")
age = st.text_input("Your age group", "")
genre = st.text_input("Genres that you prefer right now (if any)", "")
language = st.text_input("Any language prefrence?", "")
year = st.text_input("Which year's content would you prefer?","")
actor = st.text_input("Any actor you wanna watch?", "")
n_characteristics = st.number_input(
    "Depth of Search", min_value=1, max_value=10, value=1, step=1
)
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
Also only consider UNIQUE movies only
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

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
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

if st.button("Generate and Search"):
    user_data = {
        "mood": mood,
        "runtime": runtime,
        "age": age,
        "genre": genre,
        "language":language,
        "actor":actor,
        "year":year,
    }
    # Generate characteristic strings
    resp = iterative_cstring_gen(user_data, 3, 2)
    results = []
    for iterres in resp:
        similar_docs = vs.similarity_search(iterres.prompt, k=1)
        if similar_docs:
            for doc in similar_docs:
                metadata = doc.metadata
                movie_data = {
                    "Title": metadata.get("original_title", "N/A"),
                    "Year": metadata.get("release_date", "N/A"),
                    "Genre": metadata.get("genres", "N/A"),
                    "Director": metadata.get("director", "N/A"),
                    "Cast": metadata.get("cast", "N/A"),
                    "Full Description": doc.page_content,
                }
                results.append(movie_data)
    top_movies = get_top_k_movies_llm(results, k=5)
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
