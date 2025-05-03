## This file will do the following:
## 1. Query Each of the knowledge graphs for the top-k recommendations matching the desired characteristics
## 2. using that gotten knowledge, pass to llm a very good prompt with constraint output to get top recommendations
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from typing import List
import os
from pydantic import BaseModel, Field
from typing import Union
import json
os.environ["GOOGLE_API_KEY"] ="AIzaSyApbJXSZW8wV7MBXlBv0W0MA3KGHI_fDp4"
candidate_moods = ["Happy", "Sad", "Thrilling", "Romantic", "Adventurous", "Dark", "Inspiring"]
uri = "neo4j+s://fea8a723.databases.neo4j.io"
username = "neo4j"
password = "Qp3U5o9HjMkHPuLjj9M4vL91doNcq3Hj4fGFpZV7-XI"
driver = GraphDatabase.driver(uri, auth=(username, password))
class MoviePreferences(BaseModel):
    mood: str = Field(description="User's desired mood. Choose from: " + ", ".join(candidate_moods))
    runtime: Union[int, str] = Field(description="Maximum runtime in minutes or 'any'")
    director: str = Field(description="Preferred director or 'any'")
    rating: Union[float, str] = Field(description="Minimum rating or 'any'")
prompt_template = PromptTemplate.from_template("""
You are an intelligent movie assistant. Extract structured movie preferences from the following user message.

Message: "{user_input}"

Return a JSON object with the following fields:
- mood: One of {candidate_moods} or 'any'
- runtime: Desired maximum runtime in minutes, or 'any'
- director: Preferred director, or 'any'
- rating: Minimum rating (e.g., 8.0), or 'any'
""".strip())

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
def get_top_k_movies_on_mood(mood, k):
    query = f"""
    MATCH (mo:Mood {{name: $mood}})-[:RECOMMENDS]->(m:Movie)
    RETURN m.title AS title, m.director AS director, m.year AS year, m.rating AS rating
    ORDER BY m.rating DESC
    LIMIT $k
    """
    
    with driver.session() as session:
        result = session.run(query, mood=mood, k=k)
        movies = result.data()
        return movies


def get_top_k_movies_less_than_runtime(runtime, k):
    query = """
    MATCH (m:Movie)
    WHERE m.runtime <= $runtime
    RETURN m.title AS title, m.director AS director, m.year AS year, m.rating AS rating
    ORDER BY m.runtime ASC
    LIMIT $k
    """
    with driver.session() as session:
        result = session.run(query, runtime=runtime, k=k)
        return result.data()

def get_movies_by_director_and_runtime(director, runtime, k):
    query = """
    MATCH (d:Director {name: $director})-[:DIRECTED]->(m:Movie)
    WHERE m.runtime <= $runtime
    RETURN m.title AS title, m.director AS director, m.year AS year, m.rating AS rating
    ORDER BY m.runtime ASC
    LIMIT $k
    """
    with driver.session() as session:
        result = session.run(query, director=director, runtime=runtime, k=k)
        return result.data()
def get_top_k_movies_by_rating(min_rating, k):
    query = """
    MATCH (m:Movie)
    WHERE m.rating >= $min_rating
    RETURN m.title AS title, m.director AS director, m.year AS year, m.rating AS rating
    ORDER BY m.rating DESC
    LIMIT $k
    """
    with driver.session() as session:
        result = session.run(query, min_rating=min_rating, k=k)
        return result.data()


def gettopK(cls, spec, k):
    if(cls == "mood"):
        mood_movies = get_top_k_movies_on_mood(spec["mood"], k)
        return mood_movies
    if(cls == "runtime"):
        runtime_movies = get_top_k_movies_less_than_runtime(int(spec["runtime"]), k)
        return runtime_movies
    if(cls == "director"):
        director_movies = get_movies_by_director_and_runtime(spec["director"], int(spec["runtime"]), k)
        return director_movies
    if(cls == "rating"):
        rating_movies = get_top_k_movies_by_rating(float(spec["rating"]), k)
        return rating_movies





def dict_to_prompt(spec: dict) -> str:
    """
    Convert a spec dict into a natural-language prompt sentence.
    Fields with value "any" will be skipped or phrased generically.
    """
    parts = []
    mood = spec.get("mood", "any")
    if mood != "any":
        parts.append(f"a {mood.lower()} movie")
    else:
        parts.append("any kind of movie")
    runtime = spec.get("runtime", "any")
    if isinstance(runtime, int):
        parts.append(f"up to {runtime} minutes long")
    elif runtime != "any":
        parts.append(f"up to {int(runtime)} minutes long")
    director = spec.get("director", "any")
    if director != "any":
        parts.append(f"directed by {director}")
    rating = spec.get("rating", "any")
    if isinstance(rating, (int, float)):
        parts.append(f"with rating ≥{rating}")
    elif rating != "any":
        parts.append(f"with rating ≥{float(rating)}")

    # join into one sentence
    prompt = "I want " + ", ".join(parts) + "."
    return prompt
def parse_llm_response(response_content: str) -> dict:
    try:
        cleaned_response = response_content.strip('```json\n').strip('```')
        parsed_data = json.loads(cleaned_response)
        prefs = MoviePreferences(**parsed_data)
        return prefs.model_dump()
    except Exception as e:
        return {
            "mood": "any",
            "runtime": "any",
            "director": "any",
            "rating": "any"
        }

def Enhance_User_Prompt(user_input: str) -> dict:
    prompt = prompt_template.invoke({
        "user_input": user_input,
        "candidate_moods": candidate_moods
    })
    response = llm.invoke(prompt)
    try:
        return parse_llm_response(response.content)
    except Exception as e:
        return parse_llm_response(response.content)


def run_KG_Fetch(user_input, k):
    string = dict_to_prompt(user_input)
    enhanced = Enhance_User_Prompt(string)
    recommendations = []
    if(enhanced['mood'] != 'any'):
        mood = gettopK("mood", enhanced, k)
        for i in mood:
            recommendations.append(i)
    if(enhanced['runtime'] != "any"):
        runtime = gettopK("runtime", enhanced, k)
        for i in runtime:
            recommendations.append(i)
    if(enhanced['director'] != "any"):
        direc = gettopK('director', enhanced, k)
        for i in direc:
            recommendations.append(i)
    if(enhanced['rating'] != "any"):
        rating = gettopK("rating", enhanced, k)
        for i in rating:
            recommendations.append(i)
    return recommendations

def KG_pipeline(user_input, k):
    recommendations_from_KG = run_KG_Fetch(user_input, k)
    return recommendations_from_KG


if __name__=="__main__":
    spec = {"mood":"Death", "runtime":"210", "rating":"7.8"}
    # print(gettopK("mood", spec, 5))
    # print(gettopK("runtime", spec, 5))
    # print(gettopK("director", spec, 5))
    # print(gettopK("rating", spec, 5))
    # stri = dict_to_prompt(spec)
    # print(Enhance_User_Prompt(stri))
    print(run_KG_Fetch(spec, 5))
    from neo4j._sync.driver import Driver as _Neo4jDriver
    def _noop_del(self):
        return None
    # override the finalizer so it never runs the broken code
    _Neo4jDriver.__del__ = _noop_del