## This file will do the following:
## 1. Query Each of the knowledge graphs for the top-k recommendations matching the desired characteristics
## 2. using that gotten knowledge, pass to llm a very good prompt with constraint output to get top recommendations
from neo4j import GraphDatabase
uri = "neo4j+s://fea8a723.databases.neo4j.io"
username = "neo4j"
password = "Qp3U5o9HjMkHPuLjj9M4vL91doNcq3Hj4fGFpZV7-XI"
driver = GraphDatabase.driver(uri, auth=(username, password))
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
    RETURN m.title AS title, m.runtime AS runtime
    ORDER BY m.runtime ASC
    LIMIT $k
    """
    with driver.session() as session:
        result = session.run(query, runtime=runtime, k=k)
        return result.data()

def gettopK(cls, spec, k):
    if(cls == "mood"):
        mood_movies = get_top_k_movies_on_mood(spec["mood"], k)
        return mood_movies
    if(cls == "runtime"):
        runtime_movies = get_top_k_movies_less_than_runtime(spec["runtime"], k)
        return runtime_movies
    # if(cls == "")
candidate_moods = ["Happy", "Sad", "Thrilling", "Romantic", "Adventurous", "Dark", "Inspiring"]
if __name__=="__main__":
    spec = {"mood":"Sad", "runtime":"159"}
    print(gettopK("mood", spec, 5))
    print(gettopK("runtime", spec, 5))



