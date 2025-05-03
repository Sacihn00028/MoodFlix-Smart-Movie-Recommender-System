
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from typing import List
from pydantic import BaseModel, Field
os.environ["GOOGLE_API_KEY"] ="AIzaSyApbJXSZW8wV7MBXlBv0W0MA3KGHI_fDp4"
class CharacteristicString(BaseModel):
    # The description here is important to guide the LLM
    prompt : str = Field(description="Detailed characteristic string describing a potential movie or show plot/theme based on user preferences. Avoid specific titles or character names.")

class Modelresponse(BaseModel):
    prompts : List[CharacteristicString] = Field(description="A list of characteristic strings describing potential entertainment content.")


def questions_to_cstring(user_data, n):
    mood = user_data.get('mood', 'neutral') # Use .get() with a default for safety
    genre = user_data.get('genre', 'any')
    runtime = user_data.get('runtime', 'any')
    age_group = user_data.get('age', 'any')

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Enhanced Prompt Template
    prompt_template = PromptTemplate.from_template(
        "Based on the following user preferences, generate {n} detailed characteristic strings. "
        "Each string should describe a potential movie or show plot, theme, or scenario that fits the criteria. "
        "Focus on the narrative, atmosphere, and elements of the content. "
        "Please note that the described scenario, plot, movies, must cater to the people with the given mood and not necessarily entail such moods. "
        "Try to Keep the content limited to 30 words"
        "Do NOT include specific titles, character names, actors, directors, or any named entities. "
        "Each characteristic string should be a descriptive summary of the content type.\n\n"
        "User Preferences:\n"
        "- Mood: {mood}\n"
        "- Available Time/Runtime: {runtime}\n"
        "- Age Group: {age}\n"
        "- Preferred Genre: {genre}\n\n"
        "Provide the response in the specified JSON format."
    )
    structured_llm = llm.with_structured_output(Modelresponse)
    chain = (
        {
            "mood": RunnablePassthrough(),
            "runtime": RunnablePassthrough(),
            "age": RunnablePassthrough(),
            "genre": RunnablePassthrough(),
            "n": RunnablePassthrough()
        }
        | prompt_template
        | structured_llm
    )

    resp = chain.invoke({
        "mood": mood,
        "runtime": runtime,
        "age": age_group,
        "genre": genre,
        "n": n
    })
    return resp

def questions_to_cstring_iter(user_data, n, notincl):
    mood = user_data.get('mood', 'neutral') # Use .get() with a default for safety
    genre = user_data.get('genre', 'any')
    runtime = user_data.get('runtime', 'any')
    age_group = user_data.get('age', 'any')
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    # Enhanced Prompt Template
    prompt_template = PromptTemplate.from_template(
        "Based on the following user preferences, generate {n} detailed characteristic strings. "
        "Each string should describe a potential movie or show plot, theme, or scenario that fits the criteria. "
        "Focus on the narrative, atmosphere, and elements of the content. "
        "Please note that the described scenario, plot, movies, must cater to the people with the given mood and not necessarily entail such moods. "
        "Try to Keep the content limited to 30 words"
        "Do NOT include specific titles, character names, actors, directors, or any named entities. "
        "Also DO NOT Generate stories of the form {notincl}"
        "Each characteristic string should be a descriptive summary of the content type.\n\n"
        "User Preferences:\n"
        "- Mood: {mood}\n"
        "- Available Time/Runtime: {runtime}\n"
        "- Age Group: {age}\n"
        "- Preferred Genre: {genre}\n\n"
        "Provide the response in the specified JSON format."
    )
    structured_llm = llm.with_structured_output(Modelresponse)
    chain = (
        {
            "mood": RunnablePassthrough(),
            "runtime": RunnablePassthrough(),
            "age": RunnablePassthrough(),
            "genre": RunnablePassthrough(),
            "n": RunnablePassthrough(),
            "notincl":RunnablePassthrough()
        }
        | prompt_template
        | structured_llm
    )

    resp = chain.invoke({
        "mood": mood,
        "runtime": runtime,
        "age": age_group,
        "genre": genre,
        "n": n,
        "notincl":notincl
    })
    return resp

def iterative_cstring_gen(user_data, n_iter = 3, cstring_per_iter = 5):

    resp = questions_to_cstring(user_data, cstring_per_iter)
    not_incl = ""
    total = []
    for _ in range(n_iter):
        if(_ == 0):
            not_incl = "\n"
            for i, cstring in enumerate(resp.prompts):
                not_incl += f"{cstring.prompt}\n"
                total.append(cstring)
        else:
            resp = questions_to_cstring_iter(user_data, cstring_per_iter, not_incl)
            not_incl = "\n"
            for i, cstring in enumerate(resp.prompts):
                not_incl += f"{cstring.prompt}\n"
                total.append(cstring)
    return total




                    
if __name__ == "__main__":
    # Sample usage
    user_data = {"mood": "Happy", "runtime": "2 hours", "age": "20", "genre": "Horror"}
    n_characteristics = 10
    # resp = questions_to_cstring(user_data, n_characteristics)
    resp = iterative_cstring_gen(user_data, 3, 5)
    for i in resp:
        print(i)