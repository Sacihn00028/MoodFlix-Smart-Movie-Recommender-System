import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from typing import List
from pydantic import BaseModel, Field


os.environ["GOOGLE_API_KEY"] = "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM" # Replace with your actual key

class Recommendation(BaseModel):
    """Represents a single movie or show recommendation."""
    title: str = Field(description="The title of the movie or show.")
    reason: str = Field(description="A brief explanation of why this is a good recommendation.")

class MoviePreferenceResponse(BaseModel):
    """Represents the structured response for movie preferences."""
    recommendations: List[Recommendation] = Field(
        description="A list of recommended movies or shows."
    )
    conclusion: str = Field(
        description="A general concluding remark about the recommendations."
    )

def get_structured_preference_response_langchain(time: int, mood: str, language: str):


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
    prompt_template = PromptTemplate.from_template(
        "I am free for {time} hours, I am feeling {mood} today "
        "and I would like to watch a show/movie in {language} language. "
        "Please suggest some options based on these preferences. "
        "Provide the response in the specified JSON format."
    )

    # 3. Create a Langchain Chain with Structured Output
    # We wrap the LLM with with_structured_output, specifying the Pydantic schema
    structured_llm = llm.with_structured_output(MoviePreferenceResponse)

    # 4. Create the Full Chain
    chain = (
        {"time": RunnablePassthrough(), "mood": RunnablePassthrough(), "language": RunnablePassthrough()}
        | prompt_template
        | structured_llm # Use the LLM with structured output
    )

    # 5. Invoke the Chain with Input
    # The output will be automatically parsed into the MoviePreferenceResponse Pydantic model
    structured_response = chain.invoke({"time": time, "mood": mood, "language": language})

    # The result is the parsed Pydantic model
    preference_dict = {
        "time": time,
        "mood": mood,
        "language": language,
        "gemini_response": structured_response  # Store the Pydantic model here
    }
    return preference_dict

if __name__ == "__main__":
    # Sample usage
    example_time = 2
    example_mood = "Happy"
    example_language = "Hindi"

    # Set your API key before calling the function if you're not using environment variables
    # os.environ["GOOGLE_API_KEY"] = "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM"

    preferences = get_structured_preference_response_langchain(example_time, example_mood, example_language)

    print("Preferences and Structured Response:")
    print(f"Time: {preferences['time']}")
    print(f"Mood: {preferences['mood']}")
    print(f"Language: {preferences['language']}")

    print("\nGemini Structured Response:")
    gemini_response_model = preferences['gemini_response']

    print("Recommendations:")
    for recommendation in gemini_response_model.recommendations:
        print(f"- Title: {recommendation.title}")
        print(f"  Reason: {recommendation.reason}")

    print("\nConclusion:")
    print(gemini_response_model.conclusion)

    # You can access the structured data easily
    first_recommendation_title = gemini_response_model.recommendations[0].title
    print(f"\nTitle of the first recommendation: {first_recommendation_title}")
