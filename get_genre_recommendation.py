import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # Import RunnablePassthrough

# Ensure you have set your API key as an environment variable
# Or you can set it directly in the code (less recommended for security)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM" # Replace with your actual key

def get_preference_response_langchain(time: int, mood: str, language: str):
    """
    Generates a movie/show preference response using Langchain and the Gemini API.

    Args:
        time: The number of hours the user is free.
        mood: The user's current mood.
        language: The preferred language for the movie/show.

    Returns:
        A dictionary containing the user's preferences and the Gemini API response.
    """
    # 1. Define the Language Model (LLM)
    # Using ChatGoogleGenerativeAI for the chat-based Gemini models
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7) # Using gemini-pro for chat

    # 2. Define the Prompt Template
    prompt_template = PromptTemplate.from_template(
        "I am free for {time} hours, I am feeling {mood} today "
        "and I would like to watch a show/movie in {language} language. "
        "Please suggest some options based on these preferences."
    )

    # 3. Define the Output Parser
    # We want the raw text response from the model
    output_parser = StrOutputParser()

    # 4. Create a Langchain Chain
    # This chain takes the input variables, formats the prompt, and passes it to the LLM
    chain = (
        {"time": RunnablePassthrough(), "mood": RunnablePassthrough(), "language": RunnablePassthrough()} # Pass inputs directly
        | prompt_template
        | llm
        | output_parser
    )

    # 5. Invoke the Chain with Input
    # Pass the input variables as a dictionary to the chain
    gemini_reply = chain.invoke({"time": time, "mood": mood, "language": language})

    preference_dict = {
        "time": time,
        "mood": mood,
        "language": language,
        "gemini_reply": gemini_reply
    }
    return preference_dict

if __name__ == "__main__":
    # Sample usage
    example_time = 2
    example_mood = "Happy"
    example_language = "Hindi"

    # Set your API key before calling the function if you're not using environment variables
    # os.environ["GOOGLE_API_KEY"] = "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM"

    preferences = get_preference_response_langchain(example_time, example_mood, example_language)
    print(preferences)
