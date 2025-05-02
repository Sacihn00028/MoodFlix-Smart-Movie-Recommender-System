
import requests
from google import genai

def call_gemini_api(prompt, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

def get_preference_response(time, mood, language):
    prompt = f"I am free for {time} hours, I am feeling {mood} today and I would like to watch a show/movie in {language} language."
    gemini_reply = call_gemini_api(prompt, api_key="AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM")

    preference_dict = {
        "time": time,
        "mood": mood,
        "language": language,
        "gemini_reply": gemini_reply  # Should now be just a list of movies
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
