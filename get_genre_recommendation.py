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
    # Updated prompt for concise output
    prompt = (
        f"I am free for {time} hours, I am feeling {mood} today, and I would like "
        f"to watch a show/movie in {language} language. Please respond with a list "
        f"of recommended movies only—no extra text."
    )
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
    example_mood = "happy"
    example_language = "English"
    
    preferences = get_preference_response(2, "Happy", "Hindi")
    print(preferences)