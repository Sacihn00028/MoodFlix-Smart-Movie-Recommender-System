import google.generativeai as genai
import os

# Use your API key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM")
genai.configure(api_key=API_KEY)

# Use the correct model name
# The 'gemini-pro' model should be accessed as 'models/gemini-pro'
model = genai.GenerativeModel('gemini-2.0-flash')

# Movie descriptions
movie_descriptions = {
    "Inception": "A skilled thief is given a chance at redemption if he can successfully perform an inception: planting an idea into someone's subconscious.",
    "The Pursuit of Happyness": "A struggling salesman takes custody of his son as he's poised to begin a life-changing professional endeavor.",
    "The Conjuring": "Paranormal investigators help a family terrorized by a dark presence in their farmhouse.",
    "Inside Out": "After young Riley is uprooted from her Midwest life, her emotions—Joy, Fear, Anger, Disgust and Sadness—conflict on how best to navigate a new city.",
}

emotion_list = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

def infer_moods(description , k=3 , emotions=emotion_list):
    prompt = f"""Given the following movie description, list the possible moods or emotional tones it conveys. Strictly output the top {k} most applicable moods from {emotions} list. Return the answer as a Python list of mood words.

Movie Description: "{description} , Note that I only want the moods and not the reasons or any other information. Do not include any additional text or explanation. Just provide the list of moods in a Python list format, like this: ['mood1', 'mood2', 'mood3']."
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Run the mood analysis
for title, desc in movie_descriptions.items():
    print(f"\n🎬 {title}")
    try:
        moods = infer_moods(desc)
        print(f"Moods: {moods}")
    except Exception as e:
        print(f"Error: {e}")