import os
import google.generativeai as genai
import speech_recognition as sr

# --- Configure Gemini API ---
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Your emotion list (unchanged) ---
emotion_list = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral",
]

def infer_moods(description: str, k: int = 3, emotions=emotion_list) -> list:
    prompt = f"""Given the following movie description, list the possible moods or emotional tones it conveys. \
Strictly output the top {k} most applicable moods from {emotions} list. \
Return the answer as a Python list of mood words.

Movie Description: "{description}"

Note that I only want the moods and not the reasons or any other information. \
Do not include any additional text or explanation. \
Just provide the list of moods in a Python list format, like this: ['mood1', 'mood2', 'mood3'].
"""
    response = model.generate_content(prompt)
    # The API returns text like "['joy', 'excitement', 'surprise']"
    # We can safely use eval() here because we fully control the format.
    return eval(response.text.strip())

# --- Step 1: Record 20 seconds of audio ---
r = sr.Recognizer()
with sr.Microphone() as mic:
    print("🎤 Please speak your movie description now (recording for 20 seconds)...")
    r.adjust_for_ambient_noise(mic, duration=1)
    audio_data = r.record(mic, duration=10)
    print("⏹️ Recording complete, transcribing…")

# --- Step 2: Transcribe to text ---
try:
    description_text = r.recognize_google(audio_data)
    print("📝 Transcribed description:", description_text)
except sr.UnknownValueError:
    print("⚠️ Google Speech Recognition could not understand audio")
    exit(1)
except sr.RequestError as e:
    print(f"⚠️ Could not request results from Google Speech Recognition service; {e}")
    exit(1)

# --- Step 3: Infer moods via Gemini ---
try:
    moods = infer_moods(description_text)
    print("🎬 Inferred moods:", moods)
except Exception as e:
    print("❌ Error calling Gemini API:", e)