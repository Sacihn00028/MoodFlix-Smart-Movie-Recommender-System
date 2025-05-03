import streamlit as st
import time
import os
import re
import ast
import google.generativeai as genai
import speech_recognition as sr

# --- Configure Gemini API ---
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDmjdwnfVSObjHaPaaytSBDG1FYVhYiqaM")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Emotion list ---
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
Return the answer as a JSON array of mood words (e.g., [\"mood1\", \"mood2\", \"mood3\"]).\n\nMovie Description: \"{description}\""""
    response = model.generate_content(prompt)
    raw = response.text.strip()
    # Extract array substring
    match = re.search(r"\[.*\]", raw)
    if match:
        raw = match.group(0)
    try:
        moods = ast.literal_eval(raw)
        if not isinstance(moods, list):
            raise ValueError("Parsed output is not a list")
    except Exception as e:
        st.error(f"Failed to parse moods: {e}")
        return []
    return moods

# --- Streamlit UI ---
st.set_page_config(page_title="MoodFlixx Voice Interface", layout="wide")
st.title("🎬 MoodFlixx Voice-Based Mood Analyzer")
st.markdown("Speak a movie description and discover its emotional tones!")

# Sidebar for recording duration
with st.sidebar:
    st.header("Settings")
    duration = st.slider("Recording Duration (seconds)", min_value=5, max_value=60, value=20, step=5)
    st.markdown("Click **Start Recording** to begin.")

# Main: recording and processing
action = st.button("▶️ Start Recording")
if action:
    r = sr.Recognizer()
    chunks = []
    timer_placeholder = st.empty()
    with sr.Microphone() as mic:
        r.adjust_for_ambient_noise(mic, duration=1)
        for sec in range(duration, 0, -1):
            timer_placeholder.info(f"🔴 Recording... {sec} seconds left")
            chunk = r.record(mic, duration=1)
            chunks.append(chunk)
        timer_placeholder.success("⏹️ Recording complete!")

    # Combine audio chunks
    sample_rate = chunks[0].sample_rate
    sample_width = chunks[0].sample_width
    raw_data = b"".join(chunk.get_raw_data() for chunk in chunks)
    full_audio = sr.AudioData(raw_data, sample_rate, sample_width)

    # Transcribe
    try:
        with st.spinner("📝 Transcribing..."):
            transcription = r.recognize_google(full_audio)
        st.subheader("📝 Transcription")
        st.write(transcription)
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        st.stop()

    # Infer moods
    with st.spinner("🤖 Inferring moods..."):
        moods = infer_moods(transcription)
    st.subheader("🎭 Predicted Moods")
    if moods:
        st.write(moods)
        st.balloons()
    else:
        st.warning("No moods could be parsed from the AI response.")

# Instructions
