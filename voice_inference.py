import speech_recognition as sr
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def ask_questions(questions):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    answers = []

    for idx, question in enumerate(questions, 1):
        st.write(f"Question {idx}: {question}")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            st.write("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                st.write(f"You said: {text}\n")
                answers.append(text)
            except sr.UnknownValueError:
                st.write("Sorry, could not understand the audio. Please try again.")
                answers.append("")
            except sr.RequestError as e:
                st.write(f"Could not request results; {e}")
                answers.append("")
    return answers


def analyze_mood(answers):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(ans)['compound'] for ans in answers if ans]
    if not scores:
        return "Neutral"

    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.05:
        return "Positive"
    elif avg_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def main():
    st.title("Voice Mood Analyzer")
    st.write("Answer the questions below to analyze your mood.")

    questions = [
        "How are you feeling today?",
        "What made you smile recently?",
        "Are you stressed or relaxed right now?",
        "What’s something that’s been on your mind?",
        "How would you describe your energy level today?",
        "Did anything disappoint you today?",
        "What are you looking forward to?",
        "Do you feel calm or anxious at this moment?",
        "What emotion best describes your mood?",
        "Is there anything you’d like to talk about?"
    ]

    if st.button("Start"):
        answers = ask_questions(questions)
        mood = analyze_mood(answers)
        st.write(f"\nBased on your responses, your overall mood is: {mood}")


if __name__ == "__main__":
    main()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def ask_questions(questions):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    answers = []

    for idx, question in enumerate(questions, 1):
        print(f"Question {idx}: {question}")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}\n")
                answers.append(text)
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio. Please try again.")
                answers.append("")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                answers.append("")
    return answers


def analyze_mood(answers):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(ans)['compound'] for ans in answers if ans]
    if not scores:
        return "Neutral"

    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.05:
        return "Positive"
    elif avg_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


if __name__ == "__main__":
    questions = [
        "How are you feeling today?",
        "What made you smile recently?",
        "Are you stressed or relaxed right now?",
        "What’s something that’s been on your mind?",
        "How would you describe your energy level today?",
        "Did anything disappoint you today?",
        "What are you looking forward to?",
        "Do you feel calm or anxious at this moment?",
        "What emotion best describes your mood?",
        "Is there anything you’d like to talk about?"
    ]

    answers = ask_questions(questions)
    mood = analyze_mood(answers)
    print(f"\nBased on your responses, your overall mood is: {mood}")
