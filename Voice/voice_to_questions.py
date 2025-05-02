import speech_recognition as sr

def voice_to_text_timed(duration=5):
    """
    Captures voice input from the microphone for a fixed duration and converts it to text.

    Args:
        duration (int): The number of seconds to listen for audio.

    Returns:
        The recognized text, or None if recognition fails.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"Say something for {duration} seconds!")
        # Optional: Adjust for ambient noise before listening for a fixed duration
        # recognizer.adjust_for_ambient_noise(source, duration=1) # Adjust for 1 second

        # Listen for the specified duration
        try:
            audio = recognizer.listen(source, timeout=duration)
        except sr.WaitTimeoutError:
            print(f"No speech detected within {duration} seconds.")
            return None # Exit if no speech is detected within the timeout

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == "__main__":
    # Listen for 5 seconds
    recognized_text = voice_to_text_timed(duration=5)
    if recognized_text:
        print(f"The recognized text was: '{recognized_text}'")
