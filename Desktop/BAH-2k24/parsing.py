import os
import sys
import queue
import sounddevice as sd
import vosk
import json

model_path = "vosk-model-small-en-us-0.15"

if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model'")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognize_speech():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("#" * 80)
        print("Please speak...")
        print("#" * 80)

        rec = vosk.KaldiRecognizer(model, 16000)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = json.loads(result)["text"]
                print("Recognized:", text)
                return text
            else:
                partial_result = rec.PartialResult()
                print("Partial:", json.loads(partial_result)["partial"])

def parse_command(recognized_text):
    commands = {
        "zoom in": "zoom_in",
        "zoom out": "zoom_out",
        "show roads": "show_roads",
        "show satellite": "show_satellite",
        "locate": "locate",
        "weather": "weather"
    }

    for command in commands:
        if command in recognized_text:
            return commands[command], recognized_text

    return None, None

def zoom_in():
    print("Zooming in on the map...")

def zoom_out():
    print("Zooming out on the map...")

def show_roads():
    print("Showing roads layer...")

def show_satellite():
    print("Showing satellite layer...")

def locate_place(place_name):
    print(f"Locating {place_name} on the map...")

def show_weather():
    print("Showing weather data...")

def execute_command(command, recognized_text):
    if command == "zoom_in":
        zoom_in()
    elif command == "zoom_out":
        zoom_out()
    elif command == "show_roads":
        show_roads()
    elif command == "show_satellite":
        show_satellite()
    elif command == "locate":
        place_name = recognized_text.replace("locate", "").strip()
        locate_place(place_name)
    elif command == "weather":
        show_weather()
    else:
        print("Command not recognized.")

if __name__ == "__main__":
    try:
        recognized_text = recognize_speech()
        command, recognized_text = parse_command(recognized_text)
        execute_command(command, recognized_text)
    except KeyboardInterrupt:
        print("\nDone")
    except Exception as e:
        print(str(e))
