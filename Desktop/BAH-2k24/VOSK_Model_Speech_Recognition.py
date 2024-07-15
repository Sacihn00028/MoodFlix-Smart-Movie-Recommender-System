import os
import sys
import queue
import sounddevice as sd
import vosk
import json

model_path = "vosk-model-small-en-us-0.15"
model = vosk.Model(model_path)

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Function to recognize speech from audio input
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

# Main function to run the speech recognition
if __name__ == "__main__":
    try:
        recognize_speech()
    except KeyboardInterrupt:
        print("\nDone")
    except Exception as e:
        print(str(e))
