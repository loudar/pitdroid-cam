import os
from multiprocessing import Process

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr

# Configuration
CHUNK_DURATION = 5  # seconds
listening = False
SAMPLE_RATE = 44100  # Hz
CHANNELS = 1  # change to 2 for stereo sound
DEVICE = None  # change to specific device if needed

r = sr.Recognizer()

if os.path.exists("temp.wav"):
    os.remove("temp.wav")


def record_audio(transcript_file):
    recording = sd.rec(int(SAMPLE_RATE * CHUNK_DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, device=DEVICE)
    sd.wait()  # Wait for the recording to finish
    print("Recording finished")

    # Convert recording to wave audio and save
    scaled = np.int16(recording / np.max(np.abs(recording)) * 32767)
    write('temp.wav', SAMPLE_RATE, scaled)

    # Transcribe audio
    with sr.AudioFile('temp.wav') as source:
        audio = r.record(source)
        transcript = r.recognize_sphinx(audio)
        if transcript is not None:
            print("Transcript: ", transcript)
            with open(transcript_file, "w") as f:
                f.write(transcript)


def create_audio_thread(transcript_file):
    audio_thread = Process(target=record_audio, args=(transcript_file,), daemon=True)
    audio_thread.start()
    return audio_thread
