import math
import os
import sys
from multiprocessing import Process
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr
from pydub import AudioSegment

# Configuration
CHUNK_DURATION = 0.5  # 100ms
SILENCE_THRESHOLD = -17.0  # silence threshold in dB
SILENCE_DURATION = 2  # seconds
SAMPLE_RATE = 44100  # Hz
AUDIO_CLIPPING_LEVEL = 0.5
CHANNELS = 1  # change to 2 for stereo sound
DEVICE = None  # change to specific device if needed
r = sr.Recognizer()

if os.path.exists("temp.wav"):
    os.remove("temp.wav")


def record_audio(transcript_file):
    chunks = []
    while True:
        recording = sd.rec(int(SAMPLE_RATE * CHUNK_DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, device=DEVICE)
        sd.wait()  # Wait for the recording to finish

        # Convert recording to wave data
        scaled = np.int16(recording / AUDIO_CLIPPING_LEVEL * 32767)
        chunks.append(scaled)

        # check silence
        num_of_chunks = math.ceil(SILENCE_DURATION / CHUNK_DURATION)

        # Only consider the last "SILENCE_DURATION" seconds within the chunks array
        last_chunks = chunks[-num_of_chunks:]
        combined_data = b"".join(last_chunks)
        audio = AudioSegment(combined_data, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

        # Check silence for the last chunks only
        if audio.max_dBFS < SILENCE_THRESHOLD:
            combined_full_data = b"".join(chunks)
            full_audio = AudioSegment(combined_full_data, frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
            if full_audio.max_dBFS > SILENCE_THRESHOLD:
                print("Silence after sound detected. Transcribing audio...")
                transcribe_audio(chunks, transcript_file)
                return
        else:
            sys.stdout.write(".")


def transcribe_audio(chunks, transcript_file):
    if len(chunks) > 0:
        # Save wave audio
        write('temp.wav', SAMPLE_RATE, np.concatenate(chunks))
        # Transcribe audio
        with sr.AudioFile('temp.wav') as source:
            audio = r.record(source)
            transcript = r.recognize_sphinx(audio)
            if transcript is not None:
                print("Transcript: ", transcript)
                with open(transcript_file, "w") as f:
                    f.write(transcript)


def create_audio_thread(transcript_file):
    audio_thread = Process(target=record_audio, daemon=True, args=(transcript_file,))
    audio_thread.start()
    return audio_thread
