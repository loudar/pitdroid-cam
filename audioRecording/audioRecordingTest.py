import math
import os
import sys
import uuid
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from openai import OpenAI
from multiprocessing import Process
from scipy.io.wavfile import write
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()
# Configuration
CHUNK_DURATION = 0.5  # 100ms
SILENCE_THRESHOLD = -30.0  # silence threshold in dB
SILENCE_DURATION = 2  # seconds
SAMPLE_RATE = 44100  # Hz
AUDIO_CLIPPING_LEVEL = 0.5
CHANNELS = 1  # change to 2 for stereo sound
DEVICE = None  # change to specific device if needed
r = sr.Recognizer()
recognition_engine = "whisper"

openai = None
if recognition_engine == "whisper":
    openai = OpenAI()


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
                write_and_transcribe_audio(chunks, transcript_file)
                chunks = []
        else:
            sys.stdout.write(".")


def write_and_transcribe_audio(chunks, transcript_file):
    if len(chunks) > 0:
        id = uuid.uuid4()
        audio_file = f'temp_{id}.wav'
        write(audio_file, SAMPLE_RATE, np.concatenate(chunks))
        Process(target=transcribe_audio, daemon=True, args=(transcript_file, audio_file)).start()


def transcribe_audio(transcript_file, audio_file):
    print(f"Transcribing {audio_file}...")
    transcript = recognize_text_whisper(audio_file)
    if transcript is not None:
        print(f"Transcript for {audio_file}: ", transcript)
        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(transcript + "\n")
    else:
        print(f"No transcript for {audio_file}")


def recognize_text_sphinx(audio):
    return r.recognize_sphinx(audio)


def recognize_text_whisper(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        os.remove(audio_file_path)
        return transcription.text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))
    return None


def create_audio_thread(transcript_file):
    audio_thread = Process(target=record_audio, args=(transcript_file,))
    audio_thread.start()
    return audio_thread
