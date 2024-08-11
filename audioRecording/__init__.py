import os
import threading
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from pydub import AudioSegment

from lib.steps.Shared.ContentType import ContentType
from lib.steps.Shared.QueueHandler import QueueHandler
from lib.steps.Shared.QueueItem import QueueItem
from lib.steps.Shared.step_interface import StepInterface
from lib.steps.record_audio.modules.desktop_module import RecordAudioDesktopModule


class RecordAudioStep(StepInterface):
    def __init__(self, mode="google"):
        self.language = None
        self.item_count = 0
        self.transcript_files = None
        self.conversation_id = None
        self.mode = mode
        self.tmp_folder = "tmp/"
        # self.desktop_module = RecordAudioDesktopModule()
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
        self.desktop_file = self.tmp_folder + "desktop_audio.wav"
        self.mic_file = self.tmp_folder + "microphone_audio.wav"
        self.recognizer = sr.Recognizer()

    def init(self, conversation_id):
        self.conversation_id = conversation_id

    def run(self, conversation_id, out_queues, stop_event):
        self.init(conversation_id)
        while not stop_event.is_set():
            result_audio = self.record_voice()
            if result_audio is None:
                continue
            print(f"Task {self.item_count} - Audio recorded")
            q = QueueItem(ContentType.AUDIO, result_audio)
            QueueHandler.push_message_to_queues(out_queues, q)

    def record_voice(self):
        fs = 44100
        duration = 20

        def record_desktop_audio():
            print("Recording desktop audio...", end="")
            try:
                # self.desktop_module.record(self.desktop_file)
                print("Done recording desktop audio.")
            except Exception as e:
                print(f"Exception while recording desktop audio: {e}")

        def record_microphone_audio():
            print("Recording microphone audio...", end="")
            try:
                mic_audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, blocking=True)
                sf.write(self.mic_file, mic_audio, fs)
                print("Done recording microphone audio.")
            except Exception as e:
                print(f"Exception while recording microphone audio: {e}")

        def record_mic_legacy():
            with sr.Microphone(device_index=0) as source:
                print("Recording...", end="")
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=20, )
                    sf.write(self.mic_file, audio.frame_data, fs)
                    sound1 = AudioSegment.from_wav(self.mic_file)
                    self.item_count += 1
                    return sound1
                except sr.WaitTimeoutError:
                    print("no speech detected. Trying again...")
                    return None

        # return record_mic_legacy()
        # desktop_audio_thread = threading.Thread(target=record_desktop_audio)
        microphone_audio_thread = threading.Thread(target=record_microphone_audio)

        # desktop_audio_thread.start()
        microphone_audio_thread.start()

        # desktop_audio_thread.join()
        microphone_audio_thread.join()
        print("Done recording.")

        sound1 = AudioSegment.from_wav(self.mic_file)
        # sound2 = AudioSegment.from_wav(self.desktop_file)
        self.item_count += 1
        return sound1
