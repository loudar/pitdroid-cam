import pyaudio
import wave
import sounddevice as sd

from lib.steps.record_audio.modules.ra_module_interface import RecordAudioInterface


class RecordAudioDesktopModule(RecordAudioInterface):
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.RECORD_SECONDS = 20
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            print(f"{i}: {dev['name']} - Channels: {dev['maxInputChannels']} - Rate: {dev['defaultSampleRate']}")
            print(f"   - {dev}")

    def record(self, filename):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             input_device_index=sd.default.device[1],
                             output=True,
                             output_device_index=sd.default.device[1],
                             frames_per_buffer=self.CHUNK)

        print("* recording desktop audio")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording desktop audio")

        stream.stop_stream()
        stream.close()
        self.p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
