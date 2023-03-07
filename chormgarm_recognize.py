from typing import List
import librosa
import numpy as np
import pandas as pd
import os
import threading
from queue import Queue, PriorityQueue
import pyaudio as pa
import time

'''
read data from audio stream and store it to dict['audio_data']
'''

def get_labels_form_chormgram(chormgam_data: np.ndarray) -> List[str]:
    ans = []
    for i in range(chormgam_data.shape[1]):
        # 遍历每一个帧的数据
        frame_chormgam_data = chormgam_data[:, i]
        frame_data = pd.Series(frame_chormgam_data)
        recognize = frame_data.sort_values(ascending=False).index.map(lambda x : labels[x]).tolist()
        ans.append(recognize)
        print(" > ".join(recognize))
    return ans

def callback(in_data, frame_count, time_info, status):
    variables["data"] = np.frombuffer(in_data, dtype=np.float32)
    chormgram = librosa.feature.chroma_stft(y=variables["data"])
    get_labels_form_chormgram(chormgram)
    return (in_data, pa.paContinue)
    pass


def read_audio(vars: dict):
    pyaudio = pa.PyAudio()
    print("stream listening ~ ")
    vars["stream"] = pyaudio.open(format=pa.paFloat32,
                 rate=vars["rate"],
                 channels=1,
                 input=True,
                 frames_per_buffer=vars["chunk"],
                 stream_callback=callback)
    vars["stream"].start_stream();
    pass


variables = {
        "stream": None,
        "rate": 2000,
        "chunk": 2048,
        "data": None
}

labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


if __name__ == "__main__":
    read_audio(variables)
    while True:
        time.sleep(100)
        pass
    pass

