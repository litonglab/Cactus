'''aigc.py'''
import os
import queue
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QImage
from .inference import Wav2LipModel

import numpy as np
import cv2
import Util.audio as audio
import soundfile as sf

class Wav2LipThread(QThread):
    # 用于发送处理完成后的帧列表
    result_ready = pyqtSignal(list)
    log_text_signal = pyqtSignal(str)
    def __init__(self, image_data, audio_data, model, id):
        super().__init__()
        self.image_data = image_data
        self.audio_data = audio_data
        self.temp_audio_path = 'temp/temp_audio.wav'
        self.model = model
        self.fps = 25
        self.id = id
        # 设置保存音频的目录路径，默认为 "saved_audio"
        self.save_audio_dir = 'saved_audio'
        # 如果目录不存在，则创建目录
        if not os.path.exists(self.save_audio_dir):
            os.makedirs(self.save_audio_dir)

    def image_to_cv(self, qimage):
        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def save_audio(self, audio_array):
        # 使用时间戳作为文件名，避免重复覆盖文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 拼接完整的保存路径，例如 "saved_audio/audio_20231010_123456.wav"
        audio_file_path = os.path.join(self.save_audio_dir, f"audio_{timestamp}.wav")
        # 保存音频文件，采样率为 16kHz（16000）
        sf.write(audio_file_path, audio_array, 16000)
        # 打印日志或发送信号通知保存成功（可选）
        self.log_text_signal.emit(f"[AUDIO] Saved audio to {audio_file_path}\n")

    def run(self):
        full_frames = [self.image_to_cv(self.image_data)]
        audio_array = np.frombuffer(self.audio_data, dtype=np.float32)

        sf.write(self.temp_audio_path, audio_array, 16000)
        wav = audio.load_wav(self.temp_audio_path, 16000)
        mel = audio.melspectrogram(wav)
        self.log_text_signal.emit(f"[AUDIO] Mel shape: {mel.shape}\n")

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + 16])
            i += 1

        if len(mel_chunks) <= 1: return
        threading.Thread(target=self.save_audio, args=(audio_array,), daemon=True).start()

        if len(full_frames) == 1:
            full_frames = full_frames * len(mel_chunks)
        if len(full_frames) != len(mel_chunks):
            raise ValueError("Number of frames and mel chunks do not match after expansion.")

        try:
            print(f"frame shape: {len(full_frames)}")
            print(f"mel chunk shape: {len(mel_chunks)}")
            self.model.inference(full_frames, mel_chunks, self.fps, audio_array, self.id)
        except Exception as e:
            print(f"Exception occurred during aigc run: {e}")

