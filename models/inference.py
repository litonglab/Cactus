'''inference.py'''
import os
import queue
import traceback

import sounddevice as sd
import numpy as np
import cv2
import pyaudio
import torch, face_detection
import threading
import time

from tqdm import tqdm
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from models import Wav2Lip
from .data_generator import DataGenerator
from .face_detector import FaceDetector
from Util import audio

class Wav2LipModel:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))
        self.model = self.load_model(checkpoint_path)
        print("Model loaded")

    def _load_checkpoint(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load_checkpoint(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def infer(self, mel_batch, img_batch):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
        with torch.no_grad():
            pred = self.model(mel_batch, img_batch)

        return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255

class Inference(QObject):
    change_pixmap_signal = pyqtSignal(QImage)
    progress_signal = pyqtSignal(int)  # 定义一个整数类型的信号用于表示百分比

    def __init__(self, checkpoint_path):
        super().__init__()
        self.wav2lip_batch_size = 128
        self.face_detector = FaceDetector('cuda' if torch.cuda.is_available() else 'cpu')
        self.wav2lip_model = Wav2LipModel(checkpoint_path)
        self.data_generator = DataGenerator(self.face_detector, self.wav2lip_batch_size)
        # 缓存机制
        self.frame_buffer = queue.PriorityQueue(maxsize=15)
        self.audio_buffer = queue.PriorityQueue(maxsize=15)
        self.play_audio_begin = False

        # 保存所有帧
        self.saved_frames = []
        # 创建一个目录用于保存帧
        self.output_dir = "receiver_output_frames"
        os.makedirs(self.output_dir, exist_ok=True)

        threading.Thread(target=self.display_generated_frames, daemon=True).start()
        # threading.Thread(target=self.display_audio, daemon=True).start()

        # 进度条
        self.num = 0
        self.num_steps = 300
        # 锁机制
        self.lock = threading.Lock()

        self.runtime = []
        self.gen_frames = []

    def inference(self, full_frames, mels, fps, audio_array, id):
        batch_size = self.wav2lip_batch_size
        gen = self.data_generator.datagen(full_frames.copy(), mels)
        a = 0
        temp_buffer = []
        temp_save_buffer = []
        try:
            # 开始计时
            start_time = time.time()
            for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
                pred = self.wav2lip_model.infer(mel_batch, img_batch)
                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    if (x2 - x1) < 3 or (y2 - y1) < 3:
                        print(f"Skipping frame due to small crop size: {(x2 - x1)}x{(y2 - y1)}")
                        continue
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    # Convert frame to QImage for displaying in QLabel
                    rgb_image = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                    # 存储生成结果
                    temp_buffer.append(qt_image)
                    temp_save_buffer.append(f.copy())  # 将索引和帧放入队列

                    self.num += 1
                    a += 1
                    # 发射当前进度百分比
                    progress_percentage = int((self.num / self.num_steps) * 100)
                    self.progress_signal.emit(progress_percentage)

                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break
            # 结束计时并打印耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            # with self.lock:
            #     for i in temp_save_buffer:
            #         self.saved_frames.append(i)
                    # print("Save one frame.\n")
            with self.lock:
                if not self.frame_buffer.full():
                    self.frame_buffer.put((id, temp_buffer))
                if not self.audio_buffer.full():
                    self.audio_buffer.put((id, audio_array))
                # for i in temp_buffer:
                #     if not self.frame_buffer.full():
                #         self.frame_buffer.put(i)
                #     else:
                #         break
            print(f"Already inference {a} num frames.\n")
            # self.runtime.append(elapsed_time)
            # self.gen_frames.append(a)
        except Exception as e:
            print(f"Exception occurred during inference: {e}")

    def save_all_frames(self):
        print("Saving all frames to disk...")
        with self.lock:
            for idx, frame in enumerate(self.saved_frames):
                filename = os.path.join(self.output_dir, f"receiver_frame_1pic_{idx:05d}.jpg")
                cv2.imwrite(filename, frame)  # 保存为图像文件
        print(f"All frames saved to {self.output_dir}.")

    def display_generated_frames(self):
        # 从结果队列取出生成的帧并显示到界面
        # time.sleep(240)
        # while not self.frame_buffer.empty():
        #     frame_id, generated_frames = self.frame_buffer.get()
        #     audio_id, audio_array = self.audio_buffer.get()
        #     print(f"Rest frame queue size : {self.frame_buffer.qsize()}.")
        #     print(f"Rest audio queue size : {self.audio_buffer.qsize()}.")
        #     self.display_audio(audio_array)
        #     # threading.Thread(target=self.display_audio, args=(audio_array,), daemon=True).start()
        #     for pic in generated_frames:
        #         self.change_pixmap_signal.emit(pic)
        #         time.sleep(0.04)  # 帧率20fps
        while True:
            try:
                if self.frame_buffer.qsize() <= 4: continue
                with self.lock:
                    frame_id, generated_frames = self.frame_buffer.get()
                    audio_id, audio_array = self.audio_buffer.get()
                    print(f"Rest frame queue size : {self.frame_buffer.qsize()}.")
                    print(f"Rest audio queue size : {self.audio_buffer.qsize()}.")
                    threading.Thread(target=self.display_audio, args=(audio_array, ), daemon=True).start()
                for pic in generated_frames:
                    self.change_pixmap_signal.emit(pic)
                    time.sleep(0.05)  # 帧率20fps
            except Exception as e:
                print(f"Exception occurred during play generated frames {e}.\n")

    def display_audio(self, audio_array):
        try:
            print("begin play audio.")
            sd.play(audio_array, samplerate=16000)
            print("end play audio.")
        except Exception as e:
            print(f"Exception occurred during playing audios: {e}.\n")

    def end_system(self):
        return self.runtime, self.gen_frames