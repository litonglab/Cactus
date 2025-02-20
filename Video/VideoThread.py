import os
import queue
import struct
import time
import threading

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QMutex
from PyQt6.QtGui import QImage
from PyQt6 import QtWidgets
import pyqtgraph as pg
import cv2
import pickle
from time import sleep

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_bitrate_signal = pyqtSignal(list)
    update_text_signal = pyqtSignal(str)

    def __init__(self, socket_comm, picture_gap, bw, a, alpha):
        super().__init__()
        self.socket_comm = socket_comm
        self.picture_gap = picture_gap
        self.bw = bw                    # 带宽 (Mbps)
        self.a = a                      # 音频码率 (Mbps) 实际: 256kbps
        self.alpha = alpha              # SSIM 阈值
        self.save_image_gap = 5

        # 创建目录用于存储捕获的图像
        self.output_dir = "sender_captured_frames"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize additional variables for SSIM calculation and image transmission
        self.p0 = None  # Last sent reference frame
        self.L = []  # List to store intervals

    def send_image(self, image_data):
        try:
            self.socket_comm.send_data(image_data, self.socket_comm.conn)
        except Exception as e:
            print(f"Error sending image: {e}")

    def ssim_constraint(self, pic):
        if self.p0 is None:
            self.p0 = pic.copy()
            _, buffer = cv2.imencode('.jpg', pic)
            image_data = b"Image:" + buffer.tobytes()
            # threading.Thread(target=self.send_image, args=(image_data,)).start()
            self.send_image(image_data)
            self.last_send_time = time.time()
            self.update_text_signal.emit(f"Sending picture 0...\n\n")
            return

        gray_P0 = cv2.cvtColor(self.p0, cv2.COLOR_RGB2GRAY)  # Convert reference frame to grayscale
        gray_Pi = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)  # Convert current frame to grayscale
        ssim_score, _ = compare_ssim(gray_P0, gray_Pi, full=True)
        print(f"SSIM={ssim_score:.4f}")
        self.update_text_signal.emit(f"SSIM={ssim_score:.4f}\n")

        current_I = time.time() - self.last_send_time
        if ssim_score < self.alpha or current_I >= self.picture_gap:  # If similarity is below threshold, consider sending the current frame
            if current_I >= self.I0 and (np.mean(self.L) + current_I) / 2 >= self.I0:
                print(f"Sending frame due to low SSIM ({ssim_score:.4f})...")
                self.update_text_signal.emit(f"Sending frame due to low SSIM ({ssim_score:.4f}) or long interval...\n\n")
                _, buffer = cv2.imencode('.jpg', pic)
                image_data = b"Image:" + buffer.tobytes()

                threading.Thread(target=self.send_image, args=(image_data,)).start()

                setattr(self, 'last_send_time', time.time())
                self.p0 = pic.copy()
                self.L.append(current_I)

    def run(self):
        '''Cactus---dynamically adjust picture interval'''

        cap = cv2.VideoCapture(0)

        frame_count = 0

        start_time = time.time()
        last_send_time = time.time()
        bytes_sent = 0

        I0_initial_value = int(0.5 / (self.bw - self.a))
        I0_initial_value = max(I0_initial_value, 1)

        print("I0: ", I0_initial_value)
        self.update_text_signal.emit(f"Interval 0: {I0_initial_value}\n\n")

        self.I0 = I0_initial_value
        self.L.append(self.I0)

        while cap.isOpened():
            ret, cv_img = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p_scaled = convert_to_Qt_format.scaled(600, 460, Qt.AspectRatioMode.KeepAspectRatio)

                # Emit signal to update the GUI with the new image.
                self.change_pixmap_signal.emit(p_scaled)
                threading.Thread(target=self.ssim_constraint, args=(cv_img, ), daemon=True).start()

                _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                image_data = b"Image:" + buffer.tobytes()
                bytes_sent += len(image_data)
                bytes_sent += 6400  # audio bytes
                elapsed_time = time.time() - start_time
                bitrate_mbps = (bytes_sent * 8) / 1000 / 1000 / elapsed_time
                self.update_bitrate_signal.emit([bytes_sent, elapsed_time, bitrate_mbps])

                frame_count += 1

            else:
                break

        cap.release()  # Release camera resources


    # def run(self):
    #     '''Cactus---dynamically adjust picture interval'''
    #     # 打开摄像头
    #     cap = cv2.VideoCapture(0)
    #     frame_count = 0  # 帧计数器，用于确定何时保存图像
    #     start_time = time.time()
    #     last_send_time = time.time()
    #     bytes_sent = 0
    #     I0_initial_value = int(0.5 / (self.bw - self.a))
    #     I0_initial_value = max(I0_initial_value, 1)
    #     self.I0 = I0_initial_value
    #     self.L.append(self.I0)
    #     self.update_text_signal.emit(f"Minimum value of the average picture interval: {self.I0}.\n\n")
    #
    #     while cap.isOpened():
    #         ret, cv_img = cap.read()
    #         if ret:
    #             # 将图像转换为QImage对象
    #             rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #             h, w, ch = rgb_image.shape
    #             bytes_per_line = ch * w
    #             convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #             p = convert_to_Qt_format.scaled(600, 460, Qt.AspectRatioMode.KeepAspectRatio)
    #             self.change_pixmap_signal.emit(p)
    #
    #             # # 保存图像到磁盘（每隔 save_image_gap 帧）
    #             # if frame_count % self.save_image_gap == 0:
    #             #     filename = os.path.join(self.output_dir, f"sender_frame_dynamic_{frame_count}.jpg")
    #             #     cv2.imwrite(filename, cv_img)  # 保存原始BGR格式的图像
    #             #     print(f"已保存图像: {filename}")
    #
    #             # 如果没有基准帧，则初始化 p0
    #             if self.p0 is None:
    #                 self.p0 = rgb_image.copy()
    #                 _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #                 image_data = b"Image:" + buffer.tobytes()
    #                 self.send_image(image_data)
    #                 continue
    #
    #             # 每隔一段时间计算 SSIM 值
    #             gray_P0 = cv2.cvtColor(self.p0, cv2.COLOR_RGB2GRAY)  # 基准帧灰度化
    #             gray_Pi = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # 当前帧灰度化
    #             ssim_score, _ = compare_ssim(gray_P0, gray_Pi, full=True)
    #             print(f"Frame {frame_count}: SSIM={ssim_score:.4f}")
    #             self.update_text_signal.emit(f"Frame{frame_count}:SSIM={ssim_score:.3f}\n")
    #
    #             if ssim_score < self.alpha:  # 如果两帧相似性低于阈值，则考虑发送当前帧
    #                 current_I = time.time() - last_send_time
    #
    #                 if current_I >= self.I0 and (np.mean(self.L) + current_I) / 2 >= self.I0:
    #                     print(f"Sending frame {frame_count} due to low SSIM ({ssim_score:.4f})...")
    #                     print(f"Current time Interval : {current_I}")
    #                     self.update_text_signal.emit(f"Sending frame {frame_count} due to low SSIM ({ssim_score:.4f})...\n\n")
    #                     self.update_text_signal.emit(f"Current time Interval : {current_I}\n\n")
    #
    #                     _, buffer = cv2.imencode('.jpg', cv_img)
    #                     # _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #                     image_data = b"Image:" + buffer.tobytes()
    #
    #                     threading.Thread(target=self.send_image, args=(image_data,)).start()
    #                     last_send_time = time.time()
    #
    #                     # 更新基准帧和时间间隔列表
    #                     self.p0 = rgb_image.copy()
    #                     self.L.append(current_I)
    #
    #             # 计算视频码率
    #             bytes_sent += len(image_data)
    #             elapsed_time = time.time() - start_time
    #             bitrate_mbps = (bytes_sent * 8) / 1000 / 1000 / elapsed_time
    #             self.update_bitrate_signal.emit([bytes_sent, elapsed_time, bitrate_mbps])
    #
    #             frame_count += 1  # 增加帧计数器
    #         else:
    #             break
    #     # 释放摄像头资源
    #     cap.release()


    # def run(self):
    #     '''Origin video transmission'''
    #     # 打开摄像头
    #     # cap = cv2.VideoCapture(0)
    #     #
    #     # if not cap.isOpened():
    #     #     print("Error: Could not open video.")
    #     #     return
    #     #
    #     # print(f"默认FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    #     # print(f"默认分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    #     #
    #     # frame_count = 0  # 帧计数器，用于确定何时保存图像
    #     #
    #     # start_time = time.time()
    #     #
    #     # captured_images = []  # 用于存储捕获的帧文件路径
    #     #
    #     # while cap.isOpened():
    #     #     ret, cv_img = cap.read()
    #     #     if ret:
    #     #         # 将图像转换为QImage对象以供显示
    #     #         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     #         h, w, ch = rgb_image.shape
    #     #         bytes_per_line = ch * w
    #     #         convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #     #         p = convert_to_Qt_format.scaled(600, 460, Qt.AspectRatioMode.KeepAspectRatio)
    #     #         self.change_pixmap_signal.emit(p)
    #     #
    #     #         frame_count += 1
    #     #
    #     #         image_path = os.path.join(self.output_dir, f"frame_{frame_count}.jpg")
    #     #         cv2.imwrite(image_path, cv_img)  # 保存帧为图像文件
    #     #         captured_images.append(image_path)  # 保存路径到列表中
    #     #
    #     #     else:
    #     #         break
    #     #
    #     #     # 停止条件（比如运行5min后停止）
    #     #     if time.time() - start_time > 300:
    #     #         break
    #     #
    #     # cap.release()
    #
    #     # 合成视频并使用 H.264 编码器
    #     # if captured_images:
    #     output_video_path = "output_video.mp4"
    #
    #     # 发送生成的视频文件
    #     if output_video_path:
    #         self.send_video(output_video_path)
    #
    #
    # def create_h264_video(self, images):
    #     """
    #     将捕获的图片合成为一个 H.264 视频文件。
    #     :param images: 图片路径列表。
    #     """
    #     output_video_path = "output_video.mp4"  # 输出视频文件名
    #
    #     # 获取第一张图片的宽高（假设所有图片尺寸相同）
    #     first_frame = cv2.imread(images[0])
    #
    #     height, width, layers = first_frame.shape
    #
    #     fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用 H.264 编码器
    #
    #     fps = 6  # 设置帧率（可以根据需要调整）
    #
    #     out_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    #
    #     for img_path in images:
    #         frame = cv2.imread(img_path)
    #
    #         out_video_writer.write(frame)  # 写入每一帧到视频
    #
    #         print(f"写入帧: {img_path}")
    #
    #     out_video_writer.release()
    #
    #     print(f"H.264 视频已保存为 {output_video_path}")
    #
    #     return output_video_path
    #
    # def send_video(self, video_file_path):
    #     try:
    #         with open(video_file_path, 'rb') as video_file:
    #             print("开始传输视频...")
    #             # 通知接收端即将开始传输视频（可选）
    #             self.socket_comm.send_data(b"VideoStart", self.socket_comm.conn)
    #             while True:
    #                 chunk = video_file.read(4096)  # 每次读取4KB数据块
    #                 if not chunk:
    #                     break
    #                 self.socket_comm.send_data(chunk, self.socket_comm.conn)
    #             # 通知接收端传输结束（可选）
    #             self.socket_comm.send_data(b"VideoEnd", self.socket_comm.conn)
    #             print("视频传输完成！")
    #     except Exception as e:
    #         print(f"Error sending video file: {e}")

    # def run(self):
    #     '''Fixed picture interval'''
    #     # 打开摄像头
    #     cap = cv2.VideoCapture(0)
    #     print(f"默认FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    #     print(f"默认分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    #
    #     frame_count = 0  # 帧计数器，用于确定何时保存图像
    #
    #     start_time = time.time()
    #     last_send_time = time.time()
    #     bytes_sent = 0
    #
    #     while cap.isOpened():
    #         ret, cv_img = cap.read()
    #         if ret:
    #             # 将图像转换为QImage对象
    #             rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #             h, w, ch = rgb_image.shape
    #             bytes_per_line = ch * w
    #             convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #             p = convert_to_Qt_format.scaled(600, 460, Qt.AspectRatioMode.KeepAspectRatio)
    #             self.change_pixmap_signal.emit(p)
    #
    #             # 保存图像到磁盘（每隔 save_image_gap 帧）
    #             if frame_count % self.save_image_gap == 0:
    #                 filename = os.path.join(self.output_dir, f"sender_frame_1pic_{frame_count}.jpg")
    #                 cv2.imwrite(filename, cv_img)  # 保存原始BGR格式的图像
    #                 print(f"已保存图像: {filename}")
    #
    #             # _, buffer = cv2.imencode('.jpg', cv_img)
    #             _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #             image_data = b"Image:" + buffer.tobytes()
    #             # 每隔Interval传输一张image
    #             current_time = time.time()
    #             if current_time - last_send_time >= self.picture_gap:
    #                 threading.Thread(target=self.send_image, args=(image_data,)).start()
    #                 last_send_time = time.time()
    #
    #             # 计算视频码率
    #             bytes_sent += len(image_data)
    #             elapsed_time = current_time - start_time
    #             bitrate_mbps = (bytes_sent * 8) / 1000 / 1000 / elapsed_time
    #
    #             self.update_bitrate_signal.emit([bytes_sent, elapsed_time, bitrate_mbps])
    #             frame_count += 1  # 增加帧计数器
    #         else:
    #             break
    #     # 释放摄像头资源
    #     cap.release()
