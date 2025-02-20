import sys
import time

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

class PlotWindow(QtWidgets.QWidget):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.plot_widget = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self.curve = self.plot_widget.plot()

    def update_plot(self, data):
        x = np.arange(100)
        y = np.array(data)
        self.curve.setData(x, y)

class BandwidthPlotThread(QtCore.QThread):
    update_plot_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.data = []

    def run(self):
        while True:
            length = len(self.data)
            if length > 3:
                print(f"bigger than 3, current size {length}, should update plot!\n")
                # # self.curve.setData(x, self.data)
                print("ababababababab?\n")
                # self.curve.setData(x, self.data)
                # print("aaaaaaaaa???\n")
                # self.plot_window.show()
                self.update_plot_signal.emit(self.data)
                time.sleep(0.1)
            else:
                print("length less than 3!\n")

    def update_bandwidth(self, list_bandwidth):
        bandwidth_mbps = list_bandwidth[2]
        self.data.append(bandwidth_mbps)
        print(f"Append bandwidth data {bandwidth_mbps}, current data size {len(self.data)}.\n")
        if len(self.data) > 100:
            self.data.pop(0)

class BitratePlotThread(QtCore.QThread):
    update_plot_signal = QtCore.pyqtSignal(list)

    def __init__(self, plot_widget, plot_window):
        super().__init__()
        self.data = []
        self.plot_widget = plot_widget
        self.plot_window = plot_window

    def run(self):
        x = np.arange(100)
        while True:
            if len(self.data) > 3:
                self.plot_widget.plot().setData(x, self.data)
                self.plot_window.show()
                self.msleep(500)  # 每500ms更新一次

    def update_bitrate(self, list_bitrate):
        # bytes_sent = list_bitrate[0]
        # elapsed_time = list_bitrate[1]
        bitrate_mbps = list_bitrate[2]
        self.data.append(bitrate_mbps)
        print(f"Append bitrate data {bitrate_mbps}, current data size {len(self.data)}.\n")
        if len(self.data) > 100:
            self.data.pop(0)
