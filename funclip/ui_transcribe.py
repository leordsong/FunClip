# main_ui.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from video_transcriber import VideoTranscriber  # 之前实现的转换类


class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(bool)
    message_signal = pyqtSignal(str)

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir

    def run(self):
        try:
            self.message_signal.emit("初始化转换器...")
            transcriber = VideoTranscriber()

            self.message_signal.emit("转换视频到音频...")
            audio_path = transcriber.video_to_audio(self.video_path, self.output_dir)
            if not audio_path:
                self.result_signal.emit(False)
                return

            self.message_signal.emit("语音识别中...")
            transcriber.transcribe_audio(audio_path)

            self.progress_signal.emit(100)
            self.result_signal.emit(True)
        except Exception as e:
            self.message_signal.emit(f"错误: {str(e)}")
            self.result_signal.emit(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setWindowTitle("视频转文字工具")

    def init_ui(self):
        # 创建组件
        self.input_label = QLabel("输入视频:")
        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("浏览...")
        
        self.output_label = QLabel("输出目录:")
        self.output_edit = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        
        self.progress = QProgressBar()
        self.status_label = QLabel("准备就绪")
        self.convert_btn = QPushButton("开始转换")

        # 布局设置
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_btn)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.input_label)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.output_label)
        main_layout.addLayout(output_layout)
        main_layout.addWidget(self.progress)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.convert_btn)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 事件绑定
        self.input_btn.clicked.connect(self.select_input_file)
        self.output_btn.clicked.connect(self.select_output_dir)
        self.convert_btn.clicked.connect(self.start_conversion)

    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.input_edit.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_edit.setText(dir_path)

    def start_conversion(self):
        video_path = self.input_edit.text()
        output_dir = self.output_edit.text()

        if not video_path or not os.path.exists(video_path):
            QMessageBox.critical(self, "错误", "请选择有效的视频文件")
            return

        if not output_dir:
            output_dir = os.path.join(os.path.dirname(video_path), "output")

        self.output_edit.setText(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 创建并启动工作线程
        self.worker = WorkerThread(video_path, output_dir)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.result_signal.connect(self.handle_result)
        self.worker.message_signal.connect(self.update_status)
        self.convert_btn.setEnabled(False)
        self.worker.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def handle_result(self, success):
        self.convert_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", "转换成功！")
        else:
            QMessageBox.critical(self, "错误", "转换过程中发生错误")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 200)
    window.show()
    sys.exit(app.exec_())