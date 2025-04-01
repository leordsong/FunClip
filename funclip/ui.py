import sys
from os.path import join

from PIL import Image
import ffmpeg
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QLineEdit, QFileDialog, QSizePolicy,
    QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from logger import logger
from llm.openai_api import openai_call, openai_call_reasoning
from utils.io import image_np_to_base64


class VideoAIEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU-Accelerated Video AI Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Video variables
        self.video_info = None
        self.current_frame_idx = 0
        self.is_playing = False
        self.timer = QTimer(self)
        
        # AI variables
        self.model = None
        self.processor = None
        self.system_prompt = """你是一个专业的直播视频分析AI，任务是从视频中精准筛选出符合直播销售要求的片段。"""
        self.user_prompt = (
            "请根据以下标准判断当前画面是否符合直播销售要求"
            "1. 主体为女性，正面清晰展示所售服装  "
            "2. 表情自然放松，无遮挡或夸张面部动作，不能闭眼！"
            "3. 动作优雅得体，完整展示服装版型"
            "4. 服装在画面中占比≥40%，光照良好无阴影遮挡"
            "符合所有条件时返回\"对\"，否则返回\"不\""
        )
        
        # Initialize UI
        self.init_ui()
        # self.init_AI()
        self.timer.timeout.connect(self.play_video)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main horizontal layout (left-right)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins for mobile
        
        # Left side: Video display (16:9 aspect ratio)
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 10, 0)  # Right margin
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)  # 16:9 base size for mobile
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label)
        
        # Frame slider under video
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_moved)
        video_layout.addWidget(self.frame_slider)
        
        main_layout.addWidget(video_container, stretch=2)  # 2/3 width for video

        # Right side: Controls
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 0, 0, 0)  # Left margin
        
        # Navigation buttons (horizontal)
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("←")
        self.prev_btn.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton("⏯")
        self.play_btn.clicked.connect(self.toggle_play)
        nav_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton("→")
        self.next_btn.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_btn)
        controls_layout.addLayout(nav_layout)

        # AI Prompts
        controls_layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setText(self.system_prompt)
        # self.system_prompt_input.setPlaceholderText("Describe the scene...")
        controls_layout.addWidget(self.system_prompt_input)
        
        controls_layout.addWidget(QLabel("User Prompt:"))
        self.user_prompt_input = QTextEdit()
        self.user_prompt_input.setText(self.user_prompt)
        # self.user_prompt_input.setPlaceholderText("What to look for...")
        controls_layout.addWidget(self.user_prompt_input)

        # Action buttons
        self.load_btn = QPushButton("📁 Load Video")
        self.load_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("💾 Save Frame")  # New button
        self.save_btn.clicked.connect(self.save_current_frame)
        self.save_btn.setEnabled(False)  # Disabled until video loads
        controls_layout.addWidget(self.save_btn)

        # Model Selection Dropdown
        controls_layout.addWidget(QLabel("AI Model:"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            "Qwen2.5-VL-7B-Instruct",
            "qvq-max",
            'qwen-vl-max-2025-01-25'
        ])
        self.model_dropdown.setCurrentText("Qwen2.5-VL-7B-Instruct")  # Default selection
        controls_layout.addWidget(self.model_dropdown)
        
        self.ai_btn = QPushButton("🤖 Call AI")
        self.ai_btn.clicked.connect(self.call_ai)
        self.ai_btn.setEnabled(False)
        controls_layout.addWidget(self.ai_btn)
        
        # Add stretch to push controls up
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_container, stretch=1)  # 1/3 width for controls

        # Mobile-friendly adjustments
        for btn in [self.prev_btn, self.play_btn, self.next_btn, self.load_btn, self.ai_btn, self.save_btn]:
            btn.setMinimumHeight(40)  # Larger touch targets

    def init_AI(self, model_name):
        if self.model is not None and self.model.name_or_path == model_name:
            return
        if model_name == "Qwen2.5-VL-7B-Instruct":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
        
    def get_frame(self, frame_num):
        """实时提取指定帧（GPU加速）"""
        if self.video_info is None:
            return None
        # return self.frames[frame_num]

        # out, _ = (
        #     ffmpeg
        #     .input(self.video_path, hwaccel='cuda')  # GPU解码
        #     .filter('select', f'eq(n,{frame_num})')  # 精确跳帧
        #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')  # RGB格式
        #     .run(capture_stdout=True, quiet=True)
        # )
        if self.frame_cache is not None and self.frame_cache[0] == frame_num:
            return self.frame_cache[1]
        
        out, _ = (  # Capture both stdout and stderr
            ffmpeg
            .input(self.video_path)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
            # .filter('select', f'eq(n,{frame_num})')
            # .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
            # .run(capture_stdout=True, quiet=True)
        )
        width, height = self.video_info[0], self.video_info[1]
        frame = np.frombuffer(out, np.uint8).reshape([height, width, 3])
        self.frame_cache = (frame_num, frame)  # Cache the frame
        return frame

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if path:
            self.statusBar().showMessage("Loading video with GPU acceleration...")
            
            try:
                # GPU-accelerated video loading
                probe = ffmpeg.probe(path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                width = int(video_info['width'])
                height = int(video_info['height'])
                total_frames = int(video_info.get('nb_frames', 0))
                if total_frames == 0:
                    total_frames = int(video_info['duration_ts'])
                self.video_info = (width, height, total_frames, path)
                self.frame_cache = None
                
                # Read all frames into memory with GPU decoding
                # out, _ = (
                #     ffmpeg
                #     .input(path, hwaccel='cuda')
                #     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                #     .run(capture_stdout=True, capture_stderr=True)
                # )
                # self.frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                # assert total_frames == len(self.frames)
                
                self.frame_slider.setRange(0, total_frames - 1)
                self.current_frame_idx = 0
                self.frame_cache = None
                self.display_frame()
                self.ai_btn.setEnabled(True)
                self.save_btn.setEnabled(True)  # Enable save button
                self.statusBar().showMessage(f"Loaded: {path} (GPU accelerated)")
                
            except Exception as e:
                self.statusBar().showMessage(f"Error: {str(e)}")

    def display_frame(self):
        if self.video_info:
            frame = self.get_frame(self.current_frame_idx)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
            self.frame_slider.setValue(self.current_frame_idx)

    def slider_moved(self, position):
        self.current_frame_idx = position
        self.display_frame()

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_frame()

    @property
    def num_frames(self):
        return self.video_info[2] if self.video_info else 0
    
    @property
    def video_path(self):
        return self.video_info[3] if self.video_info else ""

    def next_frame(self):
        if self.current_frame_idx < self.num_frames - 1:
            self.current_frame_idx += 1
            self.display_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer.start(33)  # ~30fps
        else:
            self.timer.stop()

    def play_video(self):
        if self.current_frame_idx < self.num_frames - 1:
            self.current_frame_idx += 1
            self.display_frame()
        else:
            self.timer.stop()
            self.is_playing = False

    def call_qwenvl(self, system_prompt, user_prompt, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def call_ai(self):
        self.system_prompt = self.system_prompt_input.toPlainText()
        self.user_prompt = self.user_prompt_input.toPlainText()
        
        if self.num_frames == 0:
            self.statusBar().showMessage("No video loaded!")
            return
            
        current_frame = self.get_frame(self.current_frame_idx)
        self.statusBar().showMessage("Calling AI with current frame and prompts...")
        
        if self.model_dropdown.currentText() == "Qwen2.5-VL-7B-Instruct":
            # Call the Qwen model
            self.init_AI("Qwen2.5-VL-7B-Instruct")
            output = self.call_qwenvl(
                self.system_prompt,
                self.user_prompt,
                Image.fromarray(current_frame)
            )
        elif self.model_dropdown.currentText() == "qvq-max":
            model = self.model_dropdown.currentText()
            output, _ = openai_call_reasoning(
                "sk-a66794170cbd42c08f7605f24f72e38f",
                model=model,
                user_content=self.user_prompt,
                image=image_np_to_base64(current_frame),
                system_content=self.system_prompt,
                include_usage=True
            )
        elif self.model_dropdown.currentText() == "qwen-vl-max-2025-01-25":
            model = self.model_dropdown.currentText()
            output = openai_call(
                "sk-a66794170cbd42c08f7605f24f72e38f",
                model=model,
                user_content=self.user_prompt,
                image=image_np_to_base64(current_frame),
                system_content=self.system_prompt
            )
        
        logger.info(f"AI called with:\nSystem: {self.system_prompt}\nUser: {self.user_prompt}\nOutput: {output}")
        self.statusBar().showMessage(output)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):
            self.save_current_frame()
        else:
            super().keyPressEvent(event)

    def save_current_frame(self):
        frame = self.get_frame(self.current_frame_idx)
        if frame is None:
            self.statusBar().showMessage("No frame to save!")
            return
        # Use PIL image to save the frame
        image = Image.fromarray(frame)
        save_path = './outputs/frame_{}.png'.format(self.current_frame_idx)
        image.save(save_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAIEditor()
    window.show()
    sys.exit(app.exec_())