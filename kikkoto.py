# -*- coding: utf-8 -*-
import os
import sys
import threading
import queue
import datetime
import tempfile
import time
import numpy as np
import speech_recognition as sr
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import librosa
from gtts import gTTS
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QPushButton, QHBoxLayout, QMainWindow, QFrame
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import google.generativeai as genai
import warnings

# 環境設定と警告の抑制
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# --- APIキーの設定 ---
# 環境変数 'GOOGLE_API_KEY' から読み込みます
API_KEY = os.environ.get('GOOGLE_API_KEY', '')

if not API_KEY:
    print("警告: GOOGLE_API_KEYが設定されていません。AI機能は動作しません。")
    gemini_model = None
else:
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Geminiモデル設定完了")
    except Exception as e:
        print(f"Geminiモデルの設定に失敗しました: {e}")
        gemini_model = None

# Whisperモデルのロード
whisper_model = whisper.load_model("small")

# 履歴ファイルの生成
HISTORY_FILE = datetime.datetime.now().strftime("%Y%m%d") + "_chat_history.txt"
if not os.path.exists(HISTORY_FILE):
    open(HISTORY_FILE, "w", encoding="utf-8").close()

# カラー設定
COLOR_USER1 = "#83a8ae"
COLOR_USER2 = "#ffffff"
COLOR_AI = "#447880"
COLOR_BG = "#eef1f3"
COLOR_LINE = "#447880"

# --- ユーティリティ関数 ---
def save_to_history(speaker, text):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{speaker}\t{text}\n")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip().split("\t")[1] for line in f if "\t" in line]

def find_similar_texts(text, history_texts, threshold=0.6):
    if not history_texts:
        return []
    corpus = history_texts + [text]
    vec = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
    similar_indices = [i for i, s in enumerate(sims) if s > threshold]
    return [history_texts[i] for i in similar_indices]

def get_funny_response(prompt, similar_texts):
    if not gemini_model: return "・・・"
    try:
        similar_summary = "\n".join(similar_texts)
        full_prompt = (
            "以下の類似した話題を踏まえて、繰り返し言っていることを一切示唆しないユーモアを交えた返答を1つ生成してください。"
            "会話に参加するように自然な文章でお願いします。\n\n"
            "似た話題:\n" + (similar_summary if similar_summary else "なし") + "\n\n"
            "今回の発言:\n" + prompt
        )
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip() if response and hasattr(response, 'text') else "・・・"
    except Exception as e:
        print("Geminiエラー:", e)
        return "・・・"

def synthesize_and_play(text, player):
    try:
        tts = gTTS(text=text, lang='ja')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        player.setMedia(QMediaContent(QUrl.fromLocalFile(tmp.name)))
        player.play()
        def cleanup():
            time.sleep(10)
            try: os.unlink(tmp.name)
            except: pass
        threading.Thread(target=cleanup, daemon=True).start()
    except Exception as e:
        print("再生エラー:", e)

# --- UIコンポーネント ---
class BubbleWidget(QWidget):
    def __init__(self, text, align_right=False, color="#ffffff", is_ai=False, play_callback=None):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            background-color: {color};
            border-radius: 15px;
            padding: 10px;
            font-size: 14px;
            color: {"white" if is_ai else "black"};
        """)

        if align_right:
            layout.addStretch()
            if is_ai and play_callback:
                play_btn = QPushButton("▶")
                play_btn.setFixedSize(30, 30)
                play_btn.setStyleSheet("border: none; background: transparent; color: white; font-size: 18px;")
                play_btn.clicked.connect(play_callback)
                layout.addWidget(label)
                layout.addWidget(play_btn)
            else:
                layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addStretch()

        self.setLayout(layout)

# --- 話者認識・メインアプリ ---
speaker_features = []
kmeans_model = None
min_samples_for_clustering = 10

def extract_mfcc(wav_path):
    y, sr_librosa = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_librosa, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def classify_speaker(feature):
    global kmeans_model, speaker_features
    if len(speaker_features) < min_samples_for_clustering:
        speaker_features.append(feature)
        return 0 
    if kmeans_model is None:
        kmeans_model = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=1024, n_init='auto')
        kmeans_model.fit(speaker_features)
    label = kmeans_model.predict([feature])[0]
    speaker_features.append(feature)
    kmeans_model.partial_fit([feature])
    return label

class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ユーモアAIチャット")
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.resize(500, 700)
        self.media_player = QMediaPlayer()
        self.recognizer = sr.Recognizer()
        self.chat_queue = queue.Queue()
        self.history = load_history()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        now = datetime.datetime.now().strftime("%m月%d日 %H:%M トーク開始")
        header = QLabel(now)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size:14px; color: #666; margin: 10px;")
        layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_contents = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_contents)
        self.scroll_layout.addStretch()
        self.scroll.setWidget(self.scroll_contents)
        layout.addWidget(self.scroll)

        threading.Thread(target=self.listen_loop, daemon=True).start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chat)
        self.timer.start(1000)

    def listen_loop(self):
        mic = sr.Microphone()
        while True:
            try:
                with mic as source:
                    print("話してください...")
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio.get_wav_data())
                    wav_path = tmp.name

                result = whisper_model.transcribe(wav_path, language='ja')
                text = result.get("text", "").strip()
                
                if text:
                    feature = extract_mfcc(wav_path)
                    speaker_id = classify_speaker(feature)
                    self.chat_queue.put((speaker_id, text))
                
                os.unlink(wav_path)
            except Exception as e:
                print("認識エラー:", e)

    def update_chat(self):
        last_text = None
        while not self.chat_queue.empty():
            spk_id, text = self.chat_queue.get()
            last_text = text
            old_hist = self.history.copy()
            self.history.append(text)
            save_to_history(f"ユーザー{spk_id}", text)
            self.add_message(text, spk_id)

        if last_text:
            sims = find_similar_texts(last_text, old_hist)
            if len(sims) >= 2:
                ai_reply = get_funny_response(last_text, sims)
                if ai_reply:
                    save_to_history("AI", ai_reply)
                    self.add_message(ai_reply, is_ai=True)
            self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def add_message(self, text, speaker_id=0, is_ai=False):
        color = COLOR_AI if is_ai else (COLOR_USER1 if speaker_id == 0 else COLOR_USER2)
        bubble = BubbleWidget(
            text, align_right=(is_ai or speaker_id == 1), color=color, is_ai=is_ai,
            play_callback=(lambda t=text: synthesize_and_play(t, self.media_player)) if is_ai else None
        )
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, bubble)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
