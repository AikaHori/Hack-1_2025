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

# MiniBatchKMeansの警告対策用
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# --- 初期設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# ==============================================================================
# ▼▼▼【重要】ここにあなたのGoogle AI (Gemini) のAPIキーを設定してください ▼▼▼
# ==============================================================================
os.environ['GOOGLE_API_KEY'] = '' 
# ==============================================================================

API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY or API_KEY == '':
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

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

whisper_model = whisper.load_model("small")

HISTORY_FILE = datetime.datetime.now().strftime("%Y%m%d") + "_chat_history.txt"
if not os.path.exists(HISTORY_FILE):
    open(HISTORY_FILE, "w", encoding="utf-8").close()

COLOR_USER1 = "#83a8ae"
COLOR_USER2 = "#ffffff"
COLOR_AI = "#447880"
COLOR_BG = "#eef1f3"
COLOR_LINE = "#447880"

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
    try:
        similar_summary = "\n".join(similar_texts)
        full_prompt = (
            "以下の類似した話題を踏まえて、繰り返し言っていることを一切示唆しないユーモアを交えた返答を1つ生成してください。"
            "会話に参加するように自然な文章でお願いします。\n\n"
            "似た話題:\n" + (similar_summary if similar_summary else "なし") + "\n\n"
            "今回の発言:\n" + prompt
        )
        response = gemini_model.generate_content(full_prompt)
        if response and hasattr(response, 'text') and response.text.strip():
            return response.text.strip()
        else:
            return "・・・"
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
            time.sleep(10) # 再生時間を考慮して少し待つ
            try:
                # player.stop() # 必要であれば再生を停止
                # player.setMedia(QMediaContent()) # メディアコンテンツをクリア
                os.unlink(tmp.name)
            except Exception as e:
                print(f"一時音声ファイル {tmp.name} の削除エラー（synthesize_and_play）: {e}")
        threading.Thread(target=cleanup, daemon=True).start()
    except Exception as e:
        print("再生エラー:", e)

class BubbleWidget(QWidget):
    def __init__(self, text, align_right=False, color="#ffffff", is_ai=False, play_callback=None):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"""
            background-color: {color};
            border-radius: 15px;
            padding: 10px;
            font-size: 14px;
        """)

        if align_right:
            if is_ai:
                play_btn = QPushButton("\u25b6")
                play_btn.setFixedSize(30, 30)
                play_btn.setStyleSheet("border: none; background: transparent; color: white;") # AIバブルの再生ボタンは白色
                if play_callback:
                    play_btn.clicked.connect(play_callback)
                layout.addStretch()
                layout.addWidget(label)
                layout.addWidget(play_btn)
            else:
                layout.addStretch()
                layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addStretch()

        self.setLayout(layout)
        self.setStyleSheet("background: transparent;")

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

def get_color_by_speaker_id(speaker_id):
    return COLOR_USER1 if speaker_id == 0 else COLOR_USER2

class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ユーモアAIチャット")
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.resize(600, 800)
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
        header.setStyleSheet("font-size:16px; font-weight:bold;")
        layout.addWidget(header)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLOR_LINE}; height:3px;")
        layout.addWidget(line)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
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
        try:
            mic = sr.Microphone()
        except Exception as e:
            print("マイクエラー:", e)
            return

        while True:
            wav_path = None  # Initialize wav_path
            try:
                with mic as source:
                    print("話してください...")
                    # recognizer.adjust_for_ambient_noise(source) # 環境ノイズへの適応（必要に応じて）
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio.get_wav_data())
                    wav_path = tmp.name

                result = whisper_model.transcribe(wav_path, language='ja')
                text_result = result.get("text", "").strip()
                
                if not text_result:
                    continue # テキストが空なら次のループへ (wav_pathはfinallyで処理)

                feature = extract_mfcc(wav_path)
                speaker_id = classify_speaker(feature)
                self.chat_queue.put((speaker_id, text_result))

            except sr.WaitTimeoutError:
                print("音声入力がタイムアウトしました。")
                # wav_path が作成されていれば finally で削除
                continue
            except Exception as e:
                print("認識エラー:", e)
                # wav_path が作成されていれば finally で削除
            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.unlink(wav_path)
                    except Exception as unlink_e:
                        print(f"一時ファイル {wav_path} の削除エラー（listen_loop）: {unlink_e}")
                wav_path = None # 次のイテレーションのためにリセット

    def update_chat(self):
        last_processed_text_in_queue = None
        history_for_last_text_in_queue = None

        while not self.chat_queue.empty():
            speaker_id, text_from_queue = self.chat_queue.get()
            
            last_processed_text_in_queue = text_from_queue
            history_for_last_text_in_queue = self.history.copy()
            
            self.history.append(text_from_queue)
            save_to_history(f"ユーザー{speaker_id}", text_from_queue)
            self.add_message(text_from_queue, speaker_id)

        if last_processed_text_in_queue is not None and history_for_last_text_in_queue is not None:
            similar_texts = find_similar_texts(last_processed_text_in_queue, history_for_last_text_in_queue)
        
            ai_reply = ""
            # 修正点: 類似テキストが2つ以上見つかった場合にAIが応答
            if len(similar_texts) >= 2:
                ai_reply = get_funny_response(last_processed_text_in_queue, similar_texts)

            if ai_reply:
                save_to_history("AI", ai_reply)
                self.add_message(ai_reply, is_ai=True)

        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def add_message(self, text, speaker_id=0, is_ai=False):
        if is_ai:
            bubble = BubbleWidget(
                text, align_right=True, color=COLOR_AI, is_ai=True,
                play_callback=lambda t=text: synthesize_and_play(t, self.media_player)
            )
        else:
            align_right = (speaker_id == 1)
            bubble = BubbleWidget(text, align_right=align_right, color=get_color_by_speaker_id(speaker_id))
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, bubble)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
