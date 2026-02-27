# -*- coding: utf-8 -*-
import os
import sys
import threading
import queue
import datetime
import tempfile
import time
import re # ファイル名チェック用に正規表現モジュールをインポート
import numpy as np
import speech_recognition as sr
import whisper # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from gtts import gTTS
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QPushButton, QHBoxLayout, QMainWindow, QFrame, QStackedWidget,
    QSpacerItem, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QBrush, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import google.generativeai as genai # type: ignore
import warnings

# --- 初期設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# ==============================================================================
# ▼▼▼【重要】ここにあなたのGoogle AI (Gemini) のAPIキーを設定してください ▼▼▼
# ==============================================================================
# 'AIza...' から始まるご自身のAPIキーに書き換えてください。
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCEw7xXAVVBWfdbQrOiIrZrSHjXrHFC6o8'
# ==============================================================================

API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY or 'AIzaSyDMgxOQC' in API_KEY or 'AIzaSyDA4gjC' in API_KEY or 'AIzaSyCsI-aZf' in API_KEY:
    print("警告: GOOGLE_API_KEYが設定されていないか、デモ用のキーのままです。AI機能は動作しません。")
    gemini_model = None
else:
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Geminiモデル設定完了")
    except Exception as e:
        print(f"Geminiモデルの設定に失敗しました: {e}")
        gemini_model = None

whisper_model = None
try:
    whisper_model = whisper.load_model("small")
    print("Whisperモデルのロード完了")
except Exception as e:
    print(f"Whisperモデルのロードに失敗しました: {e}")

# --- グローバル定数 ---
HISTORY_FILE_POSTFIX = ".txt"
COLOR_USER1 = "#83a8ae"
COLOR_USER2 = "#ffffff"
COLOR_AI = "#447880"
COLOR_BG_APP = "#eef1f3"
COLOR_LINE = "#447880"
COLOR_HISTORY_SCREEN_BG = "#EEF1F3"
COLOR_HISTORY_SCREEN_TEXT = "#447880"

# --- 履歴関連関数 ---
def get_new_session_filename():
    """新しい会話セッション用のユニークなファイル名を生成"""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + HISTORY_FILE_POSTFIX

def save_text_to_history_file(file_name_to_save, speaker, text):
    try:
        with open(file_name_to_save, "a", encoding="utf-8") as f: f.write(f"{speaker}\t{text}\n")
    except IOError as e: print(f"履歴ファイル '{file_name_to_save}' への書き込み失敗: {e}")

def load_texts_from_history_file(file_name_to_load):
    if not os.path.exists(file_name_to_load): return []
    texts = []
    try:
        with open(file_name_to_load, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2: texts.append(parts[-1])
                elif len(parts) == 1 and parts[0]: texts.append(parts[0])
    except IOError as e: print(f"履歴ファイル '{file_name_to_load}' からのテキスト読み込み失敗: {e}"); return []
    return texts

def load_structured_history_for_display(file_name_to_display):
    if not os.path.exists(file_name_to_display): return []
    history_entries = []
    try:
        with open(file_name_to_display, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) == 2: history_entries.append({"time": f"Entry {line_num+1}", "speaker": parts[0], "text": parts[1]})
    except IOError as e: print(f"構造化履歴ファイル '{file_name_to_display}' の読み込み失敗: {e}"); return []
    return history_entries

def get_all_history_files_info():
    history_files_info = []
    current_dir = os.getcwd()
    filename_pattern = re.compile(r"^\d{14}\.txt$")
    try:
        for filename in os.listdir(current_dir):
            if filename_pattern.match(filename):
                try:
                    date_str = filename.split(HISTORY_FILE_POSTFIX)[0]
                    date_obj = datetime.datetime.strptime(date_str, "%Y%m%d%H%M%S")
                    display_text = date_obj.strftime("%m月%d日 %H:%M")
                    history_files_info.append({"date": date_obj, "display_text": display_text, "file_name": filename})
                except ValueError: continue
        history_files_info.sort(key=lambda x: x["date"], reverse=True)
        return history_files_info[:5]
    except Exception as e:
        print(f"get_all_history_files_infoでエラー: {e}")
        QMessageBox.warning(None, "履歴エラー", f"履歴ファイルの読み込み中にエラーが発生しました:\n{e}")
        return []

# --- AIおよび音声関連関数 ---
# === ★★★ 変更箇所 (類似度判定を緩く) ★★★ ===
def find_similar_texts(text, history_texts, threshold=0.5): # 閾値を0.6から0.5に変更
    if not history_texts: return []
    corpus = history_texts + [text]
    try:
        vec = TfidfVectorizer().fit_transform(corpus)
        sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
        similar_indices = [i for i, s in enumerate(sims) if s > threshold]
        return [history_texts[i] for i in similar_indices]
    except ValueError: return []
# === ★★★ 変更箇所ここまで ★★★ ===

def get_funny_response(prompt, similar_texts):
    if not gemini_model: return "・・・（AIモデルが設定されていません）"
    try:
        similar_summary = "\n".join(similar_texts)
        full_prompt = (
            "以下の類似した話題を踏まえて、繰り返し言っていることを一切示唆しないユーモアを交えた返答を1つ生成してください。"
            "会話に参加するように自然な文章でお願いします。\n\n"
            "似た話題:\n" + (similar_summary if similar_summary else "なし") + "\n\n"
            "今回の発言:\n" + prompt)
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip() if response and hasattr(response, 'text') and response.text.strip() else "・・・"
    except Exception as e: print(f"Geminiエラー: {e}"); return "・・・"

def synthesize_and_play(text, player):
    try:
        tts = gTTS(text=text, lang='ja')
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name); tmp_file.close()
        player.setMedia(QMediaContent(QUrl.fromLocalFile(tmp_file.name)))
        player.play()
        def cleanup_audio(file_path_to_delete):
            time.sleep(10)
            try:
                if os.path.exists(file_path_to_delete): os.unlink(file_path_to_delete)
            except Exception as e_clean: print(f"一時音声ファイル {file_path_to_delete} の削除エラー: {e_clean}")
        threading.Thread(target=cleanup_audio, args=(tmp_file.name,), daemon=True).start()
    except Exception as e: print(f"再生エラー: {e}")

# --- UIコンポーネント ---
class BubbleWidget(QWidget):
    def __init__(self, text, align_right=False, color="#ffffff", is_ai=False, play_callback=None):
        super().__init__()
        layout = QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        label = QLabel(text); label.setWordWrap(True)
        label.setStyleSheet(f"""
            background-color: {color};
            border-radius: 20px;
            padding: 12px;
            font-family: 'Inter';
            font-size: 27px;
        """)
        if align_right:
            if is_ai:
                play_btn = QPushButton("\u25b6"); play_btn.setFixedSize(30, 30)
                play_btn.setStyleSheet("border: none; background: transparent; color: white;")
                if play_callback: play_btn.clicked.connect(play_callback)
                layout.addStretch(); layout.addWidget(label); layout.addWidget(play_btn)
            else: layout.addStretch(); layout.addWidget(label)
        else: layout.addWidget(label); layout.addStretch()
        self.setLayout(layout); self.setStyleSheet("background: transparent;")

# --- 話者識別関連 ---
speaker_centroids = [None, None]
speaker_sample_counts = [0, 0]

def reset_speaker_identification():
    global speaker_centroids, speaker_sample_counts
    speaker_centroids = [None, None]; speaker_sample_counts = [0, 0]

def extract_mfcc_feature(wav_path):
    try:
        y, sr_librosa = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr_librosa, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception: return np.zeros(13)

def classify_speaker(feature):
    global speaker_centroids, speaker_sample_counts
    if speaker_centroids[0] is None:
        speaker_centroids[0] = feature; speaker_sample_counts[0] = 1; return 0
    if speaker_centroids[1] is None:
        speaker_centroids[1] = feature; speaker_sample_counts[1] = 1; return 1
    norm_feature = feature / np.linalg.norm(feature)
    norm_c0 = speaker_centroids[0] / np.linalg.norm(speaker_centroids[0])
    norm_c1 = speaker_centroids[1] / np.linalg.norm(speaker_centroids[1])
    sim0 = np.dot(norm_feature, norm_c0); sim1 = np.dot(norm_feature, norm_c1)
    assigned_id = 0 if sim0 > sim1 else 1
    old_centroid = speaker_centroids[assigned_id]; old_count = speaker_sample_counts[assigned_id]
    new_centroid = (old_centroid * old_count + feature) / (old_count + 1)
    speaker_centroids[assigned_id] = new_centroid; speaker_sample_counts[assigned_id] += 1
    return assigned_id

def get_user_color_by_speaker_id(speaker_id):
    return COLOR_USER1 if speaker_id == 0 else COLOR_USER2

# --- 画面クラス ---
class ConversationHistoryScreen(QWidget):
    def __init__(self, main_app_logic):
        super().__init__()
        self.main_app_logic = main_app_logic; self.setStyleSheet(f"background-color: {COLOR_HISTORY_SCREEN_BG};")
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 20); layout.setSpacing(0)
        header_widget = QWidget(); header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 5); header_layout.setSpacing(10)
        self.back_button = QPushButton("<")
        self.back_button.setStyleSheet(f"QPushButton {{ background-color: transparent; border: none; font-size: 28px; font-weight: bold; color: {COLOR_HISTORY_SCREEN_TEXT}; padding: 5px; }} QPushButton:hover {{ color: #608a90; }}")
        self.back_button.setFixedSize(40,40); self.back_button.clicked.connect(self.main_app_logic.show_chat_screen_default)
        header_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        header_layout.addStretch()
        layout.addWidget(header_widget)
        
        line_frame = QFrame(); line_frame.setFrameShape(QFrame.NoFrame)
        line_frame.setFixedHeight(3)
        line_frame.setStyleSheet(f"background-color: {COLOR_LINE};"); layout.addWidget(line_frame)
        
        title_label = QLabel("会話の記録"); title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {COLOR_HISTORY_SCREEN_TEXT}; padding-top: 15px; padding-bottom: 22px;")
        layout.addWidget(title_label)
        
        self.history_buttons_layout = QVBoxLayout()
        self.history_buttons_layout.setContentsMargins(50, 0, 50, 0); self.history_buttons_layout.setSpacing(15)
        self.populate_history_buttons(); layout.addLayout(self.history_buttons_layout); layout.addStretch(1)
        
        footer_container = QWidget()
        footer_layout = QVBoxLayout(footer_container)
        footer_layout.setContentsMargins(0,0,0,0)
        footer_layout.setSpacing(-5)
        footer_layout.setAlignment(Qt.AlignCenter)

        start_conversation_label = QLabel("会話を始める"); start_conversation_label.setAlignment(Qt.AlignCenter)
        start_conversation_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        start_conversation_label.setStyleSheet(f"font-size: 36px; font-family: 'Inter'; color: {COLOR_HISTORY_SCREEN_TEXT}; background-color: transparent;")
        
        self.mic_button = QPushButton()
        start_button_icon = QIcon("startBottun.png")
        if not start_button_icon.isNull():
            button_size = 320
            self.mic_button.setIcon(start_button_icon)
            self.mic_button.setIconSize(QSize(button_size, button_size))
            self.mic_button.setFixedSize(button_size, button_size)
            self.mic_button.setStyleSheet("QPushButton { border: none; background: transparent; }")
        else:
            print("警告: startBottun.png が見つかりません。絵文字ボタンで代替します。")
            self.mic_button.setText("🎙️")
            self.mic_button.setFixedSize(70, 70)
            self.mic_button.setStyleSheet(f"QPushButton {{ background-color: {COLOR_HISTORY_SCREEN_TEXT}; color: white; font-size: 35px; border-radius: 35px; border: none; padding-bottom: 5px;}} QPushButton:hover {{ background-color: #3a656c; }}")
        
        self.mic_button.clicked.connect(self.main_app_logic.start_new_conversation)
        
        footer_layout.addWidget(self.mic_button)
        footer_layout.addWidget(start_conversation_label)

        layout.addWidget(footer_container)

    def populate_history_buttons(self):
        for i in reversed(range(self.history_buttons_layout.count())):
            widget = self.history_buttons_layout.itemAt(i).widget()
            if widget: widget.setParent(None)
        history_files = get_all_history_files_info()
        if not history_files:
            no_history_label = QLabel("利用可能な会話履歴はありません."); no_history_label.setAlignment(Qt.AlignCenter)
            no_history_label.setStyleSheet(f"font-family: 'Inter'; font-size: 28px; color: {COLOR_HISTORY_SCREEN_TEXT};")
            self.history_buttons_layout.addWidget(no_history_label, alignment=Qt.AlignCenter); return
        for info in history_files:
            btn = QPushButton(info["display_text"]); btn.setMinimumHeight(50)
            btn.setStyleSheet("QPushButton { background-color: white; border: none; border-radius: 15px; padding: 10px; font-family: 'Inter'; font-size: 28px; color: black; } QPushButton:hover { background-color: #f0f0f0; }")
            btn.clicked.connect(lambda checked, fn=info["file_name"]: self.main_app_logic.show_specific_history_chat_screen(fn))
            self.history_buttons_layout.addWidget(btn)
    def refresh_history_list(self): self.populate_history_buttons()

class ChatApp(QMainWindow):
    def __init__(self, main_app_logic, history_file_to_view=None):
        super().__init__()
        self.main_app_logic = main_app_logic
        self.is_history_view_mode = bool(history_file_to_view)
        self.start_time = datetime.datetime.now()

        if self.is_history_view_mode:
            self.current_session_file = history_file_to_view
            self.history_texts_for_similarity = []
        else:
            self.current_session_file = get_new_session_filename()
            self.history_texts_for_similarity = []
            reset_speaker_identification()

        self.setStyleSheet(f"background-color: {COLOR_BG_APP};"); self.resize(600, 800)
        self.media_player = QMediaPlayer(); self.recognizer = sr.Recognizer()
        self.chat_queue = queue.Queue()
        central = QWidget(); self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0); self.main_layout.setSpacing(0)

        self.header_widget = QWidget(); header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5); header_layout.setSpacing(5)
        self.back_to_history_btn = QPushButton("<")
        self.back_to_history_btn.setStyleSheet("QPushButton { background-color: transparent; border: none; font-size: 28px; font-weight: bold; color: #000000; padding: 5px;} QPushButton:hover { color: #555555; }")
        self.back_to_history_btn.setFixedSize(40,40); self.back_to_history_btn.clicked.connect(self.main_app_logic.show_history_screen)
        header_layout.addWidget(self.back_to_history_btn)
        self.title_label = QLabel()
        if self.is_history_view_mode:
            try:
                dt = datetime.datetime.strptime(self.current_session_file.split(HISTORY_FILE_POSTFIX)[0], "%Y%m%d%H%M%S")
                self.title_label.setText(dt.strftime("%Y年%m月%d日 %H:%M の会話"))
            except ValueError: self.title_label.setText("会話履歴")
        else:
            self.title_label.setText(self.start_time.strftime("%Y年%m月%d日 %H:%M トーク開始"))
        
        self.title_label.setAlignment(Qt.AlignCenter); self.title_label.setStyleSheet("font-family: 'Inter'; font-size:24px; font-weight:bold; color:black;")
        header_layout.addWidget(self.title_label, 1); header_layout.addSpacerItem(QSpacerItem(40, 40, QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.main_layout.addWidget(self.header_widget)

        line = QFrame(); line.setFrameShape(QFrame.NoFrame); line.setFixedHeight(3)
        line.setStyleSheet(f"background-color: {COLOR_LINE};"); self.main_layout.addWidget(line)
        self.main_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True);
        self.scroll_contents = QWidget(); self.scroll_layout = QVBoxLayout(self.scroll_contents)
        self.scroll_layout.setContentsMargins(10, 5, 10, 5); self.scroll_layout.addStretch()
        self.scroll.setWidget(self.scroll_contents); self.main_layout.addWidget(self.scroll)
        
        self.image_on_chat = QLabel(self.centralWidget())
        pixmap = QPixmap("noticeWomen.png")
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaledToWidth(150, Qt.SmoothTransformation)
            self.image_on_chat.setPixmap(scaled_pixmap)
            self.image_on_chat.resize(scaled_pixmap.size())
            self.image_on_chat.setAttribute(Qt.WA_TransparentForMouseEvents)
        else:
            print("警告: noticeWomen.png が見つからないか、読み込めません。")
            self.image_on_chat = None

        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll.viewport().setStyleSheet("background: transparent;")
        
        # === ★★★ 変更箇所 (スクロールバーを常に非表示に) ★★★ ===
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        if self.is_history_view_mode:
            self.load_and_display_chat_history()
        else:
            if whisper_model: threading.Thread(target=self.listen_loop, daemon=True).start()
            else: self.add_message_to_ui("音声認識モデル利用不可。", is_system_message=True)
            self.timer = QTimer(self); self.timer.timeout.connect(self.update_chat_ui); self.timer.start(1000)
        # === ★★★ 変更箇所ここまで ★★★ ===
    
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._position_image_on_chat)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_image_on_chat()

    def _position_image_on_chat(self):
        if hasattr(self, 'image_on_chat') and self.image_on_chat and self.image_on_chat.isVisible():
            padding = 15
            img_size = self.image_on_chat.size()
            x = self.width() - img_size.width() - padding
            y = self.height() - img_size.height() - padding
            self.image_on_chat.move(x, y)
            self.image_on_chat.lower()

    def load_and_display_chat_history(self):
        entries = load_structured_history_for_display(self.current_session_file)
        for entry in entries:
            is_ai = "AI" in entry["speaker"]; speaker_id_val = 0
            if "ユーザー" in entry["speaker"]:
                try: speaker_id_val = int(entry["speaker"].replace("ユーザー", ""))
                except ValueError: speaker_id_val = 0
            self.add_message_to_ui(entry["text"], speaker_id_val, is_ai)
        
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(0))

    def listen_loop(self):
        if self.is_history_view_mode: return
        try: mic = sr.Microphone()
        except Exception as e: print(f"マイクエラー: {e}"); self.chat_queue.put((-1, "マイクエラー発生")); return
        try:
            with mic as source: self.recognizer.adjust_for_ambient_noise(source, duration=0.7)
        except Exception as e_noise: print(f"ノイズ適応エラー: {e_noise}")
        while not self.is_history_view_mode:
            wav_path_temp = None
            try:
                with mic as source: audio_data = self.recognizer.listen(source, phrase_time_limit=7, timeout=5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(audio_data.get_wav_data()); wav_path_temp = tmp_wav.name
                if not whisper_model: self.chat_queue.put((-1, "(音声認識不可)")); continue
                transcription_result = whisper_model.transcribe(wav_path_temp, language='ja')
                recognized_text = transcription_result.get("text", "").strip()
                if not recognized_text:
                    if wav_path_temp and os.path.exists(wav_path_temp): os.unlink(wav_path_temp)
                    wav_path_temp = None; continue
                audio_feature = extract_mfcc_feature(wav_path_temp)
                speaker_identity = classify_speaker(audio_feature)
                self.chat_queue.put((speaker_identity, recognized_text))
            except sr.WaitTimeoutError: continue
            except sr.UnknownValueError: print("音声を認識できませんでした。"); continue
            except Exception as e_listen: print(f"認識ループエラー: {e_listen}")
            finally:
                if wav_path_temp and os.path.exists(wav_path_temp):
                    try: os.unlink(wav_path_temp)
                    except Exception as e_unlink: print(f"一時ファイル削除エラー: {e_unlink}")

    def update_chat_ui(self):
        if self.is_history_view_mode: return
        last_user_text_for_ai = None
        history_snapshot_for_ai = self.history_texts_for_similarity.copy()
        while not self.chat_queue.empty():
            speaker_id_val, text_val = self.chat_queue.get()
            if speaker_id_val == -1: self.add_message_to_ui(text_val, is_system_message=True); continue
            self.add_message_to_ui(text_val, speaker_id_val, is_ai=False)
            save_text_to_history_file(self.current_session_file, f"ユーザー{speaker_id_val}", text_val)
            self.history_texts_for_similarity.append(text_val)
            last_user_text_for_ai = text_val
        if last_user_text_for_ai:
            similar = find_similar_texts(last_user_text_for_ai, history_snapshot_for_ai)
            if len(similar) >= 1:
                ai_response_text = get_funny_response(last_user_text_for_ai, similar)
                if ai_response_text:
                    self.add_message_to_ui(ai_response_text, is_ai=True)
                    save_text_to_history_file(self.current_session_file, "AI", ai_response_text)
                    self.history_texts_for_similarity.append(ai_response_text)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def add_message_to_ui(self, text, speaker_id=0, is_ai=False, is_system_message=False):
        if is_system_message:
            sys_label = QLabel(text); sys_label.setAlignment(Qt.AlignCenter)
            sys_label.setStyleSheet("font-style: italic; color: gray; margin: 10px;"); sys_label.setWordWrap(True)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, sys_label); return
        align_bubble_right = is_ai
        bubble_bg_color = COLOR_AI if is_ai else get_user_color_by_speaker_id(speaker_id)
        audio_play_callback = (lambda t=text: synthesize_and_play(t, self.media_player)) if is_ai else None
        new_bubble = BubbleWidget(text, align_bubble_right, bubble_bg_color, is_ai, audio_play_callback)
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, new_bubble)

    def closeEvent(self, event):
        self.is_history_view_mode = True
        if hasattr(self, 'timer') and self.timer.isActive(): self.timer.stop()
        if self.media_player.state() == QMediaPlayer.PlayingState: self.media_player.stop()
        super().closeEvent(event)

# --- メインアプリケーションロジック ---
class MainApplicationLogic:
    def __init__(self, app_instance):
        self.app = app_instance
        self.stacked_widget = QStackedWidget()
        self.history_screen = ConversationHistoryScreen(self)
        self.current_chat_screen_instance = None
        self.stacked_widget.addWidget(self.history_screen)
        self.show_history_screen()
        self.stacked_widget.resize(650, 850); self.stacked_widget.setWindowTitle("AIチャット")
        self.stacked_widget.show()

    def show_history_screen(self):
        if self.current_chat_screen_instance:
            self.current_chat_screen_instance.is_history_view_mode = True
        self.history_screen.refresh_history_list()
        self.stacked_widget.setCurrentWidget(self.history_screen)

    def show_chat_screen_default(self): self.start_new_conversation()
    def start_new_conversation(self): self._create_and_show_chat_screen(history_file_to_view=None)
    def show_specific_history_chat_screen(self, history_file_name): self._create_and_show_chat_screen(history_file_to_view=history_file_name)

    def _create_and_show_chat_screen(self, history_file_to_view=None):
        if self.current_chat_screen_instance:
            self.stacked_widget.removeWidget(self.current_chat_screen_instance)
            self.current_chat_screen_instance.close(); self.current_chat_screen_instance.deleteLater()
        self.current_chat_screen_instance = ChatApp(self, history_file_to_view)
        self.stacked_widget.addWidget(self.current_chat_screen_instance)
        self.stacked_widget.setCurrentWidget(self.current_chat_screen_instance)

# --- アプリケーション実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font_path = os.path.join("inter", "Inter-Regular.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            if font_families: print(f"フォント '{font_families[0]}' をロードしました。")
        else: print(f"警告: フォントファイル {font_path} のロードに失敗しました。")
    else: print(f"警告: フォントファイル {font_path} が見つかりません。")
    
    app_logic = MainApplicationLogic(app)
    sys.exit(app.exec_())