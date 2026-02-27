# --- 初期設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# ==============================================================================
# ▼▼▼【重要】ここにあなたのGoogle AI (Gemini) のAPIキーを設定してください ▼▼▼
# ==============================================================================
# セキュリティのため、GitHub等にアップロードする際はここを空にするか、
# 環境変数から読み込むようにしてください。
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
