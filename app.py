import streamlit as st
import os
import sys
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import zipfile
import shutil
import dotenv
import pytz
import time
from datetime import datetime
import requests
import re
import streamlit.components.v1 as components
import locales
import importlib
importlib.reload(locales)
from locales import TRANSLATIONS

dotenv.load_dotenv()

# Отключаем progress-бары HuggingFace, которые крашат Streamlit в Windows из-за ошибки sys.stderr.flush
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Исправляем зависание (deadlock) PyTorch и OpenMP в многопоточной среде Streamlit на Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Патч для Streamlit: перехватываем ошибку Errno 22 при flush и write
if hasattr(sys.stderr, "flush"):
    original_flush = sys.stderr.flush
    def safe_flush():
        try:
            original_flush()
        except OSError:
            pass
    sys.stderr.flush = safe_flush

if hasattr(sys.stderr, "write"):
    original_write = sys.stderr.write
    def safe_write(*args, **kwargs):
        try:
            return original_write(*args, **kwargs)
        except OSError:
            pass
    sys.stderr.write = safe_write

# --- ПУТИ К ДАННЫМ (Persistent Storage) ---
APP_DATA_DIR = "app_data"
FORMS_DIR = os.path.join(APP_DATA_DIR, "forms")
CHROMA_PATH = os.path.join(APP_DATA_DIR, "chroma_db")

os.makedirs(APP_DATA_DIR, exist_ok=True)
os.makedirs(FORMS_DIR, exist_ok=True)

# Устанавливаем пути в session_state для совместимости
if "chroma_path" not in st.session_state:
    st.session_state.chroma_path = CHROMA_PATH
if "forms_dir" not in st.session_state:
    st.session_state.forms_dir = FORMS_DIR

# --- RAG LOGIC ---
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def get_best_model(api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "gemini-1.5-flash"
        models = response.json().get('models', [])
        valid_models = [m['name'].replace('models/', '') for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
        
        for pref in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]:
            for m in valid_models:
                if pref in m:
                    return m
        if valid_models:
            return valid_models[0]
    except:
        pass
    return "gemini-1.5-flash"

def index_pdf(pdf_path: str, api_key: str, chroma_path: str):
    docs = load_documents(pdf_path)
    chunks = split_text(docs)
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0, text=f"Подготовка {total_chunks} фрагментов текста...")
    
    progress_bar.progress(0.4, text="Анализ частотности слов (TF-IDF)...")
    retriever = TFIDFRetriever.from_documents(chunks)
    retriever.k = 8
    
    progress_bar.progress(0.8, text="Сохранение индекса в память...")
    with open(f"{chroma_path}.pkl", "wb") as f:
        pickle.dump(retriever, f)
        
    progress_bar.progress(1.0, text="Все фрагменты успешно проиндексированы!")
    return total_chunks

def format_docs(docs):
    return "\n\n".join(f"--- SOURCE PAGE: {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}" for doc in docs)

def ask_question(question: str, api_key: str, chroma_path: str, chat_history: list = None, forms_dir: str = None, prompt_lang: str = "RUSSIAN", vessel_info: dict = None):
    index_file = f"{chroma_path}.pkl"
    if not os.path.exists(index_file):
        return {"answer": "Error: Database is empty.", "sources": [], "retrieved_chunks": []}
        
    with open(index_file, "rb") as f:
        retriever = pickle.load(f)
    
    best_model_name = get_best_model(api_key)
    llm = ChatGoogleGenerativeAI(model=best_model_name, google_api_key=api_key, temperature=0.0)
    
    formatted_history = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "USER" if msg["role"] == "user" else "AI"
            formatted_history += f"{role}: {msg['content']}\n"
            
    available_forms = []
    if forms_dir and os.path.exists(forms_dir):
        for root, dirs, files in os.walk(forms_dir):
            for file in files:
                available_forms.append(file)
    forms_text = "\n".join(available_forms) if available_forms else "Нет загруженных форм"
    
    vessel_context = ""
    if vessel_info:
        vessel_context = f"\nYou are currently deployed on the vessel: '{vessel_info.get('name', 'Unknown')}' (IMO: {vessel_info.get('imo', 'Unknown')}). Next Port: {vessel_info.get('port', 'Unknown')}. Keep this in mind."
    
    translation_prompt = PromptTemplate.from_template(
        "You are an expert maritime assistant. Extract the core search keywords from the user's query and translate them to English. "
        "These keywords will be used to search a ship's Safety Management System (SMS) manual. "
        "Output ONLY the English keywords separated by spaces. No full sentences. "
        "For example, if the query is 'Какие формы заполнять по приезду на судно?', output 'join vessel arrival familiarization forms checklist'.\n\n"
        "User Query: {question}\n"
        "English Keywords:"
    )
    english_keywords = (translation_prompt | llm | StrOutputParser()).invoke({"question": question})
    
    template = f"""You are an expert Safety Management System (SMS) AI assistant for Neptune Marine, designed to help all crew members.{vessel_context}
Your ONLY task is to answer the user's questions STRICTLY based on the provided CONTEXT from the company's SMS manual.

RULES:
1. Answer strictly based on the CONTEXT. Do not hallucinate or make up procedures/forms.
2. If the answer is not in the CONTEXT, say that there is no answer in the provided document.
3. ALWAYS cite the page numbers in your answer.
4. The user may ask questions in {prompt_lang}, but the CONTEXT is in English. YOU MUST TRANSLATE AND RESPOND IN {prompt_lang}.
5. If the user asks about a form, list exactly what the form is called and what needs to be done.
6. CRITICAL: If the CONTEXT mentions ANY specific forms, checklists, or permits that the user must fill out, you MUST end your response by offering them a choice. Say exactly something like: "I can fill out the form [FORM NAME] for you — just give me the necessary input data. Or you can open and fill it yourself." Provide this choice in {prompt_lang}.
7. CRITICAL: For EVERY single form or checklist you mention in your response, you MUST look at the AVAILABLE FORMS list. Pick the filename that sounds the most similar (even if there are typos or number mismatches) and append the exact tag `[DOWNLOAD_FORM: filename.ext]` at the end of your response. If you mention 2 forms, output 2 tags. Do this ALWAYS. If the AVAILABLE FORMS list is "Нет загруженных форм", ignore this rule.

AVAILABLE FORMS:
{{forms_list}}

CHAT HISTORY:
{{chat_history}}

CONTEXT:
{{context}}

USER QUESTION: {{question}}

HELPFUL AND ACCURATE ANSWER IN {prompt_lang}:"""
    prompt = PromptTemplate.from_template(template)
    
    docs = retriever.invoke(english_keywords)
    
    rag_chain = (
        {
            "context": lambda x: format_docs(docs), 
            "question": RunnablePassthrough(),
            "chat_history": lambda x: formatted_history,
            "forms_list": lambda x: forms_text
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    sources = set([str(doc.metadata.get('page', 'Unknown')) for doc in docs])
    
    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        raise Exception(f"Failed to call model '{best_model_name}': {e}")
    
    return {"answer": answer, "sources": list(sources), "retrieved_chunks": [doc.page_content for doc in docs]}

# --- UI LOGIC ---
st.set_page_config(page_title="Neptune AI", page_icon="🚢", layout="wide")

# --- CUSTOM NEPTUNE MARINE CSS ---
neptune_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

    /* Глобальный шрифт */
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif !important;
    }

    /* Заголовки */
    h1, h2, h3 {
        color: #1A2B4C !important;
        font-weight: 700 !important;
    }

    /* Боковая панель */
    [data-testid="stSidebar"] {
        background-color: #1A2B4C;
    }
    
    /* Убираем огромный отступ сверху на главной странице (для разных версий Streamlit) */
    .block-container,
    div[data-testid="stAppViewBlockContainer"],
    div[data-testid="block-container"],
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
    }
    
    /* Скрываем системный заголовок Streamlit, который тоже занимает место */
    header[data-testid="stHeader"],
    .stApp > header {
        display: none !important;
        height: 0px !important;
    }
    
    /* Убираем огромный отступ сверху в боковой панели */
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Радио-кнопки в сайдбаре */
    div[role="radiogroup"] label {
        background-color: transparent !important;
        padding: 10px 15px;
        border-radius: 5px;
        transition: 0.3s;
        cursor: pointer;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Кнопки */
    .stButton>button, .stDownloadButton>button {
        background-color: #1A2B4C !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        padding: 0.5rem 1rem !important;
        transition: 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #0d2b56 !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
    }
    
    /* Универсальная зачистка серого фона внизу экрана */
    .stAppBottomBlock,
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stChatFloatingInputContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Анимация плавного появления */
    @keyframes chatSlideUp {
        0% { transform: translateY(150px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    /* Делаем полностью прозрачным главный контейнер */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        animation: chatSlideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* Форсируем прозрачность для ВСЕХ вложенных div */
    div[data-testid="stChatInput"] div {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* Glassmorphism: применяем стиль ТОЛЬКО к первой обертке! */
    div[data-testid="stChatInput"] > div:first-child {
        background-color: rgba(13, 43, 86, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        padding: 5px !important;
    }
    
    /* Текстовое поле */
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: white !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255,255,255,0.7) !important;
    }
    div[data-testid="stChatInput"] button {
        background-color: transparent !important;
        color: white !important;
    }
    [data-testid="stChatInput"] button {
        background-color: transparent !important;
        color: white !important;
    }

    /* Плавающая кнопка очистки чата */
    .element-container:has(.clear-btn-wrapper) + .element-container {
        position: fixed !important;
        bottom: 120px !important;
        right: 20px !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    
    /* Инфографика (Метрики) */
    [data-testid="stMetricValue"] {
        color: #1A2B4C !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Глобальный фон делаем прозрачным, чтобы видеть Canvas */
    .stApp, header[data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Текст внутри дашборда делаем светлым (Dark Mode) */
    html, body, p, span, div, label, li {
        color: #E2E8F0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Карточки чата */
    [data-testid="stChatMessage"] {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 1rem;
        backdrop-filter: blur(5px);
    }
    
    /* Поле ввода */
    [data-testid="stChatInput"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: white !important;
    }
</style>
"""
st.markdown(neptune_css, unsafe_allow_html=True)

# --- МОРСКИЕ ВОЛНЫ (Инъекция JS в родительский DOM) ---
particles_js = """
<script>
(function() {
    const parentDoc = window.parent.document;
    if (parentDoc.getElementById('neptune-waves-script')) return; // Скрипт уже работает в родителе

    // Внедряем скрипт напрямую в head главного окна, чтобы он не умирал при обновлении iframe Streamlit
    const script = parentDoc.createElement('script');
    script.id = 'neptune-waves-script';
    script.innerHTML = `
        (function() {
            if (document.getElementById('neptune-waves')) return;

            const canvas = document.createElement('canvas');
            canvas.id = 'neptune-waves';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.zIndex = '-1';
            canvas.style.pointerEvents = 'none';
            document.body.prepend(canvas);

            const ctx = canvas.getContext('2d');
            let width = canvas.width = window.innerWidth;
            let height = canvas.height = window.innerHeight;

            let mouse = { x: width/2, y: height/2 };

            window.addEventListener('mousemove', function(event) {
                mouse.x = event.x;
                mouse.y = event.y;
            });

            window.addEventListener('resize', function() {
                width = canvas.width = window.innerWidth;
                height = canvas.height = window.innerHeight;
            });

            let time = 0;
            
            // Светящиеся неоновые волны
            const waves = [
                { yOffset: 0.65, amplitude: 40, length: 0.003, speed: 0.015, color: 'rgba(56, 189, 248, 0.15)' },
                { yOffset: 0.75, amplitude: 60, length: 0.002, speed: 0.020, color: 'rgba(56, 189, 248, 0.25)' },
                { yOffset: 0.85, amplitude: 90, length: 0.0015, speed: 0.025, color: 'rgba(14, 165, 233, 0.4)' }
            ];

            function animate() {
                window.requestAnimationFrame(animate);
                
                // Рисуем приятный темный градиент на фоне, чтобы не слепить мостик
                let grad = ctx.createLinearGradient(0, 0, width, height);
                grad.addColorStop(0, '#0f172a'); // Темно-серый
                grad.addColorStop(1, '#1e3a8a'); // Глубокий морской синий
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, width, height);
                
                waves.forEach(wave => {
                    ctx.beginPath();
                    ctx.moveTo(0, height);
                    
                    for (let i = 0; i < width; i += 10) {
                        // Волны двигаются постоянно и плавно, независимо от мыши
                        let y = height * wave.yOffset + Math.sin(i * wave.length + time * wave.speed) * wave.amplitude;
                        
                        let dist = Math.abs(i - mouse.x);
                        if (dist < 300) {
                            let effect = (300 - dist) / 300;
                            y -= effect * 60 * (mouse.y / height);
                        }
                        
                        ctx.lineTo(i, y);
                    }
                    ctx.lineTo(width, height);
                    ctx.lineTo(0, height);
                    ctx.fillStyle = wave.color;
                    ctx.fill();
                    ctx.closePath();
                });

                time++;
            }

            animate();
        })();
    `;
    parentDoc.head.appendChild(script);
})();
</script>
"""
components.html(particles_js, height=0, width=0)

if "lang" not in st.session_state:
    default_lang = "EN"
    try:
        # Автоопределение языка по настройкам браузера пользователя
        acc_lang = st.context.headers.get("Accept-Language", "").lower()
        primary_lang = acc_lang.split(",")[0]
        if "nl" in primary_lang:
            default_lang = "NL"
        elif "ru" in primary_lang:
            default_lang = "RU"
        elif "uk" in primary_lang:
            default_lang = "UA"
        elif "tl" in primary_lang or "ph" in primary_lang:
            default_lang = "PH"
    except:
        pass
    st.session_state.lang = default_lang
saved_api_key = os.getenv("GEMINI_API_KEY", "")

t = TRANSLATIONS[st.session_state.lang]

# ----------------- TOP BAR (TIME) -----------------
import pytz
from datetime import datetime

tz_ship = pytz.utc
tz_holland = pytz.timezone('Europe/Amsterdam')
tz_kiev = pytz.timezone('Europe/Kiev')
tz_manila = pytz.timezone('Asia/Manila')
now = datetime.now()

time_html = f"""
<div style="text-align: right; font-size: 12px; color: #888; margin-bottom: 10px; padding-right: 10px; font-family: sans-serif;">
    <b>{t['ship_time']}:</b> {now.astimezone(tz_ship).strftime('%H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <b>{t['rtm_time']}:</b> {now.astimezone(tz_holland).strftime('%H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <b>{t['kiev_time']}:</b> {now.astimezone(tz_kiev).strftime('%H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; 
    <b>{t['mnl_time']}:</b> {now.astimezone(tz_manila).strftime('%H:%M')}
</div>
"""
st.markdown(time_html, unsafe_allow_html=True)

# ----------------- TOP NAVIGATION BAR -----------------
current_menu = st.query_params.get("menu", "chat")

VESSEL_DATA = {
    "Neptun 11": {"imo": "9571208", "type": "EuroTug 2710", "length": "27.27m", "bollard_pull": "42t"},
    "Neptun Commander": {"imo": "9751690", "type": "EuroTug 3210", "length": "32.0m", "bollard_pull": "45t"},
    "Neptun Foxtrot": {"imo": "9605449", "type": "EuroTug 3515", "length": "35.0m", "bollard_pull": "82t"},
    "Neptun Fury": {"imo": "9705718", "type": "EuroTug 3210", "length": "32.0m", "bollard_pull": "52t"},
    "Neptun Master": {"imo": "9646560", "type": "EuroTug 2710", "length": "27.27m", "bollard_pull": "42t"},
    "Neptun Power": {"imo": "1071393", "type": "EuroTug 2710", "length": "27.27m", "bollard_pull": "40t"}
}
eurotugs_fleet = list(VESSEL_DATA.keys())
if "current_vessel" not in st.session_state:
    st.session_state.current_vessel = "Neptun Fury"

query_vessel = st.query_params.get("vessel")
if query_vessel:
    st.session_state.current_vessel = query_vessel
    st.query_params.pop("vessel", None)
    st.rerun()

current_vessel = st.session_state.current_vessel
if current_vessel not in eurotugs_fleet:
    eurotugs_fleet.append(current_vessel)

# Check query params for lang selection
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "EN"

query_lang = st.query_params.get("setlang")
if query_lang and query_lang in ["EN", "NL", "PH", "RU", "UA"]:
    st.session_state.current_lang = query_lang
    st.query_params.pop("setlang", None)
    st.rerun()

current_lang = st.session_state.current_lang

# Custom CSS for slanted navigation tabs
nav_css = """
<style>
.nav-container {
    display: flex;
    align-items: center;
    background-color: transparent;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 4px;
}
.nav-item {
    background-color: #0d2b56;
    color: white !important;
    padding: 10px 30px;
    transform: skewX(-15deg);
    text-decoration: none !important;
    font-weight: bold;
    font-size: 16px;
    font-family: sans-serif;
    transition: all 0.2s ease-in-out;
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}
.nav-item > div {
    transform: skewX(15deg);
}
.nav-item:hover {
    background-color: #1a437a;
}
.nav-item.active {
    background-color: #ffffff !important;
    color: #0d2b56 !important;
}
.nav-item.active div {
    color: #0d2b56 !important;
}

/* Dropdown CSS */
.dropdown {
    position: relative;
    display: inline-block;
}
.dropdown-content {
    display: none;
    position: absolute;
    background-color: rgba(13, 43, 86, 0.95);
    min-width: 220px;
    z-index: 9999;
    border: 1px solid #1a437a;
    top: 100%;
    left: 0;
    box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.5);
}
.dropdown:hover .dropdown-content {
    display: block;
}
.dropdown-content a {
    color: white !important;
    padding: 12px 16px;
    text-decoration: none !important;
    display: block;
    font-family: sans-serif;
    font-weight: normal;
    text-align: left;
}
.dropdown-content a:hover {
    background-color: #1a437a;
}
</style>
"""

if os.path.exists("assets/logo_transparent.png"):
    import base64
    with open("assets/logo_transparent.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    # Render Logo and Nav side-by-side using Streamlit columns
    col_logo, col_nav = st.columns([1, 3], vertical_alignment="center")
    
    with col_logo:
        st.markdown(f'''
        <div style="margin-top: -20px; margin-bottom: -10px;">
            <a href="/?menu=chat" target="_self">
                <img src="data:image/png;base64,{img_b64}" style="height: 140px; width: auto; object-fit: contain; cursor: pointer; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
            </a>
        </div>
        ''', unsafe_allow_html=True)
        
    with col_nav:
        # Сгенерируем ссылки для судов
        vessel_links = ""
        for v in eurotugs_fleet:
            vessel_links += f'<a href="/?menu={current_menu}&vessel={v.replace(" ", "%20")}" target="_self">{v}</a>'

        @st.cache_data(ttl=600, show_spinner=False)
        def get_vessel_status(vessel_name, imo=None):
            if imo:
                try:
                    import requests
                    import re
                    from bs4 import BeautifulSoup
                    
                    url = f"https://www.vesselfinder.com/vessels/details/{imo}"
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    resp = requests.get(url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        text = BeautifulSoup(resp.text, 'html.parser').get_text()
                        
                        loc_match = re.search(r'is\s+at\s+(.*?)\s+reported', text)
                        loc = loc_match.group(1).strip() if loc_match else "Unknown location"
                        
                        status_match = re.search(r'Navigation Status\s+(.*?)\s+Position received', text, re.DOTALL)
                        status = status_match.group(1).strip() if status_match else "Unknown status"
                        
                        return loc, status
                except Exception:
                    pass
            
            # Fallback (mock)
            import hashlib
            from datetime import datetime
            seed = f"{vessel_name}_{datetime.utcnow().strftime('%Y-%m-%d %H')}"
            h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
            locations = ["North Sea", "Baltic Sea", "Rotterdam Port", "English Channel", "Norwegian Sea", "Bay of Biscay"]
            statuses = ["Underway using engine", "At anchor", "Moored", "Underway using engine"]
            loc = locations[h % len(locations)]
            status = statuses[(h // 10) % len(statuses)]
            if "Underway" in status:
                speed = 5.0 + (h % 50) / 10.0
                status = f"{status} ({speed} kn)"
            return loc, status

        loc_text = ""
        v_info = VESSEL_DATA.get(current_vessel, {})
        if v_info:
            loc, status = get_vessel_status(current_vessel, v_info.get("imo"))
            loc_text = f'<div style="font-size: 11px; color: #888; position: absolute; top: 110%; left: 0px; white-space: nowrap; font-family: sans-serif;">📍 {loc} | {status}</div>'

        langs = {"English": "EN", "Dutch": "NL", "Tagalog": "PH", "Русский": "RU", "Українська": "UA"}
        lang_links = ""
        for l_name, l_code in langs.items():
            lang_links += f'<a href="/?menu={current_menu}&setlang={l_code}" target="_self">{l_name} ({l_code})</a>'

        nav_html = f"""{nav_css}
<div class="nav-container" style="margin-top: -30px;">
    <a href="/?menu=chat" target="_self" class="nav-item {'active' if current_menu == 'chat' else ''}"><div>{t['menu_chat'].replace('💬 ', '')}</div></a>
    <a href="/?menu=settings" target="_self" class="nav-item {'active' if current_menu == 'settings' else ''}"><div>{t['menu_settings'].replace('⚙️ ', '')}</div></a>
    <div class="dropdown">
        <div class="nav-item"><div>{current_vessel} ▾</div></div>
        <div class="dropdown-content">
            {vessel_links}
        </div>
        {loc_text}
    </div>
    <div class="dropdown">
        <div class="nav-item"><div>🌍 Lang ({st.session_state.lang}) ▾</div></div>
        <div class="dropdown-content">
            {lang_links}
        </div>
    </div>
</div>"""
        st.markdown(nav_html, unsafe_allow_html=True)

st.markdown("---")

# ----------------- DIALOGS -----------------
if current_menu == "lang":
    @st.dialog("🌍 Select Language")
    def select_lang_dialog():
        langs = ["English", "Dutch", "Tagalog", "Русский", "Українська"]
        lang_codes = ["EN", "NL", "PH", "RU", "UA"]
        curr_idx = lang_codes.index(st.session_state.lang)
        lang_choice = st.selectbox("Language", langs, index=curr_idx, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        if col1.button("✅ Apply", use_container_width=True):
            st.session_state.lang = lang_codes[langs.index(lang_choice)]
            st.query_params.menu = "chat"
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.query_params.menu = "chat"
            st.rerun()
    select_lang_dialog()

# Set menu variable for downstream logic
menu = t["menu_settings"] if current_menu == "settings" else t["menu_chat"]

if menu == t["menu_settings"]:
    # Проверка прав администратора
    if not st.session_state.get("is_admin", False):
        st.title("🔒 Admin Access Required")
        st.markdown("Please enter the administrator password to access the system settings.")
        
        correct_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        try:
            if "ADMIN_PASSWORD" in st.secrets:
                correct_password = st.secrets["ADMIN_PASSWORD"]
        except:
            pass
            
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == correct_password:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.title(t["settings_title"])
        st.markdown(t["settings_desc"])
        
        if st.button("🚪 Logout from Admin", type="secondary"):
            st.session_state.is_admin = False
            st.rerun()
        
        st.header(t["api_key_header"])
        api_key = st.text_input(t["api_key_input"], value=saved_api_key, type="password")
        if api_key and api_key != saved_api_key:
            dotenv.set_key(".env", "GEMINI_API_KEY", api_key)
            st.success(t["api_saved"])
            st.rerun()
            
        st.markdown(t["get_api_key"])
        
        st.divider()

        st.header(t["db_header"])
        uploaded_file = st.file_uploader(t["upload_pdf"], type="pdf")
        
        if uploaded_file is not None:
            if st.button(t["index_doc"]):
                if not saved_api_key:
                    st.error(t["err_no_key"])
                else:
                    with st.spinner(t["spin_indexing"]):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            num_chunks = index_pdf(tmp_path, saved_api_key, st.session_state.chroma_path)
                            st.success(f"{t['success_index']} ({num_chunks})")
                        except Exception as e:
                            import traceback
                            error_trace = traceback.format_exc()
                            st.error(f"{t['err_index']} {e}\n\n```python\n{error_trace}\n```")
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

        st.divider()
        
        st.header(t["forms_header"])
        st.markdown(t["forms_desc"])
        uploaded_forms = st.file_uploader(t["upload_forms"], type=["pdf", "docx", "xlsx", "zip"], accept_multiple_files=True)
        
        if st.button(t["save_forms"]):
            if uploaded_forms:
                saved_count = 0
                for form_file in uploaded_forms:
                    if form_file.name.endswith('.zip'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                            tmp_zip.write(form_file.getvalue())
                            tmp_zip_path = tmp_zip.name
                        try:
                            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                                for zip_info in zip_ref.infolist():
                                    if zip_info.is_dir():
                                        continue
                                    zip_info.filename = os.path.basename(zip_info.filename)
                                    if zip_info.filename:
                                        zip_ref.extract(zip_info, st.session_state.forms_dir)
                            st.success(f"{t['success_unzip']} {form_file.name}")
                            saved_count += len(zip_ref.namelist())
                        except Exception as e:
                            st.error(f"{t['err_unzip']} {e}")
                        finally:
                            try:
                                os.unlink(tmp_zip_path)
                            except: pass
                    else:
                        file_path = os.path.join(st.session_state.forms_dir, form_file.name)
                        with open(file_path, "wb") as f:
                            f.write(form_file.getvalue())
                        saved_count += 1
                st.success(f"{t['success_forms']} {saved_count}")
            else:
                st.warning(t["warn_no_forms"])
                
        # Показать сохраненные формы
        if os.path.exists(st.session_state.forms_dir):
            available = []
            for root, dirs, files in os.walk(st.session_state.forms_dir):
                for file in files:
                    available.append(file)
            if available:
                st.markdown(t["forms_in_db"].format(len(available)))
                if st.button(t["clear_forms"]):
                    shutil.rmtree(st.session_state.forms_dir)
                    os.makedirs(st.session_state.forms_dir)
                    st.rerun()

elif menu == t["menu_chat"]:
    def type_text(text):
        for char in text:
            yield char
            time.sleep(0.03)

    if "welcome_typed" not in st.session_state:
        st.write_stream(type_text(f"# {t['chat_title']}"))
        st.session_state.welcome_typed = True
    else:
        st.title(t["chat_title"])
        
    st.markdown(t["chat_desc"])
    
    # -----------------------------------
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        avatar = "assets/logo_transparent.png" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            display_content = re.sub(r"\[DOWNLOAD_FORM:.*?\]", "", message["content"]).strip()
            st.markdown(display_content)
            
            if i == len(st.session_state.messages) - 1 and message["role"] == "assistant":
                forms = re.findall(r"\[DOWNLOAD_FORM:\s*(.*?)\]", message["content"])
                if forms:
                    st.markdown(t["action_header"])
                    cols = st.columns(len(forms) + 1)
                    for i_col, form_name in enumerate(forms):
                        form_filename = form_name.strip()
                        form_path = None
                        for root, dirs, files in os.walk(st.session_state.forms_dir):
                            if form_filename in files:
                                form_path = os.path.join(root, form_filename)
                                break
                                
                        if form_path and os.path.isfile(form_path):
                            with open(form_path, "rb") as file:
                                with cols[i_col]:
                                    st.download_button(
                                        label=t["dl_button"].format(form_filename),
                                        data=file,
                                        file_name=form_filename,
                                        mime="application/octet-stream",
                                        key=f"dl_{message['content'][:5]}_{i_col}"
                                    )
                    
                    with cols[-1]:
                        if st.button(t["fill_for_me"]):
                            st.session_state.pending_action = f"Пожалуйста, заполни за меня форму {forms[0].strip()}. Какие данные тебе для этого нужны?"
                            st.rerun()

    st.markdown('<div class="clear-btn-wrapper"></div>', unsafe_allow_html=True)
    if st.button("🗑️", help=t["clear_chat"]):
        st.session_state.messages = []
        st.rerun()

    prompt = st.chat_input(t["chat_placeholder"])

    if "pending_action" in st.session_state:
        prompt = st.session_state.pending_action
        del st.session_state.pending_action

    if prompt:
        if not saved_api_key:
            st.warning(t["warn_api_chat"])
        else:
            st.chat_message("user", avatar="👤").markdown(prompt)
            
            with st.spinner(t["spin_chat"]):
                try:
                    imo = VESSEL_DATA.get(current_vessel, {}).get("imo", "Unknown")
                    v_info = {"name": current_vessel, "imo": imo, "port": "Unknown"}
                    response_data = ask_question(prompt, saved_api_key, st.session_state.chroma_path, st.session_state.messages, st.session_state.forms_dir, t["ai_prompt_lang"], v_info)
                    
                    if isinstance(response_data, str):
                        answer = response_data
                    else:
                        answer = response_data["answer"]
                        sources = response_data["sources"]
                        retrieved_chunks = response_data["retrieved_chunks"]
                        if sources:
                            answer += f"\n\n*{t['sources_pages']} {', '.join(sources)}*"

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"{t['err_rag']} {e}")

    # Empty bottom space for padding
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
