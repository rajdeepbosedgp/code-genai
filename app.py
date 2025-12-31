import streamlit as st
import requests
import json
from datetime import datetime
from PIL import Image
import pytesseract
import io

# Page config
st.set_page_config(
    page_title="CODE-GenAI", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False

if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Theme settings
if st.session_state.dark_mode:
    primary_bg = "#1a1f2e"
    secondary_bg = "#0f1419"
    card_bg = "#1e293b"
    text_color = "#e2e8f0"
    subtext_color = "#94a3b8"
    border_color = "#334155"
    accent_color = "#60a5fa"
else:
    primary_bg = "#ffffff"
    secondary_bg = "#f8fafc"
    card_bg = "#f1f5f9"
    text_color = "#1e293b"
    subtext_color = "#64748b"
    border_color = "#cbd5e1"
    accent_color = "#3b82f6"

# Optimized CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background-color: {primary_bg};
    }}
    
    #MainMenu, footer {{visibility: hidden;}}
    
    /* Top bar */
    header[data-testid="stHeader"] {{
        background-color: {secondary_bg} !important;
        border-bottom: 1px solid {border_color} !important;
        height: 60px !important;
    }}
    
    header[data-testid="stHeader"]::before {{
        content: "✦ CODE-GenAI";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        color: {text_color};
        font-size: 1.1rem;
        font-weight: 700;
        top: 50%;
        transform: translate(-50%, -50%);
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {secondary_bg};
        border-right: 1px solid {border_color};
        padding-top: 0.5rem;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"], 
    [data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}
    
    .stButton button {{
        width: 100%;
        border-radius: 8px;
        background-color: {card_bg};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.15s ease;
        font-size: 0.95rem;
    }}
    
    .stButton button:hover {{
        border-color: {accent_color};
        background-color: {border_color};
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(96, 165, 250, 0.2);
    }}
    
    .stTextInput input {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }}
    
    .stChatMessage {{
        background-color: transparent !important;
        padding: 1rem 0 !important;
    }}
    
    .stChatMessage > div {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: {text_color} !important;
        line-height: 1.6;
    }}
    
    .stChatInput textarea {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }}
    
    .stChatInput textarea:focus {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }}
    
    .main .block-container {{
        padding-top: 1rem;
        max-width: 1400px;
    }}
    
    .stSpinner > div {{
        border-color: {accent_color} transparent transparent transparent !important;
    }}
    
    code {{
        background-color: {secondary_bg} !important;
        color: {accent_color} !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
    }}
    
    pre {{
        background-color: {secondary_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
</style>
""", unsafe_allow_html=True)

# Function to extract text from image using OCR
def extract_text_from_image(image):
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to call Ollama API with streaming
def call_ollama_stream(prompt):
    try:
        response = requests.post(
            f"{st.session_state.ollama_url}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if "response" in json_response:
                        full_response += json_response["response"]
            return full_response
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on http://localhost:11434"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model is taking too long to respond. Try a shorter prompt or restart Ollama."
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 1rem 0 1rem 0;">
        <span style="font-size: 1.3rem; color: {accent_color};">✦</span>
        <span style="color: {text_color}; font-size: 1.15rem; font-weight: 700;">CODE-GenAI</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New Chat", key="new_chat_btn"):
        if len(st.session_state.messages) > 0:
            chat_title = st.session_state.messages[0]["content"][:40] + "..."
            st.session_state.chat_history.insert(0, {
                "title": chat_title,
                "messages": st.session_state.messages.copy(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        st.session_state.messages = []
        st.session_state.waiting_for_response = False
        st.rerun()
    
    st.divider()
    
    if len(st.session_state.chat_history) > 0:
        st.markdown(f"<p style='color: {subtext_color}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>Chat History</p>", unsafe_allow_html=True)
        for idx, chat in enumerate(st.session_state.chat_history[:10]):
            if st.button(f"💬 {chat['title']}", key=f"history_{idx}"):
                st.session_state.messages = chat["messages"].copy()
                st.session_state.waiting_for_response = False
                st.rerun()
        st.divider()
    
    st.markdown(f"<p style='color: {text_color}; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>⚙️ Configuration</p>", unsafe_allow_html=True)
    
    # Ollama Status Check
    try:
        test_response = requests.get(f"{st.session_state.ollama_url}/api/tags", timeout=2)
        if test_response.status_code == 200:
            st.success("✅ Ollama Connected")
        else:
            st.error("❌ Ollama Not Connected")
    except:
        st.error("❌ Ollama Not Running")
    
    st.markdown(f"<p style='color: {subtext_color}; font-size: 0.8rem; margin-top: 0.5rem;'>Using Llama 3 via Ollama</p>", unsafe_allow_html=True)
    
    st.divider()
    
    if len(st.session_state.messages) > 0:
        if st.button("💾 Export Chat"):
            export_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "messages": st.session_state.messages
            }
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    if st.button(f"{'☀️ Light Mode' if st.session_state.dark_mode else '🌙 Dark Mode'}"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Main Content
if len(st.session_state.messages) == 0:
    # Centered welcome screen
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Icon - left aligned
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; margin-left: 2rem;">
        <svg style="width: 70px; height: 70px; color: {accent_color};" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/>
            <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    
    # Title - left aligned
    st.markdown(f"""
    <h1 style="color: {text_color}; text-align: left; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.8rem; margin-left: 2rem;">
        Welcome to CODE-GenAI
    </h1>
    """, unsafe_allow_html=True)
    
    # Subtitle - left aligned
    st.markdown(f"""
    <p style="color: {subtext_color}; text-align: left; font-size: 1rem; margin-bottom: 2.5rem; line-height: 1.5; margin-left: 2rem;">
        Your AI-powered coding assistant. Ask me to write, debug, explain, or optimize code!
    </p>
    """, unsafe_allow_html=True)
    
    # Action buttons centered
    col1, col2, col3 = st.columns([0.8, 1.5, 0.8])
    
    with col2:
        col_a, col_b = st.columns(2, gap="medium")
        
        with col_a:
            if st.button("🐛 Debug Code", key="btn_debug", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Help me debug this code:"})
                st.session_state.waiting_for_response = True
                st.rerun()
            
            if st.button("⚡ Optimize Code", key="btn_optimize", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Help me optimize this code:"})
                st.session_state.waiting_for_response = True
                st.rerun()
        
        with col_b:
            if st.button("📖 Explain Code", key="btn_explain", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Explain this code to me:"})
                st.session_state.waiting_for_response = True
                st.rerun()
            
            if st.button("✨ Generate Function", key="btn_generate", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Generate a function for:"})
                st.session_state.waiting_for_response = True
                st.rerun()
    
    # Footer
    st.markdown(f"""
    <p style="color: {subtext_color}; text-align: center; font-size: 0.82rem; margin-top: 2.5rem;">
        CODE-GenAI can make mistakes. Always verify important code.
    </p>
    """, unsafe_allow_html=True)

else:
    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Generate response if waiting
    if st.session_state.waiting_for_response:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("🤔 Thinking... (This may take a minute for complex questions)"):
                    ai_response = call_ollama_stream(last_message["content"])
                    message_placeholder.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.waiting_for_response = False
                    st.rerun()

# Chat Input - Always visible with file upload
col_upload, col_input = st.columns([1, 6])

with col_upload:
    uploaded_file = st.file_uploader(
        "📷",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
        key="chat_uploader"
    )
    
    if uploaded_file is not None:
        # Display uploaded image in a popup/expander
        with st.expander("📷 Uploaded Image", expanded=True):
            image = Image.open(uploaded_file)
            st.image(image, caption="Code Image", width=300)
            
            if st.button("🔍 Extract & Send", use_container_width=True):
                with st.spinner("Extracting text..."):
                    extracted = extract_text_from_image(image)
                    if extracted and not extracted.startswith("Error"):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"Here's the code from image:\n\n{extracted}"
                        })
                        st.session_state.waiting_for_response = True
                        st.rerun()
                    else:
                        st.error(extracted)

with col_input:
    user_input = st.chat_input("Ask CODE-GenAI anything about coding...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.waiting_for_response = True
    st.rerun()