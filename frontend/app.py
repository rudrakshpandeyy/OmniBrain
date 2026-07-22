"""
OmniBrain - AI Research Assistant
----------------------------------
Ek Streamlit-based frontend jisme:
 - Left sidebar: file upload + uploaded files list + settings
 - Main area: welcome screen -> chat interface
 - Quick action buttons (Summarize, Translate, Notes, Key Points)
 - Right panel: document info
 - Footer

NOTE: Ye abhi sirf FRONTEND + DUMMY AI hai.
Real AI jawaab ke liye "get_ai_response()" function ko
apne LangChain / OpenAI / Anthropic API call se replace karo.
"""

import os
import requests
import streamlit as st
from datetime import datetime

# Configurable backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Optional libs for reading real file info (agar installed nahi hai to app phir bhi chalega)
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ----------------------------------------------------------------------
# 1. PAGE CONFIG  --  ye sabse pehle call hona chahiye
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="OmniBrain",
    page_icon="OB",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# 2. CUSTOM CSS (dark navy + blue theme jaisa tumne bataya)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall background */
        .stApp {
            background-color: #0F172A;
            color: #FFFFFF;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #1E293B;
        }

        /* Card-like containers */
        .ob-card {
            background-color: #1E293B;
            padding: 1rem 1.2rem;
            border-radius:12px;
            margin-bottom: 0.8rem;
            border: 1px solid #334155;
        }

        /* Top navbar */
        .ob-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 1.2rem;
            background-color: #1E293B;
            border-radius:12px;
            margin-bottom: 1.2rem;
        }
        .ob-navbar h2 {
            margin: 0;
            color: #FFFFFF;
        }

        /* Buttons -> blue accent */
        div.stButton > button {
            background-color: #3B82F6;
            color: white;
            border: none;
            border-radius:10px;;
            padding: 0.4rem 1rem;
        }
        div.stButton > button:hover {
            background-color: #2563EB;
            color: white;
        }

        /* Footer */
        .ob-footer {
            text-align: center;
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 3. SESSION STATE  --  Streamlit har baar rerun hota hai, isliye
#    "memory" (files, chat history) ko session_state me store karte hain.
# ----------------------------------------------------------------------
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []   # list of dicts: name, pages, chunks

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of dicts: role, content

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None


# ----------------------------------------------------------------------
# 4. BACKEND CHAT RESPONSE FUNCTION
# ----------------------------------------------------------------------
def get_ai_response(user_query: str) -> str:
    """Calls the backend chat endpoint with actual request/response schema."""
    if not st.session_state.uploaded_files_info:
        return "Pehle koi document upload karo, fir main uske base par jawaab dunga."
    
    # Use the document_id of the latest uploaded file
    latest = st.session_state.uploaded_files_info[-1]
    document_id = latest.get("document_id") or st.session_state.get("document_id")
    
    if not document_id:
        return "Pehle koi document upload karo, fir main uske base par jawaab dunga."
    
    payload = {
        "document_id": document_id,
        "question": user_query
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            res_data = response.json()
            answer = res_data.get("answer", "")
            sources = res_data.get("sources", [])
            
            # Format display with sources if present
            if sources:
                sources_str = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                return f"{answer}{sources_str}"
            return answer
        else:
            try:
                err_msg = response.json().get("detail", response.text)
            except Exception:
                err_msg = response.text
            st.error(f"API Error: {err_msg}")
            return f"Error: {err_msg}"
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return f"Connection Error: {str(e)}"


def process_uploaded_file(file):
    """File ko 'index' karta hai — abhi dummy hai, sirf page count nikalta hai agar PDF ho."""
    pages = None
    if file.name.lower().endswith(".pdf") and PdfReader is not None:
        try:
            reader = PdfReader(file)
            pages = len(reader.pages)
        except Exception:
            pages = None
    if pages is None:
        pages = 1  # fallback dummy value

    chunks = pages * 4  # dummy formula, just for UI display

    return {
        "name": file.name,
        "pages": pages,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%H:%M:%S"),
    }


# ----------------------------------------------------------------------
# 5. TOP NAVBAR
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="ob-navbar">
        <h2> OmniBrain</h2>
        <div>Workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 6. SIDEBAR  --  Upload + File list + Settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("###  Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        for file in uploaded:
            already = any(f["name"] == file.name for f in st.session_state.uploaded_files_info)
            if not already:
                # Call backend upload endpoint using actual request format
                files = {"file": (file.name, file.getvalue(), file.type or "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 201:
                        res_data = response.json()
                        document_id = res_data.get("document_id")
                        st.session_state.document_id = document_id
                        
                        info = process_uploaded_file(file)
                        info["document_id"] = document_id
                        st.session_state.uploaded_files_info.append(info)
                        st.success(f"{file.name} uploaded and indexed. ID: {document_id}")
                    else:
                        try:
                            err_msg = response.json().get("detail", response.text)
                        except Exception:
                            err_msg = response.text
                        st.error(f"Failed to upload {file.name}: {err_msg}")
                except Exception as e:
                    st.error(f"Connection error to backend for {file.name}: {str(e)}")

    st.markdown("---")
    st.markdown("### Uploaded Files")
    if st.session_state.uploaded_files_info:
        for f in st.session_state.uploaded_files_info:
            st.markdown(f"• {f['name']}")
    else:
        st.caption("Koi file upload nahi hui abhi.")

    st.markdown("---")
    st.markdown("### Settings")
    language = st.selectbox("Language", ["English", "Hindi", "Hinglish"])
    model = st.selectbox("Model", ["GPT-4o", "Claude", "Gemini"])
    theme = st.selectbox("Theme", ["Dark", "Light"])


# ----------------------------------------------------------------------
# 7. MAIN AREA  -- Welcome screen OR Chat + Right panel
# ----------------------------------------------------------------------
main_col, right_col = st.columns([3, 1])

with main_col:
    if not st.session_state.uploaded_files_info:
        # ---- Welcome screen ----
        st.markdown(
            """
            <div class="ob-card" style="text-align:center; padding:3rem;">
                <h1> OmniBrain</h1>
                <p style="color:#94A3B8;">AI-powered Research Workspace</p>
                <p>Upload documents and ask questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ---- Indexed file summary card (latest file) ----
        latest = st.session_state.uploaded_files_info[-1]
        st.markdown(
            f"""
            <div class="ob-card">
                 <b>{latest['name']}</b><br>
                 Status: Indexed Successfully<br>
                Pages: {latest['pages']} &nbsp;|&nbsp; Chunks: {latest['chunks']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Quick action buttons ----
        st.markdown("#### Quick Actions")
        qa1, qa2, qa3, qa4 = st.columns(4)
        if qa1.button(" Summarize"):
            st.session_state.pending_prompt = "Summarize this document."
        if qa2.button(" Translate"):
            st.session_state.pending_prompt = "Translate this document."
        if qa3.button(" Notes"):
            st.session_state.pending_prompt = "Make notes from this document."
        if qa4.button(" Key Points"):
            st.session_state.pending_prompt = "Give key points of this document."

        st.markdown("---")

        # ---- Chat history ----
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # ---- Handle quick-action pending prompt ----
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            st.session_state.messages.append({"role": "user", "content": prompt})
            reply = get_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # ---- Chat input box ----
        user_input = st.chat_input("Ask anything...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            reply = get_ai_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

with right_col:
    st.markdown("#### Document Info")
    if st.session_state.uploaded_files_info:
        latest = st.session_state.uploaded_files_info[-1]
        st.markdown(
            f"""
            <div class="ob-card">
                <b>File Name:</b> {latest['name']}<br>
                <b>Pages:</b> {latest['pages']}<br>
                <b>Language:</b> {language}<br>
                <b>Keywords:</b> Available after indexing<br>
                <b>Last Updated:</b> {latest['uploaded_at']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Document upload karne ke baad yahan info dikhegi.")


# ----------------------------------------------------------------------
# 8. FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="ob-footer">
         Built with Streamlit, FastAPI, LangChain & LLM
    </div>
    """,
    unsafe_allow_html=True,
)
