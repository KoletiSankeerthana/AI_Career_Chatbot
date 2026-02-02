import os
import json
import uuid
import datetime
import streamlit as st
import warnings
from dotenv import load_dotenv
from utils import load_json, save_json, save_api_key
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# --- Configuration & Setup ---
load_dotenv()
warnings.filterwarnings('ignore')

# Constants & Paths
HISTORY_PATH = "chat_store.json"
PROFILE_PATH = ".profile_store.json"
ENV_PATH = ".env"

# --- Core RAG Logic ---
class CareerChatbotRAG:
    def __init__(self, persist_directory="./vector_store", knowledge_base_path="./knowledge_base"):
        self.persist_directory = persist_directory
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.llm = None
        
    def load_engine(self):
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except:
                self.vector_store = self._create_new_db()
        else:
            self.vector_store = self._create_new_db()
            
    def _create_new_db(self):
        documents = []
        if os.path.exists(self.knowledge_base_path):
            for file in os.listdir(self.knowledge_base_path):
                file_path = os.path.join(self.knowledge_base_path, file)
                try:
                    if file.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                    elif file.endswith(".txt"):
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                except:
                    continue
        
        if not documents:
            documents = [
                Document(page_content="Data Science includes roles like Data Analyst, ML Engineer, and Research Scientist.", metadata={"source": "Internal"}),
                Document(page_content="Software Engineering requires mastery of algorithms, systems design, and version control.", metadata={"source": "Internal"})
            ]
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        try:
            db.persist()
        except:
            pass
        return db

    def chat(self, user_query, history, api_key, user_profile):
        try:
            if not self.llm:
                self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.5)
            
            docs = self.vector_store.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            
            profile_context = f"User Profile: Skills({user_profile.get('skills', 'Unknown')}), Education({user_profile.get('education', 'Unknown')}), Interest({user_profile.get('interest', 'General')})."
            
            messages = [
                SystemMessage(content=f"""You are a Technical AI Career Guidance Assistant. 
                {profile_context}
                Use the following context to provide professional, mentor-grade advice: {context}
                
                Rules:
                1. Provide structured, clear, and encouraging guidance.
                2. Use professional typography: bullet points for steps and skills.
                3. Tailor recommendations specifically to the user's provided profile.
                4. Maintain a calm, neutral, and helpful tone.""")
            ]
            
            for msg in history:
                role = msg.get("role")
                if role == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif role == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=user_query))
            
            response = self.llm.invoke(messages)
            return response.content, docs
            
        except Exception as e:
            return f"❌ System Error: {str(e)}", []

# --- UI Setup ---
st.set_page_config(page_title="AI Career Guidance Assistant", page_icon="🎓", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-color: #ffffff;
        --sidebar-bg: #f8fafc;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --accent-blue: #2563eb;
        --border-color: #e2e8f0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-color);
    }

    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Chat Styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #f1f5f9;
        border: none;
    }

    .timestamp {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        display: block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: var(--text-primary);
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid var(--border-color);
    }
    
    .stButton>button:hover {
        border-color: var(--accent-blue);
        color: var(--accent-blue);
    }

</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'history' not in st.session_state:
    st.session_state.history = load_json(HISTORY_PATH, {"sessions": {}, "active_id": None})

if 'profile' not in st.session_state:
    st.session_state.profile = load_json(PROFILE_PATH, {"skills": "", "education": "Student", "interest": ""})

if 'rag' not in st.session_state:
    with st.spinner("Initializing Intelligence Engine..."):
        rag = CareerChatbotRAG()
        rag.load_engine()
        st.session_state.rag = rag

# --- API Key Handling ---
# Load from environment or state
api_key = os.getenv("GROQ_API_KEY")

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">🎓 Career Assistant</p>', unsafe_allow_html=True)
    
    # 1. New Chat Button (Refined)
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        active_id = st.session_state.history.get("active_id")
        # If the active chat is already empty, just stay there
        if active_id and active_id in st.session_state.history["sessions"] and \
           not st.session_state.history["sessions"][active_id]["messages"]:
            pass 
        else:
            new_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history["sessions"][new_id] = {
                "name": "Career Planning", 
                "messages": [], 
                "time": timestamp
            }
            st.session_state.history["active_id"] = new_id
            save_json(HISTORY_PATH, st.session_state.history)
        st.rerun()

    # 2. Chat History Reveal (Expandable)
    st.markdown("### 💬 Chat History")
    all_sessions = st.session_state.history.get("sessions", {})
    history_sessions = {sid: sdata for sid, sdata in all_sessions.items() if sdata["messages"]}
    
    with st.expander("Explore Past Consultations", expanded=False):
        if not history_sessions:
            st.markdown('<p style="color: #94a3b8; font-size: 0.9rem;">No past conversations</p>', unsafe_allow_html=True)
        else:
            sorted_sessions = sorted(history_sessions.items(), key=lambda x: x[1].get('time', ''), reverse=True)
            for sid, sdata in sorted_sessions:
                is_active = (sid == st.session_state.history.get("active_id"))
                btn_style = "primary" if is_active else "secondary"
                display_name = sdata['name'] if len(sdata['name']) < 25 else sdata['name'][:22] + "..."
                
                if st.button(f"📄 {display_name}", key=sid, use_container_width=True, type=btn_style):
                    st.session_state.history["active_id"] = sid
                    save_json(HISTORY_PATH, st.session_state.history)
                    st.rerun()

    st.markdown("---")

    # 3. Career Profile
    st.markdown("### 🧑‍💼 Your Profile")
    with st.expander("Adjust Career Context", expanded=not any(st.session_state.profile.values())):
        skills = st.text_area("Mastered Skills", value=st.session_state.profile["skills"], placeholder="e.g. Python, Machine Learning")
        edu = st.selectbox("Education Level", ["Student", "Undergraduate", "Graduate", "PhD/Professional"], 
                           index=["Student", "Undergraduate", "Graduate", "PhD/Professional"].index(st.session_state.profile["education"]))
        interest = st.text_input("Aspirational Roles", value=st.session_state.profile["interest"], placeholder="e.g. Data Scientist")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Profile", use_container_width=True):
                st.session_state.profile = {"skills": skills, "education": edu, "interest": interest}
                save_json(PROFILE_PATH, st.session_state.profile)
                st.success("Saved!")
        with col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.profile = {"skills": "", "education": "Student", "interest": ""}
                save_json(PROFILE_PATH, st.session_state.profile)
                st.rerun()

    # 4. API Setup (Only if missing)
    if not api_key:
        st.markdown("---")
        st.markdown("### ⚙️ Engine Setup")
        with st.expander("Connect Groq API", expanded=True):
            input_key = st.text_input("Groq API Key", type="password", help="Enter your key to enable the AI engine.")
            if st.button("Connect Engine", use_container_width=True):
                if input_key.startswith("gsk_"):
                    save_api_key(input_key)
                    st.success("Successfully Connected!")
                    st.rerun()
                else:
                    st.error("Invalid key format. Should start with 'gsk_'.")

# --- Main Interface ---
st.markdown('<p class="main-header">AI Career Guidance Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional mentoring and strategic career mapping powered by Llama 3.</p>', unsafe_allow_html=True)

# Ensure an active session exists
if not st.session_state.history.get("active_id") or st.session_state.history["active_id"] not in st.session_state.history["sessions"]:
    new_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history["sessions"][new_id] = {
        "name": "Career Planning", 
        "messages": [], 
        "time": timestamp
    }
    st.session_state.history["active_id"] = new_id

active_id = st.session_state.history["active_id"]
session_data = st.session_state.history["sessions"][active_id]

# Show conversation history
if not session_data["messages"]:
    st.container().info("💡 **Getting Started:** Share your career goals or ask about specific job paths. Use the sidebar to provide your background for personalized guidance.")
else:
    for m in session_data["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "time" in m:
                st.markdown(f'<span class="timestamp">{m["time"]}</span>', unsafe_allow_html=True)

# Chat Input Area
if prompt := st.chat_input("Message your career mentor..."):
    if not api_key:
        st.info("👋 **Setup Required:** Please enter your Groq API Key in the sidebar 'Engine Setup' section to begin.")
    else:
        # Auto-rename session
        if not session_data["messages"]:
            session_data["name"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
        
        now_time = datetime.datetime.now().strftime("%H:%M")
        
        # User Message
        session_data["messages"].append({"role": "user", "content": prompt, "time": now_time})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(f'<span class="timestamp">{now_time}</span>', unsafe_allow_html=True)

        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing career paths..."):
                response_content, sources = st.session_state.rag.chat(
                    prompt, 
                    session_data["messages"][:-1], 
                    api_key, 
                    st.session_state.profile
                )
                st.markdown(response_content)
                st.markdown(f'<span class="timestamp">{now_time}</span>', unsafe_allow_html=True)
                
                if sources:
                    with st.expander("🔍 Strategic Context"):
                        for s in sources:
                            st.write(f"• {s.page_content[:200]}...")
        
        # Save persistence
        session_data["messages"].append({"role": "assistant", "content": response_content, "time": now_time})
        session_data["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_json(HISTORY_PATH, st.session_state.history)
