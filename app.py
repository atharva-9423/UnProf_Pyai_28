import os
import tempfile
import warnings

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

st.set_page_config(
    page_title="Document AI — Day 28",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #1e1e1e; }
    [data-testid="stSidebar"] * { color: #e5e5e5 !important; }
    .sidebar-title { font-size: 1.1rem; font-weight: 700; color: #ffffff !important; letter-spacing: -0.5px; margin-bottom: 0.25rem; }
    .sidebar-subtitle { font-size: 0.75rem; color: #888 !important; margin-bottom: 1.5rem; }
    .file-badge { display: inline-flex; align-items: center; gap: 6px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; color: #ccc !important; margin-bottom: 6px; width: 100%; }
    .file-badge-dot { width: 7px; height: 7px; background: #22c55e; border-radius: 50%; flex-shrink: 0; }
    .main-title { font-size: 2rem; font-weight: 800; letter-spacing: -1px; color: #111827; margin-bottom: 0; }
    .main-subtitle { color: #9ca3af; font-size: 0.9rem; margin-bottom: 2rem; }
    .stChatMessage { background: transparent !important; border-bottom: 1px solid #f3f4f6; padding-bottom: 1rem; }
    .stChatInputContainer { border: 1px solid #e5e7eb !important; border-radius: 12px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important; }
    .status-ready { display: inline-block; background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; border-radius: 99px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; margin-bottom: 1rem; }
    .status-empty { display: inline-block; background: #fefce8; color: #ca8a04; border: 1px solid #fde68a; border-radius: 99px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 

def validate_query(query: str) -> tuple:
    if not query or not query.strip():
        return False, "⚠️ Please type a question before submitting."
    if len(query.strip()) < 3:
        return False, "⚠️ Your question is too short. Please ask something more specific."
    if len(query) > 2000:
        return False, "⚠️ Your question is too long (max 2000 characters). Please shorten it."
    return True, ""

def validate_uploaded_file(filename: str, file_size: int) -> tuple:
    if not filename:
        return False, "No file was provided."
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"❌ Unsupported file type **'{ext}'**. Only **.pdf** and **.txt** files are allowed."
    if file_size == 0:
        return False, "❌ The uploaded file is **empty**. Please upload a file with actual content."
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"❌ File is **too large** ({file_size // 1024 // 1024} MB). Maximum allowed size is 10 MB."
    return True, ""


@st.cache_resource(show_spinner="⚙️ Booting AI engine...")
def load_base_components():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        st.error(
            "🔑 **Missing API Key!**\n\n"
            "Please add your Gemini API key to the `.env` file:\n"
            "```\nGEMINI_API_KEY=AIza...\n```\n"
            "Then restart the app."
        )
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs_folder = "documents"
    all_splits = []
    if os.path.exists(docs_folder):
        try:
            loader = DirectoryLoader(docs_folder, glob="**/*.txt", loader_cls=TextLoader)
            docs = loader.load()
            if docs:
                all_splits = text_splitter.split_documents(docs)
        except Exception:
            pass 

    if not all_splits:
        all_splits = [Document(page_content="Welcome! Upload a PDF or TXT document to get started.")]

    vectorstore = FAISS.from_documents(all_splits, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    return embeddings, text_splitter, vectorstore, llm


def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def get_context_question(inputs):
        history = inputs.get("chat_history", [])
        if not history:
            return inputs["input"]
        last_human = next((t for r, t in reversed(history) if r == "human"), "")
        return f"{last_human} {inputs['input']}"

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for question-answering tasks. "
         "Use the retrieved context below to answer the question. "
         "If the answer is not in the context, honestly say you don't know. "
         "Keep answers concise — 3 sentences max.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    return (
        RunnablePassthrough.assign(standalone_question=get_context_question)
        | RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(
                doc.page_content for doc in retriever.invoke(x["standalone_question"])
            )
        )
        | qa_prompt | llm | StrOutputParser()
    )


embeddings, text_splitter, vectorstore, llm = load_base_components()

for key, default in [
    ("messages", []),
    ("langchain_history", []),
    ("uploaded_files", []),
    ("vectorstore", vectorstore),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown('<p class="sidebar-title">🧪 Document AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">Day 28 — Tested & Error-Hardened</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_size = len(file_bytes)

        is_valid, err_msg = validate_uploaded_file(uploaded_file.name, file_size)

        if not is_valid:
            st.error(err_msg)
        elif uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"📥 Indexing **{uploaded_file.name}**..."):
                try:
                    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                    docs = loader.load()

                    if not docs or all(not d.page_content.strip() for d in docs):
                        st.error("❌ Could not extract any text from this file. It may be scanned/image-based.")
                    else:
                        splits = text_splitter.split_documents(docs)
                        st.session_state.vectorstore.add_documents(splits)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"✅ Indexed {len(splits)} chunks from **{uploaded_file.name}**!")

                    os.remove(tmp_path)

                except Exception as e:
                    st.error(f"❌ Failed to process file: {str(e)}")
        else:
            st.info(f"ℹ️ **{uploaded_file.name}** is already indexed.")

    if st.session_state.uploaded_files:
        st.markdown("**Indexed Documents**")
        for fname in st.session_state.uploaded_files:
            st.markdown(
                f'<div class="file-badge"><span class="file-badge-dot"></span>{fname}</div>',
                unsafe_allow_html=True
            )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.rerun()


st.markdown('<p class="main-title">Chat with your Documents</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Tested, secure, and production-ready.</p>', unsafe_allow_html=True)

if st.session_state.uploaded_files:
    n = len(st.session_state.uploaded_files)
    st.markdown(f'<span class="status-ready">● {n} file(s) indexed — Ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-empty">● No files uploaded — Using default knowledge base</span>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your documents..."):
    is_valid, err_msg = validate_query(query)
    if not is_valid:
        st.warning(err_msg)
    else:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        chain = build_rag_chain(st.session_state.vectorstore, llm)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in chain.stream({
                    "input": query,
                    "chat_history": st.session_state.langchain_history
                }):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.langchain_history.append(("human", query))
                st.session_state.langchain_history.append(("ai", full_response))

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    placeholder.error(
                        "⏳ **Rate limit hit.** You've sent too many requests. "
                        "Please wait ~60 seconds and try again."
                    )
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    placeholder.error(
                        "🔧 **Google's servers are busy.** This is temporary. "
                        "Please try again in a moment."
                    )
                elif "401" in error_str or "UNAUTHENTICATED" in error_str:
                    placeholder.error(
                        "🔑 **Invalid API Key.** Please check your `.env` file "
                        "and make sure `GEMINI_API_KEY` starts with `AIza`."
                    )
                else:
                    placeholder.error(f"❌ **Unexpected error:** {error_str}")
