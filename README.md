<div align="center">

# 🧪 Day 28 — Testing & Error Handling
### Reliable, Crash-Proof, Production-Ready

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Unit%20Tests-orange?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Coverage](https://img.shields.io/badge/Tests-15%20Unit%20Tests-brightgreen?style=for-the-badge)]()

*Building confidence through automated testing and bulletproof error handling.*

</div>

---

## 📖 Overview

Day 28 is the final quality gate before deployment. We added **15 unit tests** using `pytest` to verify our RAG pipeline logic, and we hardened the Streamlit UI with comprehensive, user-friendly error handling for every possible failure scenario.

## ✨ Key Features

### 🧪 Unit Tests (`tests/test_rag.py`)
| Test Group | Tests | What is Covered |
|---|---|---|
| `TestTextSplitter` | 4 tests | Chunking logic, empty docs, size limits |
| `TestRetriever` | 4 tests | FAISS search, K results, relevance, edge queries |
| `TestInputValidation` | 5 tests | Empty, whitespace, too short, too long queries |
| `TestFileValidation` | 7 tests | Invalid extensions, empty files, oversized files |

### ⚠️ Graceful Error Handling (`app.py`)
- **Invalid Query:** Warning shown before the LLM is ever called (saves API quota!)
- **Invalid File Upload:** Descriptive red error message for wrong type, empty, or oversized files
- **Rate Limit (429):** Clear message telling user to wait 60 seconds
- **Server Busy (503):** Friendly "Google's servers are busy" message
- **Invalid API Key (401):** Direct instruction to check the `.env` file
- **Missing API Key:** Startup error with exact instructions to fix it

## 📁 Project Structure

```text
day28/
├── app.py               # Streamlit app with full error handling
├── requirements.txt     # Python dependencies (+ pytest)
├── README.md            # This file
├── .env                 # ⛔ YOUR SECRETS (never commit!)
├── .env.example         # ✅ Template (safe to commit)
├── .gitignore           # Ignores .env
├── documents/           # Default knowledge base
│   └── sample_notes.txt
└── tests/               # All unit tests live here
    ├── __init__.py
    └── test_rag.py      # 15 unit tests for RAG components
```

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
cd day28
pip install -U -r requirements.txt
```

### 2. Configure `.env`
```env
GEMINI_API_KEY=AIza...your_key_here
```

### 3. Run Unit Tests ✅
```bash
pytest tests/ -v
```
You should see all 15 tests **PASSED** (no API key needed — tests run offline!).

### 4. Run the App
```bash
streamlit run app.py
```

---
<div align="center">
<i>Built for the 100 Days of Code challenge. Phase 4 Final Day.</i>
</div>
