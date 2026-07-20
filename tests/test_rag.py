import os
import pytest
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)


@pytest.fixture(scope="session")
def sample_vectorstore(embeddings):
    docs = [
        Document(page_content="Artificial Intelligence is the simulation of human intelligence by machines."),
        Document(page_content="Machine Learning is a subset of AI that learns from data without being explicitly programmed."),
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It enhances LLMs with external knowledge."),
        Document(page_content="FAISS is a library for efficient similarity search developed by Facebook."),
        Document(page_content="Transformers are deep learning models that use self-attention mechanisms."),
    ]
    return FAISS.from_documents(docs, embeddings)


class TestTextSplitter:

    def test_splits_long_text_into_chunks(self, text_splitter):
        long_text = "AI is amazing. " * 50  # ~750 chars
        docs = [Document(page_content=long_text)]
        chunks = text_splitter.split_documents(docs)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_short_text_stays_single_chunk(self, text_splitter):
        short_doc = [Document(page_content="Short text.")]
        chunks = text_splitter.split_documents(short_doc)
        assert len(chunks) == 1, "Short text should produce exactly 1 chunk"

    def test_empty_document_list(self, text_splitter):
        chunks = text_splitter.split_documents([])
        assert chunks == [], "Empty docs should return empty list"

    def test_chunk_size_respected(self, text_splitter):
        long_text = "Word " * 200
        docs = [Document(page_content=long_text)]
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 300, "Chunk size exceeded limit with overlap"



class TestRetriever:

    def test_retriever_returns_results(self, sample_vectorstore):
        retriever = sample_vectorstore.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("What is AI?")
        assert len(results) > 0, "Retriever should return results for a valid query"

    def test_retriever_returns_correct_k(self, sample_vectorstore):
        retriever = sample_vectorstore.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke("machine learning and neural networks")
        assert len(results) == 3, "Retriever should return exactly k=3 documents"

    def test_retriever_returns_relevant_content(self, sample_vectorstore):
        retriever = sample_vectorstore.as_retriever(search_kwargs={"k": 1})
        results = retriever.invoke("What is RAG?")
        assert len(results) == 1
        assert "RAG" in results[0].page_content, "Top result should contain 'RAG'"

    def test_retriever_with_unrelated_query(self, sample_vectorstore):
        retriever = sample_vectorstore.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("cooking pasta recipe")
        assert len(results) == 2, "FAISS always returns k results even for unrelated queries"



class TestInputValidation:

    def validate_query(self, query: str) -> tuple[bool, str]:
        if not query or not query.strip():
            return False, "Query cannot be empty."
        if len(query.strip()) < 3:
            return False, "Query is too short. Please ask a complete question."
        if len(query) > 2000:
            return False, "Query is too long. Please keep it under 2000 characters."
        return True, ""

    def test_empty_string_is_invalid(self):
        valid, msg = self.validate_query("")
        assert not valid
        assert "empty" in msg.lower()

    def test_whitespace_only_is_invalid(self):
        valid, msg = self.validate_query("   ")
        assert not valid
        assert "empty" in msg.lower()

    def test_single_char_is_invalid(self):
        valid, msg = self.validate_query("a")
        assert not valid
        assert "short" in msg.lower()

    def test_normal_query_is_valid(self):
        valid, msg = self.validate_query("What is machine learning?")
        assert valid
        assert msg == ""

    def test_very_long_query_is_invalid(self):
        valid, msg = self.validate_query("a" * 2001)
        assert not valid
        assert "long" in msg.lower()


class TestFileValidation:

    ALLOWED_EXTENSIONS = {".pdf", ".txt"}

    def validate_file(self, filename: str, file_size_bytes: int) -> tuple[bool, str]:

        if not filename:
            return False, "No file provided."

        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type '{ext}'. Only PDF and TXT files are allowed."

        if file_size_bytes == 0:
            return False, "The uploaded file is empty. Please upload a file with content."

        max_size = 10 * 1024 * 1024  # 10 MB
        if file_size_bytes > max_size:
            return False, "File is too large. Maximum allowed size is 10 MB."

        return True, ""

    def test_valid_pdf_passes(self):
        valid, msg = self.validate_file("document.pdf", 1024)
        assert valid

    def test_valid_txt_passes(self):
        valid, msg = self.validate_file("notes.txt", 512)
        assert valid

    def test_unsupported_extension_rejected(self):
        valid, msg = self.validate_file("image.png", 1024)
        assert not valid
        assert "Unsupported" in msg

    def test_docx_is_rejected(self):
        valid, msg = self.validate_file("report.docx", 2048)
        assert not valid

    def test_empty_file_rejected(self):
        valid, msg = self.validate_file("empty.pdf", 0)
        assert not valid
        assert "empty" in msg.lower()

    def test_oversized_file_rejected(self):
        valid, msg = self.validate_file("huge.pdf", 11 * 1024 * 1024)
        assert not valid
        assert "large" in msg.lower()

    def test_no_filename_rejected(self):
        valid, msg = self.validate_file("", 512)
        assert not valid
