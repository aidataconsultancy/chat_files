from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, VectorStore
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any
from io import BytesIO
from pypdf import PdfReader
import re
from config import STUFF_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from io import StringIO
import docx2txt
from langchain import OpenAI
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
# Define answer generation function

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

def answer(_docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(llm=OpenAI(temperature=0, openai_api_key=st.session_state.get("OPENAI_API_KEY")), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore

    # Cohere doesn't work very well as of now.
    # chain = load_qa_with_sources_chain(Cohere(temperature=0), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore
    answer = chain(
        {"input_documents": _docs, "question": query}, return_only_outputs=True
    )
    return answer

def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def embedding(_docs: List[Document]) -> VectorStore:
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"),model='text-embedding-ada-002')
        #embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        index = Chroma.from_documents(_docs, embeddings)
        return index


def search_docs(_index: VectorStore, query: str) -> List[Document]:
    # Search for similar chunks
    docs = _index.similarity_search(query, k=config.k)
    return docs


def ingest_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


def ingest_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def ingest_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def ingest_csv(uploaded_file):
    # To read file as bytes:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    return string_data

def ingest_any(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    return string_data


