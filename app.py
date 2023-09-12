import funciones as functions
import streamlit as st
from streamlit_chat import message
import os
import tempfile
from openai.error import OpenAIError

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

def clear_submit():
    st.session_state["submit"] = False

# Creating the chatbot interface
st.title("ChatBot")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'submit' not in st.session_state:
    st.session_state['submit'] = False

# Define a function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask me a question", key="input", on_change=clear_input_text)
    if st.button("Submit"):
        st.session_state['submit'] = True
    return input_text

def main():
    # Sidebar
    index = None
    doc = None
    with st.sidebar:
        user_secret = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )
        if user_secret:
            set_openai_api_key(user_secret)

    user_input = get_text()

    uploaded_file = st.file_uploader('Select your PDF file', 
                                     type=["pdf", "docx", "txt", "csv", "js", "py", "json", "html", "css", "md"],
                                     help="This document type is not available",
                                     on_change=clear_submit)

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

                # Execute the ingestion process of the PDF file in the background
            with st.spinner('Loading a moment ...'):
                doc = functions.ingest_pdf(uploaded_file)
                text = functions.text_to_docs(doc)
                index = functions.embedding(text)

                # Delete the temporary file
            os.remove(temp_file.name)

        elif uploaded_file.name.endswith(".csv"):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            with st.spinner('Loading a moment ...'):
                doc = functions.ingest_csv(uploaded_file)
                text = functions.text_to_docs(doc)
                index = functions.embedding(text)

        elif uploaded_file.name.endswith(".txt"):
            with st.spinner('Loading a moment ...'):
                doc = functions.ingest_txt(uploaded_file)
                text = functions.text_to_docs(doc)
                index = functions.embedding(text)

        elif uploaded_file.name.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            with st.spinner('Loading a moment ...'):
                doc = functions.ingest_docx(uploaded_file)
                text = functions.text_to_docs(doc)
                index = functions.embedding(text)

        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            with st.spinner('Loading a moment ...'):
                doc = functions.ingest_any(uploaded_file)
                text = functions.text_to_docs(doc)
                index = functions.embedding(text)
            # Set the session state variable to indicate that the file has been uploaded and ingested
            st.session_state['uploaded_file'] = True

    else:
        st.warning('Please select a file to continue')
        return

    if user_input and st.session_state['submit']:    
        sources = functions.search_docs(index, user_input)
        output = functions.answer(sources, user_input)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["output_text"].split("SOURCES: ")[0])
        st.session_state['submit'] = False

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="Botts")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Run the app
if __name__ == "__main__":
    main()