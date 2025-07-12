import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pyttsx3
import speech_recognition as sr

# === Voice Input/Output ===
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn’t catch that."

def speak_response(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# === STEP 1: Load and split your text ===
def load_and_split(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    return raw_text, chunks

# === STEP 2: Create FAISS Vector Store ===
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# === STEP 3: Build QA Chain using local Hugging Face model ===
def build_qa_chain(vectorstore):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        device=0 if torch.cuda.is_available() else -1
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever)
    return qa_chain, pipe

# === STEP 4: Streamlit UI ===
def main():
    st.title("🧠 Local LLM FAQ Chatbot with Voice Assistant")

    uploaded_file = st.file_uploader("Upload your policy or FAQ text file", type="txt")

    if uploaded_file is not None:
        # Save uploaded file locally
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.read())

        # Run indexing and summary once and store in session
        if "vectorstore" not in st.session_state:
            with st.spinner("Indexing document..."):
                raw_text, chunks = load_and_split("temp.txt")
                vectorstore = create_vector_store(chunks)
                qa_chain, pipe = build_qa_chain(vectorstore)

                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain
                st.session_state.pipe = pipe
                st.session_state.raw_text = raw_text

                # Generate summary once
                detailed_prompt = (
                    "You are an expert analyst. Read the below business document and describe what this document is about, its key themes, and what insights or conclusions are mentioned, in 4-5 sentence summary "
                    "Keep it factual, clean, and formal.\n\n"
                    f"{raw_text[:2000]}"
                )
                summary = pipe(detailed_prompt)[0]['generated_text']
                st.session_state.summary = summary

            st.success("Document indexed. Ask your questions below.")

        # Sidebar summary (reuses cached summary)
        st.sidebar.markdown("### 📝 File Overview")
        st.sidebar.success("File analyzed.")
        st.sidebar.markdown(f"**Summary:** {st.session_state.summary}")

        # === TEXT input ===
        query = st.text_input("Ask a question about your document:")
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(query)
                st.markdown(f"**Answer:** {response}")
                speak_response(response)

        # === VOICE input ===
        if st.button("🎙️ Ask using Voice"):
            spoken_query = record_audio()
            st.write(f"**You said:** {spoken_query}")
            if spoken_query.strip():
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain.run(spoken_query)
                    st.markdown(f"**Answer:** {response}")
                    speak_response(response)


if __name__ == '__main__':
    main()