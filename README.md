# 📚 VoiceFAQ: A Local LLM-Powered Voice-Enabled FAQ Assistant

This project is a voice-interactive FAQ assistant powered by a local Hugging Face language model. It allows users to upload a `.txt` document (such as policies or FAQs), index its content using semantic embeddings and FAISS, and ask questions using either text or voice. The assistant will respond in text and optionally via text-to-speech.

---

## 🚀 Features

- Upload a `.txt` document (e.g., company policies, FAQ, reports)
- Automatic chunking and semantic indexing using HuggingFace Embeddings + FAISS
- Question-answering over document using a local LLM (`flan-t5-base`)
- Voice input via microphone (SpeechRecognition)
- Voice output using text-to-speech (pyttsx3)
- Document summarization displayed in the sidebar
- GPU support (if available)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) - UI
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - LLM & Tokenizer
- [LangChain](https://www.langchain.com/) - QA chain and retrieval
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) - Voice input
- [pyttsx3](https://pypi.org/project/pyttsx3/) - Voice output (TTS)

---

# Install requirements
pip install -r requirements.txt
