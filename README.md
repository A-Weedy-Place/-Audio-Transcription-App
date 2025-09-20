# 🎙️ Audio Transcription App (IBM Guided Project)

This project is an **IBM Guided Project** that demonstrates how to build an **Audio Transcription App** using state-of-the-art AI models. It combines **speech-to-text** transcription with **LLM-powered summarization** to process audio files and extract key insights.

---

## 🚀 Features
- Upload an audio file and get the **transcribed text**.
- Automatically **summarizes and extracts key points** from the transcription.
- Uses **IBM Watsonx** with **LLaMA 3 Vision Instruct** for summarization.
- Powered by **OpenAI Whisper Tiny** for speech recognition.
- Simple and interactive **Gradio interface**.

---

## 🛠️ Tech Stack & Libraries
- **[PyTorch](https://pytorch.org/):** Deep learning framework.
- **[Transformers](https://huggingface.co/transformers/):** Whisper ASR pipeline.
- **[LangChain](https://www.langchain.com/):** Prompt handling and chaining LLMs.
- **[IBM Watson Machine Learning](https://www.ibm.com/products/watsonx):** LLaMA 3 large language model.
- **[Gradio](https://gradio.app/):** Web-based UI for audio upload and result display.

---

## 📂 Project Workflow
1. **Upload Audio** → via Gradio UI.  
2. **Transcription** → Audio converted to text using *Whisper Tiny (English)*.  
3. **Summarization** → The transcription is passed into *LLaMA 3 (Watsonx)* with a custom prompt to extract key points.  
4. **Results** → Displayed in the UI as clean text.

---

## ▶️ How to Run
1. Install required libraries:
   ```bash
   pip install torch gradio langchain transformers ibm-watson-machine-learning
