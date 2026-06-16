# 🤖 NeuraFAQ — AI-Powered FAQ Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-154360?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**A production-quality, AI-powered FAQ chatbot with semantic search, PDF Q&A, voice interaction, and real-time analytics.**

[✨ Features](#-features) · [🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📁 Project Structure](#-project-structure) · [📈 Portfolio](#-portfolio-value)

</div>

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🧠 **Semantic FAQ Matching** | TF-IDF + Cosine Similarity (no exact keyword matching needed) |
| 📝 **NLP Pipeline** | Lowercasing, tokenization, stopword removal, lemmatization via NLTK |
| 🎯 **Confidence Scoring** | Every response displays a confidence percentage |
| 📄 **PDF Q&A** | Upload any PDF and query its content intelligently |
| 🎙️ **Voice Input** | Ask questions via microphone (SpeechRecognition) |
| 🔊 **Text-to-Speech** | Chatbot reads answers aloud (pyttsx3) |
| 📊 **Analytics Dashboard** | 6 interactive Plotly charts with real-time metrics |
| ⚙️ **FAQ Admin Panel** | Add, edit, delete, import/export FAQs without touching code |
| 🌙 **Premium Dark UI** | Glassmorphism design with smooth animations |
| 💾 **Persistent History** | All queries logged to CSV for audit and analytics |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Windows / macOS / Linux
- Microphone (optional, for voice input)

### 1. Clone / Download the Project

```bash
git clone https://github.com/yourusername/neurafaq.git
cd neurafaq
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

> ⚠️ **PyAudio Note (Windows):** If `pip install pyaudio` fails, install it via pre-built wheel:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
> If you don't need voice input, you can skip PyAudio entirely and delete that line from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🏗️ Architecture

```
User Input (Text / Voice)
        │
        ▼
┌─────────────────────┐
│   NLP Preprocessor  │  ← Lowercase, Tokenize, Remove Stopwords, Lemmatize
│      (utils.py)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   TF-IDF Index      │     │   PDF TF-IDF Index  │
│   (FAQ Questions)   │     │   (Uploaded Doc)    │
└─────────┬───────────┘     └──────────┬──────────┘
          │                             │
          └────────────┬────────────────┘
                       ▼
            ┌──────────────────┐
            │  Cosine Similarity│ ← Find best matching chunk
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │ Confidence Check │ ← Above threshold? Return answer. Else: fallback.
            └────────┬─────────┘
                     │
          ┌──────────▼──────────┐
          │    ChatResponse     │ ← answer, confidence, source, category, timing
          └──────────┬──────────┘
                     │
        ┌────────────┴───────────┐
        ▼                        ▼
   Streamlit UI            Analytics Logger
   (Chat Bubbles)          (logs/query_log.csv)
```

### How Semantic Matching Works

Unlike traditional keyword matching, NeuraFAQ uses **TF-IDF (Term Frequency-Inverse Document Frequency)** combined with **Cosine Similarity**:

1. **Preprocessing**: Both user query and FAQ questions go through the NLP pipeline.
2. **Vectorization**: Each question is converted to a high-dimensional vector using TF-IDF (with unigrams + bigrams).
3. **Similarity Computation**: The query vector is compared against all FAQ vectors using Cosine Similarity.
4. **Threshold Check**: If the best score ≥ threshold (default 25%), the matched answer is returned with a confidence score.

**Example:**
- User asks: *"How much is the college fee?"*
- FAQ contains: *"What is the admission fee?"*
- Common NLP tokens: `college`, `fee`, `admission` → High similarity → Correct answer returned ✅

---

## 📁 Project Structure

```
faq chatbot/
│
├── app.py               ← Main Streamlit application (UI + routing)
├── chatbot.py           ← TF-IDF engine, semantic search, conversation history
├── faq_manager.py       ← CRUD operations on FAQ knowledge base
├── pdf_processor.py     ← PDF text extraction and chunking
├── voice_handler.py     ← Speech-to-Text and Text-to-Speech
├── analytics.py         ← Query logging and Plotly dashboard charts
├── utils.py             ← NLP preprocessing pipeline
│
├── data/
│   └── faq_data.json    ← FAQ knowledge base (editable without code changes)
│
├── uploads/             ← Temporary storage for uploaded files
├── logs/
│   ├── app.log          ← Application logs
│   └── query_log.csv    ← All query records for analytics
│
├── nltk_data/           ← Auto-downloaded NLTK packages (local, no internet needed after first run)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All settings are adjustable from the **sidebar** in the running app:

| Setting | Default | Description |
|---------|---------|-------------|
| Confidence Threshold | 25% | Minimum similarity score to return an answer |
| Text-to-Speech | ON | Speak answers aloud using pyttsx3 |
| Voice Input | ON | Capture questions via microphone |

---

## 📊 FAQ Knowledge Base

The FAQ dataset is stored in `data/faq_data.json`. Each entry has this structure:

```json
{
  "id": "faq_001",
  "category": "Admissions",
  "question": "What is the admission fee?",
  "answer": "The admission fee is ₹50,000.",
  "tags": ["fee", "admission", "cost"],
  "created_at": "2026-01-01T00:00:00",
  "updated_at": "2026-01-01T00:00:00"
}
```

**To add FAQs without coding:**
1. Open the app → **FAQ Manager** tab → **Add FAQ**.
2. Or edit `data/faq_data.json` directly.
3. Or import from a CSV/JSON file via the **Import** section.

---

## 🎙️ Voice Interaction

- **Voice Input**: Click the 🎤 button in the chat interface. The app listens for up to 5 seconds and transcribes your speech using Google Web Speech API (requires internet).
- **Text-to-Speech**: When enabled, the chatbot will read answers aloud after each response.
- Both features can be toggled off in the sidebar if not needed.

---

## 📄 PDF Question Answering

1. Go to the **Document Indexer** tab.
2. Upload any PDF file (college brochure, manual, policy document, etc.).
3. Click **"⚡ Index Document"**.
4. Return to the **Chat Portal** and ask questions about the PDF content.
5. Responses from the PDF will be labeled with a **PDF** badge.

---

## 🧪 Technical Details

### NLP Preprocessing Pipeline (`utils.py`)

```python
Input: "What is the Admission Fee for 1st year?"
  ↓ Unicode normalization
  ↓ Lowercase: "what is the admission fee for 1st year"
  ↓ Remove noise: "what is the admission fee for st year"
  ↓ Tokenize: ["what", "is", "the", "admission", "fee", "for", "st", "year"]
  ↓ Remove stopwords: ["admission", "fee", "st", "year"]
  ↓ Lemmatize: ["admission", "fee", "st", "year"]
Output: "admission fee st year"
```

### TF-IDF + Cosine Similarity (`chatbot.py`)

- **Vectorizer settings**: `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=1`
- **Similarity**: `sklearn.metrics.pairwise.cosine_similarity`
- **Default threshold**: `0.25` (adjustable from 0.10 to 0.80 via sidebar)

---

## 📈 Portfolio Value

**This project demonstrates:**

- ✅ Natural Language Processing (NLP) with NLTK
- ✅ Machine Learning techniques (TF-IDF, Cosine Similarity)
- ✅ Information Retrieval and Semantic Search
- ✅ Object-Oriented Programming (OOP) in Python
- ✅ Full-Stack Web Development with Streamlit
- ✅ PDF Document Processing and Chunking
- ✅ Voice AI Integration (STT + TTS)
- ✅ Data Analytics and Visualization with Plotly
- ✅ RESTful Data Management (JSON-based CRUD)
- ✅ Production-ready code architecture
- ✅ Logging, error handling, and thread safety

---

## 📝 Resume Description

```
NeuraFAQ — AI-Powered FAQ Chatbot  |  Python · Streamlit · NLP · Machine Learning

Architected and developed a production-ready intelligent FAQ chatbot using TF-IDF
vectorization and cosine similarity for semantic question matching. The system
preprocesses natural language queries through an NLTK pipeline (tokenization,
stopword removal, lemmatization) and retrieves the most relevant answers from
a JSON knowledge base or uploaded PDF documents. Features a premium dark-mode
Streamlit UI with real-time analytics dashboards (Plotly), voice interaction
(SpeechRecognition + pyttsx3), and an admin panel for no-code FAQ management.
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'pyaudio'` | Install via `pipwin install pyaudio` (Windows) or `brew install portaudio && pip install pyaudio` (macOS) |
| NLTK download error | Check internet connection; NLTK data downloads to `nltk_data/` on first run |
| Voice not recognized | Ensure microphone is connected and browser/system has microphone permission |
| PDF shows no chunks | The PDF may be scanned (image-based). Use a PDF with selectable text |
| Low confidence scores | Lower the threshold in the sidebar, or add more relevant FAQs |

---

## 📜 License

MIT License — Free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using Python, Streamlit, NLTK, and Scikit-Learn<br>
⭐ Star this project if you found it useful!
</div>
#   A I - F A Q - C h a t b o t  
 