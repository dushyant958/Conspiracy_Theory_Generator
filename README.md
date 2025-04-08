# 🕵️‍♂️ Conspiracy Theory Generator - RAG System  
*Unveiling Theories from the Dark Web of Knowledge*

---

## ⚡ About the Project

This project is an end-to-end *Conspiracy Theory Generator* powered by a Retrieval-Augmented Generation (RAG) Pipeline. The system scrapes verified sources like government docs (CIA & FBI), Reddit, Wikipedia, and News articles — processes the data, and generates mind-blowing conspiracy theories using OpenAI's GPT-3.5 Turbo model.


![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented-blueviolet)
![LLM](https://img.shields.io/badge/LLM-Mistral%207B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🔑 APIs Used  

| API | Purpose | Variable Name in .env |
|-----|---------|-----------------------|
| OpenAI GPT-3.5 | Text Generation | `OPENAI_API_KEY` |
| Reddit API | Scraping Reddit Data | `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` |
| News API | Fetching News Articles | `NEWS_API_KEY` |

---

## 🚀 RAG Pipeline - Flow Overview

1. Scrape data from:
   - CIA Docs
   - FBI Archives
   - Reddit Threads
   - Wikipedia Articles
   - Latest News

2. Clean & Preprocess Text.

3. Convert text into Vectors.

4. Store vectors in Vector DB (FAISS).

5. Query relevant chunks → Pass to GPT-3.5 Turbo → Generate Conspiracy Theories.

6. (Optional) Fact Check & Validate.

---

## 🖥️ Frontend Preview  

| Landing Page | Generated Theories | Data Sources |
|--------------|-------------------|--------------|
| ![Frontend Screenshot 1](your-ss-link-here) | ![Frontend Screenshot 2](your-ss-link-here) | ![Frontend Screenshot 3](your-ss-link-here) |

---

## 🏃‍♂️ Running the Project  

1. Clone the Repo:
2. Install Dependencies:
3. Setup `.env` file with your API Keys:
4. Run Backend + Vector Creation:
5. Launch Streamlit Frontend:



> Additional terminal instructions provided in:  
`Running APO and streamlit server.txt`

---

## ⚠️ Note

> The provided API keys in the repo are non-functional (limited to author's usage).  
> Kindly add your own valid API Keys in `.env` for full functionality.

---

## ✨ Features  

- Auto Web Scraping from credible sources.
- Powerful RAG-based Theory Generation.
- Real-time API-driven GPT-3.5 Turbo Integration.
- Customizable Vector DB Search.
- Simple Streamlit UI for Exploration.

---

## 🧠 Future Improvements

- LLM Fine-tuning for Conspiracy Styles.
- Dynamic Source Addition.
- Visualizing Theory Graphs.
- Fact-Checking Automation.
- Multi-modal integration (images, documents).

---

## 🤝 Contributions

PRs and suggestions are welcome! Feel free to fork, enhance, and create your own crazy theory generator!

---
