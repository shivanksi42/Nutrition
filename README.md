# 🌿 NutriCare AI – Personalized Dietary Recommendations with RAG

<img src="media/nutrition.png" width="400" alt="NutriCare Logo" />
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Build-Stable-success)]()
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)]()
[![Frontend-React](https://img.shields.io/badge/Frontend-React.js-61DAFB.svg)]()
[![Backend-Flask](https://img.shields.io/badge/Backend-Flask%2FFastAPI-yellow.svg)]()
[![Database-ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-blueviolet.svg)]()

**NutriCare AI** is a personalized nutrition recommendation web application powered by Retrieval-Augmented Generation (RAG). It provides custom diet plans based on a user's health conditions (e.g., diabetes, high blood pressure) and allergies (e.g., dairy, gluten). Designed for healthcare-conscious individuals and professionals, it intelligently queries curated medical nutrition articles to suggest optimal meal plans tailored to each patient's needs.

---

## 🚀 Features

- ✅ Input diagnosis + allergy via natural language
- 🧠 Retrieval-Augmented Generation for contextual dietary planning
- 🔍 Real-time querying of health-specific article databases
- 🥗 Personalized recommendations using AI-driven text understanding
- 🌐 Modern UI with React frontend
- 📊 Scalable with **ChromaDB** vector database

---

## 🛠️ Tech Stack

| Frontend | Backend | AI Model | Database | Web Scraping |
|----------|---------|----------|----------|--------------|
| React.js | Flask / FastAPI | LangChain + OpenAI (RAG) | **ChromaDB** | BeautifulSoup |

---

## 📷 Demo

![App Demo](https://your-image-hosting-link/nutricare-demo.gif)

---

## 🧩 System Architecture

```plaintext
User Input → React UI → Flask API → LangChain Query
         → Article Context via ChromaDB
         → RAG Engine → AI Response → Personalized Diet Plan
