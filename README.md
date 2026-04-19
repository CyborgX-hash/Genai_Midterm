# 🏠 EstateAI v2.0 — AI-Powered Real Estate Intelligence

**Agentic Property Valuation · FAISS Market Retrieval · Structured Investment Advisory**

---

## 📖 Overview

EstateAI is an advanced, AI-powered real estate investment platform built with **Streamlit** and **LangGraph**. It combines traditional Machine Learning property price prediction with a Retrieval-Augmented Generation (RAG) agentic workflow to deliver comprehensive, data-driven investment advisory reports.

---

## 🎯 Objectives & Capabilities

- **Predictive Valuation**: ML-based price prediction using a Random Forest model.
- **Market Intelligence (RAG)**: Retrieves relevant real estate market patterns and laws using **FAISS** vector search and HuggingFace sentence embeddings.
- **Comparable Properties (Comps)**: Dynamically filters the historical dataset to find similar properties and compares their valuation details.
- **AI Investment Advisor**: Leverages the `google/flan-t5-base` LLM to generate structured advisory insights (Valuation Summary, Recommendation, Risk Factors).
- **Agentic Workflow**: Orchestrated by **LangGraph**, ensuring sequential, structured execution from prediction to final advice.

---

## 🔄 System Workflow (LangGraph)

The application models its logic as a typel-safe state graph moving through the following sequential nodes:

1. **`predict_node`**: Predicts property price using a pre-trained Random Forest model.
2. **`rag_node`**: Uses FAISS to query a knowledge corpus of Indian real estate market trends based on property specifications.
3. **`comps_node`**: Scans the dataset for comparable properties (+/- 30% area matching room count).
4. **`advisor_node`**: Ingests property metrics, predicted price, and retrieved market insights into an LLM (`Flan-T5`) to formulate structured advice.

---

## 🧠 Tech Stack & Requirements

- **UI Framework**: Streamlit (with Custom Luxury CSS)
- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`
- **Agent Orchestration**: `langgraph`
- **RAG & Embeddings**: `faiss-cpu`, `langchain`, `sentence-transformers`
- **LLM**: `transformers`, `torch` (`google/flan-t5-base`)

### 📦 Installation

Ensure you have Python 3.10+ installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 🚀 Usage

Run the Streamlit application locally:
```bash
streamlit run real_state_project.py
```

---

## 📊 Evaluation Metrics (Random Forest)

The predictive model is evaluated on standard regression metrics displayed in the sidebar:
| Metric | Description |
|--------|-------------|
| **R² Score** | Measures variance explained by the model |
| **MAE** | Mean Absolute Error (Average prediction error in INR) |
| **RMSE** | Root Mean Squared Error (Penalizes large errors) |

---

## ⚖️ Disclaimer

This application and the generated reports are for **informational and educational purposes only**. They do not constitute professional financial, investment, or legal advice. Predictions are based on historical data and LLMs. Always consult a certified investment advisor before making financial decisions.
