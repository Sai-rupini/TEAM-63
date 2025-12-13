# 💊 Med-Reminder Chatbot (Label-Aware)

A Retrieval-Augmented Generation (RAG) system designed to answer complex questions over drug labels and generate structured medication reminder plans. This project leverages local LLMs via Ollama to ensure data privacy and efficient local development.

---

## 🎯 Problem Statement

The goal is to develop a **Medication Reminder Chatbot (Label-Aware)** capable of two core functions:

1.  **Label-Aware Q&A:** Answering user questions based on comprehensive drug label data (e.g., side effects, contraindications, dosage instructions).
2.  **Schedule Reminders:** Generating a structured, sample JSON plan for medication scheduling, dose, and frequency based on user input and label data.

*Note: This solution focuses purely on the backend logic and does not require integration with external phone/SMS services.*

---

## ✨ Key Outcome & Features

The primary function is to demonstrate a robust RAG pipeline:

### Core Function: `ask_drug(question)`
* Retrieves relevant sections from the vectorized drug labels (stored in ChromaDB).
* Uses the retrieved context to generate accurate and contextualized answers using the local **Mistral** LLM.

### Sample Reminder Generation
* Generates a sample JSON structure detailing the drug, dosage, frequency, and time slots.
* *Example Output:*
    ```json
    {
      "drug_name": "AMOXICILLIN CAPSULE",
      "dosage": "500 mg",
      "frequency": "Three times daily",
      "schedule": [
        {"time": "8:00 AM", "action": "Take 1 capsule"},
        {"time": "2:00 PM", "action": "Take 1 capsule"},
        {"time": "8:00 PM", "action": "Take 1 capsule"}
      ]
    }
    ```

---

## 🛠️ Technology Stack (GenAI Hackathon Focus)

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python (3.10+) | Core development environment. |
| **Framework** | LangChain | Orchestration of the RAG pipeline (Loaders, Splitting, Chains). |
| **Local LLM** | Ollama (Mistral) | Hosting the **Mistral 7B** model for inference, ensuring fast, local execution. |
| **Vector DB** | ChromaDB | Storing high-dimensional embeddings of drug label text for efficient retrieval. |
| **Data Source** | openFDA Drug Label | Used as the source for raw drug label data (e.g., indications, warnings, dosage). |
| **Serving (Optional)** | FastAPI | For creating a simple REST API endpoint to interact with the chatbot logic. |

---

## 🚀 Setup and Installation

Follow these steps to get your environment running locally.

### 1. Ollama Setup

1.  **Install Ollama:** Download and install Ollama for your operating system from the official website.
2.  **Pull the LLM:** Use the command line to pull the required Mistral model:
    ```bash
    ollama pull mistral
    ```
    *(The LLM service runs in the background, ready to be called by LangChain.)*

### 2. Python Environment

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Sai-rupini/TEAM-63.git](https://github.com/Sai-rupini/TEAM-63.git)
    cd TEAM-63
    git checkout feature/med-chatbot # Switch to the working branch
    ```
2.  **Setup Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows PowerShell
    # source venv/bin/activate # On Linux/macOS
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` includes: `langchain`, `langchain-community`, `ollama`, `chromadb`, `pydantic`)*

### 3. Data Ingestion

1.  **Download Sample Data:** Place the downloaded openFDA Drug Label JSON files into a dedicated `data/` directory.
2.  **Run the Ingestion Script:** This script processes the raw JSON, splits the text, generates embeddings, and loads them into ChromaDB.
    ```bash
    python ingest_data.py
    ```

---

## 🧪 Usage

Once setup is complete, you can interact with the chatbot logic.

### Running the Q&A Function

Execute the main script to test the RAG functionality:

```bash
python main_app.py

