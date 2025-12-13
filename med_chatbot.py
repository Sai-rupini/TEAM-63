import os
import json
import re
import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from deep_translator import GoogleTranslator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Constants
VECTOR_STORE_PATH = "chroma_db_meds"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Define Pydantic schema for reminder plan
class ReminderPlan(BaseModel):
    drug_name: str = Field(description="Name of the medication.")
    dosage: str = Field(description="Dosage, e.g., '200 mg'.")
    schedule: list[str] = Field(description="Times to take, e.g., ['8:00 AM', '8:00 PM'].")
    rationale: str = Field(description="Brief reason for the schedule.")

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Russian": "ru"
}

# Load or create the vector store
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Setup the language model
@st.cache_resource
def get_llm():
    try:
        return ChatOllama(model="mistral:7b-instruct-v0.2-q4_0", temperature=0.0, num_ctx=2048, num_predict=256)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None

# Build the RAG question-answering chain
@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
    llm = get_llm()
    if llm is None:
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Reduce k for speed

    template = """
You are a helpful Medication Assistant. Answer ONLY based on the following context from drug labels.
If the answer is not in the context, say so explicitly.

CONTEXT:
{context}

QUESTION:
{question}

Also, list any side effects and warnings if available.
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n---\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | (lambda x: x.content)
    )
    return rag_chain

# Dosage calculator
def calculate_dosage(drug_name: str, age: int, weight: int, condition: str) -> str:
    if drug_name.lower() == "ibuprofen":
        if age >= 12 and weight >= 40:
            return "200-400 mg every 4-6 hours"
        else:
            return "Consult a healthcare provider"
    return "Consult a healthcare provider"

# Drug search and suggestion
def suggest_drug(drug_name: str, drug_list: list) -> str:
    best_match, score = process.extractOne(drug_name, drug_list)
    if score > 70:
        return best_match
    return None

# Multi-drug reminder plans
def generate_multi_drug_reminder_json(questions: list) -> list:
    plans = []
    for question in questions:
        plan = generate_reminder_json(question)
        plans.append(plan)
    return plans

# Export reminder plan
def export_reminder_plan(plan: dict, format: str = "json"):
    if format == "json":
        return json.dumps(plan, indent=4)
    # Add CSV logic if needed

# Medication schedule calendar
def create_calendar(schedule: list):
    df = pd.DataFrame(schedule, columns=["Time"])
    return df

# Multi-language support
def translate_text(text: str, target_lang: str) -> str:
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# Generate reminder JSON
def generate_reminder_json(question: str):
    drug_match = re.search(r"paracetamol|ibuprofen|aspirin|amoxicillin|metformin|drug|medication", question, re.IGNORECASE)
    dosage_match = re.search(r"(\d+)\s*mg", question, re.IGNORECASE)
    frequency_match = re.search(r"twice|two times|thrice|three times|once|four times", question, re.IGNORECASE)

    drug_name = drug_match.group(0).title() if drug_match else "Generic Drug"
    dosage = f"{dosage_match.group(1)} mg" if dosage_match else "Standard dose"

    if frequency_match:
        freq = frequency_match.group(0).lower()
        if "twice" in freq or "two times" in freq:
            schedule = ["8:00 AM", "8:00 PM"]
        elif "thrice" in freq or "three times" in freq:
            schedule = ["8:00 AM", "2:00 PM", "8:00 PM"]
        elif "four times" in freq:
            schedule = ["8:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"]
        else:
            schedule = ["8:00 AM"]
    else:
        schedule = ["8:00 AM", "8:00 PM"]

    rationale = f"Default {drug_name} {dosage} schedule. Adjust based on actual drug and your healthcare provider's advice."

    sample_plan = ReminderPlan(
        drug_name=drug_name,
        dosage=dosage,
        schedule=schedule,
        rationale=rationale
    )
    return sample_plan.model_dump()

# Load symptom data
def load_symptom_data():
    with open("symptoms_to_drugs.json", "r") as f:
        return json.load(f)

# Hybrid search function
def hybrid_search(symptom: str, symptom_data: list) -> list:
    for item in symptom_data:
        if item["symptom"].lower() == symptom.lower():
            return item["drugs"]
    return []

# Main Streamlit app
def main():
    st.set_page_config(page_title="Medication Assistant", layout="wide")
    st.title("💊 Medication Assistant Chatbot")

    # Initialize chat messages and user profile
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {"age": None, "weight": None, "medications": []}

    # Load or create RAG chain
    rag_chain = get_rag_chain()
    if rag_chain is None:
        st.error("Failed to load resources. Check data and environment.")
        return

    # Load symptom data
    symptom_data = load_symptom_data()

    # Sidebar for user profile and language selection
    with st.sidebar:
        st.header("User Profile")
        st.session_state.user_profile["age"] = st.number_input("Age", min_value=1, max_value=120, value=25)
        st.session_state.user_profile["weight"] = st.number_input("Weight (kg)", min_value=1, max_value=200, value=70)

        st.header("Language")
        selected_language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))
        target_lang = SUPPORTED_LANGUAGES[selected_language]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask a question about a drug, symptoms, or request a reminder..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer and reminder plan
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Check if the prompt is about symptoms
                    symptom_match = re.search(r"symptom|headache|fever|muscle pain|cough|cold", prompt, re.IGNORECASE)
                    if symptom_match:
                        symptom = symptom_match.group(0)
                        drugs = hybrid_search(symptom, symptom_data)
                        if drugs:
                            st.markdown(f"**Recommended drugs for {symptom}:** {', '.join(drugs)}")
                            for drug in drugs:
                                drug_prompt = f"What are the indications and dosage for {drug}?"
                                answer = rag_chain.invoke(drug_prompt)
                                st.markdown(f"**Drug Info for {drug}:**\n{answer}")
                                dosage = calculate_dosage(drug, st.session_state.user_profile["age"], st.session_state.user_profile["weight"], "pain")
                                st.markdown(f"**Dosage for {drug}:** {dosage}")
                        else:
                            st.markdown("No drugs found for this symptom.")
                    else:
                        answer = rag_chain.invoke(prompt)
                        st.markdown(f"**Drug Info:**\n{answer}")

                        # Dosage calculator
                        dosage = calculate_dosage("ibuprofen", st.session_state.user_profile["age"], st.session_state.user_profile["weight"], "pain")
                        st.markdown(f"**Dosage:** {dosage}")

                        # Drug search and suggestion
                        drug_list = ["paracetamol", "ibuprofen", "aspirin", "amoxicillin", "metformin"]
                        suggested_drug = suggest_drug(prompt, drug_list)
                        if suggested_drug:
                            st.markdown(f"**Did you mean:** {suggested_drug}?")

                        # Multi-drug reminder plans
                        multi_drug_plans = generate_multi_drug_reminder_json([prompt])
                        for plan in multi_drug_plans:
                            st.markdown("### Reminder Plan (JSON)")
                            st.json(plan)

                            # Medication schedule calendar
                            st.markdown("### Medication Schedule Calendar")
                            st.table(create_calendar(plan["schedule"]))

                        # Export reminder plan
                        st.download_button(
                            label="Download Reminder Plan",
                            data=export_reminder_plan(plan),
                            file_name="reminder_plan.json",
                            mime="application/json"
                        )

                        # Multi-language support
                        translated = translate_text(answer, target_lang)
                        st.markdown(f"**Translated ({selected_language}):** {translated}")

                        # Append to chat history
                        full_response = f"**Drug Info:**\n{answer}\n\n**Dosage:** {dosage}\n\n---"
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error during RAG chaining: {e}")

if __name__ == "__main__":
    main()
