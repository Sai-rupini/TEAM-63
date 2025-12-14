import os
import json
import re
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
import easyocr
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


# Initialize OCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])


# Initialize speech recognizer
recognizer = sr.Recognizer()


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
    "Russian": "ru",
    "Telugu": "te"
}


# Voice input function
def voice_to_text():
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return ""
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service")
        return ""
    except Exception as e:
        st.error(f"Voice recognition error: {e}")
        return ""


# Image OCR function
def image_to_text(image):
    try:
        reader = get_ocr_reader()
        result = reader.readtext(np.array(image))
        text = " ".join([detection[1] for detection in result])
        return text
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""


# Load or create the vector store
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


# Get all unique drug names from Chroma DB
@st.cache_resource
def get_all_drugs():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []
    result = vectorstore.get()
    drugs = set()
    for metadata in result.get("metadatas", []):
        drug = metadata.get("generic_name", "").strip()
        if drug:
            drugs.add(drug)
    return list(drugs)


# Search drugs by symptom from Chroma DB
def search_drugs_by_symptom(symptom: str, vectorstore):
    query = f"drugs for {symptom}"
    docs = vectorstore.similarity_search(query, k=5)
    drugs = []
    for doc in docs:
        drug = doc.metadata.get("generic_name", "").strip()
        if drug:
            drugs.append(drug)
    return drugs


# Setup the language model
@st.cache_resource
def get_llm():
    try:
        return ChatOllama(model="mistral:7b-instruct-v0.2-q4_0", temperature=0.0, num_ctx=2048, num_predict=256)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None


@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Vectorstore not loaded. Check your Chroma DB path and data.")
        return None
    llm = get_llm()
    if llm is None:
        st.error("LLM not loaded. Check your Ollama connection and model.")
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

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


# Dosage calculator (dynamic from Chroma DB)
def get_dosage_info(drug_name: str, vectorstore):
    query = f"dosage for {drug_name}"
    docs = vectorstore.similarity_search(query, k=1)
    if docs:
        return docs[0].page_content
    return "Dosage information not found"


# Drug search and suggestion (dynamic)
def suggest_drug(drug_name: str, drug_list: list) -> str:
    best_match, score = process.extractOne(drug_name, drug_list)
    if score > 70:
        return best_match
    return None


# Multi-drug reminder plans (dynamic)
def generate_multi_drug_reminder_json(questions: list, vectorstore):
    plans = []
    for question in questions:
        plan = generate_reminder_json(question, vectorstore)
        plans.append(plan)
    return plans


# Export reminder plan
def export_reminder_plan(plan: dict, format: str = "json"):
    if format == "json":
        return json.dumps(plan, indent=4)


# Medication schedule calendar
def create_calendar(schedule: list):
    df = pd.DataFrame(schedule, columns=["Time"])
    return df


# Multi-language support
def translate_text(text: str, target_lang: str) -> str:
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)


# Generate reminder JSON (dynamic)
def generate_reminder_json(question: str, vectorstore):
    drug_match = re.search(r"(\w+)", question, re.IGNORECASE)
    drug_name = drug_match.group(0).title() if drug_match else "Generic Drug"
    dosage_info = get_dosage_info(drug_name, vectorstore)
    dosage = dosage_info if dosage_info else "Standard dose"

    # Default schedule (can be made dynamic based on dosage info)
    schedule = ["8:00 AM", "8:00 PM"]
    rationale = f"Default {drug_name} {dosage} schedule. Adjust based on actual drug and your healthcare provider's advice."

    sample_plan = ReminderPlan(
        drug_name=drug_name,
        dosage=dosage,
        schedule=schedule,
        rationale=rationale
    )
    return sample_plan.model_dump()


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
        st.error("Failed to load RAG chain. Check your resources.")
        return

    # Load vectorstore
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Failed to load vector store. Check your Chroma DB path and data.")
        return

    # Get all drugs from Chroma DB
    drug_list = get_all_drugs()

    # Sidebar for user profile and language selection
    with st.sidebar:
        st.header("User Profile")
        st.session_state.user_profile["age"] = st.number_input("Age", min_value=1, max_value=120, value=25)
        st.session_state.user_profile["weight"] = st.number_input("Weight (kg)", min_value=1, max_value=200, value=70)

        st.header("Language")
        selected_language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))
        target_lang = SUPPORTED_LANGUAGES[selected_language]

        st.header("Input Method")
        input_method = st.radio("Select Input Method", ["Text", "Voice", "Image"])

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if input_method == "Text":
        prompt = st.chat_input("Ask a question about a drug, symptoms, or request a reminder...")
    elif input_method == "Voice":
        if st.button("🎙️ Record Voice"):
            prompt = voice_to_text()
            st.text_area("Voice Input", prompt)
        else:
            prompt = ""
    elif input_method == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
            if st.button("Extract Text"):
                prompt = image_to_text(image)
                st.text_area("Extracted Text", prompt)
            else:
                prompt = ""
        else:
            prompt = ""
    else:
        prompt = ""

    if prompt:
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
                        drugs = search_drugs_by_symptom(symptom, vectorstore)
                        if drugs:
                            st.markdown(f"**Recommended drugs for {symptom}:** {', '.join(drugs)}")
                            for drug in drugs:
                                drug_prompt = f"What are the indications and dosage for {drug}?"
                                try:
                                    answer = rag_chain.invoke(drug_prompt)
                                    st.markdown(f"**Drug Info for {drug}:**\n{answer}")
                                    dosage = get_dosage_info(drug, vectorstore)
                                    st.markdown(f"**Dosage for {drug}:** {dosage}")
                                except Exception as e:
                                    st.error(f"Error retrieving info for {drug}: {e}")
                        else:
                            st.markdown("No drugs found for this symptom.")
                    else:
                        try:
                            answer = rag_chain.invoke(prompt)
                            st.markdown(f"**Drug Info:**\n{answer}")
                        except Exception as e:
                            st.error(f"Error retrieving drug info: {e}")

                    # Dosage calculator (dynamic)
                    dosage = get_dosage_info("ibuprofen", vectorstore)
                    st.markdown(f"**Dosage:** {dosage}")

                    # Drug search and suggestion (dynamic)
                    suggested_drug = suggest_drug(prompt, drug_list)
                    if suggested_drug:
                        st.markdown(f"**Did you mean:** {suggested_drug}?")

                    # Multi-drug reminder plans (dynamic)
                    multi_drug_plans = generate_multi_drug_reminder_json([prompt], vectorstore)
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
                    try:
                        translated = translate_text(answer, target_lang)
                        st.markdown(f"**Translated ({selected_language}):** {translated}")
                    except Exception as e:
                        st.error(f"Translation error: {e}")

                    # Append to chat history
                    full_response = f"**Drug Info:**\n{answer}\n\n**Dosage:** {dosage}\n\n---"
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error during RAG chaining: {e}")


if __name__ == "__main__":
    main()
