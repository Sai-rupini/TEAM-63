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
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
VECTOR_STORE_PATH = "chroma_db_meds"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# keywords to decide whether a query is really medical
MEDICAL_KEYWORDS = [
    "tablet", "capsule", "injection", "syrup", "ointment", "cream",
    "dose", "dosage", "side effect", "adverse", "contraindication",
    "indication", "drug", "medicine", "medication", "mg", "mcg",
    "fever", "pain", "infection", "antibiotic", "analgesic",
    "antacid", "antihistamine", "antidepressant", "blood pressure",
    "diabetes","contradictions"
]


# -------------------------------------------------------------------
# Helpers for safety / intent
# -------------------------------------------------------------------
def looks_medical(prompt: str) -> bool:
    """Return True only if the question clearly looks medication‑related."""
    p = prompt.lower()
    return any(k in p for k in MEDICAL_KEYWORDS)


# -------------------------------------------------------------------
# OCR and speech
# -------------------------------------------------------------------
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])


recognizer = sr.Recognizer()


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


def image_to_text(image):
    try:
        reader = get_ocr_reader()
        result = reader.readtext(np.array(image))
        text = " ".join([detection[1] for detection in result])
        return text
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""


# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------
class ReminderPlan(BaseModel):
    drug_name: str = Field(description="Name of the medication.")
    dosage: str = Field(description="Dosage, e.g., '200 mg'.")
    schedule: list[str] = Field(description="Times to take, e.g., ['8:00 AM', '8:00 PM'].")
    rationale: str = Field(description="Brief reason for the schedule.")


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


# -------------------------------------------------------------------
# Vector store and LLM
# -------------------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


@st.cache_resource
def get_all_drugs():
    """All generic names present in Chroma metadata – used for suggestions."""
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


def search_drugs_by_symptom(symptom: str, vectorstore):
    query = f"drugs used to treat {symptom}"
    docs = vectorstore.similarity_search(query, k=5)
    drugs = []
    for doc in docs:
        drug = doc.metadata.get("generic_name", "").strip()
        if drug:
            drugs.append(drug)
    return drugs


@st.cache_resource
def get_llm():
    try:
        return ChatOllama(
            model="mistral:7b-instruct-v0.2-q4_0",
            temperature=0.0,
            num_ctx=2048,
            num_predict=256
        )
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
You are a STRICT Medication Assistant.

Rules:
- You ONLY answer questions about human medications and drug labels contained in the context.
- If the question is about a person, place, event, or any non‑drug topic, answer exactly:
  "This assistant only answers questions about medications and drug information."
- If the needed information is not in the context, answer exactly:
  "This information is not available in the provided context."
- NEVER invent, guess, or extrapolate clinical advice beyond what the context states.

CONTEXT:
{context}

QUESTION:
{question}

If relevant and present in the context, list indications, dosage, side effects, and warnings.
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


# -------------------------------------------------------------------
# Domain helpers
# -------------------------------------------------------------------
def get_dosage_info(drug_name: str, vectorstore):
    query = f"dosage and administration for {drug_name}"
    docs = vectorstore.similarity_search(query, k=1)
    if docs:
        return docs[0].page_content
    return "Dosage information not found in the provided labels."


def suggest_drug(drug_name: str, drug_list: list) -> str | None:
    if not drug_list:
        return None
    best_match, score = process.extractOne(drug_name, drug_list)
    if score > 80:
        return best_match
    return None


def generate_multi_drug_reminder_json(questions: list, vectorstore):
    return [generate_reminder_json(q, vectorstore) for q in questions]


def export_reminder_plan(plan: dict, fmt: str = "json"):
    if fmt == "json":
        return json.dumps(plan, indent=4)
    return ""


def create_calendar(schedule: list):
    return pd.DataFrame(schedule, columns=["Time"])


def translate_text(text: str, target_lang: str) -> str:
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)


def generate_reminder_json(question: str, vectorstore):
    drug_match = re.search(r"([A-Za-z][A-Za-z0-9_-]*)", question)
    drug_name = drug_match.group(1).title() if drug_match else "Medication"

    dosage_info = get_dosage_info(drug_name, vectorstore)
    # Keep dosage short in the plan
    short_dosage = dosage_info[:250] + "..." if len(dosage_info) > 250 else dosage_info

    schedule = ["8:00 AM", "8:00 PM"]
    rationale = (
        f"Example schedule for {drug_name}. Confirm exact timing and dosage with "
        "a licensed healthcare professional."
    )

    plan = ReminderPlan(
        drug_name=drug_name,
        dosage=short_dosage,
        schedule=schedule,
        rationale=rationale
    )
    return plan.model_dump()


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Medication Assistant", layout="wide")
    st.title("💊 Medication Assistant Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {"age": None, "weight": None, "medications": []}

    rag_chain = get_rag_chain()
    if rag_chain is None:
        st.error("Failed to load RAG chain. Check your resources.")
        return

    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Failed to load vector store. Check your Chroma DB path and data.")
        return

    drug_list = get_all_drugs()

    with st.sidebar:
        st.header("User Profile")
        st.session_state.user_profile["age"] = st.number_input("Age", 1, 120, 25)
        st.session_state.user_profile["weight"] = st.number_input("Weight (kg)", 1, 200, 70)

        st.header("Language")
        selected_language = st.selectbox("Select Language", list(SUPPORTED_LANGUAGES.keys()))
        target_lang = SUPPORTED_LANGUAGES[selected_language]

        st.header("Input Method")
        input_method = st.radio("Select Input Method", ["Text", "Voice", "Image"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # collect user prompt
    if input_method == "Text":
        prompt = st.chat_input("Ask a question about a drug, symptoms, or request a reminder...")
    elif input_method == "Voice":
        if st.button("🎙️ Record Voice"):
            prompt = voice_to_text()
            st.text_area("Voice Input", prompt)
        else:
            prompt = ""
    else:  # Image
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

    if not prompt:
        st.info("This assistant only answers questions about medications and drug information.")
        return

    # log user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # 1. HARD GUARD: reject non‑medical queries up front
                if not looks_medical(prompt):
                    msg = (
                        "This assistant only answers questions about medications and drug "
                        "information from official drug labels. Please ask a medication‑related question."
                    )
                    st.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    return

                # 2. symptom‑style detection (still within medical domain)
                symptom_match = re.search(
                    r"symptom|fever|pain|headache|cough|cold|infection",
                    prompt,
                    re.IGNORECASE
                )

                if symptom_match:
                    symptom = symptom_match.group(0)
                    drugs = search_drugs_by_symptom(symptom, vectorstore)
                    if drugs:
                        st.markdown(f"**Possible drugs mentioned in labels for {symptom}:** {', '.join(drugs)}")
                        for drug in drugs:
                            q = f"What do the labels say about {drug}?"
                            answer = rag_chain.invoke(q)
                            st.markdown(f"**Drug Info for {drug}:**\n{answer}")
                            dosage = get_dosage_info(drug, vectorstore)
                            st.markdown(f"**Dosage (from labels) for {drug}:** {dosage}")
                    else:
                        st.markdown("No specific drugs for this symptom were found in the indexed labels.")
                else:
                    # 3. normal drug / label question
                    answer = rag_chain.invoke(prompt)
                    st.markdown(f"**Drug Info:**\n{answer}")

                # 4. optional suggestion based on known drugs
                suggested_drug = suggest_drug(prompt, drug_list)
                if suggested_drug:
                    st.markdown(f"**Did you mean the medication:** {suggested_drug}?")

                # 5. reminder plan and schedule (purely informational)
                plans = generate_multi_drug_reminder_json([prompt], vectorstore)
                for plan in plans:
                    st.markdown("### Reminder Plan (JSON)")
                    st.json(plan)
                    st.markdown("### Medication Schedule Calendar")
                    st.table(create_calendar(plan["schedule"]))

                st.download_button(
                    label="Download Reminder Plan",
                    data=export_reminder_plan(plans[-1]),
                    file_name="reminder_plan.json",
                    mime="application/json"
                )

                # translation (non‑critical)
                try:
                    translated = translate_text(answer, target_lang)
                    st.markdown(f"**Translated ({selected_language}):** {translated}")
                except Exception as e:
                    st.error(f"Translation error: {e}")

                full_response = f"**Drug Info:**\n{answer}\n\n---"
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error during processing: {e}")

    st.info(
        "Safety notice: This chatbot only reads information from approved drug labels and "
        "does not replace a consultation with a licensed healthcare professional."
    )


if __name__ == "__main__":
    main()
