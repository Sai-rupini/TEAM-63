import os
import json
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_DIR = "D:\\med_chatbot\\drug_labels"
VECTOR_STORE_PATH = "chroma_db_meds"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

documents = []
for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Processing {jf}, found {len(data.get('results', []))} records")
    for record in data.get("results", []):
        content = ""
        if "openfda" in record:
            content += f"GENERIC NAME: {record['openfda'].get('generic_name', ['N/A'])[0]}\n"
            content += f"BRAND NAME: {record['openfda'].get('brand_name', ['N/A'])[0]}\n"
        for key in ["indications_and_usage", "warnings", "contraindications", "adverse_reactions", "dosage_and_administration"]:
            if key in record:
                content += f"### {key.replace('_', ' ').upper()}\n{' '.join(record[key])}\n"
        if not content.strip() and "description" in record:
            content += record["description"]
        if content.strip():
            documents.append(Document(page_content=content, metadata={"source": jf}))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=VECTOR_STORE_PATH
)
print("Index built and saved!")
