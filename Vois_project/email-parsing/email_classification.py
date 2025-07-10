import streamlit as st
import datetime as dt
import random
import string
from pathlib import Path
from typing import List, Dict

import pandas as pd
import spacy
from email.message import EmailMessage
from email import policy
from email.parser import BytesParser

# Optional: only import Hugging Face if zero-shot enabled
from transformers import pipeline

# ────────────────── Configuration
OUT_DIR = Path("emails")
OUT_DIR.mkdir(exist_ok=True)

SENDERS = [
    "billing@company.com", "hr@startup.io", "support@webapp.org",
    "sales@techcorp.com", "info@example.org"
]
RECIPIENTS = ["user@example.com"]
SUBJECTS = [
    "Invoice overdue", "Payment confirmation",
    "Job application for Data Analyst", "Issue with logging in",
    "New product launch update", "Resume submission for HR position",
    "Need support with account", "Sales inquiry for bulk pricing"
]
BODIES = [
    "Hello, I would like to confirm that I’ve made the payment for the last invoice.",
    "There’s an issue with my login credentials. Can you help me fix it?",
    "I’m applying for the Data Analyst position. Please find my resume attached.",
    "We are launching a new product next month. Here's the marketing plan.",
    "Can I get a quote for purchasing 500 units of your product?",
    "Please help me reset my password. I can't access my account.",
    "Attached is my resume for the HR Assistant role.",
    "Reminder: Your invoice from last month is still unpaid."
]
ZERO_LABELS = [
    "billing", "technical support", "human resources",
    "marketing", "sales", "general"
]

def generate_random_email(save_path: Path) -> None:
    msg = EmailMessage()
    msg["From"] = random.choice(SENDERS)
    msg["To"] = random.choice(RECIPIENTS)
    msg["Subject"] = random.choice(SUBJECTS)
    msg["Date"] = dt.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    uid = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    msg.set_content(f"{random.choice(BODIES)}\n\n--ID:{uid}--")
    save_path.write_bytes(bytes(msg))

def parse_eml(path: Path) -> Dict[str, str]:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part else ""
    return {
        "file": path.name,
        "from": msg.get("from", ""),
        "subject": msg.get("subject", ""),
        "date": msg.get("date", ""),
        "body": body,
    }

def rule_intent(text: str) -> str:
    txt = text.lower()
    if any(k in txt for k in ("invoice", "payment")):
        return "billing"
    if any(k in txt for k in ("resume", "job")):
        return "human resources"
    if any(k in txt for k in ("issue", "problem", "password", "support", "login")):
        return "technical support"
    if any(k in txt for k in ("quote", "pricing", "purchase")):
        return "sales"
    if any(k in txt for k in ("launch", "marketing", "product")):
        return "marketing"
    return "general"

@st.cache_resource
def load_model():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier


# ────────────────── Streamlit App
st.set_page_config(page_title="📧 Email Categorizer", layout="wide")
st.title("📧 Email Parser & Categorizer")

col1, col2 = st.columns(2)
count = col1.slider("Number of Random Emails", 1, 10, 3)
enable_zero_shot = col2.toggle("Enable Zero-Shot Classification (Hugging Face)", value=False)

if st.button("Generate and Analyze"):
    with st.spinner("Generating and parsing emails..."):
        nlp = spacy.load("en_core_web_sm")

        if enable_zero_shot:
            classifier = load_model()

        paths = []
        for i in range(count):
            path = OUT_DIR / f"email_{i+1}.eml"
            generate_random_email(path)
            paths.append(path)

        data: List[Dict[str, str]] = []

        for p in paths:
            rec = parse_eml(p)
            rec["intent_rule"] = rule_intent(rec["body"])

            doc = nlp(rec["body"])
            rec["persons"] = ", ".join(e.text for e in doc.ents if e.label_ == "PERSON")

            if enable_zero_shot:
                result = classifier(rec["body"], ZERO_LABELS, multi_label=False)
                rec["intent_zero"] = result["labels"][0]
                rec["zero_score"] = round(result["scores"][0], 3)

            data.append(rec)

        df = pd.DataFrame(data)
        display_cols = ["file", "from", "subject", "intent_rule", "persons"]
        if enable_zero_shot:
            display_cols += ["intent_zero", "zero_score"]

        st.success("Emails parsed and categorized!")
        st.dataframe(df[display_cols], use_container_width=True)
