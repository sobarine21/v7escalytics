import streamlit as st
import google.generativeai as genai
from langdetect import detect
from textblob import TextBlob
from fpdf import FPDF
from io import BytesIO
import concurrent.futures
import json
import docx2txt
from PyPDF2 import PdfReader
import re
import base64
from cryptography.fernet import Fernet
import email
from email import policy
from email.parser import BytesParser
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

st.set_page_config(page_title="Advanced Email AI", page_icon="📧", layout="wide")
st.title("📨 Advanced Email AI Analysis & Insights")
st.write("Extract insights, generate professional responses, and analyze emails with AI.")

# Add custom CSS to hide the header and the top-right buttons
hide_streamlit_style = """
    <style>
        .css-1r6p8d1 {display: none;} /* Hides the Streamlit logo in the top left */
        .css-1v3t3fg {display: none;} /* Hides the star button */
        .css-1r6p8d1 .st-ae {display: none;} /* Hides the Streamlit logo */
        header {visibility: hidden;} /* Hides the header */
        .css-1tqja98 {visibility: hidden;} /* Hides the header bar */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

features = {
    "sentiment": True,
    "highlights": True,
    "response": True,
    "export": True,
    "tone": True,
    "urgency": False,
    "task_extraction": True,
    "subject_recommendation": True,
    "category": True,
    "politeness": True,
    "emotion": True,
    "spam_check": True,
    "readability": True,
    "root_cause": True,
    "clarity": True,
    "best_response_time": True,
    "scenario_responses": True,
    "attachment_analysis": True,
    "complexity_reduction": True,
    "phishing_detection": True,
    "sensitive_info_detection": True,
    "confidentiality_rating": True,
    "attachment_summarization": True,
    "bias_detection": True,
    "conflict_detection": True,
    "argument_mining": True,
    "metadata_extraction": True,
}

selected_scenario = st.selectbox("Select a scenario for suggested response:", scenario_options)

@st.cache_data(ttl=3600)
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return ""

def get_sentiment(email_content):
    return TextBlob(email_content).sentiment.polarity

def get_readability(email_content):
    return round(TextBlob(email_content).sentiment.subjectivity * 10, 2)

def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

def analyze_phishing_links(email_content):
    phishing_keywords = ["login", "verify", "update account", "account suspended", "urgent action required", "click here"]
    phishing_links = []
    urls = re.findall(r'(https?://\S+)', email_content)
    for url in urls:
        for keyword in phishing_keywords:
            if keyword.lower() in url.lower():
                phishing_links.append(url)
    return phishing_links

def detect_sensitive_information(email_content):
    sensitive_info_patterns = {
        "phone_number": r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?|\d{3})[\s\-]?\d{3}[\\s\-]?\d{4}",
        "email_address": r"[\w\.-]+@[\w\.-]+\.\w+",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b"
    }

    sensitive_data = {}
    for key, pattern in sensitive_info_patterns.items():
        matches = re.findall(pattern, email_content)
        if matches:
            sensitive_data[key] = matches
    return sensitive_data

def confidentiality_rating(email_content):
    keywords = ["confidential", "private", "restricted", "not for distribution"]
    rating = sum(1 for keyword in keywords if keyword.lower() in email_content.lower())
    return min(rating, 5)

def analyze_attachment(file):
    try:
        if file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(file)
        elif file.type in ["message/rfc822", "application/vnd.ms-outlook"]:
            msg = BytesParser(policy=policy.default).parsebytes(file.getvalue())
            return msg.get_body(preferencelist=('plain')).get_content()
        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Error analyzing attachment: {e}"

def extract_email_metadata(email_file):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(email_file.getvalue())
        metadata = {
            "from": msg.get("From"),
            "to": msg.get("To"),
            "subject": msg.get("Subject"),
            "date": msg.get("Date"),
            "message_id": msg.get("Message-ID"),
            "in_reply_to": msg.get("In-Reply-To"),
            "references": msg.get("References"),
            "attachments": [part.get_filename() for part in msg.iter_attachments()]
        }
        return metadata
    except Exception as e:
        return f"Error extracting metadata: {e}"

def countdown_timer(duration):
    with st.spinner("Processing..."):
        for i in range(duration, 0, -1):
            st.write(f"⏳ {i} seconds remaining...")
            time.sleep(1)

def visualize_argument_mining(argument_mining):
    arguments = argument_mining.split("\n")
    arguments = [arg for arg in arguments if arg.strip()]
    data = pd.DataFrame({"Argument": arguments})
    st.bar_chart(data["Argument"].value_counts())

def visualize_conflict_detection(conflict_detection):
    conflicts = conflict_detection.split("\n")
    conflicts = [conflict for conflict in conflicts if conflict.strip()]
    data = pd.DataFrame({"Conflict": conflicts})
    st.bar_chart(data["Conflict"].value_counts())

def visualize_metrics(metrics, title, xlabel, ylabel, palette):
    data = pd.DataFrame(metrics.items(), columns=[xlabel, ylabel])
    st.bar_chart(data.set_index(xlabel))

if (email_content or uploaded_file or uploaded_email_file) and st.button("🔍 Generate Insights"):
    try:
        countdown_timer(5)

        if uploaded_email_file:
            msg = BytesParser(policy=policy.default).parsebytes(uploaded_email_file.getvalue())
            email_content = msg.get_body(preferencelist=('plain')).get_content()
            metadata = extract_email_metadata(uploaded_email_file)

        detected_lang = detect(email_content)
        if detected_lang != "en":
            st.error("⚠️ Only English language is supported.")
        else:
            with st.spinner("⚡ Processing email insights..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_summary = executor.submit(get_ai_response, "Summarize this email concisely:\n\n", email_content) if features["highlights"] else None
                    future_response = executor.submit(get_ai_response, "Generate a professional response to this email:\n\n", email_content) if features["response"] else None
                    future_highlights = executor.submit(get_ai_response, "Highlight key points:\n\n", email_content) if features["highlights"] else None
                    future_tone = executor.submit(get_ai_response, "Detect the tone of this email:\n\n", email_content) if features["tone"] else None
                    future_tasks = executor.submit(get_ai_response, "List actionable tasks:\n\n", email_content) if features["task_extraction"] else None
                    future_subject = executor.submit(get_ai_response, "Suggest a professional subject line:\n\n", email_content) if features["subject_recommendation"] else None
                    future_clarity = executor.submit(get_ai_response, "Rate the clarity of this email:\n\n", email_content) if features["clarity"] else None
                    future_complexity_reduction = executor.submit(get_ai_response, "Explain this email in the simplest way possible:\n\n", email_content) if features["complexity_reduction"] else None

                    scenario_prompt = f"Generate a response for a {selected_scenario.lower()}:\n\n"
                    future_scenario_response = executor.submit(get_ai_response, scenario_prompt, email_content) if features["scenario_responses"] else None

                    attachment_text = analyze_attachment(uploaded_file) if uploaded_file and features["attachment_analysis"] else None
                    future_attachment_analysis = executor.submit(get_ai_response, "Analyze this attachment content:\n\n", attachment_text) if attachment_text else None

                    phishing_links = analyze_phishing_links(email_content) if features["phishing_detection"] else []

                    sensitive_info = detect_sensitive_information(email_content) if features["sensitive_info_detection"] else {}

                    confidentiality = confidentiality_rating(email_content) if features["confidentiality_rating"] else 0

                    future_bias_detection = executor.submit(get_ai_response, "Identify potential biases in this email:\n\n", email_content) if features["bias_detection"] else None
                    future_conflict_detection = executor.submit(get_ai_response, "Detect conflicts in this email thread:\n\n", email_content) if features["conflict_detection"] else None
                    future_argument_mining = executor.submit(get_ai_response, "Analyze the arguments presented in this email:\n\n", email_content) if features["argument_mining"] else None

                    if features["metadata_extraction"] and uploaded_email_file:
                        email_metadata = extract_email_metadata(uploaded_email_file)
                    else:
                        email_metadata = None

                    summary = future_summary.result() if future_summary else None
                    response = future_response.result() if future_response else None
                    highlights = future_highlights.result() if future_highlights else None
                    tone = future_tone.result() if future_tone else None
                    tasks = future_tasks.result() if future_tasks else None
                    subject_recommendation = future_subject.result() if future_subject else None
                    clarity_score = future_clarity.result() if future_clarity else None
                    readability_score = get_readability(email_content)
                    complexity_reduction = future_complexity_reduction.result() if future_complexity_reduction else None
                    scenario_response = future_scenario_response.result() if future_scenario_response else None
                    attachment_analysis = future_attachment_analysis.result() if future_attachment_analysis else None
                    bias_detection = future_bias_detection.result() if future_bias_detection else None
                    conflict_detection = future_conflict_detection.result() if future_conflict_detection else None
                    argument_mining = future_argument_mining.result() if future_argument_mining else None

                if summary:
                    st.subheader("📌 Email Summary")
                    st.write(summary)
                    visualize_metrics({"Summary Length": len(summary)}, "Summary Metrics", "Metric", "Value", "Blues")

                if response:
                    st.subheader("✉️ Suggested Response")
                    st.write(response)
                    visualize_metrics({"Response Length": len(response)}, "Response Metrics", "Metric", "Value", "Greens")

                if highlights:
                    st.subheader("🔑 Key Highlights")
                    st.write(highlights)
                    visualize_metrics({"Highlights Length": len(highlights)}, "Highlights Metrics", "Metric", "Value", "Purples")

                if features["sentiment"]:
                    st.subheader("💬 Sentiment Analysis")
                    sentiment = get_sentiment(email_content)
                    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                    st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")
                    visualize_metrics({"Sentiment Polarity": sentiment}, "Sentiment Metrics", "Metric", "Value", "Oranges")

                if tone:
                    st.subheader("🎭 Email Tone")
                    st.write(tone)
                    visualize_metrics({"Tone Length": len(tone)}, "Tone Metrics", "Metric", "Value", "Reds")

                if tasks:
                    st.subheader("📝 Actionable Tasks")
                    st.write(tasks)
                    visualize_metrics({"Tasks Length": len(tasks)}, "Tasks Metrics", "Metric", "Value", "Greens")

                if subject_recommendation:
                    st.subheader("📬 Subject Line Recommendation")
                    st.write(subject_recommendation)
                    visualize_metrics({"Subject Length": len(subject_recommendation)}, "Subject Metrics", "Metric", "Value", "Blues")

                if clarity_score:
                    st.subheader("🔍 Email Clarity Score")
                    st.write(clarity_score)
                    visualize_metrics({"Clarity Score": clarity_score}, "Clarity Metrics", "Metric", "Value", "Purples")

                if complexity_reduction:
                    st.subheader("🔽 Simplified Explanation")
                    st.write(complexity_reduction)
                    visualize_metrics({"Complexity Reduction Length": len(complexity_reduction)}, "Complexity Metrics", "Metric", "Value", "Oranges")

                if scenario_response:
                    st.subheader("📜 Scenario-Based Suggested Response")
                    st.write(f"**{selected_scenario}:**")
                    st.write(scenario_response)
                    visualize_metrics({"Scenario Response Length": len(scenario_response)}, "Scenario Metrics", "Metric", "Value", "Reds")

                if attachment_analysis:
                    st.subheader("📎 Attachment Analysis")
                    st.write(attachment_analysis)
                    visualize_metrics({"Attachment Analysis Length": len(attachment_analysis)}, "Attachment Metrics", "Metric", "Value", "Greens")

                if phishing_links:
                    st.subheader("⚠️ Phishing Links Detected")
                    st.write(phishing_links)
                    visualize_metrics({"Number of Phishing Links": len(phishing_links)}, "Phishing Metrics", "Metric", "Value", "Reds")

                if sensitive_info:
                    st.subheader("⚠️ Sensitive Information Detected")
                    st.json(sensitive_info)
                    visualize_metrics({"Number of Sensitive Info Types": len(sensitive_info)}, "Sensitive Info Metrics", "Metric", "Value", "Blues")

                if confidentiality:
                    st.subheader("🔐 Confidentiality Rating")
                    st.write(f"Confidentiality Rating: {confidentiality}/5")
                    visualize_metrics({"Confidentiality Rating": confidentiality}, "Confidentiality Metrics", "Metric", "Value", "Purples")

                if bias_detection:
                    st.subheader("⚖️ Bias Detection")
                    st.write(bias_detection)
                    visualize_metrics({"Bias Detection Length": len(bias_detection)}, "Bias Metrics", "Metric", "Value", "Oranges")

                if conflict_detection:
                    st.subheader("🚨 Conflict Detection")
                    st.write(conflict_detection)
                    visualize_conflict_detection(conflict_detection)

                if argument_mining:
                    st.subheader("💬 Argument Mining")
                    st.write(argument_mining)
                    visualize_argument_mining(argument_mining)

                if email_metadata:
                    st.subheader("📅 Email Metadata")
                    metadata_df = pd.DataFrame(list(email_metadata.items()), columns=["Field", "Value"])
                    st.table(metadata_df)
                    visualize_metrics({"Number of Metadata Fields": len(email_metadata)}, "Metadata Metrics", "Metric", "Value", "Reds")

                if features["export"]:
                    export_data = {
                        "summary": summary,
                        "response": response,
                        "highlights": highlights,
                        "clarity_score": clarity_score,
                        "complexity_reduction": complexity_reduction,
                        "scenario_response": scenario_response,
                        "attachment_analysis": attachment_analysis,
                        "phishing_links": phishing_links,
                        "sensitive_info": sensitive_info,
                        "confidentiality": confidentiality,
                        "bias_detection": bias_detection,
                        "conflict_detection": conflict_detection,
                        "argument_mining": argument_mining,
                        "metadata": email_metadata,
                    }
                    export_json = json.dumps(export_data, indent=4)
                    st.download_button("📥 Download JSON", data=export_json, file_name="analysis.json", mime="application/json")

                    pdf_data = export_pdf(json.dumps(export_data, indent=4))
                    st.download_button("📥 Download PDF", data=pdf_data, file_name="analysis.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("✏️ Paste email content and click 'Generate Insights' to begin.")
