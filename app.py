
import streamlit as st

import pickle
import re
import nltk
from PyPDF2 import PdfReader
import docx 
import base64
# === STEP 1 & 2: Base64 background image from local file ===
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f'''
    <style>
    .stApp {{
        background-image: url("data:res.jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

# For background image
set_background("res.jpg")  
st.markdown("""
    <style>
        body {
            background-color: #0000FF;
            color: white; /* To make text readable on blue background */
            font-family: Arial, sans-serif;
        }
        .css-1v0mbdj {
            background-color: #0000FF !important; /* Ensuring Streamlit's default styles don't override */
        }
    </style>
""", unsafe_allow_html=True)

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Function to check if resume is poor
def is_poor_resume(text):
    word_count = len(text.split())
    keywords = ["education", "experience", "skills", "projects"]

    # Basic rules
    if word_count < 50:
        return True
    if not any(keyword.lower() in text.lower() for keyword in keywords):
        return True
    return False

# Web interface
def main():
    st.title("AI Resume Screening")
    st.write("Upload your resume and get the predicted job category.")
    upload_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    if upload_file is not None:
        try:
            if upload_file.name.endswith(".pdf"):
                with open("temp_resume.pdf", "wb") as f:
                    f.write(upload_file.getbuffer())
                resume_text = extract_text_from_pdf("temp_resume.pdf")
            elif upload_file.name.endswith(".docx"):
                resume_text = extract_text_from_docx(upload_file)
            else:
                resume_bytes = upload_file.read()
                resume_text = resume_bytes.decode('utf-8')
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # Check if resume is poor
        if is_poor_resume(resume_text):
            st.warning("⚠️ Your resume seems to have very little information or missing important sections. Please upload a better version for accurate prediction.")
        else:
            cleaned_resume = tfidfd.transform([resume_text])
            prediction_id = clf.predict(cleaned_resume)[0]

            categoryMapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "Dotnet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate"
            }
            category_name = categoryMapping.get(prediction_id, "Unknown Category")

            st.success(f"✅ Category : {category_name}")

if __name__ == "__main__":
    main()
