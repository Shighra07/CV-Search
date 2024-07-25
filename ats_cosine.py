import streamlit as st
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_reader):
  text = ""
  for page in file_reader.pages:
    text += page.extract_text()
  return text

def extract_text_from_docx(file):
  return docx2txt.process(file)

def extract_text_from_txt(file):
  return file.read().decode("utf-8")  # Decode for proper text handling

def extract_text(file):
  if isinstance(file, str):  # Handle string file path
    if file.endswith('.pdf'):
      with open(file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return extract_text_from_pdf(reader)
    elif file.endswith('.docx'):
      return extract_text_from_docx(file)
    elif file.endswith('.txt'):
      with open(file, 'r') as f:
        return extract_text_from_txt(f)
    else:
      return ""
  else:  # Handle uploaded file object
    if file.name.endswith('pdf'):
      reader = PyPDF2.PdfReader(file)
      return extract_text_from_pdf(reader)
    elif file.name.endswith('docx'):
      return extract_text_from_docx(file)
    elif file.name.endswith('txt'):
      return extract_text_from_txt(file)  # Use stream for uploaded files
    else:
      return ""

st.title('Resume Matcher')

job_description = st.text_area('Enter Job Description')
uploaded_resumes = st.file_uploader('Upload Resumes (PDF, DOCX, TXT)',type=['pdf','docx','txt'],accept_multiple_files=True) #upload multiple files

if st.button('Match Resumes'):
  if not uploaded_resumes or not job_description:
    st.error("Please upload resumes and enter a job description.")
  else:
    resumes = []
    for resume in uploaded_resumes:
      resumes.append(extract_text(resume))

    # Vectorize job description and resumes
    vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
    vectors = vectorizer.toarray()

    # Calculate cosine similarities
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_vector], resume_vectors)[0]

    # Get top 3 resumes and their similarity scores
    # top_indices = similarities.argsort()[-3:][::-1]
    # top_resumes = [uploaded_resumes[i].name for i in top_indices]
    # similarity_scores = [round(similarities[i], 2) for i in top_indices]

    # st.subheader('Top Matching Resumes:')
    # for i, (resume, score) in enumerate(zip(top_resumes, similarity_scores)):
    #   st.write(f"{i+1}. {resume} (Similarity Score: {score})")

    top_indices = similarities.argsort()[-3:][::-1]  # Get top 3 indices (adjustable)
    threshold = 0.3  # Adjust this value to your desired threshold

    top_resumes = [uploaded_resumes[i].name for i in top_indices if similarities[i] > threshold]
    similarity_scores = [round(similarities[i], 2) for i in top_indices if similarities[i] > threshold]

    # Check if any resumes meet the threshold
    if top_resumes:
      st.subheader('Top Matching Resumes:')
      for i, (resume, score) in enumerate(zip(top_resumes, similarity_scores)):
        st.write(f"{i+1}. {resume} (Similarity Score: {score})")
    else:
      st.write("No resumes found with a similarity score above", threshold)
  

