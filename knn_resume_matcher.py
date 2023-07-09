import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from docx import Document
import nltk
import os
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')


def read_doc_file(file_path):
    doc = Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    word_tokens = word_tokenize(text)

    # Remove punctuations
    word_tokens = [word for word in word_tokens if word.isalnum()]

    # Remove stopwords and common words
    stop_words = set(stopwords.words('english'))
    # Add more common words to remove if needed
    common_words = set(['from', 'what', 'and'])
    word_tokens = [
        word for word in word_tokens if not word in stop_words and not word in common_words]

    # Stemming
    ps = PorterStemmer()
    word_tokens = [ps.stem(word) for word in word_tokens]

    return ' '.join(word_tokens)


# Load and preprocess job description
job_desc_text = read_txt_file("job_desc.txt")
job_desc_processed = preprocess(job_desc_text)
job_desc_skills = set(job_desc_processed.split())

# Load and preprocess resumes
df = pd.read_excel("resume.xlsx")
df['Processed_Content'] = df['Content'].apply(preprocess)

# Combine all documents for TF-IDF
documents = [job_desc_processed] + df['Processed_Content'].tolist()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Separate job description vector from resume vectors
job_desc_vector = X[0]
resume_vectors = X[1:]

# Create and train the KNN model
knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
knn_model.fit(resume_vectors)

# Use the KNN model to find the k most similar resumes
distances, indices = knn_model.kneighbors(job_desc_vector.toarray())

# Create output dictionary
output = {
    "Job Description": "Software Engineer",
    "Resumes": []
}

# Add the resumes to the output
for i in range(len(indices[0])):
    # Get the resume details
    resume_text = df['Processed_Content'].iloc[indices[0][i]]
    resume_skills = set(resume_text.split())
    matched_skills = job_desc_skills.intersection(resume_skills)

    output["Resumes"].append({
        "Identifier": df['Title'].iloc[indices[0][i]],
        "Similarity Score": 1 - distances[0][i],
        "Rank": i+1,
        "Percentage Match": (1 - distances[0][i]) * 100,
        "Job Description Skills": list(job_desc_skills),
        "Resume Skills": list(resume_skills),
        "Matched Skills": list(matched_skills)
    })

# Convert output to JSON format
output_json = json.dumps(output, indent=4)

# Write output to a text file
with open("outputknnxl.txt", "w") as file:
    file.write(output_json)

print("Output written to output.txt file.")
