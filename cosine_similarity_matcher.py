from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from docx import Document
import json

# Read .doc file


def read_doc_file(file_path):
    doc = Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Read .txt file


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


resume_text = read_doc_file("resume.docx")
job_desc_text = read_txt_file("job_desc.txt")

nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing


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

    return word_tokens


resume_tokens = preprocess(resume_text)
job_desc_tokens = preprocess(job_desc_text)

# Feature Extraction
# Combine all documents for TF-IDF
documents = []
documents.append(' '.join(resume_tokens))
documents.append(' '.join(job_desc_tokens))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Get the feature vectors for the resume and job description
resume_vector = X[0]
job_desc_vector = X[1]

# Calculate cosine similarity
similarity = cosine_similarity(resume_vector, job_desc_vector)[0][0]

# Calculate percentage match
percentage_match = round(similarity * 100, 2)

# Extract skills from the job description
job_desc_skills = set(job_desc_tokens)

# Extract skills from the resume
resume_skills = set(resume_tokens)

# Find the matched skills
matched_skills = job_desc_skills.intersection(resume_skills)

# Create output dictionary
output = {
    # Update with your actual job description
    "Job Description": "Software Engineer",
    "Resumes": [
        {
            "Identifier": "resume.docx",  # Update with the identifier or name of the resume
            "Similarity Score": similarity,
            "Rank": 1,
            "Percentage Match": percentage_match,
            "Job Description Skills": list(job_desc_skills),
            "Resume Skills": list(resume_skills),
            "Matched Skills": list(matched_skills)
        }
    ]
}

# Convert output to JSON format
output_json = json.dumps(output, indent=4)

# Write output to a text file
with open("output1.txt", "w") as file:
    file.write(output_json)

print("Output written to output.txt file.")
