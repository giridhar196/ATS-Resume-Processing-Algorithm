# ATS Resume Processing Algorithm

## Description
This repository contains two Python scripts used for matching resumes to a job description. Both scripts use natural language processing techniques to clean and preprocess the text data, then apply different methods for measuring the similarity between the job description and the resumes.

1. `knn_resume_matcher.py`: This script uses a K-Nearest Neighbors (KNN) model and a TF-IDF vectorizer to match resumes to a job description based on cosine similarity.

2. `cosine_similarity_matcher.py`: This script directly calculates the cosine similarity between the TF-IDF feature vectors of a single resume and the job description.

Both scripts output the result in a JSON format, detailing the similarity score, percentage match, and matched skills for each resume. 

## Dependencies
This project uses the following Python libraries:
- Pandas
- Scikit-learn
- NLTK
- Python-docx
- JSON

To install these dependencies, run the following command in your terminal:
```
pip install -r requirements.txt
```

In addition, you need to download NLTK's `punkt` and `stopwords` data using the following commands in a Python interpreter:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage
Before you can run the scripts, you need to have your data ready. The job description should be a .txt file, while the resumes can either be in .docx or .xlsx format. 

**`knn_resume_matcher.py`:**

To use this script, first ensure your resumes are in an Excel file where the 'Content' column contains the resume content and the 'Title' column contains a unique identifier or name for each resume.

Your directory should look like this:
```
|--- resume.xlsx
|--- job_desc.txt
|--- knn_resume_matcher.py
```
Run the script and the output will be a .txt file named `outputknnxl.txt`, which will contain the ranked resumes.

**`cosine_similarity_matcher.py`:**

To use this script, ensure your single resume is in a .docx file.

Your directory should look like this:
```
|--- resume.docx
|--- job_desc.txt
|--- cosine_similarity_matcher.py
```
Run the script and the output will be a .txt file named `output1.txt`.

## Contributing
Feel free to fork this project and make your own changes. If you have any suggestions or any issues, please create a new issue in the GitHub repository.

## License
This project is licensed under the MIT License.

