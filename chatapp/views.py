''' Imports Used in the Program '''

from django.http import HttpResponse
import boto3
from django.conf import settings
import PyPDF2 as pdf
from io import BytesIO
import json
from django.views.decorators.csrf import csrf_exempt
import csv
import nltk
from docx import Document
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import PyPDF2 as pdf
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from polyfuzz import PolyFuzz
from nltk import word_tokenize, sent_tokenize
import language_tool_python  
my_tool =language_tool_python.LanguageTool('en-US')  

chunk_size = 500
arrayFilesName = []
newDictionaryData ={}
dictionary={}

''' Main Function For Extracting Text, Calling Functions and Returning Results...'''

@csrf_exempt
def getting_details(request):

    # User Input -->
    if request.method == 'POST':
        user_input = request.POST.get('question')

        # S3 Files Access -->

        s3 = boto3.client('s3', aws_access_key_id=settings.AWS_S3_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY)
        response_files_name = s3.list_objects_v2(Bucket=settings.AWS_STORAGE_BUCKET_NAME)
        if 'Contents' in response_files_name:
             for obj in response_files_name['Contents']:
                object_key = obj['Key']
                print(f"Object Key: {object_key}")
                arrayFilesName.append(object_key)

        # Extracting Data from Files -->
        fullData = ''
        for I in arrayFilesName:
            file_response = s3.get_object(Bucket='dl-chat-bucket',Key=I)
            file_extension = file_response['ContentType']
            pdf_data = file_response['Body'].read()
            text=''
            if file_extension == 'text/plain':
                text = pdf_data.decode('utf-8')
            elif file_extension == 'text/csv':
                text = ""
                csv_data = pdf_data.decode('utf-8').splitlines()
                reader = csv.reader(csv_data)
                for row in reader:
                    text += ','.join(row) + '\n'
            elif file_extension == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                document = Document(BytesIO(pdf_data))
                text = '\n'.join([p.text for p in document.paragraphs])
            
            elif file_extension == 'application/pdf':
                pdf_reader = pdf.PdfReader(BytesIO(pdf_data))
                num_pages = len(pdf_reader.pages)
                text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    text=text.lower()
            else:
                text = "Unsupported file type"
            fullData+=text

        # implement cleaning function to clean the text
        cleaned_sentences=text_cleaning(fullData)
        tokens= chunk_text(cleaned_sentences, chunk_size)
        cleaned_sentences_with_stopwords=[remove_stop_words(sentences) for sentences in tokens]
       
        # clean teh user input
        clean_question=text_cleaning(user_input)
        question=remove_stop_words(clean_question)
        
        # convert strings into vector form
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences_with_stopwords)
        input_vector = vectorizer.transform([question])
        
        # find the similarity 
        similarity_scores = tfidf_matrix.dot(input_vector.T).toarray().flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Sorting in descending order
        similarity_threshold = 0.3
        similar_sentences = [tokens[i] for i in sorted_indices if similarity_scores[i] > similarity_threshold]
        
        # get the answer
        final_answer = ""
        for sentence in similar_sentences:
            if similarity_scores[tokens.index(sentence)] > similarity_threshold:
                final_answer = sentence
                
        if final_answer == "":
            model = PolyFuzz("TF-IDF")
            model.match([question],cleaned_sentences_with_stopwords)
            result=model.get_matches()
            pd.set_option('display.max_colwidth', None)
            answer=result['To']
            
            # get the answer
            final_answer=''
            for ans in answer:
                for idx, tokenized_sentence in enumerate(tokens):
                    if ans in cleaned_sentences_with_stopwords[idx]:
                        final_answer=' '.join(tokenized_sentence.split())
        final_output=my_tool.correct(final_answer)
        response_data = {'answer': final_output.title()}
        return HttpResponse(json.dumps(response_data), content_type='application/json')


def text_cleaning(text):
  sentence = text.lower()
  sentence = re.sub(r'[^a-z0-9\s]', '', sentence)                                                 # REMOVE THE EXTRA SPACES
  return sentence

# make a function for removing the unneccsary words from the text
def remove_stop_words(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def chunk_text(text, chunk_size):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        else:
            while end < len(text) and text[end] != ' ':
                end -= 1
        chunks.append(text[start:end])
        start = end + 1
    return chunks

