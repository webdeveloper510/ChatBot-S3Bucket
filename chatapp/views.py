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
from nltk.corpus import stopwords
import nltk
import string
from nltk.corpus import stopwords
import PyPDF2 as pdf
import re
from keybert import KeyBERT
import pandas as pd
from polyfuzz import PolyFuzz

''' Global Variables'''

chunk_size = 500
overlap_size = 100
arrayFilesName = []
newDictionaryData ={}
dictionary={}

''' Main Function For Extracting Text, Calling Functions and Returning Results...'''

@csrf_exempt
def getting_details(request):

    # User Input -->
    if request.method == 'POST':
        question = request.POST.get('question')

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

        # implement cleaning function to clean the text --> 

        cleaned_sentences=text_cleaning(fullData)

        # divide text into sentences and small chunk size of piece -->

        tokens= chunk_text(cleaned_sentences, chunk_size, overlap_size)

        # Tokenizing Function --> 

        answer = token_summary(question , tokens)
        answer = answer.to_string()
        response_data = {'answer': answer}
        return HttpResponse(json.dumps(response_data), content_type='application/json')




def text_cleaning(text):
  punch_to_remove=string.punctuation
  input_text=text.lower()                                                                   # LOWER THE TEXT
  input_text=''.join(char for char in input_text if char not in punch_to_remove)            # REMOVE PUNCTUTATION
  input_text=''.join(word for word in input_text if not word.isdigit())                     # REMOVE THE NUMBER
  input_text=' '.join(input_text.split())                                                   # REMOVE THE EXTRA SPACES
  input_text=re.sub(r'(!)1+','',input_text)                                                       # REM0VE THE REPITATION OF THE PUNCTUATION
  return input_text


# make a function for removing the unneccsary words from the text
def stop_words(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def chunk_text(text, chunk_size, overlap_size):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        start += chunk_size - overlap_size
    return chunks



def token_summary(user_input,tokens):
    cleaned_sentences_with_stopwords=[stop_words(sentences) for sentences in tokens]

    for i , value in enumerate(tokens):
        dictionary[i]=value

    extracted_labels = []
    kw_model = KeyBERT()
    for value in cleaned_sentences_with_stopwords:
        keywords = kw_model.extract_keywords(value)
        keyword= kw_model.extract_keywords(value, keyphrase_ngram_range=(1, 2), stop_words='english',top_n=2)
        label = ' '.join(k[0] for k in keyword)
        extracted_labels.append(label)

    result_list = []
    for item in extracted_labels:
        words = item.split() # Split the item into words
        unique_words = list(set(words)) # Create a set to remove duplicates and convert back to list
        cleaned_item = ' '.join(unique_words) # Join the unique words back together
        result_list.append(cleaned_item)


    count = 0
    for key, value in dictionary.items():
        newDictionaryData[result_list[count]]=value
        count +=1
        
        clean_question=text_cleaning(user_input)
        question=stop_words(clean_question)

        model = PolyFuzz("TF-IDF")
        model.match([question],tokens)
        result=model.get_matches()
        pd.set_option('display.max_colwidth', None)
        answer=result['To']
        return answer

