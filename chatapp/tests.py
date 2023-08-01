# from django.test import TestCase

# # Create your tests here.

# #https://colab.research.google.com/drive/1z-Zl2hftMrFXabYfmz8o9YZpgYx6sGeW?usp=sharing#scrollTo=iFSyHQOoKJEI

import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline

def get_soup(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup

def clean_wiki_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace('[edit]', "")
    return text

def get_paragraph_text(p):
    paragraph_text = ""
    for tag in p.children:
        paragraph_text = paragraph_text + tag.text
    return paragraph_text

def get_wiki_text(url):
    soup = get_soup(url)
    headers = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    wiki_extract = []
    for tag in soup.find_all():
        if tag.name in headers and tag.text != "Contents":
            p = " "
            for ne in tag.next_elements:
                if ne.name == 'p':
                    p = p + get_paragraph_text(ne)
                if ne.name in headers:
                    break

            if p != "":
                section = [clean_wiki_text(tag.text), tag.name, clean_wiki_text(p)]
                wiki_extract.append(section)

    # Concatenate the text from each section into a single string
    full_text = " ".join(section[2] for section in wiki_extract)
    return full_text

url = "https://cloud.google.com/learn/what-is-artificial-intelligence"
wiki_extract = get_wiki_text(url)
question = "what is unsupervisd learning"

question_answerer = pipeline("question-answering", model='deepset/roberta-base-squad2')
result = question_answerer(question=question, context=wiki_extract,max_answer_len=128)
print("context--------------->>>",wiki_extract)
print('\n')
print('Result--------------->>>>', result)
