import torch
import PyPDF2 as pdf
import pdfplumber
import re
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
max_length = model.config.max_position_embeddings  # Account for [CLS] and [SEP] tokens

pdf_path="/home/codenomad/Desktop/machine_learning_model/InterviewQuestions.pdf"

def pdf_reader(pdf_path):
    file=open(pdf_path,"rb")
    doc=pdf.PdfReader(file)
    page_number=len(doc.pages)
    extracted_text=" "
    for i in range(page_number):
        current_page=doc.pages[i]
        text=current_page.extract_text()
        extracted_text += text.strip() + " "  # Remove leading/trailing spaces and add a space between pages
    extracted_text = " ".join(extracted_text.split())  
    return extracted_text


def question_answer(question,text):
    input_ids=tokenizer.encode(question,text,max_length=max_length,truncation=True)
    sep_index=input_ids.index(tokenizer.sep_token_id)
    num_seg_a=sep_index+1
    num_seg_b=len(input_ids)-num_seg_a
    segment_ids=[0]*num_seg_a+[1]*num_seg_b
    assert len(segment_ids)==len(input_ids)
    outputs = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]),return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start=torch.argmax(start_scores)
    answer_end=torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    answer=tokens[answer_start]
    for i in range(answer_start+1,answer_end+1):
        if tokens[i][0:2]=='##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]
    return answer


text=pdf_reader(pdf_path)
question="List is mutable or not"
answer=question_answer(question,text)
print("Answer-------->>",answer)

