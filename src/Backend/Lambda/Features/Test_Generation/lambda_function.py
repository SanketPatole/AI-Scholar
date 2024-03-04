import json
import os
import string
import random
import boto3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import concurrent.futures

def download_s3_folder(s3_client, bucket_name, s3_folder, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith("/"):
                local_file_path = os.path.join(local_dir, key[len(s3_folder):])
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)
                s3_client.download_file(bucket_name, key, local_file_path)

def get_question_100_words(llm, subject, age, context):
    question_100_words = ResponseSchema(name="question_100_words", description="A question keeping only the provided context in mind. It must be only one question.")
    output_parser = StructuredOutputParser.from_response_schemas([question_100_words])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with some part of a chapter from a textbook for the subject of {subject},
  delimited within {delimiter} delimiter. The textbook is intended towards students of age {age}.
  Please generate a question based upon only the context provided.
  The question must require students at least 100 words to solve.
  Please ask only one  question.
  Keep in mind that students will not have access to the text/excerpt.
  
  <CHAPTER PART >
  {delimiter}{context}{delimiter}
  
  Let me remind again.
  Please generate a question based upon only the context provided.
  The question must require students at least 100 words to solve.
  Please ask only one  question.
  Keep in mind that students will not have access to the text/excerpt.

  {instructions}
  
  < NOTE >
  Following is an example of a wrong question.
  "Based on the excerpt provided, discuss the significance of dreams and superstitions in the story. How do dreams play a role in the survival of"
  The question is wrong because of 2 reasons.
  1. Students do not have access to the text/excerpt.
  2. The question is incomplete.
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "delimiter",
                    "instructions", "age", "subject"])
    prompt_inputs = {"input_documents": context, "delimiter": "###",
    "instructions": instructions, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict

def get_question_40_words(llm, subject, age, context):
    question_40_words = ResponseSchema(name="question_40_words", description="A question keeping only the provided context in mind. It must be only one question.")
    output_parser = StructuredOutputParser.from_response_schemas([question_40_words])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with some part of a chapter from a textbook for the subject of {subject},
  delimited within {delimiter} delimiter. The textbook is intended towards students of age {age}.
  Please generate a question based upon only the context provided.
  The question must require students at most 40 words to solve.
  Please ask only one  question.
  Keep in mind that students will not have access to the text/excerpt.
  
  <CHAPTER PART >
  {delimiter}{context}{delimiter}
  
  Let me remind again.
  Please generate a question based upon only the context provided.
  The question must require students at most 40 words to solve.
  Please ask only one  question.
  Keep in mind that students will not have access to the text/excerpt.

  {instructions}
  
  < NOTE >
  Following is an example of a wrong question.
  "Based on the excerpt provided, discuss the significance of dreams and superstitions in the story. How do dreams play a role in the survival of"
  The question is wrong because of 2 reasons.
  1. Students do not have access to the text/excerpt.
  2. The question is incomplete.
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "delimiter",
                    "instructions", "age", "subject"])
    prompt_inputs = {"input_documents": context, "delimiter": "###",
    "instructions": instructions, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict

def get_true_false_question(llm, subject, age, context):
    question = ResponseSchema(name="question", description="A true/false question keeping only the provided context in mind. Do not return answer with the question.")
    output_parser = StructuredOutputParser.from_response_schemas([question])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with some part of a chapter from a textbook for the subject of {subject},
  delimited within {delimiter} delimiter. The textbook is intended towards students of age {age}.
  Please generate a true/false question based upon only the context provided.
  Please ask only one  question. Do not return answer with the question.
  Keep in mind that students will not have access to the text/excerpt.
  
  <CHAPTER PART >
  {delimiter}{context}{delimiter}
  
  Let me remind again.
  Please generate a true/false question based upon only the context provided.
  Please ask only one  question. Do not return answer with the question.
  Keep in mind that students will not have access to the text/excerpt.

  {instructions}
  
  < NOTE >
  Following is an example of a wrong question.
  "Based on the excerpt provided, the woman in the story had never thought highly of her dreams as a means of survival. (True/False).
  1. Students do not have access to the text/excerpt.
  2. The question must not have (true/false) mentioned in the question.
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "delimiter",
                    "instructions", "age", "subject"])
    prompt_inputs = {"input_documents": context, "delimiter": "###",
    "instructions": instructions, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict

def get_fill_in_the_blanks(llm, subject, age, context):
    fill_in_the_blank_statement = ResponseSchema(name="fill_in_the_blank_statement", description="Fill in the blank statement with exactly one blank ('____') word or phrase. Statement must be 10 words long.")
    option_1 = ResponseSchema(name="option_1", description="First option for fill in the blanks question.")
    option_2 = ResponseSchema(name="option_2", description="Second option for fill in the blanks question.")
    option_3 = ResponseSchema(name="option_3", description="Third option for fill in the blanks question.")
    output_parser = StructuredOutputParser.from_response_schemas([fill_in_the_blank_statement, option_1, option_2, option_3])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with some part of a chapter from a textbook for the subject of {subject},
  delimited within {delimiter} delimiter. The textbook is intended towards students of age {age}.
  Please generate a fill in the blank statement using provided chapter-part as a context in mind.
  Statement must be 10 words long.
  The blank word or phrase must exactly be "____".
  Exactly one word or phrase in the statement must be blank ("____").
  You must provide 3 options along with the statement.
  
  <CHAPTER PART >
  {delimiter}{context}{delimiter}
  
  Let me remind again.
  Statement must be 10 words long.
  The blank word or phrase must exactly be "____".
  Exactly one word or phrase in the statement must be blank ("____").

  {instructions}
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "delimiter",
                    "instructions", "age", "subject"])
    prompt_inputs = {"input_documents": context, "delimiter": "###",
    "instructions": instructions, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict

def get_questions(contexts, llm_100tok, subject, cls, type_of_question):
    if type_of_question == 'fill_in_the_blanks':
        fill_in_the_blanks_list = []
        for context in contexts[:5]:
            question = get_fill_in_the_blanks(llm_100tok, subject, int(cls)+6, [context])
            question = [{'option' if 'option' in key else key: question[key]} for key in question]
            fill_in_the_blanks_list.append(question)
        return fill_in_the_blanks_list
    elif type_of_question == 'true_false':
        true_false_list = []
        for context in contexts[5:10]:
            question = get_true_false_question(llm_100tok, subject, int(cls)+6, [context])
            true_false_list.append(question)
        return true_false_list
    elif type_of_question == 'question_100_words':
        question_100_words_list = []
        for i in range(0, 6, 2):
            question = get_question_100_words(llm_100tok, subject, int(cls)+6, contexts[i:i+2])
            question_100_words_list.append(question)
        return question_100_words_list
    elif type_of_question == 'question_40_words':
        question_40_words_list = []
        for i in range(6, 10, 2):
            question = get_question_40_words(llm_100tok, subject, int(cls)+6, contexts[i:i+2])
            question_40_words_list.append(question)
        return question_40_words_list

def lambda_handler(event, context):
    new_event = event['queryStringParameters']
    cls = new_event['class']
    subject = new_event['subject']
    chapter = new_event['chapter'].split(" ")[1]
    if len(chapter) == 1:
        chapter = "0" + chapter
    chatgpt3_5_turbo_100tok = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=100)
    llm_100tok = chatgpt3_5_turbo_100tok
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    s3 = boto3.client('s3')
    bucket_name = os.environ['BUCKET_NAME']
    file_key = f"{cls}/{subject}/{chapter}/"
    download_s3_folder(s3, bucket_name, file_key, '/tmp/vector/')
    new_db = FAISS.load_local('/tmp/vector/', embedding)
    db_query = "What all topics are present in this chapter?"
    contexts = new_db.similarity_search(db_query, k=10)
    questions = {'fill_in_the_blanks': [], 'true_false': [], 'question_100_words': [], 'question_40_words': []}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_key = {
            executor.submit(get_questions, contexts, llm_100tok, subject, cls, key): key
            for key in questions.keys()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                questions[key] = future.result()
            except Exception as exc:
                print(f'{key} generated an exception: {exc}')

    return {'statusCode': 200,
            "headers": {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": True},
            'body': json.dumps(questions)}
