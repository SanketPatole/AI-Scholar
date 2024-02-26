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

def get_notes(summaries, llm, age, subject):
    notes_schema = []
    for i in range(1, 6):
        notes_schema.append(ResponseSchema(name=f"note_{i}", description=f"Note number {i} in output, strictly keeping input short notes in mind."))
    output_parser = StructuredOutputParser.from_response_schemas(notes_schema)
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
    You will be provided with a list short exam revision notes generated
    on different parts of a chapter extracted from a textbook
    for the subject of {subject}.
    The short notes are included within {delimiter} delimiter.
    The textbook is intended for the students of age {age}.
    You are supposed to build exactly 5 pointer exam notes for the
    student keeping only the short notes in mind.
    
    < SHORT NOTES >
    
    {delimiter}{summaries}{delimiter}
    
    {instructions}
    
    < NOTE >
  Revision notes must explain some topics in the chapter.
  It must not have any action item for the student.
  For example: "Read works by Gabriel Garcia Marquez" is 
  not an exam revision note because it contains action item for a student.
    """
    template = PromptTemplate(template=prompt_template_text,
            input_variables=["delimiter", "age", "instructions",
            "subject", "summaries"])
    chain = LLMChain(llm=llm, prompt=template)
    prompt_inputs = {"delimiter": "###", "age": age, "subject": subject,
    "instructions": instructions, "summaries": summaries}
    response = chain(prompt_inputs, return_only_outputs=True)
    return output_parser.parse(response["text"])

def get_topic_summary(llm, subject, age, context):
    summary = ResponseSchema(name="summary", description="Exactly 5 pointer exam revision notes")
    output_parser = StructuredOutputParser.from_response_schemas([summary])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with some part of a chapter from a textbook for the subject of {subject},
  delimited within {delimiter} delimiter.
  The textbook is intended towards students of age {age}.
  Please write a 3 pointer exam revision notes strictly keeping the chapter part in mind.
  
  <CHAPTER PART >
  {delimiter}{context}{delimiter}

  {instructions}

  Let me remind again. Output should have 3 pointer exam notes.
  And notes should explain some topics in the chapter.
  It must not have any action item for student.
  For example: "Read works by Gabriel Garcia Marquez" is 
  not an exam revision note because it contains action item for a student.
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "delimiter",
                    "instructions", "age", "subject"])
    prompt_inputs = {"input_documents": context, "delimiter": "###",
    "instructions": instructions, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict['summary']

def lambda_handler(event, context):
    new_event = event['queryStringParameters']
    cls = new_event['class']
    subject = new_event['subject']
    chapter = new_event['chapter'].split(" ")[1]
    if len(chapter) == 1:
        chapter = "0" + chapter
    chatgpt3_5_turbo = ChatOpenAI(model="gpt-3.5-turbo")
    llm = chatgpt3_5_turbo
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    s3 = boto3.client('s3')
    bucket_name = os.environ['BUCKET_NAME']
    file_key = f"{cls}/{subject}/{chapter}/"
    download_s3_folder(s3, bucket_name, file_key, '/tmp/vector/')
    new_db = FAISS.load_local('/tmp/vector/', embedding)
    question = "What all topics are present in this chapter?"
    contexts = new_db.similarity_search(question, k=3)
    summary_list = []
    summaries = ""
    for context in contexts:
        summary = get_topic_summary(llm, subject, int(cls)+6, [context])
        summary_list.append(summary)
    for i in range(len(summary_list)):
        #summaries += f"\n\nSummary number {i+1}\n"
        summaries += "\n" + summary_list[i].replace(".", ".\n")
    final_notes = []
    index = 1
    notes = get_notes(summaries, llm, int(cls)+6, subject)
    for key in notes:
        final_notes.append({'index': index, "note": notes[key]})
        index += 1
    notes = [{'notes': notes[key]} for key in notes]
    return {'statusCode': 200, 'body': json.dumps(notes)}
