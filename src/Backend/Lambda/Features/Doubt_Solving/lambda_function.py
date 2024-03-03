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

def is_question_appropriate(question, llm, age):
    is_appropriate = ResponseSchema(name="is_appropriate", description="Is the question appropriate as per the age? {Yes/No}")
    output_parser = StructuredOutputParser.from_response_schemas([is_appropriate])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
    Following is the question asked by a student with age {age}
    The question is delimited with {delimiter}
    Your task is to identify if the question is appropriate based upon the age of the student.
    {delimiter}{question}{delimiter}
    {instructions}
    
    """
    template = PromptTemplate(template=prompt_template_text,
            input_variables=["delimiter", "age",
            "instructions", "question"])
    chain = LLMChain(llm=llm, prompt=template)
    prompt_inputs = {"delimiter": "###", "age": age,
    "instructions": instructions, "question": question}
    response = chain(prompt_inputs, return_only_outputs=True)
    return output_parser.parse(response["text"])['is_appropriate']

def get_answer(question, llm, subject, age, context):
    answer = ResponseSchema(name="answer", description="Answer to the question in 2 sentences, keeping context in mind")
    output_parser = StructuredOutputParser.from_response_schemas([answer])
    instructions = output_parser.get_format_instructions()
    prompt_template_text = """
  You will be provided with a question (delimited with {question_delimiter}) related to the {subject} subject asked by a student of age {age}.
  You will also be provided with a relevant context from the textbook delimited with {context_delimiter}.

  Please answer the question keeping only the context in mind.
  Answer must be two sentences long.


  {question_delimiter}{question}{question_delimiter}

  

  {context_delimiter}{context}{context_delimiter}

  {instructions}

  Let me remind again. Answer must be two sentences long.
    
    """
    prompt = PromptTemplate(template=prompt_template_text,
    input_variables=["context", "context_delimiter", "question_delimiter",
    "instructions", "question", "age", "subject"])
    prompt_inputs = {"input_documents": context, "context_delimiter": "###",
    "question_delimiter": "$$$", "instructions": instructions,
    "question": question, "age": age, "subject": subject}
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    response = chain(prompt_inputs, return_only_outputs=True)
    response_dict = output_parser.parse(response["output_text"])
    return response_dict['answer']

def lambda_handler(event, context):
    new_event = event['queryStringParameters']
    cls = new_event['class']
    subject = new_event['subject']
    chapter = new_event['chapter'].split(" ")[1]
    if len(chapter) == 1:
        chapter = "0" + chapter
    question = new_event['message']
    chatgpt3_5_turbo = ChatOpenAI(model="gpt-3.5-turbo")
    llm = chatgpt3_5_turbo
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    s3 = boto3.client('s3')
    bucket_name = os.environ['BUCKET_NAME']
    file_key = f"{cls}/{subject}/{chapter}/"
    download_s3_folder(s3, bucket_name, file_key, '/tmp/vector/')
    new_db = FAISS.load_local('/tmp/vector/', embedding)
    context = new_db.similarity_search(question, k=2)
    result = is_question_appropriate(question, llm, int(cls)+6)
    id = ''.join(random.choices(string.ascii_lowercase+string.digits, k=4))
    if result == 'No':
        result = "As per our community guidelines, the question seems inappropriate. Please ask a different question."
        result = {"from": {"type": "gpt"}, "msg": {"message": result}, "id": id}
        return {'statusCode': 200, 'body': json.dumps(result)}
    else:
        result = get_answer(question, llm, subject, int(cls)+6, context)
        result = {"from": {"type": "gpt"}, "msg": {"message": result}, "id": id}
    return {'statusCode': 200,
            "headers": {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": True},
            'body': json.dumps(result)}
