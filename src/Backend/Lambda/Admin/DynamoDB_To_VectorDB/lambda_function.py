import json
import os
import boto3
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import time

def upload_directory_to_s3(s3_client, bucket_name, directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory_path)
            s3_path = os.path.join(relative_path)
            s3_client.upload_file(local_path, bucket_name, s3_path)

def get_splits(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                            chunk_overlap=100)
    return splitter.split_text(text)

def get_vectorstore(splits):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(splits, embedding=embedding)
    return vectorstore

def lambda_handler(event, context):
    bucket_name = os.environ['BUCKET_NAME']
    tablename = os.environ['DYNAMODB_TABLE_NAME']
    dynamodb = boto3.client('dynamodb', region_name='us-east-1')
    s3 = boto3.client('s3')
    response = dynamodb.scan(TableName=tablename)
    data = response['Items']
    while response.get('LastEvaluatedKey'):
        time.sleep(5)
        response = dynamodb.scan(TableName=tablename,
                ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    for entry in data:
        cls = entry['CLASS_SUBJECT']['S'].split('_')[0]
        subject = entry['CLASS_SUBJECT']['S'].split('_')[1]
        chapter = entry['CHAPTER']['S']
        text = entry['CONTENT']['S']
        splits = get_splits(text)
        vectorstore = get_vectorstore(splits[0])
        vectorstore.save_local(f"/tmp/{cls}/{subject}/{chapter}")
        print(f"Loaded {cls} - {subject} - {chapter}")
    upload_directory_to_s3(s3, bucket_name, directory_path=f"/tmp/")