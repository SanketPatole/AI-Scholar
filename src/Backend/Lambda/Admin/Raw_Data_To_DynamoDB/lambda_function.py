import os
import boto3
import io
import json
import urllib
import time
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor


def download_to_s3(s3, url, bucket, object_name):
    response = urllib.request.urlopen(url)
    data = response.read()
    s3.put_object(Body=data, Bucket=bucket, Key=object_name)
    print(f"Uploaded {object_name.split('/')[-1]} to S3 bucket {bucket}.")

def extract_data_from_pdf(s3, bucket_name, file_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        response = response['Body']
        raw_data = b''
        chunk_no = 1
        for chunk in response.iter_chunks(chunk_size=1024*1024):
            raw_data += chunk
            chunk_no += 1
        pdfreader = PdfReader(io.BytesIO(raw_data))
        count = len(pdfreader.pages)
        data = ""
        for i in range(count):
            page = pdfreader.pages[i]
            data += page.extract_text()
        return data
    except Exception as e:
        print(f"Unable to load file {file_key} because of the error {e}")

def insert_record_into_dynamodb(dynamodb, class_info, subject_info, chapter_info, content, table):
    try:
        # Insert new record into DynamoDB table
        dynamodb.put_item(TableName=table,
        Item={'CLASS_SUBJECT': {'S': class_info+"_"+subject_info},
        'CHAPTER': {'S': chapter_info}, 'CONTENT': {'S': content}})
        print(f"Inserted new record for Class {class_info}, Subject {subject_info}, Chapter {chapter_info}")
    except Exception as e:
        print(f"Error inserting records into DynamoDB: {e}")

def get_urls():
    base_url = "https://ncert.nic.in/textbook.php"
    source = requests.get(base_url).text
    excluded_subjects = [
        'Health and Physical Education',
        'Hindi',
        'Sanskrit',
        'Urdu',
        'Biotechnology',
        'Computers and Communication Technology',
        'Creative Writing and Translation',
        'Fine Art',
        'Graphics design',
        'Home Science',
        'Knowledge Traditions Practices of India',
        'Sangeet',
        'Sociology',
        'Psychology',
        'Science'
    ]
    classes = source.split("//this function check the classthat you have selected")[1].split("function")[0].split("}")[1:-3]
    subjects = dict()
    for cls in classes:
        lines = [line.strip() for line in cls.split("\n") if line.strip() != "" and line.strip() != "{" and not ".." in line.strip() and not line.strip().startswith("//")]
        subjects[int(lines[0].split("=")[2].split(")")[0])] = [lines[i].split('"')[1] for i in range(1, len(lines))]
    url_list = dict()
    for cls in [10, 11, 12]:
        url_list[cls] = []
        for subject in subjects[cls]:
            if subject in excluded_subjects:
                continue
            lines = source.split(f"""if((document.test.tclass.value=={cls}) && (document.test.tsubject.options[sind].text=="{subject}"))""")[1].split("}")[0].split("\n")
            lines = [line.strip() for line in lines if line.strip() != "" and line.strip() != "{" and not ".." in line.strip() and not line.strip().startswith("//")]
            if len(lines) > 0:
                code = lines[1].split("?")[1].split("=")[0]
                number_of_pages = int(lines[1].split("?")[1].split("=")[1].split("-")[1].split('"')[0])            
                for i in range(1, number_of_pages):
                    unitcode = str(i) if i > 9 else "0" + str(i)
                    url = f"https://ncert.nic.in/textbook/pdf/{code}{unitcode}.pdf"
                    url_obj = {'class': cls, 'subject': subject,
                    'chapter': unitcode, 'url': url}
                    url_list[cls].append(url_obj)
    return url_list
                    

def lambda_handler(event, context):
        bucket_name = os.environ['BUCKET_NAME']
        table = os.environ['DYNAMODB_TABLE_NAME']
        dynamodb = boto3.client('dynamodb')
        s3 = boto3.client('s3')
        
        mode = "parent"
        if 'url_list' in event.keys():
            mode = "child"
            
        if mode == 'parent':
            print("Code ran in parent mode")
            client = boto3.client('lambda')
            url_list = get_urls()
            for _class in [10, 11, 12] :
                client.invoke(FunctionName="Raw_Data_To_DynamoDB",
                    InvocationType='Event',
                    Payload=json.dumps({'url_list': url_list[_class]}))
        
        elif mode == 'child':
            print(f"Code ran in child mode for class {event['url_list'][0]['class']}")
            for url_obj in event['url_list']:
                time.sleep(5)
                cls = url_obj['class']
                subject = url_obj['subject']
                chapter = url_obj['chapter']
                url = url_obj['url']
                s3_folder = f"All Files/{cls}/{subject}"
                s3_key = f"{s3_folder}/{chapter}.pdf"

                response = dynamodb.get_item(TableName=table,
                    Key={'CLASS_SUBJECT': {'S': str(cls)+"_"+subject},
                    'CHAPTER': {'S': chapter}})
                try:
                    if 'Item' in response:
                        print(f"Already Loaded {url}")
                        continue
                except Exception as e:
                    print("Unable to get item")
                try:
                    response = urllib.request.urlopen(url)
                except Exception as e:
                    print("Unable to download data")
                    continue
                data = response.read()
                pdfreader = PdfReader(io.BytesIO(data))
                count = len(pdfreader.pages)
                data = ""
                for i in range(count):
                    page = pdfreader.pages[i]
                    data += page.extract_text()
                content = data
                try:
                    if content:
                        insert_record_into_dynamodb(dynamodb, str(cls),
                                str(subject), chapter, content, table)
                    else:
                        print("Failed to extract content from pdf.")
                except Exception as e:
                    print(f"Unable to insert into dynamodb: {e}")
