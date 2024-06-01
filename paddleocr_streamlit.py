import streamlit as st
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
"""
st.title('Image Text Extraction')

uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image_data = uploaded_file.read()
        result = ocr.ocr(image_data)
        text = ""
        for line in result:
            for word in line:
                text += word[1][0] + " "
        st.write(f"Extracted Text: {text}")
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", category=FutureWarning)

import tempfile
import os
from io import BytesIO
import mysql.connector
import fitz
import docx2txt
import streamlit as st
import pandas as pd
import requests
import html5lib
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
api_key=st.secrets["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}  "
    prompt += f"[INST] {message} [/INST]"
    return prompt


generate_kwargs = dict(
    temperature=0.7,
    max_new_tokens=3000,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    seed=42,
)

def generate_output(input_query):
    template = """
    You are an intelligent chatbot. Help the following question with brilliant answers.
    Question: {question}
    Answer:"""
    model1 = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            model_kwargs={"temperature": 0.5,
                                          'repetition_penalty': 1.1,
                                          "max_new_tokens": 3000,
                                          "max_length": 3000})
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=model1)
    answer = llm_chain.invoke(input_query)
    response = answer["text"]
    answer_index = response.find("Answer:")
    if answer_index != -1:
        answer_text = response[answer_index + len("Answer:") + 1:].strip()
        return ( answer_text.strip())
    else:
        return ( response.strip())


def generate_text(message, history):
    prompt = format_prompt(message, history)
    output = client.text_generation(prompt, **generate_kwargs)
    return output


with st.sidebar:
    chat_mode = st.selectbox("Select Chat Mode", options=["SQL Chat", "Normal Chat","Website Chat","Documents Chat","ATS-SCORE","Image Generator"])
    database = st.text_input("SQL Database Name", max_chars=100, placeholder="Enter database name")
    password = st.text_input("SQL Database Password", max_chars=100, placeholder="Enter password", type="password")
    host = st.text_input("SQL Database Host Name", max_chars=100, placeholder="Enter host name")
    user = st.text_input("SQL Database User Name", max_chars=100, placeholder="Enter user name")
    if chat_mode =="Website Chat":
        url = st.text_input("URL link", max_chars=1000, placeholder="apply URL link")
    elif chat_mode == "Documents Chat":
        file_format = ("pdf", "doc", "docx", "txt")
        file = st.file_uploader("Upload a file",type=file_format)
    elif chat_mode =="ATS-SCORE":
        file_formats = ("pdf", "doc", "docx")
        file = st.file_uploader("Upload a file", type=file_formats)
        job_description = st.text_area("Job Description",height=10,max_chars=20000)

st.title("💬 VickyGPT")
st.caption("🚀 A Streamlit chatbot specially design to generate the sql query,and general question answering. ")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def evalutate_ats_score(file_texts,job_description):
    texts = [file_texts,job_description]
    vector = TfidfVectorizer()
    vector1 = CountVectorizer()
    count_matrix = vector.fit_transform(texts)
    match = cosine_similarity(count_matrix)[0][1]
    match = round(match * 100, 2)
    return str(match)

def extract_text_url(url):
    """
    read = requests.get(url).content
    soup = BeautifulSoup(read, "html5lib")
    link = soup.find_all("a")
    all_links = []
    base_url = url
    for i in link:
        href = i.get('href')
        if href:
            complete_url = urljoin(base_url, href)
            all_links.append(complete_url)
    http_links = [link for link in all_links if link.startswith('http://') or link.startswith('https://')]
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=10)
    chunks = text_splitter.split_documents(docs)
    chunk_texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(chunk_texts, embeddings)
    docs = vector_store.similarity_search(prompt, k=6)
    page_content_list = []
    for document in docs:
        page_content = document.page_content
        page_content_list.append(page_content)
    page_content = " ".join(page_content_list)
    return page_content

def documents_similarity(file_text,prompt):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
    chunks = text_splitter.split_text(file_text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    docs = vector_store.similarity_search(prompt, k=6)
    page_content_list = []
    for document in docs:
        page_content = document.page_content
        page_content_list.append(page_content)
    page_content = " ".join(page_content_list[0:])
    return page_content


def extract_text_from_files(file):
    if file is not None:
        file_type = file.type
        all_text = ""
        if file_type == "application/pdf":
            all_text = extract_text(file.read())

        elif file_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
            file_bytes = file.read()
            with BytesIO(file_bytes) as word_file:
                all_text = docx2txt.process(word_file)

        elif file_type == "text/plain":
            file_bytes = file.read()
            all_text = file_bytes.decode('utf-8')

        return all_text

def connect_to_db(host, user, password, database):
    try:
        cnx = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            use_pure=True
        )
        return cnx
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

if "table_columns" not in st.session_state:
    st.session_state["table_columns"] = {}

if st.button("Connect to Database"):
    cnx = connect_to_db(host, user, password, database)
    if cnx:
        cursor = cnx.cursor()
        query = "SHOW TABLES;"
        cursor.execute(query)
        tables = cursor.fetchall()
        table_columns = {}
        for table in tables:
            table_name = table[0]
            column_query = f"SHOW COLUMNS FROM {table_name};"
            cursor.execute(column_query)
            columns = cursor.fetchall()
            column_names = [column[0] for column in columns]
            table_columns[table_name] = column_names
        cursor.close()
        cnx.close()
        st.session_state["table_columns"] = table_columns
        for table_name, columns in table_columns.items():
            st.write(f"Table: {table_name}")
            st.write(f"Columns: {', '.join(columns)}")

prompt = st.chat_input("Say something")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    table_columns = st.session_state.get("table_columns", {})

    if chat_mode == "SQL Chat":
        input_query = str(
            table_columns) + "\n\n\n" + "Given an SQL table name with data and a corresponding list of column names, automatically generate an SQL query that determines the table name and its corresponding column names for following query." + "\n" + prompt

    elif chat_mode =="Normal Chat":
        input_query = str(prompt)
    elif chat_mode =="Website Chat":
        url_text = extract_text_url(url)
        input_query = url_text + "\n\n\n\n\n" + "Please provide an answer to the following question using the text data above. Make sure the answer is relevant to the provided text." + "\n" + prompt
    elif chat_mode =="Documents Chat":
        file_text = extract_text_from_files(file)
        page_content =documents_similarity(file_text,prompt)
        input_query =page_content +"\n\n\n\n" + "Please provide an answer to the following question using the text data above. Make sure the answer is relevant to the provided text." +"\n"+prompt

    elif chat_mode =="ATS-SCORE":
        file_texts = extract_text_from_files(file)
        score = evalutate_ats_score(file_texts,job_description)
        input_query = score +"\n\n\n"+"above present value is similarity ATS score.please answer the following question."+"\n"+prompt
    answer = generate_output(input_query)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
