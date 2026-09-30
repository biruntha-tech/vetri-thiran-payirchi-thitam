import pandas as pd
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# read data
data = pd.read_csv("PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv")
data.head()

# preprocess the data to extract useful information
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'], errors='coerce')
data['Month'] = data['Purchase Date'].dt.month
data['Quarter'] = data['Purchase Date'].dt.quarter
data['Year'] = data['Purchase Date'].dt.year


# select useful features from the data
selected_feat = ['Total Price','Purchase Date', 'Month', 'Quarter','Year','Item Description','Item Name','Department Name'] 
df = data[selected_feat]

# drop records which have nan values at these columns 
df = df.dropna(subset=['Purchase Date','Total Price'])

# database(pgvector) connection parameters
CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
COLLECTION_NAME = 'procurementdata'

# the embeddings model
model_path = "intfloat/multilingual-e5-small"
embeddings = HuggingFaceEmbeddings(model_name=model_path)


# convert dataset rows into Documents for embedding storage
documents = [
    Document(
        page_content=row['Item Description'] if pd.notna(row['Item Description']) else "No description available",
        metadata={
            "Purchase Date": row['Purchase Date'].strftime('%Y-%m-%d'),
            "Total Price": row['Total Price'],
            "Item Name": row['Item Name'] if pd.notna(row['Item Name']) else "Unknown",
            "Department Name": row['Department Name'] if pd.notna(row['Department Name']) else "Unknown",
            "Quarter": row['Quarter'],
            "Year": row['Year'] 
        }
    )
    for _, row in df.iterrows()
]


# PGVector for embedding search
db = PGVector(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)


# load LLM Model (Qwen)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def extract_quarter(query):
    # extract the quarter number from the query (e.g., "quarter 1" should return 1)
    match = re.search(r"quarter\s*(\d+)", query)
    if match:
        return int(match.group(1))
    return None

def extract_month(query):
    # example implementation to extract a month from query
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    for month in months:
        if month in query.lower():
            return months[month]
    return None


def get_procurement_data(query):
    query_lower = query.lower()

    # total number of orders based on time period
    if "total number of orders" in query_lower:
        if "month" in query_lower:
            month = extract_month(query)
            result = df[df['Month'] == month].shape[0]
            return f"Total number of orders in month {month}: {result}"
        
        elif "quarter" in query_lower:
            quarter = extract_quarter(query)
            result = df[df['Quarter'] == quarter].shape[0]
            return f"Total number of orders in quarter {quarter}: {result}"

        elif "year" in query_lower:
            year = extract_year(query)
            result = df[df['Year'] == year].shape[0]
            return f"Total number of orders in year {year}: {result}"

    # quarter with the highest spending
    if "highest spending" in query_lower:
        highest_spending = df.groupby(['Year', 'Quarter'])['Total Price'].sum().idxmax()
        year, quarter = highest_spending
        total_spending = df.groupby(['Year', 'Quarter'])['Total Price'].sum().max()
        return f"Quarter {quarter} of {year} had the highest spending with a total of ${total_spending}."

    # analysis of frequently ordered items
    if "frequently ordered" in query_lower or "most ordered" in query_lower:
        most_frequent = df['Item Name'].value_counts().idxmax()
        count = df['Item Name'].value_counts().max()
        return f"The most frequently ordered item is '{most_frequent}' with {count} orders."

    return None


def llm_based_answer(result, question):
    text = ""

    for doc in result:
        text += f"Item Name: {doc['Item Name']}\n" \
                f"Description: {doc['embedded_text']}\n" \
                f"Year: {doc['Year']}\n" \
                f"Quarter: {doc['Quarter']}\n" \
                f"Department Name: {doc['Department Name']}\n" \
                f"Total Price: {doc['Total Price']}\n" \
                f"Purchase Date: {doc['Purchase Date']}\n\n"

    # create the messages for the LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {
            "role": "user",
            "content": f"""
                You are reviewing procurement-related content. You have item names, descriptions, and other relevant details.
                Based on the provided information, return only the accurate answer without without any hallucinations.
                Text: '''{text}'''
                Question: '''{question}'''
            """
        },
    ]

    # generate the prompt 
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Convert the prompt to model inputs
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    # extract the response to return only the answer
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]

    return response

def handle_query(query):
    # try to get an answer using structured query handling
    structured_answer = get_procurement_data(query)
    if structured_answer:
        return structured_answer

    # if no structured answer, proceed to use LLM-based approach
    doc = db.similarity_search_with_score(query, k=3)
    result = []
    for i in doc:
        k = i[0].metadata
        k['embedded_text'] = i[0].page_content
        k['score'] = i[1]
        result.append(k)

    # get the answer using LLM
    return llm_based_answer(result, query)