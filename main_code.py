#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("insight_coffee_key")


# # Creating the data!!

# In[2]:


import os
import requests
import pandas as pd

filepath1 = 'AllCoffeeFinalApril.csv'
cafe_df = pd.read_csv(filepath1, header = 0)


# In[3]:


# updated 2019
cafe_df.head()


# In[4]:


cafe_df.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)
cafe_df.head()


# In[ ]:


# create a function to combine text for embedding
def combine_text(row):
    name = row['name'] if 'name' in row else ''
    categories = ', '.join(row['categories']) if isinstance(row['categories'], list) else ''
    location = row['location'] if isinstance(row['location'], dict) else {}
    address_parts = [
        location.get('address1', ''),
        location.get('city', ''),
        location.get('state', ''),
        location.get('zip_code', '')
    ]
    address = ', '.join([part for part in address_parts if part])
    price = row['price'] if 'price' in row else ''
    rating = str(row['rating']) if 'rating' in row else ''
    review_count = str(row['review_count']) if 'review_count' in row else ''

    return f"{name}. Categories: {categories}. Location: {address}. Price: {price}. Rating: {rating}/5 from {review_count} reviews."


# In[ ]:


# apply the function to create a new column for embeddings
cafe_df['embedding_text'] = cafe_df.apply(combine_text, axis=1)
cafe_df.head()


# In[8]:


# function to get address string from location dictionary
def get_address_string(row):
    location = row['location'] if isinstance(row['location'], dict) else {}
    return ', '.join([location.get('address1', ''), location.get('city', ''), location.get('state', '')])


# In[9]:


from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from langchain_openai import ChatOpenAI

docs = [
    Document(
        page_content=row['embedding_text'],
        metadata={
            "name": row.get("name", ""),
            "address": get_address_string(row)
        }
    )
    for _, row in cafe_df.iterrows()
]

embedding_function = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_function,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

prompt_message = """
Find me cafe suggestions based on the following preferences:

Preferences:
{preferences}

Top Cafes:
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("human", prompt_message)
])

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=openai_key,
    temperature=1.0
)

rag_chain = (
    {"preferences": retriever, "input": RunnablePassthrough()}
    | prompt_template
    | llm
)


# In[14]:


query = "I want a modern korean cafe with matcha, coffee, and wifi in Montclair. I prefer cafes with plenty of seating for studying. Show me the top 3 cafes that match my preferences, including their names, locations, and a brief description of their offerings."
response = rag_chain.invoke(query)

print(response.content)

