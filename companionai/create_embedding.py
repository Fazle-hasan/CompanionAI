import os
import pandas as pd
from tqdm.auto import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance

# Set you path to create embedding
df = pd.read_csv('../Data/Final_data.csv')

# Here we have created a conbine text to create vectore of our dataframe
df['all_txt'] = "Question: " + df['question'] + " Answer: " + df['answer']

# Set you path for storing the embeddings
qclient = QdrantClient(path="../Data/Emb")

collectionName = "YourCollectionName"

# Creating collection for mistal 
qclient.create_collection(
       collection_name=collectionName,
       vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
   )
print(collectionName + " Collection created")

# Here we create a function to create embedding using mistral7B

from ollama import Client

OLLAMA_HOST = "http://localhost:11434/"

ollama_client = Client(OLLAMA_HOST)

def get_embedding_mis(txt):
    embeddings = ollama_client.embeddings(model='mistral', prompt=txt)
    return embeddings['embedding']

shape = df.shape

not_app = 0
i = 0
for x in tqdm(range(shape[0])):
    # points = []
    # Get the text from the row
    # txt = row['Merge']
    try:
        txt = df['all_txt'][x]
        print(txt)
        result=get_embedding_mis(txt)
        #print(result)
        # category = texts[x].metadata['Category']
        ids = df['ids'][x]
        print(ids)

        qclient.upsert(
            collection_name=collectionName,
            wait=True,
            points=[
            PointStruct(
                id=x,
                vector=result,
                payload={"text": txt,"ids":str(ids)}
            )
            ]
        )
        i = i + 1
        #print("Inserted " + str(x)+ " document into Qdrant for Mistral")
    except Exception as e:
        print(f"This token is not append {x} and got exception {e}")
        not_app += 1

print(not_app)
