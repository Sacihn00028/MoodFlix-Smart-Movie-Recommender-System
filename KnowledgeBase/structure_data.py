## This file is a helper file to load the dataset, and then pre-process and store it in a vector Store DB
## Steps:
##      1. First we load the data
##      2. Then we enhance the Movie Metadata
##      2. Then we iterate over it and push each of the information to the Vector Store (Knowledge Base)
##      3. Add another Model?
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
csv_file_path = 'eng.csv' 
try:
    # df = pd.read_csv(csv_file_path)
    print("Data loaded successfully!")
    vector_store = Chroma(
        collection_name="Knowledge_Base_Movies_eng",
        embedding_function=embeddings,
        persist_directory="./kb_db",  
    )
    langchain_documents = []
    # for index, row in df.iterrows():
    #     # page_content = f"Movie Name: {row['movie_name']}\n" \
    #     #             f"Year: {row['year']}\n" \
    #     #             f"Genre: {row['genre']}\n" \
    #     #             f"Overview: {row['overview']}\n" \
    #     #             f"Director: {row['director']}\n" \
    #     #             f"Cast: {row['cast']}" ## For Bollywood movies
    #     page_content = f"Movie Name: {row['Title']}\n" \
    #         f"Year: {row['Year']}\n" \
    #         f"Genre: {row['Genre']}\n" \
    #         f"Overview: {row['Description']}\n" \
    #         f"Director: {row['Director']}\n" \
    #         f"Cast: {row['Actors']}\n" \
    #         f"Runtime : {row['Runtime (Minutes)']}\n" \
    #         f"Rating : {row['Rating']}"
    #     metadata = {
    #         "movie_name": row['Title'],
    #         "year": row['Year'],
    #         "genre": row['Genre'],
    #         "director": row['Director'],
    #         "cast": row['Actors'],
    #         "metascore" : row["Metascore"]
    #     }
    #     doc = Document(page_content=page_content, metadata=metadata)
    #     langchain_documents.append(doc)
    # vector_store.add_documents(documents=langchain_documents)
    docs = vector_store.similarity_search("User wants to see some depression", k=2)
    print(*docs)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
