from llama_index import GPTChromaIndex, SimpleDirectoryReader
import chromadb

from dotenv import load_dotenv

load_dotenv()
#  https://docs.trychroma.com/embeddings
# create a Chroma vector store, by default operating purely in-memory
chroma_client = chromadb.Client()

# create a collection
chroma_collection = chroma_client.create_collection("dlchapters")
# https://docs.trychroma.com/api-reference
print(chroma_collection.count())

documents = SimpleDirectoryReader('data').load_data()

index = GPTChromaIndex.from_documents(documents, chroma_collection=chroma_collection)
print(chroma_collection.count())
print(chroma_collection.get()['documents'])
print(chroma_collection.get()['metadatas'])

index.save_to_disk("dlchapters.json")

# During query time, the index uses Chroma to query for the top k
# most similar nodes, and synthesizes an answer from the retrieved nodes.

r = index.query("How do you do regression with TensorFlow?  What about classification?")
print(r)
