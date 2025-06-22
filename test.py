import chromadb

# Creates a local database in ./chroma_db/
client = chromadb.PersistentClient(path="./data/chroma_db")

# Create a collection (like a table)
collection = client.create_collection("conversations")

print("ChromaDB working!")
