# Importing Dependencies
from langchain_chroma import Chroma

# Defining a custom class
class VectorStore:
    # Defining the instance attributes
    def __init__(self, chunks, embeddings, db_path):
        # Instantiating the path for the vector database
        self.db_path=db_path
        
        # Instantiating the chunks
        self.chunks=chunks
        
        # Instantiating the embeddings
        self.embeddings=embeddings
        
    # Defining an instance method to instantiate the embedding model
    def create_vector_db(self):
        # Instantiating the vector store
        vector_db=Chroma.from_documents(documents=self.chunks,
                                        embedding=self.embeddings,
                                        persist_directory=self.db_path)
        
        # Returning the model
        return vector_db