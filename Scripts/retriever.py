# Importing Dependencies
from sentence_transformers import CrossEncoder

# Defining a custom class for retriever
class Retriever:
    # Defining instance attributes
    def __init__(self, model_name: str, vector_store, device: str):
        # Specifying the model name
        self.model_name=model_name
        
        # Instantiating the vector store
        self.vector_store=vector_store
        
        # Instantiating the device
        self.device=device
        
        # Instantiating the model
        self.model=CrossEncoder(model_name_or_path=self.model_name, 
                                device=self.device)
        
    # Defining the instance method to retrieve documents
    def retrieve(self, query: str, top_k: int, top_n: int):
        # Extracting the documents
        documents=self.vector_store.similarity_search(query=query, k=top_k)
        
        # Extracting the content from the document
        doc_contents=[doc.page_content for doc in documents]
        
        # Creating query-document (chunk) pairs
        pairs=[[query, content] for content in doc_contents]
        
        # Calculating the relevance score of each query-document pair
        scores=self.model.predict(inputs=pairs)
        
        # Zipping the scores and documents
        zipped_scores=zip(documents, scores)
        
        # Sorting the scores
        zipped_scores=sorted(zipped_scores, key=lambda x: x[1], reverse=True)
        
        # Extracting the top N documents
        reranked_documents=[doc for doc, _ in zipped_scores[:top_n]]
        
        # Returning the documents
        return reranked_documents