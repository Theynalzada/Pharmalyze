# Importing Dependencies
from augmentation import Augmentation
from langchain_chroma import Chroma
from embedder import EmbeddingModel
from retriever import Retriever
from dotenv import load_dotenv

import transformers
import warnings
import torch
import yaml
import os

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Muting transformers notifications
transformers.logging.set_verbosity_error()

# Loading the environment variables
load_dotenv()

# Making sure the GPU is being utilized
assert torch.cuda.is_available()

# Specifying the device
device="cuda" if torch.cuda.is_available() else "cpu"

# Extracting the directory for the script
script_dir=os.path.dirname(os.path.abspath(__file__))

# Defining a directory for the configurations
config_dir=os.path.abspath(os.path.join(script_dir, "..", "Configuration"))

# Defining the filepath for the YAML file
config_filepath=os.path.join(config_dir, "config.yml")

# Making sure the Configuration folder exists
os.makedirs(name=config_dir, exist_ok=True)

# Defining a directory for the Models
models_dir=os.path.abspath(os.path.join(script_dir, "..", "Models"))

# Defining the filepath for the fine-tuned model
model_filepath=os.path.join(models_dir, "tsdae_fine_tuned")

# Making sure the Logs folder exists
os.makedirs(name=models_dir, exist_ok=True)

# Defining a directory for the data
vectorstore_dir=os.path.abspath(os.path.join(script_dir, "..", "VectorStore"))

# Making sure the VectorStore folder exists
os.makedirs(name=vectorstore_dir, exist_ok=True)

# Loading the YAML file
with open(file=config_filepath) as yaml_file:
    config=yaml.safe_load(stream=yaml_file)

# Defining a custom class to build the RAG system
class MedRAG:
    # Defining the instance attributes
    def __init__(self, config=config):
        # Loading the embedding instance with the fine tuned embedding model
        embedding_instance=EmbeddingModel(model_path=model_filepath)
        
        # Calling the instance method to load the model
        embeddings=embedding_instance.get_embeddings()
        
        # Loading the vector database
        vector_db=Chroma(embedding_function=embeddings,
                         persist_directory=vectorstore_dir)
        
        # Instantiating the retrieval module
        self.retrieval_module=Retriever(model_name=config.get("rerank_model"), 
                                        vector_store=vector_db,
                                        device=device)
        
        # Instantiating the augmentation module
        self.augmentation_module=Augmentation(config=config, device=device)
        
    # Defining the instance method
    def predict(self, query):
        # Extracting the list of document objects
        documents=self.retrieval_module.retrieve(query=query, top_k=20, top_n=5)
        
        # Applying context compression to stay within the context length limit of LLM
        context=self.augmentation_module.compress_context(query=query, context=documents)
        
        # Sending the query with compressed context to LLM
        output=self.augmentation_module.generate_response(query=query, compressed_context=context)
        
        # Returning the output
        return output