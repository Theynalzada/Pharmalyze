# Importing Dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

import transformers
import warnings
import torch

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Muting transformers notifications
transformers.logging.set_verbosity_error()

# Loading the environment
load_dotenv()

# Defining the device
device="cuda" if torch.cuda.is_available() else "cpu"

# Making sure the GPU is being utilized
assert torch.cuda.is_available()

# Defining a custom class
class EmbeddingModel:
    # Defining the instance attributes
    def __init__(self, model_path: str):
        # Instantiating the path for the fine-tuned model
        self.model_path=model_path
        
    # Defining an instance method to instantiate the embedding model
    def get_embeddings(self):
        # Instantiating the model
        embeddings=HuggingFaceEmbeddings(model_name=self.model_path,
                                         model_kwargs={"device": device},
                                         encode_kwargs={"normalize_embeddings": True})
        
        # Returning the model
        return embeddings