# Importing Dependencies
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv

import transformers
import warnings
import logging
import tqdm

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Muting notifications from transformers
transformers.logging.set_verbosity_error()

# Loading the environment variables
load_dotenv()

# Defining a custom class to create chunks
class DataSplitter:
    # Defining the instance attributes
    def __init__(self, data: list, config: dict):
        # Instantiating the data
        self.data=data
        
        # Instantiating the configuration
        self.config=config
        
    # Defining an instance method to load the tokenizer
    def load_tokenizer(self):
        # Loading the tokenizer
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.config.get("emb_model"))
        
        # Sending the log
        logging.info(msg="Tokenizer has been loaded")
        
        # Returning the tokenizer
        return tokenizer
        
    # Defining the instance method to create chunks
    def create_chunks(self):
        # Sending the log
        logging.info(msg="Data splitting process started!")
        
        # Calling the instance method to load the tokenizer
        tokenizer=self.load_tokenizer()
        
        # Defining the text splitter
        text_splitter=RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer, 
                                                                                chunk_size=self.config.get("chunk_size"), 
                                                                                chunk_overlap=self.config.get("chunk_overlap"))
        
        # Creating a list of metadata labels
        metadata_labels=self.config.get("metadata")

        # Creating an empty list to store chunks
        chunks=[]

        # Looping through each record
        for medicine in tqdm.tqdm(iterable=self.data, total=len(self.data)):
            # Extracting the metadata
            metadata={label: medicine.get(label) for label in metadata_labels}
            
            # Extracting labels for the non-metadata
            maindata=[i for i in medicine.keys() if i not in metadata_labels]
            
            # Extracting the text to chunk
            text_to_chunk=[f"Section: {section.replace("_", " ").capitalize()}\n\n{medicine.get(section)}" for section in maindata]
            
            # Creating the chunk
            chunk=text_splitter.create_documents(texts=text_to_chunk, metadatas=[metadata]*len(text_to_chunk))
            
            # Sending the log
            logging.info(msg=f"Chunk has been for {medicine.get('brand_name')}")
            
            # Extending the list
            chunks.extend(chunk)
            
        # Sending the log
        logging.info(msg=f"Chunking process finished successfully. Total {len(chunks)} chunks created!")
            
        # Returning the chunks
        return chunks