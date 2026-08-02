# Importing Dependencies
from tsdae_fine_tuning import TSDAEFineTuner
from data_splitter import DataSplitter
from data_loader import MedDataLoader
from vector_store import VectorStore
from embedder import EmbeddingModel

import pandas as pd
import warnings
import logging
import yaml
import os

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Running the script
if __name__ == "__main__":
    # Extracting the directory for the script
    script_dir=os.path.dirname(os.path.abspath(__file__))

    # Defining a directory for the configurations
    config_dir=os.path.abspath(os.path.join(script_dir, "..", "Configuration"))

    # Defining the filepath for the YAML file
    config_filepath=os.path.join(config_dir, "config.yml")
    
    # Making sure the Configuration folder exists
    os.makedirs(name=config_dir, exist_ok=True)
    
    # Defining a directory for the data
    data_dir=os.path.abspath(os.path.join(script_dir, "..", "Data"))

    # Defining the filepath for the JSON file
    json_filepath=os.path.join(data_dir, "openfda.json")
    
    # Defining the filepath for the Parquet file
    parquet_filepath=os.path.join(data_dir, "chunks.parquet.brotli")
    
    # Defining the filepath for the Parquet file
    output_filepath=os.path.join(data_dir, "synthetic_data.parquet.brotli")
    
    # Making sure the Data folder exists
    os.makedirs(name=data_dir, exist_ok=True)
    
    # Defining a directory for the logs
    log_dir=os.path.abspath(os.path.join(script_dir, "..", "Logs"))

    # Defining the filepath for the log file
    log_filepath=os.path.join(log_dir, "indexing.log")
    
    # Making sure the Logs folder exists
    os.makedirs(name=log_dir, exist_ok=True)
    
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
        
    # Creating a log file
    logging.basicConfig(filename=log_filepath,
                        filemode="w",
                        format="%(asctime)s - %(message)s - %(levelname)s",
                        level=logging.INFO)
    
    # Sending the log
    logging.info(msg="Building indexing pipeline in progress...")
        
    # Instantiating the data loader
    dataloader=MedDataLoader(filepath=json_filepath, base_url=config.get("base_url"))
    
    # Calling the instance method to load the data
    data=dataloader.fetch_medicines(epc_limit=1000, medicine_limit=1000)
    
    # Instantiating the data splitter
    datasplitter=DataSplitter(data=data, config=config)
    
    # Calling the instance method to create chunks
    chunks=datasplitter.create_chunks()
    
    # Converting chunks to a data frame
    chunks_df=pd.DataFrame(data=[{"metadata": chunk.metadata, "text": chunk.page_content} for chunk in chunks])
    
    # Saving the chunks as parquet
    chunks_df.to_parquet(path=parquet_filepath,
                         engine="fastparquet",
                         compression="brotli",
                         index=False)
    
    # Sending the log
    logging.info(msg="Embedding model is being fine-tuned...")
    
    # Instantiating the fine-tuning
    tsdae=TSDAEFineTuner(config=config, data=chunks_df)
    
    # Fine-tuning the pretrained embedding model
    tsdae.train_model(model_path=model_filepath)
    
    # Sending the log
    logging.info(msg="Embedding model has been fine-tuned")
    
    # Instantiating the embedding instance
    emb_model=EmbeddingModel(model_path=model_filepath)
    
    # Calling the instance method to load the model
    embeddings=emb_model.get_embeddings()
    
    # Sending the log
    logging.info(msg="Embeddings have been created")
    
    # Instantiating the Vector Store class
    vector_store=VectorStore(chunks=chunks, 
                             embeddings=embeddings, 
                             db_path=vectorstore_dir)
    
    # Creating the vector store
    vector_db=vector_store.create_vector_db()
    
    # Sending the log
    logging.info(msg="Building vector DB is done, indexing pipeline finished successfully!")