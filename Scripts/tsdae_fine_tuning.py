# Importing Dependencies
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss
from datasets import Dataset

import pandas as pd
import transformers
import warnings
import random

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Muting transformers notifications
transformers.logging.set_verbosity_error()

# Defining a custom class
class TSDAEFineTuner:
    # Defining the instance attributes
    def __init__(self, config: dict, data: pd.DataFrame):
        # Instantiating the configuration
        self.config=config
        
        # Instantiating the data frame
        self.data=data
        
    # Defining a function to add noise
    def add_noise(self, text, noise_ratio=0.6):
        # Splitting the text with whitespaces
        words=text.split()
        
        # Creating a damaged words
        damaged_words=[word for word in words if random.random()>noise_ratio]
        
        # Creating a condition
        if len(damaged_words)==0:
            # Randomly choosing words
            damaged_words=[random.choice(seq=words)]
            
        # Reconstructing the damaged words into damaged sentence
        damaged_sentence=" ".join(damaged_words)
        
        # Returning the damaged sentence
        return damaged_sentence

    # Defining a function to prepare data
    def prepare_tsdae_data(self, batch):
        # Creating a new column called corrupted_text
        batch["corrupted_text"]=[self.add_noise(text=text) for text in batch["text"]]
        
        # Creating a new column called original_text
        batch["original_text"]=batch["text"]
        
        # Returning the batch
        return batch
    
    # Defining an instance method to apply sampling
    def sample_data(self, sampling_count=25):
        # Extracting the EPC
        self.data["epc"]=self.data.metadata.apply(func=lambda x: x.get("epc"))

        # Sampling the data frame
        sampled_df=pd.concat(objs=[self.data.loc[self.data.epc==i].sample(n=min(self.data.loc[self.data.epc==i].shape[0], 
                                                                                sampling_count), 
                                                                          random_state=42) 
                                   for i in self.data.epc.unique()], 
                             ignore_index=True)

        # Loading the dataset
        dataset=Dataset.from_pandas(df=sampled_df)
        
        # Returning the dataset
        return dataset
    
    # Defining a function to train model
    def train_test_split(self, test_size=0.05, apply_sampling=True):
        # Creating a condition
        if apply_sampling:
            # Calling the instance method to apply sampling
            dataset=self.sample_data()
        else:
            # Loading the dataset
            dataset=Dataset.from_pandas(df=self.data)
    
        # Selecting only the text column
        dataset=dataset.select_columns(column_names="text")

        # Adding noise to the dataset
        dataset=dataset.map(function=self.prepare_tsdae_data, batched=True)

        # Selecting only the text column
        dataset=dataset.select_columns(column_names=["corrupted_text", "original_text"])

        # Splitting the dataset into train and test set
        dataset=dataset.train_test_split(test_size=test_size, seed=42)

        # Extracting the train set
        train_set=dataset["train"]

        # Extracting the test set
        test_set=dataset["test"]
        
        # Returning the train and test sets
        return train_set, test_set
    
    # Defining an instance method to train the model
    def train_model(self, model_path: str):
        # Calling the instance method to create train and test sets
        train_set, test_set = self.train_test_split()
        
        # Loading the pretrained embedding model
        model=SentenceTransformer(model_name_or_path=self.config.get("emb_model")).to(device="cuda")

        # Defining the loss
        loss=DenoisingAutoEncoderLoss(model=model,
                                      decoder_name_or_path=self.config.get("emb_model"),
                                      tie_encoder_decoder=True)

        # Instantiating the training arguments
        train_args=SentenceTransformerTrainingArguments(output_dir="Models",
                                                        num_train_epochs=1,
                                                        per_device_train_batch_size=4,
                                                        per_device_eval_batch_size=4,
                                                        gradient_accumulation_steps=4,
                                                        learning_rate=3e-5,
                                                        fp16=True,
                                                        dataloader_num_workers=2,
                                                        eval_strategy="steps",
                                                        eval_steps=10,
                                                        save_strategy="steps",
                                                        save_steps=10,
                                                        save_total_limit=2,
                                                        logging_steps=5)

        # Instantiating the trainer
        trainer=SentenceTransformerTrainer(model=model,
                                           args=train_args,
                                           train_dataset=train_set,
                                           eval_dataset=test_set,
                                           loss=loss)

        # Training the model
        trainer.train()
        
        # Saving the model
        model.save_pretrained(path=model_path)