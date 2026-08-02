# Importing Dependencies
from transformers import pipeline
from dotenv import load_dotenv

import transformers
import warnings
import torch

# Filtering out potential warnings
warnings.filterwarnings(action="ignore")

# Muting transformer notifications
transformers.logging.set_verbosity_error()

# Loading the environment variables
load_dotenv()

# Defining a custom class for augmentation
class Augmentation:
    # Defining the instance attributes
    def __init__(self, config: dict, device: str):
        # Instantiating the device
        self.device=device
        
        # Instantiating the configuration
        self.config=config
        
        # Loading the LLM
        self.llm=pipeline(task="text-generation",
                          model=self.config.get("llm"),
                          model_kwargs={"torch_dtype": torch.float16},
                          device=self.device)
        
    # Defining the instance method to compress the context
    def compress_context(self, query: str, context: list):
        # Joining the chunks
        chunks_joined="\n\n".join([i.page_content for i in context])
        
        # Defining the instructions for the system
        system_instruction = ("You are a strict data extractor. Extract and summarize ONLY key facts "
                              "from the context that directly help answer the user question. "
                              "Do NOT add outside knowledge. "
                              "If the context does not contain relevant information to answer the question, "
                              "you MUST output exactly: NO_RELEVANT_CONTEXT")
        
        # Defining the user content
        user_content=f"Context:\n{chunks_joined}\n\nQuestion: {query}"
        
        # Reformatting the prompt
        message=[{"role": "system", "content": system_instruction},
                 {"role": "user", "content": user_content}]
        
        # Sending the prompt to the LLM
        response=self.llm(text_inputs=message,
                          max_new_tokens=256,
                          do_sample=False,
                          return_full_text=False)
        
        # Extracting the summary
        summary=response[0].get("generated_text").strip()
                
        # Returning the summary
        return summary
    
    # Defining an instance method to generate text
    def generate_response(self, query: str, compressed_context: str):
        # Creating another guardrail
        if not compressed_context or "NO_RELEVANT_CONTEXT" in compressed_context:
            # Returning the pre-defined answer
            return "I do not know the answer given context"
        
        # Updating the prompt
        prompt=("You are an expert in the field of medical science. "
                "You must always answer the question by referring to the provided context. "
                "If you don't know the answer given context, you must always say 'I do not know the answer given context'.\n"
                f"Question: {query}\n\n"
                f"Context: {compressed_context}")
        
        # Reformatting the prompt
        message=[{"role": "user", "content": prompt}]
        
        # Sending the prompt to the LLM
        response=self.llm(text_inputs=message,
                          max_new_tokens=512,
                          do_sample=False,
                          return_full_text=False)
        
        # Extracting the response
        output=response[0].get("generated_text").strip()
        
        # Returning the output
        return output