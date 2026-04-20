from hardware_preparation import torch, gpu_preparation 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from datasets import load_dataset
from IPython.display import display, Markdown
import random
from huggingface_hub import login
import os


class ModelManager:
    """
    Manages the loading and configuration of a HuggingFace causal language model.
    Supports optional 4-bit quantization via BitsAndBytes for memory-efficient inference.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" # another example deepseek-ai/deepseek-coder-1.3b-instruct

    def __init__(self, model_id=DEFAULT_MODEL_ID, quantization_setting=True):
        """
            Initializing our Model class with our tokenizer and loading model
        """
        if gpu_preparation():
            self.hugging_face_auth()
            self.model_id  = model_id
            self.tokenizer = self.loading_tokenizer()
            self.model = self.loading_model(self.model_id, quantization_setting)
        else:
            raise EnvironmentError("No GPU detected. This model requires a CUDA-compatible GPU to run.")


    def hugging_face_auth(self):
        '''
            in order to download the right model to work on some of the model are gated by HuggingFace therefore we must authenticate first
        '''
        # getting token from .env
        HUGGING_FACE_TOKEN=os.environ.get("HF_TOKEN")

        # logging in
        print("Attempting Hugging Face login...")
        login(token=HUGGING_FACE_TOKEN)
        print("Login successful!")


    def loading_tokenizer(self):
        """
            this function will be returning a tokenizer object to our ModelManager to initiate our tokenizer
        """
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
    

    def loading_model(self, model_id, quantization_setting):
        """
            this function will load and prepare our model according to our preference
            1- model_id
            2- with our without quantization
        """
        kwargs = {
            "device_map": "auto",
            "dtype": "auto",
            "trust_remote_code": True,
        }

        if quantization_setting:
            kwargs["quantization_config"] = self.quantization_configuration()
        
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


    def quantization_configuration(self):
        """
            this function will return a BitsAndBytesConfig configuration for our model for quantization
        """
        return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
    
    def __repr__(self) -> str:
        return (
            f"ModelManager("
            f"model_id='{self.model_id}', "
            f"quantization={self.tokenizer}, "
            f"model={self.model}, "
            f"device='{torch.cuda.get_device_name(0)}')"
        )
    
    
###################
#x = ModelManager()
#print(x)
##############3####