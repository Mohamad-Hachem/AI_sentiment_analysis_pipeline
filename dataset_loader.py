# pip install transformers datasets ipython
from hardware_preparation import torch, gpu_preparation 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from datasets import load_dataset
from IPython.display import display, Markdown
import random

class DatasetLoader:

    def __init__(self, dataset_path="mteb/tweet_sentiment_extraction"):
        self.dataset = load_dataset(dataset_path, split="train")
        self.current_column = None

    def checking_dataset_as_pandas(self):
        """
            we will use this function to show the dataset loaded in a beautiful pandas way
        """
        return self.dataset.to_pandas()
    
    def printing_dataset(self, min=0,max=None):
        """
            we will be printing and showing our dataset if we want using this function
        """
        print(self.checking_dataset_as_pandas()[min:max])
    
    def dataset_more_information(self):
        """
            this function will allow us to know more information about our dataset such as number of columns and columns names
        """
        columns_name = self.dataset.column_names
        information = f"""
                this dataset has {len(columns_name)} columns and are the following:\n {columns_name}
        """
        print(information.strip())
        return information
    
    # the reason we are building all this functions is to allow this class to be very flexible and used on many different datasets
    def choosing_a_column_to_work_with(self, column_name):
        """
            this column would allow us to choose the column we would like to use (run the pipeline analysis on)
        """
        if column_name not in self.dataset.column_names:
            raise ValueError(
                f"Column '{column_name}' does not exist. "
                f"Available columns are: {self.dataset.column_names}"
            )
        
        self.current_column = self.dataset.select_columns([column_name]) 
        
        return self.current_column

    def ask_user_to_choose_a_column_to_work_with(self):
        """
            giving the user the option to choose a column to work with
        """
        user_column_choice = input(f"these are our columns: {self.dataset.column_names} please choose one to work with \n")

        return self.choosing_a_column_to_work_with(user_column_choice.strip())

x = DatasetLoader()
print(x.ask_user_to_choose_a_column_to_work_with())