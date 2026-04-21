import os
from huggingface_hub import login


def hugging_face_auth():
    '''
        in order to download the right model to work on some of the model are gated by HuggingFace therefore we must authenticate first
    '''
    # getting token from .env
    HUGGING_FACE_TOKEN=os.environ.get("HF_TOKEN")

    # logging in
    print("Attempting Hugging Face login...")
    login(token=HUGGING_FACE_TOKEN)
    print("Login successful!")