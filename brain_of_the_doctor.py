#Step1: Setup GROQ API key
import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#Step2: Covert image to required format
import base64 #bits to string convert ek system se dusre system m corruopt na ho encoding and decoding

#image_path="acne.jpg"
def encode_image(image_path):
    image_file=open(image_path, "rb") #read binary
    return base64.b64encode(image_file.read()).decode('utf-8')  #file read utf-8 and then decode base64 encode and store it

#Step3: Setup Multimodal LLM
from groq import Groq

query="Is there something wrong with my face?"
#model="llama-3.2-90b-vision"
model="meta-llama/llama-4-scout-17b-16e-instruct"

def analyze_image_with_query(query,model,encoded_image):
    client=Groq()
    
    #api call to groq
    messages=[
        {
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content