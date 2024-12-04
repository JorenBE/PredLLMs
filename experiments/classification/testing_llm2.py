import os


import transformers
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = '/data/joren/HF_home'
print(os.getenv('TRANSFORMERS_CACHE'))

pipeline = transformers.pipeline(
                    "text-generation",
                    model='meta-llama/Meta-Llama-3.1-8B',
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
        
system_prompt = 'you are a pirate'
text = 'What is your name?'

messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]
outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
print(outputs[0]["generated_text"][-1])