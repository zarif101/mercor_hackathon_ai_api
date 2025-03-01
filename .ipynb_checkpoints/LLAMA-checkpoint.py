import transformers
import torch
import json


class LLAMAMODEL():
    def __init__(self, model_id):
        self.model_id=model_id
        self.pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
        )
    def query(self,
        model="Llama-3.1-8B",
        messages=None,
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop=None,
              response_format=None
    ):
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries.")
    
        # Extract the latest message from the user
        user_prompt = messages[-1]["content"] if messages else ""
        
        # Generate response
        output = self.pipe(
            user_prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_length=max_tokens,
            return_full_text=False
        )
    
        # Extract generated text
        generated_text = output[0]["generated_text"]
    
        # Construct OpenAI-compatible response
        response = {
            "id": "cmpl-" + "".join(str(ord(c)) for c in user_prompt[:5]),  # Fake unique ID
            "object": "text_completion",
            "created": 1234567890,  # Dummy timestamp
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(user_prompt.split()) + len(generated_text.split())
            }
        }
        
        return response
