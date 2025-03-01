import transformers
import torch
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI()

class LLAMAMODEL:
    _instance = None
    _lock = threading.Lock()  # Ensures only one instance is created

    def __new__(cls, model_id="/home/jovyan/api/meta-llama/Llama-3.1-8B"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLAMAMODEL, cls).__new__(cls)
                cls._instance._initialize(model_id)
        return cls._instance

    def _initialize(self, model_id):
        """Load the model only once."""
        self.model_id = model_id
        self.pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def query(self, messages, temperature=0.7, top_p=0.9, max_tokens=256):
        """Generate response using the model."""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries.")
        
        # Extract latest user message
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

        # OpenAI-like JSON response
        response = {
            "id": "cmpl-" + "".join(str(ord(c)) for c in user_prompt[:5]),  # Fake unique ID
            "object": "text_completion",
            "created": 1234567890,  # Dummy timestamp
            "model": self.model_id,
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

# Load model instance once
llama_model = LLAMAMODEL()

# Define API Request Schema
class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    """Handles API requests."""
    try:
        response = llama_model.query(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
