from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv

def authenticate():
    """Handle Hugging Face auth"""

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    print(f"token loaded: {token is not None}")

    if token:
        print("logging in...")
        login(token)
    else:
        print("No token found, using cached login...")
        login()

def get_device():
    """See what devices are available"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using cuda gpu: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("using cpu")
        return device

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """Load and return model and tokenizer"""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Model loaded successfully")
    return model, tokenizer

def generate_response(model,tokenizer, prompt, max_tokens=10):
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    authenticate()
    model, tokenizer = load_model_and_tokenizer()

    test_prompt = "hello, how are you?"
    print(f"Testing with: {test_prompt}")

    response = generate_response(model, tokenizer, test_prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
