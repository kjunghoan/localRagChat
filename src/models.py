from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_device():
    """Determine best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device


def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    """Load and return model and tokenizer"""
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Model loaded successfully")
    return model, tokenizer
