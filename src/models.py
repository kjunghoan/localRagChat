from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM as ModelType
from transformers.models.auto.tokenization_auto import AutoTokenizer as TokenizerType
import torch
from typing import Tuple

from transformers.utils.quantization_config import BitsAndBytesConfig


def get_device() -> torch.device:
    """Determine best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device


def load_model_and_tokenizer(
    model_name: str, use_quantization: bool = True
) -> Tuple[ModelType, TokenizerType]:
    """Load and return model and tokenizer"""
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_quantization:
        print("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    print("Model loaded successfully")
    return model, tokenizer
