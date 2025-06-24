# src/models/mistral.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from ..interfaces.model import (
    TransformerModelInterface,
    # ModelConfig,
    ConversationHistory,
)


class MistralModel(TransformerModelInterface):
    """
    Mistral-specific model implementation.
    Handles Mistral's instruction format and response parsing.
    """

    def load(self) -> None:
        """Load Mistral model and tokenizer with quantization support"""
        print(f"Loading Mistral model: {self.config.name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization if requested
        if self.config.use_quantization:
            print("Using 4-bit quantization for Mistral")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                device_map=self.config.device_map,
                low_cpu_mem_usage=True,
            )

        print("Mistral model loaded successfully")

    def format_prompt(self, conversation_history: ConversationHistory) -> str:
        """
        Format conversation for Mistral's instruction format.
        Uses the [INST] tag format that Mistral expects.
        """
        # Build conversation context
        context = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"chatbot: {msg['content']}\n"

        # Wrap in Mistral's instruction format
        prompt = f"<s>[INST] Previous conversation:\n{context}\nPlease respond naturally to the latest message. [/INST]"

        return prompt

    def parse_response(self, raw_output: str, debug: bool = False) -> str:
        """
        Parse Mistral's output to extract clean response.
        Handles [/INST] tags and removes any prefixes.
        """
        if debug:
            print(f"\n🔍 DEBUG - Mistral raw response length: {len(raw_output)}")
            print(f"🔍 DEBUG - Raw response preview: '{raw_output[:100]}...'")

        response = raw_output

        # Handle [/INST] tag extraction
        if "[/INST]" in raw_output:
            response = raw_output.split("[/INST]")[-1].strip()
            if debug:
                print("🔍 DEBUG - Used [/INST] extraction")

        # Handle instruction-based extraction
        elif "Please respond naturally" in raw_output:
            parts = raw_output.split("Please respond naturally to the latest message.")
            if len(parts) > 1:
                response = parts[-1].strip()
                if response.startswith("."):
                    response = response[1:].strip()
                if debug:
                    print("🔍 DEBUG - Used instruction extraction")

        # Clean up common prefixes
        if response.startswith("chatbot:"):
            response = response[len("chatbot:") :].strip()

        if debug:
            print(f"🔍 DEBUG - Extracted response: '{response}'")
            print(f"🔍 DEBUG - Final length: {len(response)}")

        return response

    def generate_raw(self, prompt: str, debug: bool = False) -> str:
        """
        Generate raw response using Mistral model.
        """
        if debug:
            print(f"\n🔍 DEBUG - Prompt length: {len(prompt)}")
            print(f"🔍 DEBUG - Max tokens: {self.config.max_tokens}")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Could make this configurable
        )

        input_length = inputs["input_ids"].shape[1]
        if debug:
            print(f"🔍 DEBUG - Input token count: {input_length}")

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if debug:
            output_length = outputs[0].shape[0]
            print(f"🔍 DEBUG - Output token count: {output_length}")
            print(f"🔍 DEBUG - New tokens generated: {output_length - input_length}")

        return full_response
