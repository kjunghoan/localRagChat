# src/models/dialogpt.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..interfaces.model import (
    TransformerModelInterface,
    ModelConfig,
    ConversationHistory,
)
from ..utils.logger import create_logger


class DialoGPTModel(TransformerModelInterface):
    """
    DialoGPT-specific model implementation.
    Handles conversational format without instruction tags.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = create_logger(
            f"DialoGPTModel", False
        )  # Models use info level by default

    def load(self) -> None:
        """Load DialoGPT model and tokenizer"""
        self.logger.model(f"Loading DialoGPT model: {self.config.name}")

        # Load tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)

        # DialoGPT uses special tokens for conversation structure
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # DialoGPT is typically smaller, quantization less necessary but supported
        if self.config.use_quantization:
            self.logger.info(
                "Note: DialoGPT is smaller, quantization may not be needed"
            )
            # Could add quantization here if needed

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            low_cpu_mem_usage=True,
        )

        self.logger.success("DialoGPT model loaded successfully")

    def format_prompt(self, conversation_history: ConversationHistory) -> str:
        """
        Format conversation for DialoGPT's conversational format.
        DialoGPT expects natural conversation flow, not instruction format.
        """
        # DialoGPT works best with recent conversation context
        # Limit to last few exchanges to avoid overwhelming the model
        recent_history = (
            conversation_history[-6:]
            if len(conversation_history) > 6
            else conversation_history
        )

        # Build natural conversation
        conversation_text = ""
        # Use a placeholder EOS token if tokenizer not loaded yet
        eos_token = self.tokenizer.eos_token if self.tokenizer else "<|endoftext|>"

        for msg in recent_history:
            if msg["role"] == "user":
                conversation_text += f"{msg['content']}"
                conversation_text += eos_token  # End of user turn
            else:
                conversation_text += f"{msg['content']}"
                conversation_text += eos_token  # End of bot turn

        # The prompt is just the conversation so far
        # DialoGPT will continue the conversation naturally
        return conversation_text

    def parse_response(self, raw_output: str, debug: bool = False) -> str:
        """
        Parse DialoGPT's output to extract clean response.
        DialoGPT output is more natural, needs less aggressive parsing.
        """
        if debug:
            self.logger.debug(f"DialoGPT raw response length: {len(raw_output)}")
            self.logger.debug(f"Raw response preview: '{raw_output[:100]}...'")

        response = raw_output

        # Split by EOS tokens to get the new response
        if self.tokenizer.eos_token in raw_output:
            parts = raw_output.split(self.tokenizer.eos_token)
            # Get the last non-empty part (the new response)
            for part in reversed(parts):
                if part.strip():
                    response = part.strip()
                    break
            if debug:
                self.logger.debug("Used EOS token extraction")

        # Clean up any remaining artifacts
        response = response.strip()

        # Remove any repeated user input that might have leaked through
        # (DialoGPT sometimes echoes parts of the input)
        if debug:
            self.logger.debug(f"Extracted response: '{response}'")
            self.logger.debug(f"Final length: {len(response)}")

        return response

    def generate_raw(self, prompt: str, debug: bool = False) -> str:
        """
        Generate raw response using DialoGPT model.
        """
        if debug:
            self.logger.debug(f"DialoGPT prompt length: {len(prompt)}")
            self.logger.debug(f"Max tokens: {self.config.max_tokens}")

        # Encode the conversation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        input_length = input_ids.shape[1]
        if debug:
            self.logger.debug(f"Input token count: {input_length}")

        # Move to model device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                # DialoGPT specific settings
                no_repeat_ngram_size=3,  # Avoid repetition
                top_p=0.9,  # Nucleus sampling for more natural responses
            )

        # Decode full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if debug:
            output_length = outputs[0].shape[0]
            self.logger.debug(f"Output token count: {output_length}")
            self.logger.debug(f"New tokens generated: {output_length - input_length}")

        return full_response
