import torch
from typing import List, Dict
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass

ConversationHistory = List[Dict[str, str]]


@dataclass
class ChatConfig:
    """
    Configuration for chat settings
    """

    max_tokens: int
    max_length: int


def build_conversation_context(
    conversation_history: ConversationHistory, max_conversation_context: int = 6
) -> str:
    """
    Extract and format conversation history for prompt
    """
    context = ""
    conversation_context = 0 - max_conversation_context
    for msg in conversation_history[conversation_context:]:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        else:
            context += f"chatbot: {msg['content']}\n"
    return context


def extract_response_from_output(full_response: str) -> str:
    """
    Parse model output to extract clean response
    """
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
        # print("Used [/INST] extraction")
    elif "Please respond naturally" in full_response:
        parts = full_response.split("Please respond naturally to the latest message.")
        if len(parts) > 1:
            response = parts[-1].strip()
            if response.startswith("."):
                response = response[1:].strip()
            # print("Used instruction extraction")
        else:
            response = "I had trouble parsing that response."
    else:
        response = full_response.strip()
        # print("Used fallback - this shouldn't happen")

    # Clean up any remaining prefixes
    if response.startswith("chatbot:"):
        response = response[len("chatbot:") :].strip()

    return response


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_tokens: int,
    max_length: int,
) -> str:
    """
    Generate response from model given a formatted prompt
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response


def chat_turn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    conversation_history: ConversationHistory,
    # user_input: str,
    max_tokens: int,
    max_length: int,
) -> str:
    """
    Orchestrate a complete chat turn
    """
    # 1. Build context from history
    context = build_conversation_context(conversation_history)

    # 2. Create prompt
    prompt = f"<s>[INST] Previous conversation:\n{context}\nPlease respond naturally to the latest message. [/INST]"

    # 3. Generate response
    full_response = generate_response(model, tokenizer, prompt, max_tokens, max_length)

    # 4. Extract clean response
    clean_response = extract_response_from_output(full_response)

    return clean_response


def chat_loop(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, chat_config: ChatConfig
) -> None:
    """
    Interactive chat loop
    """
    max_tokens = chat_config.max_tokens
    max_length = chat_config.max_length

    conversation_history = []
    print("\nChat Started! Type 'exit' to end the chat.\n")

    while True:
        user_input = input("\nUser: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Ending chat session.")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Get response using updated context
        response = chat_turn(
            model,
            tokenizer,
            conversation_history,
            # user_input,
            max_tokens,
            max_length,
        )
        print(f"\nChatbot: {response}")

        # Add chatbot response to history
        conversation_history.append({"role": "chatbot", "content": response})
