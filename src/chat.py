import torch
from typing import List, Dict
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from dataclasses import dataclass

ConversationHistory = List[Dict[str, str]]


@dataclass
class ChatConfig:
    """Chat behavior configuration"""

    max_tokens: int = 150
    max_length: int = 1024
    context_messages: int = 6
    temperature: float = 0.7
    do_sample: bool = True


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


def extract_response_from_output(full_response: str, debug: bool = False) -> str:
    """
    Parse model output to extract clean response
    """
    if debug:
        print(f"\n🔍 DEBUG - Full response length: {len(full_response)}")
        if debug:  # Could add show_full_responses check here
            print(f"🔍 DEBUG - Full response: '{full_response}'")

    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
        if debug:
            print("🔍 DEBUG - Used [/INST] extraction")
    elif "Please respond naturally" in full_response:
        parts = full_response.split("Please respond naturally to the latest message.")
        if len(parts) > 1:
            response = parts[-1].strip()
            if response.startswith("."):
                response = response[1:].strip()
            if debug:
                print("🔍 DEBUG - Used instruction extraction")
        else:
            response = "I had trouble parsing that response."
    else:
        response = full_response.strip()
        if debug:
            print("🔍 DEBUG - Used fallback extraction")

    # Clean up any remaining prefixes
    if response.startswith("chatbot:"):
        response = response[len("chatbot:") :].strip()

    if debug:
        print(f"🔍 DEBUG - Extracted response: '{response}'")
        print(f"🔍 DEBUG - Extracted length: {len(response)}")

    return response


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    chat_config,  # ChatConfig from src.config
    debug: bool = False,
) -> str:
    """
    Generate response from model given a formatted prompt
    """
    if debug:
        print(f"\n🔍 DEBUG - Prompt length: {len(prompt)}")
        print(
            f"🔍 DEBUG - Max tokens: {chat_config.max_tokens}, Max length: {chat_config.max_length}"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=chat_config.max_length,
    )

    input_length = inputs["input_ids"].shape[1]  # type: ignore
    if debug:
        print(f"🔍 DEBUG - Input token count: {input_length}")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=chat_config.max_tokens,
            temperature=chat_config.temperature,
            do_sample=chat_config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_length = outputs[0].shape[0]

    if debug:
        print(f"🔍 DEBUG - Output token count: {output_length}")
        print(f"🔍 DEBUG - New tokens generated: {output_length - input_length}")

    return full_response


def chat_turn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    conversation_history: ConversationHistory,
    chat_config: ChatConfig,
    context_messages: int = 6,
    debug: bool = False,
) -> str:
    """
    Orchestrate a complete chat turn
    """
    # 1. Build context from history
    context = build_conversation_context(conversation_history, context_messages)

    # 2. Create prompt
    prompt = f"<s>[INST] Previous conversation:\n{context}\nPlease respond naturally to the latest message. [/INST]"

    # 3. Generate response
    full_response = generate_response(model, tokenizer, prompt, chat_config, debug)

    # 4. Extract clean response
    clean_response = extract_response_from_output(full_response, debug)

    return clean_response


def chat_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    chat_config,  # ChatConfig from src.config
    debug: bool = False,
    context_messages: int = 6,
    storage=None,  # Optional ConversationStorage
) -> None:
    """
    Interactive chat loop with configuration
    """
    conversation_history = []
    print("\nChat Started! Type 'exit' to end the chat.\n")

    if debug:
        print("🔍 DEBUG MODE ENABLED")

    while True:
        user_input = input("\nUser: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            # Store conversation before exiting
            if storage and conversation_history:
                try:
                    conv_id = storage.store_conversation(conversation_history)
                    print(f"💾 Conversation saved with ID: {conv_id}")
                except Exception as e:
                    print(f"⚠️  Failed to save conversation: {e}")

            print("Ending chat session.")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Get response
        response = chat_turn(
            model, tokenizer, conversation_history, chat_config, context_messages, debug
        )
        print(f"\nChatbot: {response}")

        # Add chatbot response to history
        conversation_history.append({"role": "chatbot", "content": response})
