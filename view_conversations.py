#!/usr/bin/env python3
"""
Simple script to view saved conversations
"""

from src.storage import ConversationStorage
# import json


def main():
    storage = ConversationStorage()

    print("📚 Saved Conversations:\n")

    # List all conversations
    conversations = storage.list_conversations()

    if not conversations:
        print("No conversations found. Chat first and exit to save a conversation!")
        return

    # Show conversation list
    for i, conv in enumerate(conversations):
        print(
            f"{i + 1}. ID: {conv['id'][:8]}... | Messages: {conv['message_count']} | Time: {conv['timestamp']}"
        )

    print("\n" + "=" * 60)

    # Show the most recent conversation in detail
    if conversations:
        latest = conversations[0]
        conv_id = latest["id"]

        print(f"\n📖 Most Recent Conversation (ID: {conv_id[:8]}...):\n")

        conversation = storage.get_conversation(conv_id)

        if conversation:
            for message in conversation:
                role = message["role"].title()
                content = message["content"]
                print(f"{role}: {content}")
                print()
        else:
            print("❌ Could not retrieve conversation content")

    print("=" * 60)
    print("\n💡 Want to see a specific conversation? Add the full ID as an argument")


if __name__ == "__main__":
    main()
