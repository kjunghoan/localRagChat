#!/usr/bin/env python3
"""
Debug script to inspect raw ChromaDB storage
"""

from src.storage import ConversationStorage
import json


def main():
    storage = ConversationStorage()

    print("🔍 Raw ChromaDB Data Inspection\n")

    # Get raw data from ChromaDB
    raw_result = storage.collection.get()

    print("📊 Collection Overview:")
    print(f"Total conversations: {len(raw_result['ids'])}")
    print(f"IDs: {raw_result['ids']}")
    print()

    if raw_result["ids"]:
        # Show detailed data for each conversation
        for i, conv_id in enumerate(raw_result["ids"]):
            print(f"🗂️  Conversation {i + 1}: {conv_id}")
            print("-" * 50)

            # Show metadata
            metadata = raw_result["metadatas"][i]
            print("📋 Metadata:")
            for key, value in metadata.items():
                if key == "conversation_json":
                    print(f"  {key}: [JSON data - {len(value)} chars]")
                else:
                    print(f"  {key}: {value}")

            # Show document (the text used for embedding)
            document = raw_result["documents"][i]
            print(f"\n📄 Document text (for embedding):")
            print(f"'{document[:200]}{'...' if len(document) > 200 else ''}'")

            # Show embeddings info (if available)
            if raw_result["embeddings"] and raw_result["embeddings"][i]:
                embedding = raw_result["embeddings"][i]
                print(f"\n🧮 Embedding:")
                print(f"  Vector length: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
            else:
                print(f"\n🧮 Embedding: [Generated automatically by ChromaDB]")

            # Show raw conversation JSON
            conversation_json = metadata.get("conversation_json", "[]")
            conversation = json.loads(conversation_json)
            print(f"\n💬 Raw Conversation Data:")
            print(json.dumps(conversation, indent=2))

            print("\n" + "=" * 60 + "\n")

    else:
        print("No conversations found!")


if __name__ == "__main__":
    main()
