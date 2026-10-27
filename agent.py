# agent.py
# Lab 1.1 — Coder Agent with two-tier memory
#
# The main loop is already written.
# TODO: implement the body of the chat() function.

import os
from memory import SlidingWindowBuffer, store_memory, build_context, helicone_client

# Reuse the same Helicone-routed client from memory.py
client = helicone_client
MODEL = "gpt-4.1-mini"


def chat(user_input: str, window: SlidingWindowBuffer) -> str:
    """
    Single agent turn. Steps in order:
      1. Call store_memory() to save the user input to long-term memory.
      2. Call window.add() to add the user turn to the sliding window.
      3. Call build_context() to assemble the full message list.
      4. Call the OpenAI API with the assembled messages.
      5. Call window.add() to add the assistant reply to the sliding window.
      6. Return the reply string.
    """
    # Step 1: Save user input to long term memory
    memory_id = store_memory(user_input)

    # Step 2: Add the user turn to the sliding window
    window.add("user", user_input)

    # Step 3: Assemble the full message list for the OpenAI API call
    messages = build_context(user_input, window)

    # Step 4: Call the OpenAI API with the assembled messages
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    # Get the assistant's reply from the API response
    assistant_reply = response['choices'][0]['message']['content']

    # Step 5: Add the assistant reply to the sliding window
    window.add("assistant", assistant_reply)

    # Step 6: Return the reply string
    return assistant_reply


def main():
    from memory import SLIDING_WINDOW_SIZE
    print("=== Coder Agent — Lab 1.1 ===")
    print(f"Short-term window: {SLIDING_WINDOW_SIZE} turns | Long-term: Chroma semantic store")
    print("Type 'quit' to exit.\n")

    window = SlidingWindowBuffer()
    turn = 0

    while True:
        user_input = input(f"[Turn {turn + 1}] You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        reply = chat(user_input, window)
        turn += 1
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
