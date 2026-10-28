# memory.py
# Lab 1.1 — Two-tier memory system for the Coder Agent
#
# TODO: Implement the three components marked below.
# Read the docstring for each before writing any code.
# build_context() and the Chroma setup are already done for you.

import os
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

SLIDING_WINDOW_SIZE = 6
TOP_K_MEMORIES = 3

# ── Helicone-routed client (shared with agent.py) ─────────────────────────────

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

helicone_client = OpenAI(
    # Helicone key doubles as the auth token.
    api_key=_OPENROUTER_API_KEY,
    base_url=_HELICONE_BASE,
    default_headers={
        "Helicone-Auth": f"Bearer {_HELICONE_API_KEY}",
    },
)

# ── Custom embedding function routing through Helicone ────────────────────────


class HeliconeEmbeddingFunction(EmbeddingFunction):
    """Routes all embedding calls through Helicone instead of calling OpenAI directly."""

    def __call__(self, input: Documents) -> Embeddings:
        response = helicone_client.embeddings.create(
            model="text-embedding-3-small", input=input
        )
        return [item.embedding for item in response.data]


# ── Chroma setup (already done for you) ──────────────────────────────────────

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="agent_memory",
    embedding_function=HeliconeEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)

# ── TODO 1: SlidingWindowBuffer ───────────────────────────────────────────────


class SlidingWindowBuffer:
    """Short-term, in-memory conversation history with a bounded sliding window.

    Maintains the most recent conversation turns so the agent has immediate
    conversational context for the current exchange. When the window fills
    up, the oldest turn is evicted — only the most recent ``max_turns``
    turns are retained.

    This is the first tier of the two-tier memory system:
        * Tier 1 — this buffer (short-term, in-order, lossy, bounded).
        * Tier 2 — Chroma vector store via ``store_memory`` / ``retrieve_memories``
          (long-term, semantic, persistent).

    Each turn is a dict of the form::

        {"role": "user" | "assistant", "content": str}

    matching the message format expected by OpenAI-compatible chat APIs.

    Attributes:
        max_turns: Maximum number of turns the buffer will hold. Once this
            many turns are stored, each subsequent ``add`` evicts the oldest.
            Defaults to ``SLIDING_WINDOW_SIZE`` (6).

    Examples:
        >>> buf = SlidingWindowBuffer(max_turns=3)
        >>> buf.add("user", "hello")
        >>> buf.add("assistant", "hi there")
        >>> len(buf)
        2
        >>> buf.get()
        [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi there'}]
    """

    def __init__(self, max_turns: int = SLIDING_WINDOW_SIZE):
        """Initialize an empty buffer.

        Args:
            max_turns: Maximum number of turns to retain. Older turns are
                evicted (FIFO) once this capacity is exceeded. Defaults to
                ``SLIDING_WINDOW_SIZE``.
        """
        self.max_turns = max_turns
        self._turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a new conversation turn, evicting the oldest if over capacity.

        The turn is appended to the end (most recent position). If the buffer
        then holds more than ``max_turns`` turns, the turn at index 0
        (the oldest) is removed — a FIFO sliding window.

        Args:
            role: The speaker role, typically ``"user"`` or ``"assistant"``.
            content: The message text for this turn.
        """
        # append the new turn to the end (most recent)
        self._turns.append({"role": role, "content": content})
        # evict the oldest turn if we exceed the configured max_turns
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get(self) -> list[dict]:
        """Return the current turns in chronological order (oldest first).

        Returns a shallow copy of the internal list so callers can freely
        mutate the result without affecting the buffer's state.

        Returns:
            A list of turn dicts, ordered oldest -> newest, each shaped
            ``{"role": str, "content": str}``.
        """
        # return the current list of turns in chronological order
        return list(self._turns)

    def __len__(self) -> int:
        """Return the number of turns currently held in the buffer."""
        return len(self._turns)


# ── TODO 2: store_memory ──────────────────────────────────────────────────────

_memory_counter = 0


def store_memory(text: str, metadata: dict | None = None) -> str:
    """Embed and persist a string in the Chroma long-term memory collection.

    This is the write half of Tier 2 of the two-tier memory system. The
    supplied ``text`` is embedded automatically by the collection's
    ``HeliconeEmbeddingFunction`` and stored alongside a generated unique ID,
    making it retrievable later via :func:`retrieve_memories`.

    A monotonically increasing, zero-padded ID is generated for each call
    (e.g. ``mem_0001``, ``mem_0002``), so the returned ID is unique across
    the lifetime of the process.

    Args:
        text: The raw text to embed and store.
        metadata: Optional dict of arbitrary key/value pairs to attach to the
            stored document (e.g. ``{"role": "user", "turn": 4}``). Passed
            straight through to Chroma; ``None`` means no metadata is stored.

    Returns:
        The generated unique ID string for the stored memory
        (e.g. ``"mem_0001"``).

    Examples:
        >>> store_memory("The agent uses Chroma for long-term memory.")
        'mem_0001'
        >>> store_memory("Python 3.11 introduced type parameter syntax.")
        'mem_0002'
    """
    global _memory_counter
    _memory_counter += 1
    mem_id = f"mem_{_memory_counter:04d}"

    collection.add(
        ids=[mem_id],
        documents=[text],
        metadatas=[metadata] if metadata is not None else None,
    )

    return mem_id


# ── TODO 3: retrieve_memories ─────────────────────────────────────────────────


def retrieve_memories(query: str, top_k: int = TOP_K_MEMORIES) -> list[str]:
    """
    Return the top-k most semantically similar memories for a query string.
    The query is matched against the embedded memory contents in the Chroma collection.

    If the collection is empty, return an empty list to avoid errors.
    The maximum number of results returned is limited to the smaller of
    ``top_k`` or the total number of memories stored.

    Args:
        query: The query string to compare against stored memory.
        top_k: The maximum number of results to return, defaults to
            ``TOP_K_MEMORIES``.

    Returns:
        list[str]: A list of document strings corresponding to the most semantically similar memories. The list contains only
        the document text, not metadata or IDs.
    """

    # guard against an empty collection to avoid a query error / empty results
    if collection.count() == 0:
        return []

    # never ask for more results than the collection actually holds
    n_results = min(top_k, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    # results["documents"] is a list of lists (one per query); we sent one query
    return results["documents"][0]


# ── build_context (already done for you) ─────────────────────────────────────


def build_context(query: str, window: SlidingWindowBuffer) -> list[dict]:
    """
    Assembles the full message list for an LLM call:
      1. System prompt with retrieved long-term memories injected
      2. Recent turns from the sliding window
    """
    long_term = retrieve_memories(query)

    memory_block = ""
    if long_term:
        formatted = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(long_term))
        memory_block = (
            f"\n\nRelevant memories from earlier in this session:\n{formatted}"
        )

    system_message = {
        "role": "system",
        "content": (
            "You are a Coder Agent. Help with coding tasks. "
            "Use the memories below if relevant to the current question." + memory_block
        ),
    }
    return [system_message] + window.get()
