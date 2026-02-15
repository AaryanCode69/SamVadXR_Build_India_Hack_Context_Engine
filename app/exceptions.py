"""
Exception classes raised by generate_vendor_response().

Dev B catches these and maps them to HTTP error codes for Unity.
See INTEGRATION_GUIDE.md §2.3 and rules.md §6.1 for the full contract.

    BrainServiceError  → Dev B returns 500 to Unity
    StateStoreError    → Dev B returns 503 to Unity
"""


class BrainServiceError(Exception):
    """Raised when the LLM call fails after all retries.

    Causes:
        - OpenAI API timeout (exceeded AI_TIMEOUT_MS)
        - OpenAI rate limit (429) exhausted after retries
        - JSON parse failure on LLM response after retry
        - Unexpected OpenAI SDK error

    Dev B should catch this and return HTTP 500 to Unity.
    """

    def __init__(self, message: str = "AI brain service unavailable") -> None:
        self.message = message
        super().__init__(self.message)


class StateStoreError(Exception):
    """Raised when Neo4j is unreachable or a state query fails.

    Causes:
        - Neo4j connection timeout
        - Neo4j driver error (auth, network)
        - Cypher query failure

    Dev B should catch this and return HTTP 503 to Unity.
    Game state cannot be trusted without the authoritative store.
    """

    def __init__(self, message: str = "State store unavailable") -> None:
        self.message = message
        super().__init__(self.message)
