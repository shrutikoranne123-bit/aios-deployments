import logging
from typing import List, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

MODEL_MAPPING = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/e5-large-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "gte-small": "thenlper/gte-small",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
    "all-minilm": "sentence-transformers/all-MiniLM-L12-v2",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "gte-large": "thenlper/gte-large",
    "uae-large": "WhereIsAI/UAE-Large-V1",
    "snowflake": "Snowflake/snowflake-arctic-embed-l",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "mxbai-embed-xsmall": "mixedbread-ai/mxbai-embed-xsmall-v1",
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
    "gte-base": "thenlper/gte-base",
    "e5-small-v2": "intfloat/e5-small-v2",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "paraphrase-multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Models that are served remotely (e.g. AWS Bedrock) and therefore have no
# local HuggingFace tokenizer. For chunk-size estimation we use tiktoken's
# offline cl100k_base BPE encoder as a stable, dependency-free proxy so the
# pipeline does not emit "model not supported" warnings or fall back to the
# coarse character-based estimate. Names are matched after normalization.
TIKTOKEN_PROXY_MODELS = {
    "cohere.embed-multilingual-v3",
    "cohere-embed-multilingual-v3",
    "cohere.embed-multilingual-v3.0",
    "cohere-embed-multilingual-v3.0",
    "embed-multilingual-v3",
    "cohere.embed-english-v3",
    "cohere-embed-english-v3",
}
_TIKTOKEN_PROXY_ENCODING = "cl100k_base"

_tokenizer_cache = {}


class _TiktokenTokenizer:
    """
    Minimal tokenizer adapter over a tiktoken encoding.

    Exposes the subset of the HuggingFace tokenizer API that the chunking
    layer relies on (``encode`` / ``decode`` / ``model_max_length``) so it can
    be used interchangeably with a ``PreTrainedTokenizerBase`` for token
    counting, without pulling any model weights over the network.
    """

    def __init__(self, encoding_name: str = _TIKTOKEN_PROXY_ENCODING):
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        # cl100k_base has no hard sequence limit; expose the common 512-token
        # window used elsewhere so get_max_tokens() returns a sensible value.
        self.model_max_length = 512

    def encode(self, text: str, **_: object) -> List[int]:
        return self._encoding.encode(text or "")

    def decode(self, tokens: List[int], **_: object) -> str:
        if not tokens:
            return ""
        return self._encoding.decode(tokens)


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to lowercase with hyphens."""
    return model_name.lower().replace("_", "-").strip()


def get_model_huggingface_id(model_name: str) -> Optional[str]:
    """
    Get the HuggingFace model ID for a given model name.
    
    Args:
        model_name: The model name (e.g., "bge-base", "BGE-Base")
    
    Returns:
        The HuggingFace model ID, or None if not found
    """
    normalized = normalize_model_name(model_name)
    return MODEL_MAPPING.get(normalized)


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load and cache a tokenizer for the given model using AutoTokenizer.
    
    Args:
        model_name: The model name (e.g., "bge-base", "BGE-Base")
    
    Returns:
        The loaded tokenizer
    
    Raises:
        ValueError: If the model is not supported
        Exception: If the tokenizer fails to load
    """
    normalized = normalize_model_name(model_name)
    
    # Check cache first
    if normalized in _tokenizer_cache:
        logger.debug(f"Returning cached tokenizer for {model_name}")
        return _tokenizer_cache[normalized]

    # Remotely served models (e.g. AWS Bedrock Cohere) have no local HF
    # tokenizer. Use an offline tiktoken-backed proxy for token counting.
    if normalized in TIKTOKEN_PROXY_MODELS:
        logger.info(
            "Using tiktoken '%s' proxy tokenizer for remotely served model '%s'",
            _TIKTOKEN_PROXY_ENCODING,
            model_name,
        )
        tokenizer = _TiktokenTokenizer(_TIKTOKEN_PROXY_ENCODING)
        _tokenizer_cache[normalized] = tokenizer
        return tokenizer

    # Get the HuggingFace model ID
    hf_model_id = get_model_huggingface_id(normalized)
    if not hf_model_id:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {list(MODEL_MAPPING.keys())}"
        )
    
    try:
        logger.info(
            "Loading tokenizer for %s from %s (first keyword search may take a little longer)",
            model_name,
            hf_model_id,
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        _tokenizer_cache[normalized] = tokenizer
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise


def get_max_tokens(model_name: str) -> int:
    """
    Get the maximum token length for a model.
    
    Args:
        model_name: The model name (e.g., "bge-base", "BGE-Base")
    
    Returns:
        The maximum token length, or a default value if not defined
    """
    try:
        tokenizer = get_tokenizer(model_name)
        max_tokens = tokenizer.model_max_length
        if max_tokens > 100000:
            return 512  # Default for models without explicit max length
        return max_tokens
    except Exception as e:
        logger.warning(f"Could not determine max tokens for {model_name}: {e}")
        return 512  # Default fallback


def print_model_stats():
    """Print max tokens for all supported models (for debugging)."""
    for model in MODEL_MAPPING:
        try:
            max_tokens = get_max_tokens(model)
            print(f"{model}: {max_tokens}")
        except Exception as e:
            print(f"{model}: Error loading tokenizer ({e})")
