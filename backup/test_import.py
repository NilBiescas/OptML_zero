try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
    print("WORKS")
except ImportError as e:
    print(f"FAILED: {e}")
