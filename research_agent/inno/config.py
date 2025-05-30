# research_agent/inno/config.py
from dataclasses import dataclass

@dataclass
class SmartExecutionConfig:
    """
    Configuration for smart execution features, including intelligent review,
    resource management, and checkpointing.
    """
    # --- Intelligent Review ---
    # Enables or disables the intelligent review mechanism.
    enable_smart_review: bool = True
    # The number of consecutive failures that trigger an intelligent review.
    consecutive_failure_threshold: int = 2
    # The quality score threshold below which an intelligent review is triggered.
    # This score is typically evaluated by a JudgeAgent or similar mechanism.
    quality_threshold: float = 0.3
    # The code compilation failure rate (0.0 to 1.0) above which an intelligent review is triggered.
    compilation_failure_rate_threshold: float = 0.5

    # --- Resource Management ---
    # Enables or disables automatic batch size adjustment based on GPU memory.
    auto_batch_size_adjustment: bool = True
    # The target GPU memory utilization (0.0 to 1.0). If actual utilization exceeds this,
    # adjustments (e.g., reducing batch size) may be made.
    gpu_memory_threshold: float = 0.8

    # --- Checkpoints & Backtracking ---
    # Frequency of checkpoints. The interpretation depends on the workflow,
    # e.g., 1 might mean after every major phase or agent execution.
    checkpoint_frequency: int = 1  # Per phase or major step
    # Maximum number of times the system will attempt to roll back to a checkpoint
    # upon encountering an unrecoverable error.
    max_rollback_attempts: int = 3
