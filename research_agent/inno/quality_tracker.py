# research_agent/inno/quality_tracker.py
from typing import List, Dict
from research_agent.inno.config import SmartExecutionConfig # Assuming this path is correct for imports

class QualityTracker:
    """
    Tracks various quality metrics during the AI research process to enable
    intelligent review triggers and potentially other adaptive behaviors.
    """
    def __init__(self):
        self.consecutive_failures: int = 0
        self.quality_scores: List[float] = [] # Stores recent quality scores
        self.compilation_attempts: int = 0
        self.compilation_failures: int = 0

    def increment_consecutive_failure(self) -> None:
        """Increments the count of consecutive failures."""
        self.consecutive_failures += 1

    def reset_consecutive_failures(self) -> None:
        """Resets the count of consecutive failures, typically after a success."""
        self.consecutive_failures = 0

    def add_quality_score(self, score: float) -> None:
        """Adds a new quality score to the tracker."""
        # Potentially keep only a window of recent scores if needed, for now, appends all.
        self.quality_scores.append(score)

    def record_compilation_attempt(self, successful: bool) -> None:
        """Records a compilation attempt and whether it was successful."""
        self.compilation_attempts += 1
        if not successful:
            self.compilation_failures += 1

    def get_compilation_failure_rate(self) -> float:
        """Calculates the current compilation failure rate."""
        if self.compilation_attempts == 0:
            return 0.0
        return self.compilation_failures / self.compilation_attempts

    def get_latest_quality_score(self) -> float | None:
        """Returns the most recent quality score, or None if no scores are available."""
        if not self.quality_scores:
            return None
        return self.quality_scores[-1]

    def is_review_triggered(self, config: SmartExecutionConfig) -> bool:
        """
        Checks if an intelligent review should be triggered based on the current
        state of the tracker and the provided configuration.
        """
        if not config.enable_smart_review:
            return False

        if self.consecutive_failures >= config.consecutive_failure_threshold:
            return True

        latest_score = self.get_latest_quality_score()
        if latest_score is not None and latest_score < config.quality_threshold:
            return True

        if self.get_compilation_failure_rate() > config.compilation_failure_rate_threshold:
            return True

        return False
