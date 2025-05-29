# research_agent/tests/test_config.py
import unittest
from research_agent.inno.config import SmartExecutionConfig

class TestSmartExecutionConfig(unittest.TestCase):

    def test_default_values(self):
        config = SmartExecutionConfig()
        self.assertEqual(config.enable_smart_review, True)
        self.assertEqual(config.consecutive_failure_threshold, 2)
        self.assertEqual(config.quality_threshold, 0.3)
        self.assertEqual(config.compilation_failure_rate_threshold, 0.5)
        self.assertEqual(config.auto_batch_size_adjustment, True)
        self.assertEqual(config.gpu_memory_threshold, 0.8)
        self.assertEqual(config.checkpoint_frequency, 1)
        self.assertEqual(config.max_rollback_attempts, 3)

    def test_override_values(self):
        config = SmartExecutionConfig(
            enable_smart_review=False,
            quality_threshold=0.5,
            max_rollback_attempts=5
        )
        self.assertEqual(config.enable_smart_review, False)
        self.assertEqual(config.quality_threshold, 0.5)
        self.assertEqual(config.max_rollback_attempts, 5)
        # Check a default value to ensure others are not affected
        self.assertEqual(config.consecutive_failure_threshold, 2)

if __name__ == '__main__':
    unittest.main()
