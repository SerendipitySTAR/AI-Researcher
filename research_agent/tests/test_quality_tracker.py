# research_agent/tests/test_quality_tracker.py
import unittest
from research_agent.inno.quality_tracker import QualityTracker
from research_agent.inno.config import SmartExecutionConfig

class TestQualityTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = QualityTracker()
        self.config = SmartExecutionConfig()

    def test_consecutive_failures(self):
        self.assertEqual(self.tracker.consecutive_failures, 0)
        self.tracker.increment_consecutive_failure()
        self.assertEqual(self.tracker.consecutive_failures, 1)
        self.tracker.increment_consecutive_failure()
        self.assertEqual(self.tracker.consecutive_failures, 2)
        self.tracker.reset_consecutive_failures()
        self.assertEqual(self.tracker.consecutive_failures, 0)

    def test_quality_scores(self):
        self.assertIsNone(self.tracker.get_latest_quality_score())
        self.tracker.add_quality_score(0.8)
        self.assertEqual(self.tracker.get_latest_quality_score(), 0.8)
        self.tracker.add_quality_score(0.5)
        self.assertEqual(self.tracker.get_latest_quality_score(), 0.5)
        self.assertEqual(len(self.tracker.quality_scores), 2)

    def test_compilation_failure_rate(self):
        self.assertEqual(self.tracker.get_compilation_failure_rate(), 0.0)
        self.tracker.record_compilation_attempt(successful=True)
        self.assertEqual(self.tracker.get_compilation_failure_rate(), 0.0)
        self.tracker.record_compilation_attempt(successful=False) # 1 fail, 2 attempts
        self.assertAlmostEqual(self.tracker.get_compilation_failure_rate(), 1.0 / 2.0)
        self.tracker.record_compilation_attempt(successful=False) # 2 fails, 3 attempts
        self.assertAlmostEqual(self.tracker.get_compilation_failure_rate(), 2.0 / 3.0)
        self.tracker.record_compilation_attempt(successful=True) # 2 fails, 4 attempts
        self.assertAlmostEqual(self.tracker.get_compilation_failure_rate(), 2.0 / 4.0)

    def test_is_review_triggered_disabled(self):
        self.config.enable_smart_review = False
        self.tracker.consecutive_failures = self.config.consecutive_failure_threshold
        self.assertFalse(self.tracker.is_review_triggered(self.config))

    def test_is_review_triggered_consecutive_failures(self):
        self.tracker.consecutive_failures = self.config.consecutive_failure_threshold
        self.assertTrue(self.tracker.is_review_triggered(self.config))
        self.tracker.consecutive_failures = self.config.consecutive_failure_threshold - 1
        self.assertFalse(self.tracker.is_review_triggered(self.config))

    def test_is_review_triggered_quality_score(self):
        self.tracker.add_quality_score(self.config.quality_threshold - 0.1)
        self.assertTrue(self.tracker.is_review_triggered(self.config))
        self.tracker.quality_scores = [] # Reset
        self.tracker.add_quality_score(self.config.quality_threshold + 0.1)
        self.assertFalse(self.tracker.is_review_triggered(self.config))
    
    def test_is_review_triggered_no_score(self):
         self.assertFalse(self.tracker.is_review_triggered(self.config))


    def test_is_review_triggered_compilation_rate(self):
        # Trigger: 1 fail / 1 attempt > 0.5 threshold
        self.tracker.record_compilation_attempt(successful=False)
        self.assertTrue(self.tracker.is_review_triggered(self.config))
        
        self.setUp() # Reset tracker
        # Not triggered: 1 fail / 2 attempts = 0.5 (not > 0.5)
        self.tracker.record_compilation_attempt(successful=False)
        self.tracker.record_compilation_attempt(successful=True)
        self.assertFalse(self.tracker.is_review_triggered(self.config))

if __name__ == '__main__':
    unittest.main()
