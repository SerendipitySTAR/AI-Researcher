from typing import List

# Thresholds and parameters (these can be tuned)
INITIAL_INTERVENTION_QUALITY_THRESHOLD = 0.6 # If first attempt quality is below this, intervene
CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD = 2 # If this many consecutive failures, intervene

QUALITY_PLATEAU_WINDOW = 3 # Check for plateau over this many recent scores
QUALITY_PLATEAU_THRESHOLD = 0.02 # Max improvement within window to be considered a plateau
HIGH_QUALITY_THRESHOLD = 0.9 # If quality is above this and execution is successful, consider stopping
MIN_ITERATIONS_BEFORE_STOPPING = 3 # Minimum iterations before intelligent stopping can occur (unless max_iter reached)

def should_request_human_intervention(
    code_generation_attempt: int, 
    code_quality_score: float, 
    execution_success_history: List[bool],
    is_initial_code: bool = False
) -> bool:
    """
    Determines if human intervention should be requested.
    - Intervenes on initial low-quality code.
    - Intervenes if there are too many consecutive execution failures.
    """
    if is_initial_code and code_quality_score < INITIAL_INTERVENTION_QUALITY_THRESHOLD:
        # print(f"Intervention recommended: Initial code quality ({code_quality_score:.2f}) is below threshold ({INITIAL_INTERVENTION_QUALITY_THRESHOLD}).")
        return True

    if len(execution_success_history) >= CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD:
        # Check for consecutive failures at the end of the history
        if all(not success for success in execution_success_history[-CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD:]):
            # print(f"Intervention recommended: {CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD} consecutive execution failures.")
            return True
            
    return False

def intelligent_stopping_condition(
    quality_score_history: List[float], 
    execution_success_history: List[bool], 
    iteration_count: int, 
    max_iterations: int
) -> bool:
    """
    Determines if the code optimization process should dynamically stop.
    Conditions for stopping:
    1. Max iterations reached.
    2. High quality achieved with successful execution (after min iterations).
    3. Quality score has plateaued (after min iterations).
    4. Too many consecutive failures (might lead to intervention first, then stopping if not resolved).
    """
    if iteration_count >= max_iterations:
        # print("Stopping condition met: Maximum iterations reached.")
        return True

    # Only apply other dynamic conditions after a minimum number of iterations
    if iteration_count < MIN_ITERATIONS_BEFORE_STOPPING:
        return False

    # Condition 2: High quality with successful execution
    if quality_score_history and execution_success_history:
        if quality_score_history[-1] >= HIGH_QUALITY_THRESHOLD and execution_success_history[-1]:
            # print(f"Stopping condition met: High quality ({quality_score_history[-1]:.2f}) achieved with successful execution.")
            return True

    # Condition 3: Quality score has plateaued
    if len(quality_score_history) >= QUALITY_PLATEAU_WINDOW:
        recent_scores = quality_score_history[-QUALITY_PLATEAU_WINDOW:]
        if (max(recent_scores) - min(recent_scores)) < QUALITY_PLATEAU_THRESHOLD:
            # print(f"Stopping condition met: Quality score has plateaued. Recent scores: {recent_scores}")
            return True
            
    # Condition 4 (related to intervention but can also inform stopping if not resolved):
    # This is partially handled by should_request_human_intervention. 
    # If intervention is requested and not resolved, the loop might break elsewhere.
    # However, if there's a long string of failures, it might be wise to stop regardless.
    if len(execution_success_history) >= CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD: # Using same threshold for consistency
        if all(not success for success in execution_success_history[-CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD:]):
            # print(f"Stopping condition met: Persistent consecutive execution failures ({CONSECUTIVE_FAILURE_INTERVENTION_THRESHOLD} times).")
            return True

    return False

# Example Usage (for testing functions, not for production)
if __name__ == "__main__":
    # Test should_request_human_intervention
    print("Testing should_request_human_intervention:")
    print(f"Initial low quality (0.5): {should_request_human_intervention(1, 0.5, [], is_initial_code=True)}") # True
    print(f"Initial good quality (0.7): {should_request_human_intervention(1, 0.7, [], is_initial_code=True)}") # False
    print(f"2 consecutive failures: {should_request_human_intervention(3, 0.6, [True, False, False])}") # True
    print(f"1 failure: {should_request_human_intervention(2, 0.6, [True, False])}") # False
    print(f"3 failures, but not all consecutive recently: {should_request_human_intervention(4, 0.6, [False, True, False, False])}") # True
    print(f"No failures: {should_request_human_intervention(3, 0.8, [True, True, True])}") # False

    # Test intelligent_stopping_condition
    print("\nTesting intelligent_stopping_condition:")
    print(f"Max iterations (10/10): {intelligent_stopping_condition([0.7]*10, [True]*10, 10, 10)}") # True
    print(f"Not enough iterations (2/10): {intelligent_stopping_condition([0.7]*2, [True]*2, 2, 10)}") # False
    print(f"High quality (0.95) & success (iter 5/10): {intelligent_stopping_condition([0.7, 0.8, 0.9, 0.92, 0.95], [True]*5, 5, 10)}") # True
    print(f"High quality (0.95) but failed execution (iter 5/10): {intelligent_stopping_condition([0.7, 0.8, 0.9, 0.92, 0.95], [True]*4 + [False], 5, 10)}") # False
    print(f"Quality plateau ([0.7, 0.71, 0.705]) (iter 5/10): {intelligent_stopping_condition([0.6, 0.65, 0.7, 0.71, 0.705], [True]*5, 5, 10)}") # True
    print(f"Quality increasing ([0.7, 0.8, 0.9]) (iter 5/10): {intelligent_stopping_condition([0.6, 0.65, 0.7, 0.8, 0.9], [True]*5, 5, 10)}") # False
    print(f"Persistent failures (3 consecutive): {intelligent_stopping_condition([0.5]*5, [True, True, False, False, False], 5, 10)}") # True
    print(f"Min iterations not met for plateau (iter 2/10): {intelligent_stopping_condition([0.7, 0.71], [True]*2, 2, 10)}") # False
