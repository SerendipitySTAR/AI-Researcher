import random # For placeholder logic

def get_estimated_code_quality_score(ml_agent_output: str, judge_agent_feedback: str) -> float:
    """
    Placeholder function to estimate code quality score based on ML agent output and Judge agent feedback.
    In a real scenario, this would involve NLP, code analysis, and heuristic models.
    Returns a score between 0.0 (poor) and 1.0 (excellent).
    """
    # Placeholder: Simple keyword-based scoring for demonstration
    score = 0.5 # Neutral baseline
    if "error" in judge_agent_feedback.lower() or "issue" in judge_agent_feedback.lower():
        score -= 0.2
    if "correct" in judge_agent_feedback.lower() or "good" in judge_agent_feedback.lower():
        score += 0.2
    if "complex" in ml_agent_output.lower() or len(ml_agent_output) > 2000: # Longer output might indicate complexity
        score -= 0.1
    if "simple" in ml_agent_output.lower() or len(ml_agent_output) < 500:
        score += 0.1
    
    # Ensure score is within bounds [0, 1]
    return max(0.0, min(1.0, round(score, 2)))

def get_estimated_error_count(judge_agent_feedback: str) -> int:
    """
    Placeholder function to estimate the number of errors based on Judge agent feedback.
    This would typically involve parsing the feedback to identify distinct issues.
    """
    # Placeholder: Count occurrences of "error" or "issue"
    errors = judge_agent_feedback.lower().count("error")
    issues = judge_agent_feedback.lower().count("issue")
    # A simple heuristic: if "fully_correct": true is present, error count is 0
    if '"fully_correct": true' in judge_agent_feedback.lower():
        return 0
    return errors + issues

def get_execution_success(judge_agent_feedback: str) -> bool:
    """
    Placeholder function to determine if the code execution (simulated or actual) was successful.
    This would parse Judge agent feedback for execution status.
    """
    # Placeholder: Check for keywords indicating success or failure
    if '"fully_correct": true' in judge_agent_feedback.lower(): # Explicit success
        return True
    if "error during execution" in judge_agent_feedback.lower() or "failed to run" in judge_agent_feedback.lower():
        return False
    # Default to True if no strong failure signals, assuming some partial success or successful compilation
    return True

def should_request_human_intervention(
    ml_agent_output: str,
    judge_agent_feedback: str,
    current_iteration: int,
    max_iterations: int,
    execution_success_history: list[bool], # History of execution success from previous relevant steps
    quality_score_history: list[float] # History of quality scores
    ) -> bool:
    """
    Determines whether human intervention should be requested based on various factors.

    Args:
        ml_agent_output (str): The output from the ML Agent (generated code).
        judge_agent_feedback (str): Feedback from the Judge Agent.
        current_iteration (int): The current iteration number in a loop (0 for initial review).
        max_iterations (int): The maximum number of iterations allowed for the loop.
        execution_success_history (list[bool]): History of execution success from previous relevant steps.
        quality_score_history (list[float]): History of quality scores from previous relevant steps.

    Returns:
        bool: True if human intervention is recommended, False otherwise.
    """
    estimated_quality = get_estimated_code_quality_score(ml_agent_output, judge_agent_feedback)
    estimated_errors = get_estimated_error_count(judge_agent_feedback)
    current_execution_successful = get_execution_success(judge_agent_feedback)

    # Update histories (caller should manage the actual list objects)
    # For the purpose of this function, we assume they are passed correctly reflecting current state.
    
    print(f"Intervention check for Iteration {current_iteration}:")
    print(f"  Estimated Quality: {estimated_quality}, Estimated Errors: {estimated_errors}, Execution Successful: {current_execution_successful}")
    print(f"  Execution Success History (current cycle): {execution_success_history}")
    print(f"  Quality Score History (current cycle): {quality_score_history}")


    # --- Intervention Triggers ---

    # 1. Critical Failure: Code doesn't run or has significant errors early on.
    if not current_execution_successful and current_iteration < 2: # Early failure
        print("  Intervention Trigger: Critical failure in early iteration.")
        return True

    # 2. Stagnation or Decline: Quality isn't improving or is getting worse.
    if len(quality_score_history) >= 2: # Need at least two past scores to compare
        # Simple check: if quality drops or stays same for 2 consecutive iterations
        if quality_score_history[-1] <= quality_score_history[-2] and estimated_quality <= quality_score_history[-1]:
            print("  Intervention Trigger: Stagnation or decline in code quality.")
            return True
    
    if len(execution_success_history) >= 2 and not any(execution_success_history[-2:]): # Two consecutive failures
        print("  Intervention Trigger: Consecutive execution failures.")
        return True

    # 3. High Error Count: Too many errors flagged by the Judge.
    if estimated_errors > 5: # Threshold for "too many" errors
        print("  Intervention Trigger: High estimated error count.")
        return True
    
    # 4. Low Quality Score: Persistently low quality.
    if estimated_quality < 0.3 and current_iteration > 1: # Persistently very low score after initial attempts
        print("  Intervention Trigger: Persistently low quality score.")
        return True

    # 5. Nearing Max Iterations without success: If approaching max_iterations and still not good.
    if current_iteration >= max_iterations -1 : # If this is the second to last or last iteration
        if not current_execution_successful or estimated_quality < 0.6:
             print("  Intervention Trigger: Nearing max iterations without sufficient quality/success.")
             return True


    # --- Confidence-based Intervention (Less critical, more for proactive help) ---
    # 6. Moderate quality with some errors, suggest review to speed up.
    #    This could be after a few iterations if quality is mediocre.
    if current_iteration > 1 and 0.4 <= estimated_quality <= 0.7 and estimated_errors > 1:
        print("  Intervention Trigger: Moderate quality with errors, proactive review suggested.")
        return True
        
    # --- Default: No intervention if none of the above conditions are met ---
    # Especially if quality is high and errors are low or zero.
    if estimated_quality > 0.8 and estimated_errors <= 1 and current_execution_successful:
        print("  No Intervention: Quality is high, errors are low, execution successful.")
        return False

    print("  No specific intervention trigger met based on current heuristics.")
    # Fallback decision: if not clearly good or bad, err on side of caution for early iterations
    # or if no strong signal otherwise. This part can be refined.
    # For now, if no strong reason to intervene, and not clearly excellent, don't intervene unless early.
    # This function is a placeholder, so the default non-intervention is reasonable if not explicitly triggered.
    return False # Default to no intervention if no specific rule is met.
    # A more sophisticated version might have a weighted scoring of these factors.

def intelligent_stopping_condition(
    judge_result_str: str,
    current_loop_iter_idx: int,  # 0-indexed loop variable i
    max_loop_iters: int,
    quality_score_history: list[float], 
    logger=None 
) -> bool:
    def _log(msg):
        if logger:
            logger.info(f"[StoppingCondition] {msg}") # Use info level for logger
        else:
            print(f"[StoppingCondition] {msg}")

    if not quality_score_history:
        _log("Quality score history is empty. Not stopping.")
        return False
    current_score = quality_score_history[-1]
    iteration_count_formula = current_loop_iter_idx + 1 # current_loop_iter_idx is 0-indexed

    _log(f"Inputs: current_loop_iter_idx={current_loop_iter_idx}, max_loop_iters={max_loop_iters}, current_score={current_score:.2f}, history_len={len(quality_score_history)}")

    execution_success = get_execution_success(judge_result_str) # Assuming get_execution_success is available
    if '"fully_correct": true' in judge_result_str and execution_success:
        _log("'fully_correct' is true and execution successful. Requesting stop.")
        return True

    # Check improvement rate only if there are at least two scores in history
    if len(quality_score_history) >= 2: # This means at least two iterations have completed (idx 0 and idx 1)
        previous_score = quality_score_history[-2]
        # Handle previous_score = 0 to avoid division by zero
        if abs(previous_score) > 1e-5:
            improvement_rate = (current_score - previous_score) / abs(previous_score)
        elif abs(current_score - previous_score) < 1e-5: # Both near zero
            improvement_rate = 0.0
        else: # previous_score is near zero, current_score is not
            improvement_rate = float('inf') # Effectively a large improvement

        _log(f"Improvement rate: {improvement_rate:.2f} (current: {current_score:.2f}, prev: {previous_score:.2f})")
        
        # Stop if improvement rate is low after at least one actual refinement iteration (current_loop_iter_idx >= 1)
        # current_loop_iter_idx = 0 is the first pass through the loop (second overall evaluation after initial)
        # current_loop_iter_idx = 1 is the second pass (third overall evaluation)
        # We need at least one iteration *within the loop* to measure improvement from.
        # So, this condition applies from the end of loop iteration i=1 onwards.
        if improvement_rate < 0.1 and current_loop_iter_idx >= 1: 
            _log(f"Improvement rate ({improvement_rate:.2f}) < 0.1 and current_loop_iter_idx ({current_loop_iter_idx}) >= 1. Requesting stop.")
            return True
    
    if iteration_count_formula >= max_loop_iters: 
        if current_score > 0.7:
            _log(f"Max iterations ({iteration_count_formula}) reached and current score ({current_score:.2f}) > 0.7. Requesting stop.")
            return True
        else:
            # This case means max iterations reached, but score is not high enough.
            # The loop will terminate naturally due to the for loop condition.
            # The function should return False here as the *intelligent* condition to stop early isn't met.
            _log(f"Max iterations ({iteration_count_formula}) reached but score ({current_score:.2f}) <= 0.7. Loop will terminate naturally.")
            # No explicit stop requested here; loop terminates on its own.

    _log("No intelligent stopping condition met to stop early.")
    return False
