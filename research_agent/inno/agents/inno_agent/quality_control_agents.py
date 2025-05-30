# research_agent/inno/agents/inno_agent/quality_control_agents.py
from research_agent.inno.types import Agent
from research_agent.inno.registry import register_agent
from research_agent.inno.environment.docker_env import DockerEnv # If needed
# from research_agent.inno.environment.browser_env import BrowserEnv # If needed
# from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser # If needed

# Placeholder for a function that might be used by these agents if they need to resolve a case.
# Similar to case_resolved in judge_agent.py
def qc_case_resolved(context_variables, assessment: str, status: str, suggestions: list = None):
    """
    Used by quality control agents to provide their assessment.
    Args:
       context_variables: The context variables.
       assessment: A summary of the assessment.
       status: e.g., "PlanValid", "PlanInvalid", "QualityAcceptable", "QualityLow".
       suggestions: A list of suggestions, if any.
    """
    result = {
        "assessment": assessment,
        "status": status,
        "suggestions": suggestions or []
    }
    # Storing in context_variables might not be standard for all agents,
    # but shown here for consistency if needed.
    # Ensure agent_name exists or provide a default to avoid AttributeError
    agent_name_key = context_variables.get('agent_name', 'qc_agent')
    context_variables[f"{agent_name_key.lower().replace(' ', '_')}_assessment"] = result
    return str(result) # Agents typically return strings as tool call results

@register_agent("get_plan_validator_agent")
def get_plan_validator_agent(model: str, **kwargs) -> Agent:
    code_env: DockerEnv = kwargs.get("code_env")

    def instructions(context_variables):
        plan_content = context_variables.get("plan_content", "No plan provided.")
        # Ensure agent_name is passed or available in context_variables for qc_case_resolved
        context_variables['agent_name'] = "Plan Validator Agent"
        return f"""
You are a Plan Validator Agent. Your task is to assess the executability and coherence of a given research/coding plan.

The plan you need to validate is:
--- PLAN START ---
{plan_content}
--- PLAN END ---

Consider the following:
- Are the steps clear and actionable?
- Are there any obvious contradictions or missing prerequisite steps?
- Does the plan seem feasible given standard tools and resources?
- Are the objectives well-defined?

Provide your assessment using the `qc_case_resolved` function.
If the plan is invalid or has significant issues, provide clear suggestions for improvement.
Set status to "PlanValid" or "PlanInvalid".
"""
    
    return Agent(
        name="Plan Validator Agent",
        model=model,
        instructions=instructions,
        functions=[qc_case_resolved],
        tool_choice="required" 
    )

@register_agent("get_quality_assurance_agent")
def get_quality_assurance_agent(model: str, **kwargs) -> Agent:
    code_env: DockerEnv = kwargs.get("code_env")
    # web_env: BrowserEnv = kwargs.get("web_env")
    # file_env: RequestsMarkdownBrowser = kwargs.get("file_env")

    def instructions(context_variables):
        # This agent would typically receive more context, like the code,
        # previous judge's remarks, quality tracker status, etc.
        # For now, it's a general instruction.
        ml_dev_res = context_variables.get("ml_dev_res", "No ML development result provided.")
        judge_res = context_variables.get("judge_res", "No previous Judge Agent remarks provided.")
        quality_tracker_summary = context_variables.get("quality_tracker_summary", "No quality tracker summary.")
        # Ensure agent_name is passed or available in context_variables for qc_case_resolved
        context_variables['agent_name'] = "Quality Assurance Agent"
        return f"""
You are a Quality Assurance Agent. An automated quality check has triggered a need for a more thorough review.
This may be due to repeated failures, low quality scores, or other indicators.

Relevant information:
- Last ML Agent Output: {ml_dev_res}
- Last Judge Agent Remarks: {judge_res}
- Quality Tracker Summary: {quality_tracker_summary}

Your tasks are:
1. Perform a deep analysis of the situation. Why was this review triggered?
2. Re-evaluate the quality of the ML agent's output, considering the history.
3. Provide a comprehensive assessment and actionable suggestions for improvement or resolution.
   It might be necessary to suggest reverting to a previous checkpoint or significantly altering the approach.

Use the `qc_case_resolved` function to submit your findings.
Set status to "QualityAcceptable", "QualityLowNeedsMajorRevision", or "QualityLowSuggestRollback".
"""

    return Agent(
        name="Quality Assurance Agent",
        model=model,
        instructions=instructions,
        functions=[qc_case_resolved],
        tool_choice="required"
    )
