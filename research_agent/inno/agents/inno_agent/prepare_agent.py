from research_agent.inno.types import Agent
from research_agent.inno.tools.terminal_tools import gen_code_tree_structure, read_file, execute_command, terminal_page_down, terminal_page_up, terminal_page_to
from research_agent.inno.util import make_message, make_tool_message
from research_agent.inno.registry import register_agent
from research_agent.inno.types import Result
import json
from inspect import signature
from research_agent.inno.environment.docker_env import DockerEnv, with_env
from typing import List, Dict, Optional # Added for type hinting

def case_resolved(selected_repositories: List[Dict[str, Optional[str]]]):
    """
    The function to output the determined reference codebases. Use this function only after you have carefully reviewed the existing resources and understand the task.
    Args:
        selected_repositories: A list of dictionaries. Each dictionary represents a chosen repository and should contain:
            'url': (str) The URL of the repository.
            'source_type': (str) Origin of the link, e.g., 'official_link' or 'github_search'.
            'reasoning': (str) Justification for selecting this repository.
            'cloned_path': (str, optional) The directory name if the repository was cloned (e.g., 'user_repo_name').
    """
    prepare_result = {
        "selected_code_repositories": selected_repositories
    }

    return Result(
        value=f"""\
I have determined the reference codebases based on official links and general search results.
{json.dumps(prepare_result, ensure_ascii=False, indent=4)}
""", 
        context_variables={"prepare_result": prepare_result}
    )

@register_agent("get_prepare_agent")
def get_prepare_agent(model: str, **kwargs):
    code_env: DockerEnv = kwargs.get("code_env", None)
    def instructions(context_variables):
      working_dir = context_variables.get("working_dir", None)
      # The input to this agent (the user message) will contain:
      # 1. The innovative idea/task description.
      # 2. A list of source papers.
      # 3. "User-Provided and Tool-Discovered Official Code Links".
      # 4. "General GitHub Search Results".
      return f"""
You are tasked with selecting relevant reference codebases for a given research project.
Your working directory is `/{working_dir}`. You can only access files within this directory.

Input Information:
You will receive the research project's innovative idea/task description, a list of source papers, a section detailing 'User-Provided and Tool-Discovered Official Code Links', and a section with 'General GitHub Search Results'.

Your Task:
1.  **Understand the Goal:** First, thoroughly understand the research project's innovative idea or task description. All selections must support this goal.
2.  **Review Information Sources:** Examine all provided code link sources:
    *   **User-Provided and Tool-Discovered Official Code Links:** These are listed with their original source (e.g., 'User Input', 'arXiv comment section').
    *   **General GitHub Search Results:** Broader results based on paper titles.
3.  **Prioritize and Investigate:**
    *   **Highest Priority:** Start by investigating links labeled as 'User-Provided Link' (source: 'User Input'). These are direct suggestions.
    *   **Second Priority:** Next, examine links labeled as 'Tool-Discovered Link' (sources like 'arXiv comment section', 'arXiv summary/abstract').
    *   **Third Priority:** If the above sources do not yield enough relevant repositories, or if you need more options, explore the 'General GitHub Search Results'.
    *   Use the provided tools (`execute_command git clone ...`, `gen_code_tree_structure`, `read_file`) to assess promising repositories.
4.  **Select Repositories:** Choose at least 5 repositories in total that are most relevant and useful for implementing the innovative idea.
5.  **Justify Selections:** For each repository you select, provide a detailed justification for your choice, explaining its relevance and potential utility.

Evaluation Criteria for Repositories (consider these during your investigation):
*   Relevance: How closely does the repository align with the innovative idea and the techniques mentioned in the source papers?
*   Quality: Is the code well-structured, documented, and maintained? (Stars, recent updates, README quality are indicators).
*   Completeness: Does it seem to be a full implementation or a partial one?
*   Framework/Language: Python and PyTorch are preferred for deep learning projects, but consider others if highly relevant.

Output Requirements:
When you have made your final selections, you MUST use the `case_resolved` function.
The `selected_repositories` argument for `case_resolved` must be a list of dictionaries. Each dictionary should have the following structure:
  - 'url': (string) The URL of the repository.
  - 'source_type': (string) Must be one of 'user_provided_official', 'tool_discovered_official', or 'github_search'. This should reflect how you primarily identified and verified the link's relevance (e.g., if a GitHub search result matches an official paper, and you verify it, you might still classify based on its initial discovery or how its relevance was confirmed).
  - 'reasoning': (string) Your detailed justification for why this repository was selected.
  - 'cloned_path': (string, optional) If you cloned the repository, provide the name of the directory you cloned it into (e.g., 'user_repo_name'). If not cloned, omit this key or set its value to None or an empty string.

Tools Available:
- `execute_command`: To clone repositories or run other useful commands.
- `gen_code_tree_structure`: To view the structure of cloned repositories.
- `read_file`: To read specific files from cloned repositories.
- `terminal_page_down`, `terminal_page_up`, `terminal_page_to`: For navigating large terminal outputs.
- `case_resolved`: To submit your final list of selected repositories with all required details.
      """
    tools = [gen_code_tree_structure, read_file, execute_command, case_resolved, terminal_page_down, terminal_page_up, terminal_page_to]
    tools = [with_env(code_env)(tool) if 'env' in signature(tool).parameters else tool for tool in tools]
    return Agent(
    name="Prepare Agent",
    model=model,
    instructions=instructions,
    functions=tools, 
    tool_choice = "required", 
    parallel_tool_calls = False
    )
