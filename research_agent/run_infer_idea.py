import json
import os # Make sure os is imported for file path checks
import sys
# Ensure research_agent.inno.interaction_utils can be imported.
from research_agent.inno.interaction_utils import present_text_to_user_for_review, present_choices_to_user, manage_code_review_session # Import new utility
from research_agent.inno.workflow.flowcache import FlowModule, ToolModule, AgentModule
from research_agent.inno.tools.inno_tools.paper_search import get_arxiv_paper_meta, find_official_code_for_paper # Import new tool
from research_agent.inno.tools.inno_tools.code_search import search_github_repos, search_github_code
from research_agent.inno.agents.inno_agent.plan_agent import get_coding_plan_agent
from research_agent.inno.agents.inno_agent.prepare_agent import get_prepare_agent
from research_agent.inno.agents.inno_agent.ml_agent import get_ml_agent
from research_agent.inno.agents.inno_agent.judge_agent import get_judge_agent
from research_agent.inno.agents.inno_agent.survey_agent import get_survey_agent
from research_agent.inno.agents.inno_agent.exp_analyser import get_exp_analyser_agent
from research_agent.inno.agents.inno_agent.idea_agent import get_idea_agent, get_code_survey_agent
from research_agent.inno.tools.arxiv_source import download_arxiv_source_by_title
from research_agent.inno import MetaChain
from tqdm import tqdm
from pydantic import BaseModel, Field
from research_agent.constant import DOCKER_WORKPLACE_NAME, COMPLETION_MODEL, CHEEP_MODEL
from research_agent.inno.util import single_select_menu
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig
from research_agent.inno.environment.browser_env import BrowserEnv
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
import asyncio
import argparse
import os
from typing import List, Dict, Any, Union
from research_agent.inno.logger import MetaChainLogger
import importlib
from research_agent.inno.environment.utils import setup_dataset
from research_agent.validation import validate_source_papers # Added
from research_agent.intervention_utils import ( # Added
    should_request_human_intervention,
    get_estimated_code_quality_score, 
    get_execution_success,
    intelligent_stopping_condition # Added intelligent_stopping_condition
)
from research_agent.code_quality import CodeQualityChecker # Added
from research_agent.unified_logging import UnifiedLogger # Added
# instance_path = "benchmark/gnn.json"
# task_level = "task1"
def warp_source_papers(source_papers):
    return "\n".join([f"Title: {source_paper['reference']}; You can use this paper in the following way: {source_paper['usage']}" for source_paper in source_papers])
def extract_json_from_output(output_text: str) -> dict:
    # 计数器方法来找到完整的JSON
    def find_json_boundaries(text):
        stack = []
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:  # 第一个开括号
                    start = i
                stack.append(char)
            elif char == '}':
                stack.pop()
                if not stack and start != -1:  # 找到匹配的最外层括号
                    return text[start:i+1]
        
        return None

    # 找到JSON文本
    json_str = find_json_boundaries(output_text)
    
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {}
    return {}
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, default="benchmark/gnn.json")
    parser.add_argument('--container_name', type=str, default='paper_eval')
    parser.add_argument("--task_level", type=str, default="task1")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--workplace_name", type=str, default="workplace")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--max_iter_times", type=int, default=0)
    parser.add_argument("--category", type=str, default="recommendation")
    args = parser.parse_args()
    return args

class EvalMetadata(BaseModel):
    source_papers: List[dict] = Field(description="the list of source papers")
    task_instructions: str = Field(description="the task instructions")
    date: str = Field(description="the date", pattern="^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD format
    date_limit: str = Field(description="the date limit", pattern="^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD format
def load_instance(instance_path, task_level) -> Dict:
    with open(instance_path, "r", encoding="utf-8") as f:
        eval_instance = json.load(f)
    source_papers = eval_instance["source_papers"]  
    task_instructions = eval_instance[task_level]   
    arxiv_url = eval_instance["url"]
    meta = get_arxiv_paper_meta(arxiv_url)
    if meta is None:
        date = "2024-01-01"
    else:
        date = meta["published"].strftime("%Y-%m-%d")

    return EvalMetadata(source_papers=source_papers, task_instructions=task_instructions, date=date, date_limit=date).model_dump()

def github_search(metadata: Dict) -> str:
    github_result = ""
    for source_paper in tqdm(metadata["source_papers"]):
        github_result += search_github_repos(metadata, source_paper["reference"], 10)
        github_result += "*"*30 + "\n"
    return github_result

class InnoFlow(FlowModule):
    def __init__(self, cache_path: str, log_path: Union[str, None, MetaChainLogger] = None, model: str = "gpt-4o-2024-08-06", code_env: DockerEnv = None, web_env: BrowserEnv = None, file_env: RequestsMarkdownBrowser = None):
        super().__init__(cache_path, log_path, model)
        self.load_ins = ToolModule(load_instance, cache_path)
        self.git_search = ToolModule(github_search, cache_path)
        self.prepare_agent = AgentModule(get_prepare_agent(model=CHEEP_MODEL, code_env=code_env), self.client, cache_path)
        self.download_papaer = ToolModule(download_arxiv_source_by_title, cache_path)
        self.coding_plan_agent = AgentModule(get_coding_plan_agent(model=CHEEP_MODEL, code_env=code_env), self.client, cache_path)
        self.ml_agent = AgentModule(get_ml_agent(model=COMPLETION_MODEL, code_env=code_env), self.client, cache_path)
        self.judge_agent = AgentModule(get_judge_agent(model=CHEEP_MODEL, code_env=code_env, web_env=web_env, file_env=file_env), self.client, cache_path)
        self.idea_agent = AgentModule(get_idea_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        # self.survey_agent = AgentModule(get_survey_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.code_survey_agent = AgentModule(get_code_survey_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.exp_analyser = AgentModule(get_exp_analyser_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.code_checker = CodeQualityChecker(logger=self.logger) # Added
        self.code_env = code_env # Added: Store code_env for pre_execution_check
        self.unified_logger = UnifiedLogger(main_logger=self.logger) # Added

    async def forward(self, instance_path: str, task_level: str, local_root: str, workplace_name: str, max_iter_times: int, category: str, *args, **kwargs):
        project_path_in_docker = f"/{workplace_name}/project"
        host_code_review_base_dir = os.path.abspath("./temp_human_code_review_sessions")

        metadata = self.load_ins({"instance_path": instance_path, "task_level": task_level})

        all_valid, validation_errors = validate_source_papers(metadata.get('source_papers', []))
        if not all_valid:
            error_str = "\n".join(validation_errors)
            # Assuming self.logger is available from InnoFlow's __init__
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"Source papers validation failed in run_infer_idea.py:\n{error_str}")
            else:
                print(f"CRITICAL (run_infer_idea.py): Source papers validation failed:\n{error_str}")
            raise SystemExit("Execution halted in run_infer_idea.py due to invalid source paper data.")
        else:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.info("Source papers validated successfully in run_infer_idea.py.")
            else:
                print("Source papers validated successfully in run_infer_idea.py.")
        
        context_variables = {
            "working_dir": workplace_name, # TODO: change to the codebase path
            "date_limit": metadata["date_limit"],
        }

        github_result = self.git_search({"metadata": metadata})
        data_module = importlib.import_module(f"benchmark.process.dataset_candidate.{category}.metaprompt")

        # Step 2: Gather Code Links (Revised Strategy)
        compiled_code_links_for_prompt = []
        for paper_info in metadata["source_papers"]:
            title = paper_info.get("reference", "Unknown Title")
            user_link = paper_info.get("github_link") # New optional field from benchmark JSON

            if user_link:
                # Ensure user_link is treated as a single link, not a list
                compiled_code_links_for_prompt.append(f"- Paper: '{title}', User-Provided Link: {user_link} (Source: User Input)")
            else:
                paper_arxiv_url = paper_info.get("url") # URL for the specific paper from benchmark JSON
                if paper_arxiv_url:
                    # find_official_code_for_paper returns a list of dicts
                    tool_links = find_official_code_for_paper(paper_title=title, arxiv_url=paper_arxiv_url)
                    if tool_links:
                        for link_info in tool_links:
                            compiled_code_links_for_prompt.append(f"- Paper: '{title}', Tool-Discovered Link: {link_info['url']} (Source: {link_info['source_description']})")
                    else:
                        compiled_code_links_for_prompt.append(f"- Paper: '{title}', Tool-Discovered Link: None found in arXiv metadata.")
                else:
                    # If no user_link and no paper_arxiv_url, we cannot find specific links for this paper_info item.
                    compiled_code_links_for_prompt.append(f"- Paper: '{title}', Official Link: None provided or found (no user link, no arXiv URL for tool search).")
        
        official_links_presentation_str = "\n".join(compiled_code_links_for_prompt)
        if not compiled_code_links_for_prompt:
            official_links_presentation_str = "No user-provided or tool-discovered official code links were processed for the source papers."

        # In run_infer_idea.py, PrepareAgent is called before the specific innovative idea is fully formed by the user.
        # The task_description should guide the PrepareAgent to select broadly useful codebases.
        task_description_for_prepare_agent = (
            "Your primary task is to identify and select highly relevant and valuable code repositories "
            "that could serve as foundational references for a potential research project in the domain "
            "covered by the source papers. The specific innovative idea will be developed after this step. "
            "Therefore, focus on selecting codebases that are either official implementations of the source papers, "
            "high-quality implementations of core concepts, or provide useful tools/datasets for this research area."
        )

        query_for_prepare_agent = f"""\
You are tasked with selecting reference codebases for a potential research project.
{task_description_for_prepare_agent}

Your selection should be based on the following information, presented in order of preference for your consideration:

1.  **User-Provided and Tool-Discovered Official Code Links:**
    (These are links explicitly provided by the user in the benchmark or discovered from paper metadata. They should be prioritized if relevant.)
{official_links_presentation_str}

2.  **General GitHub Search Results:**
    (These are broader search results based on paper titles. Use these to supplement if the official links are insufficient or to find alternatives.)
{github_result}

3.  **List of Source Papers (for context):**
{warp_source_papers(metadata["source_papers"])}

**Instructions for Selecting Code Repositories:**
-   Carefully review all provided information.
-   **Prioritization Strategy:**
    1.  Strongly prioritize **User-Provided Links** (from 'User Input' source).
    2.  Next, carefully consider **Tool-Discovered Official Links** (from 'arXiv comment section', 'arXiv summary/abstract', etc.).
    3.  Finally, use **General GitHub Search Results** to find alternatives or if the above sources yield insufficient relevant repositories.
-   For each link, assess its relevance to the general research area of the source papers and its potential utility for a new project in this domain.
-   You must choose at least 5 repositories in total.
-   For each chosen repository, when calling the `case_resolved` function, you MUST specify:
    -   `url`: (string) The URL of the repository.
    -   `source_type`: (string) Must be one of 'user_provided_official', 'tool_discovered_official', or 'github_search'. Base this on how you primarily identified the relevance of this specific URL.
    -   `reasoning`: (string) Your detailed justification for why this repository was selected.
    -   `cloned_path`: (string, optional) If you decide to clone the repository, provide the name of the directory you cloned it into (e.g., 'user_repo_name'). If not cloned, this can be omitted or be an empty string.

Based on your review, provide a list of the chosen reference codebases using the `case_resolved` function with the specified structure for each repository.
"""
        messages = [{"role": "user", "content": query_for_prepare_agent}]
        prepare_messages, context_variables = await self.prepare_agent(messages, context_variables)
        prepare_res = prepare_messages[-1]["content"] # This is the JSON string from case_resolved

        # TeX paper downloading remains decoupled.
        source_paper_titles_for_tex = [p.get("reference", "") for p in metadata["source_papers"] if p.get("reference")]
        source_paper_titles_for_tex = [title for title in source_paper_titles_for_tex if title.strip()]
        
        download_res = self.download_papaer({
            "paper_list": source_paper_titles_for_tex, 
            "local_root": local_root, 
            "workplace_name": workplace_name
        })
        
        dataset_description = f"""\
You should select SEVERAL datasets as experimental datasets from the following description:
{data_module.DATASET}

We have already selected the following baselines for these datasets:
{data_module.BASELINE}

The performance comparison of these datasets:
{data_module.COMPARISON}

And the evaluation metrics are:
{data_module.EVALUATION}

{data_module.REF}
"""
        
        idea_query = f"""\
I have a task related to machine learning:
{data_module.TASK}
And a list of papers for your reference:
{warp_source_papers(metadata["source_papers"])}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}
And I have also downloaded the corresponding paper in the Tex format, with the following information:
{download_res}

Your task is to thoroughly review research papers and generate innovative ideas for the given task.

Note that the math formula should be as complete as possible.
"""
        messages = [{"role": "user", "content": idea_query}]
        context_variables["notes"] = []
        self.unified_logger.log_agent_activity(agent_name="IdeaAgent", activity_type="start_call", metadata={"input_query_summary": idea_query[:200]}, message="Starting IdeaAgent execution (initial idea generation).") # Unified Log
        survey_messages, context_variables = await self.idea_agent(messages, context_variables)
        # Note: survey_res for the *very first* call isn't explicitly logged with an end_call here,
        # as the subsequent loop and selection process is more significant for the final idea.
        # The loop itself will have start/end if we decide to log each generation.
        # For now, focusing on the specified logging points.

        survey_res = survey_messages[-1]["content"]
        ideas = [survey_res]
        IDEA_NUM = 5
        for i in range(IDEA_NUM - 1):
            # Optionally log start/end for each idea generation in the loop if needed for granularity
            # self.unified_logger.log_agent_activity(agent_name="IdeaAgent", activity_type="start_call", metadata={"iteration": i+1}, message=f"Generating idea {i+2}.")
            messages.extend(survey_messages)
            messages.append({"role": "user", "content": "please survey again and give me another idea"})
            survey_messages, context_variables = await self.idea_agent(messages, context_variables, iter_times=i+1)
            survey_res = survey_messages[-1]["content"]
            ideas.append(survey_res)
            # self.unified_logger.log_agent_activity(agent_name="IdeaAgent", activity_type="end_call", metadata={"iteration": i+1, "output_summary": survey_res[:200]}, message=f"Finished generating idea {i+2}.")
        
        # messages.extend(survey_messages) # This line seems redundant as messages are already extended in the loop
        
        # Prepare messages for idea selection/refinement call
        idea_selection_query_content = """\
You have generated {} innovative ideas for the given task:
{}

Your task is to analyze multiple existing ideas, select the most novel one, enhance the idea if any key information is missing, finally give me the most novel idea with refined math formula and code implementation. Directly output the selected refined idea report.
""".format(IDEA_NUM, '\n===================\n==================='.join(ideas))
        messages = [{"role": "user", "content": idea_selection_query_content}]

        self.unified_logger.log_agent_activity(agent_name="IdeaAgent", activity_type="start_call", metadata={"input_query_summary": messages[0]['content'][:200] if messages else "N/A"}, message="Starting IdeaAgent execution (idea selection/refinement).") # Unified Log
        survey_messages, context_variables = await self.idea_agent(messages, context_variables, iter_times="select")
        survey_res = survey_messages[-1]["content"]
        self.unified_logger.log_agent_activity(agent_name="IdeaAgent", activity_type="end_call", metadata={"output_summary": survey_res[:200]}, message="IdeaAgent execution (idea selection/refinement) finished.") # Unified Log
        
        print("\n=== LLM has selected and refined an idea. User review requested. ===")
        
        choice_prompts = []
        full_idea_contents = []

        # Use chr(10) for newline within f-string expressions to avoid issues with backslashes
        choice_prompts.append(f"Review/Approve LLM's Refined Idea (Snippet: {survey_res[:200].replace(chr(10), ' ')}...)")
        full_idea_contents.append(survey_res)

        for i, idea_text in enumerate(ideas):
            choice_prompts.append(f"Review/Approve Original Idea {i+1} (Snippet: {idea_text[:200].replace(chr(10), ' ')}...)")
            full_idea_contents.append(idea_text)

        custom_path_option_text = "Provide path to a custom/edited idea file"
        custom_console_option_text = "Input custom/edited idea directly via console"
        choice_prompts.append(custom_path_option_text)
        choice_prompts.append(custom_console_option_text)

        selected_idea_content = None

        while selected_idea_content is None:
            main_prompt_for_choice = "\nPlease select an option for the research idea:"
            chosen_option_str, chosen_idx = present_choices_to_user(main_prompt_for_choice, choice_prompts)

            if chosen_idx is not None: # User selected one of the numbered options
                if 0 <= chosen_idx < len(full_idea_contents): # Index corresponds to an existing idea
                    idea_to_review = full_idea_contents[chosen_idx]
                    file_sugg_name = "llm_refined_idea.txt" if chosen_idx == 0 else f"original_idea_{chosen_idx}.txt"
                    
                    print(f"\nReviewing: {choice_prompts[chosen_idx]}")
                    edited_content = present_text_to_user_for_review(
                        content=idea_to_review, 
                        file_name_suggestion=file_sugg_name,
                        prompt_message="The full text of the selected idea has been saved for your review.",
                        edit_prompt="[A]pprove this idea as is, or [E]dit the saved file and use your version?"
                    )
                    if edited_content is not None:
                        selected_idea_content = edited_content
                    else:
                        print("Review process was cancelled or an error occurred. Please choose an option again.")
                # This case should ideally not be reached if chosen_idx is not None and choice_prompts is consistent
                # else: 
                #    print("Internal error: chosen_idx out of sync. Please try again.")

            elif chosen_option_str == custom_path_option_text:
                custom_file_path = input("Please enter the full path to your custom/edited idea file: ").strip()
                if custom_file_path and os.path.exists(custom_file_path):
                    try:
                        with open(custom_file_path, 'r', encoding='utf-8') as f: selected_idea_content = f.read()
                        print(f"Loaded idea from {custom_file_path}")
                    except Exception as e: print(f"Error reading file {custom_file_path}: {e}. Please try again.")
                elif not custom_file_path: print("No file path provided. Please try again.")
                else: print(f"File not found: {custom_file_path}. Please try again.")
            
            elif chosen_option_str == custom_console_option_text:
                print("Please paste your custom/edited idea. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) when done to submit.")
                custom_idea_lines = []
                try:
                    while True: custom_idea_lines.append(input())
                except EOFError: pass # Expected way to end multi-line input
                custom_idea = "\n".join(custom_idea_lines)
                if custom_idea.strip():
                    selected_idea_content = custom_idea; print("Custom idea captured.")
                else: print("No input received or input was empty. Please try again.")
            else: 
                print("Invalid selection or process aborted. Please try again.")

        survey_res = selected_idea_content 
        print("\n--- User idea selection complete. Proceeding with the chosen idea. ---")
        # print(survey_res) # Original print statement for survey_res

        code_survey_query = f"""\
I have an innovative idea related to machine learning:
{survey_res}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, in the directory `/workplace`, use the `list_files` tool to navigate the directory.
And I have also downloaded the corresponding paper in the Tex format, with the following information:
{download_res}

Your task is to carefully understand the innovative idea, and thoroughly review codebases and generate a comprehensive implementation report for the innovative idea. You can NOT stop to review the codebases until you have get all academic concepts in the innovative idea.

Note that the code implementation should be as complete as possible.
"""
        messages = [{"role": "user", "content": code_survey_query}]
        self.unified_logger.log_agent_activity(agent_name="CodeSurveyAgent", activity_type="start_call", metadata={"input_query_summary": code_survey_query[:200]}, message="Starting CodeSurveyAgent execution.") # Unified Log
        code_survey_messages, context_variables = await self.code_survey_agent(messages, context_variables)
        code_survey_res = code_survey_messages[-1]["content"]
        self.unified_logger.log_agent_activity(agent_name="CodeSurveyAgent", activity_type="end_call", metadata={"output_summary": code_survey_res[:200]}, message="CodeSurveyAgent execution finished.") # Unified Log
        # print(code_survey_res)
        
        context_variables["model_survey"] = code_survey_res

        plan_query = f"""\
I have an innovative ideas related to machine learning:
{survey_res}
And a list of papers for your reference:
{warp_source_papers(metadata["source_papers"])}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}

I have also understood the innovative idea, comprehensively reviewed the codebases, and generated a comprehensive implementation report:
{code_survey_res}

We have already selected the following datasets as experimental datasets:
{dataset_description}

Your task is to carefully review the existing resources and understand the task, and give me a detailed plan for the implementation.
"""
        messages = [{"role": "user", "content": plan_query}]
        plan_messages, context_variables = await self.coding_plan_agent(messages, context_variables)
        plan_res = plan_messages[-1]["content"]

        print("\n=== Coding plan generated. User review requested. ===")
        edited_plan_res = present_text_to_user_for_review(
            content=plan_res,
            file_name_suggestion="coding_plan_for_review.md", # Suggest markdown for better readability if plan is structured text
            prompt_message="The generated coding plan has been saved for your review.",
            edit_prompt="[A]pprove this plan as is, or [E]dit the saved file and use your version?"
        )

        if edited_plan_res is not None:
            plan_res = edited_plan_res
            print("\n--- User plan validation complete. Proceeding with the approved/edited plan. ---")
        else:
            print("\n--- Plan review was cancelled or an error occurred. Halting execution. ---")
            raise SystemExit("Execution halted due to plan review cancellation or error.")

        # write the model based on the model survey notes
        ml_dev_query = f"""\
INPUT:
You are given an innovative idea:
{survey_res}. 
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And I have conducted the comprehensive survey on the innovative idea and the papers, and give you the model survey notes:
{survey_res}
You should carefully go through the math formula and the code implementation, and implement the innovative idea according to the plan and existing resources.

We have already selected the following datasets as experimental datasets:
{dataset_description}
Your task is to implement the innovative idea after carefully reviewing the math formula and the code implementation in the paper notes and existing resources in the directory `/{workplace_name}`. You should select ONE most appropriate and lightweight dataset from the given datasets, and implement the idea by creating new model, and EXACTLY run TWO epochs of training and testing on the ACTUAL dataset on the GPU device. Note that EVERY atomic academic concept in model survey notes should be implemented in the project.

PROJECT STRUCTURE REQUIREMENTS:
1. Directory Organization
- Data: `/{workplace_name}/project/data/`
     * Use the dataset selected by the `Plan Agent`
     * NO toy or random datasets
- Model Components: `/{workplace_name}/project/model/`
    * All model architecture files
    * All model components as specified in survey notes
    * Dataset processing scripts and utilities

- Training: `/{workplace_name}/project/training/`
    * Training loop implementation
    * Loss functions
    * Optimization logic

- Testing: `/{workplace_name}/project/testing/`
    * Evaluation metrics
    * Testing procedures

- Data processing: `/{workplace_name}/project/data_processing/`
    * Implement the data processing pipeline

- Main Script: `/{workplace_name}/project/run_training_testing.py`
    * Complete training and testing pipeline
    * Configuration management
    * Results logging

2. Complete Implementation Requirements
   - MUST implement EVERY component from model survey notes
   - NO placeholder code (no `pass`, `...`, `raise NotImplementedError`)
   - MUST include complete logic and mathematical operations
   - Each component MUST be fully functional and tested

3. Dataset and Training Requirements
   - Select and download ONE actual dataset from references
   - Implement full data processing pipeline
   - Train for exactly 2 epochs
   - Test model performance after training
   - Log all metrics and results

4. Integration Requirements
   - All components must work together seamlessly
   - Clear dependencies between modules
   - Consistent coding style and documentation
   - Proper error handling and GPU support

EXECUTION WORKFLOW:
1. Dataset Setup
   - Choose appropriate dataset from references (You MUST use the actual dataset, not the toy or random datasets) [IMPORTANT!!!]
   - Download to data directory `/{workplace_name}/project/data`
   - Implement processing pipeline in `/{workplace_name}/project/data_processing/`
   - Verify data loading

2. Model Implementation
   - Study model survey notes thoroughly
   - Implement each component completely
   - Document mathematical operations
   - Add comprehensive docstrings

3. Training Implementation
   - Complete training loop
   - Loss function implementation
   - Optimization setup
   - Progress monitoring

4. Testing Setup
   - Implement evaluation metrics
   - Create testing procedures
   - Set up results logging
   - Error handling

5. Integration
   - Create run_training_testing.py
   - Configure for 2 epoch training
   - Add GPU support and OOM handling
   - Implement full pipeline execution

VERIFICATION CHECKLIST:
1. Project Structure
   - All directories exist and are properly organized
   - Each component is in correct location
   - Clear separation of concerns

2. Implementation Completeness
   - Every function is fully implemented
   - No placeholder code exists
   - All mathematical operations are coded
   - Documentation is complete

3. Functionality
   - Dataset downloads and loads correctly
   - Training runs for 2 epochs
   - Testing produces valid metrics
   - GPU support is implemented

Remember: 
- MUST use actual dataset (no toy data, download according to the reference codebases) [IMPORTANT!!!]
- Implementation MUST strictly follow model survey notes
- ALL components MUST be fully implemented
- Project MUST run end-to-end without placeholders
- MUST complete 2 epochs of training and testing
"""
        messages = [{"role": "user", "content": ml_dev_query}]
        ml_dev_messages, context_variables = await self.ml_agent(messages, context_variables)
        ml_dev_res = ml_dev_messages[-1]["content"]

        # ---- Initial JudgeAgent Call ----
        print("\n=== Initial code by MLAgent. Evaluating with JudgeAgent (Idea Run). ===")
        initial_judge_query = f"""\
INPUT:
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
The implementation of the project:
{ml_dev_res}
Your task is to evaluate the implementation, and give a suggestion about the implementation. Note that you should carefully check whether the implementation meets the idea, especially the atomic academic concepts in the model survey notes one by one! If not, give comprehensive suggestions about the implementation.
[IMPORTANT] You should fully utilize the existing resources in the reference codebases as much as possible.
[IMPORTANT] You should recognize every key point in the innovative idea, and check implementation.
[IMPORTANT] Some tips: Check plan step-by-step, ensure test process (train 1 dataset, 2 epochs, test), GPU usage.
"""
        # Initialize history lists in context_variables if not already present
        if "quality_score_history" not in context_variables: context_variables["quality_score_history"] = []
        if "execution_success_history" not in context_variables: context_variables["execution_success_history"] = []

        # ---- Initial Code Quality Checks ----
        self.logger.info("[run_infer_idea.py] Validating project structure (initial)...")
        if not self.code_checker.validate_project_structure(project_path_in_docker):
            self.logger.warning("[run_infer_idea.py] Initial project structure validation (placeholder) indicated issues. Continuing with caution.")
        else:
            self.logger.info("[run_infer_idea.py] Initial project structure validation (placeholder) successful.")

        self.logger.info("[run_infer_idea.py] Performing pre-execution checks (initial)...")
        if not self.code_checker.pre_execution_check(project_path_in_docker, self.code_env):
            self.logger.warning("[run_infer_idea.py] Initial pre-execution checks (placeholder) indicated issues. Continuing with caution.")
        else:
            self.logger.info("[run_infer_idea.py] Initial pre-execution checks (placeholder) successful.")
        # ---- End Initial Code Quality Checks ----
        
        initial_judge_messages_list = [{"role": "user", "content": initial_judge_query}]
        judged_initial_code_messages, context_variables = await self.judge_agent(initial_judge_messages_list, context_variables)
        initial_judge_res_str = judged_initial_code_messages[-1]["content"]
        context_variables["last_judge_res"] = initial_judge_res_str # Store for potential use

        # For intervention check, we need ml_agent_output (ml_dev_res) and judge_agent_feedback (initial_judge_res_str)
        # Histories are passed as current state (empty for first call to should_request_human_intervention)
        
        human_did_initial_review = False
        if should_request_human_intervention(
            ml_agent_output=ml_dev_res, # Pass the ML agent's code output
            judge_agent_feedback=initial_judge_res_str, 
            current_iteration=0, 
            max_iterations=max_iter_times, # max_iter_times from args for the main loop
            execution_success_history=context_variables["execution_success_history"], # Empty for first call
            quality_score_history=context_variables["quality_score_history"]   # Empty for first call
        ):
            print("\n--- Human intervention triggered for initial code (Idea Run). ---")
            review_completed_initial = manage_code_review_session(
                docker_env=self.code_env, 
                project_path_in_docker=project_path_in_docker,
                host_review_dir_base=host_code_review_base_dir,
                session_id_prefix="initial_code_review_idea_triggered" 
            )
            if not review_completed_initial:
                print("Error during triggered initial human code review (Idea Run). Halting execution.")
                raise SystemExit("Execution halted due to triggered code review error or cancellation.")
            human_did_initial_review = True
            print("--- Triggered initial human code review session concluded (Idea Run). ---")
        else:
            print("--- No human intervention triggered for initial code (Idea Run). Proceeding automatically. ---")
        
        # The 'query' for the main loop's first JudgeAgent call (if loop runs) should be based on initial_judge_res_str
        # However, the loop structure is: ML -> Judge. So initial_judge_res_str is the first feedback *to* ML.
        judge_res = initial_judge_res_str # This will be the first feedback for the loop's ML agent
        
        # `judge_messages` should be the conversation history that led to `initial_judge_res_str`
        # This will serve as the input history for the first MLAgent call in the loop.
        loop_ml_agent_input_messages = judged_initial_code_messages


        # This 'query' was for the JudgeAgent, but the first JudgeAgent call in the loop
        # will use a dynamically generated query based on the output of the first MLAgent call in the loop.
        # The original query variable here is for the first judge_agent call *after* initial MLAgent *and* initial review.
        # This is now covered by `initial_judge_query` and `initial_judge_res_str`.
        # The loop will construct its own queries.
        # initial_code_quality_score and initial_execution_success are already calculated and appended
        # in the previous step's integration of should_request_human_intervention.
        # We just need to ensure they are used correctly for the stopping condition.
        # The `quality_score_history` and `execution_success_history` in `context_variables`
        # should have one entry each at this point from the initial assessment.

        # query = f"""\  # This query is effectively replaced by initial_judge_query for the first pass
# INPUT:
# You are given an innovative idea:
# {survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
The implementation of the project:
{ml_dev_res}
Your task is to evaluate the implementation, and give a suggestion about the implementation. Note that you should carefully check whether the implementation meets the idea, especially the atomic academic concepts in the model survey notes one by one! If not, give comprehensive suggestions about the implementation.

[IMPORTANT] You should fully utilize the existing resources in the reference codebases as much as possible, including using the existing datasets, model components, and training process, but you should also implement the idea by creating new model components!

[IMPORTANT] You should recognize every key point in the innovative idea, and carefully check whether the implementation meets the idea one by one!

[IMPORTANT] Some tips about the evaluation:
1. The implementation should carefully follow the plan. Please check every component in the plan step by step.
2. The implementation should have the test process. All in all, you should train ONE dataset with TWO epochs, and finally test the model on the test dataset within one script. The test metrics should follow the plan.
3. The model should be train on GPU device. If you meet Out of Memory problem, you should try another specific GPU device.
"""
        # input_messages = [{ # This was the old way of calling the first judge_agent in loop
        #     "role": "user",
        #     "content": query
        # }]
        # judge_messages, context_variables = await self.judge_agent(input_messages, context_variables)
        # judge_res = judge_messages[-1]["content"] # Now judge_res is primed with initial_judge_res_str

        MAX_ITER_TIMES = max_iter_times
        human_edit_note_for_ml_agent = "" # Initialize human edit note

        # Add current quality score and execution success from initial assessment before loop starts
        # This is needed for the *first* call to should_request_human_intervention *inside* the loop
        context_variables["quality_score_history"].append(get_estimated_code_quality_score(ml_dev_res, initial_judge_res_str))
        context_variables["execution_success_history"].append(get_execution_success(initial_judge_res_str))

        for i in range(MAX_ITER_TIMES):
            current_loop_iteration = i + 1
            print(f"\n=== Refinement Iteration {current_loop_iteration}/{MAX_ITER_TIMES} (Idea Run). MLAgent will use feedback: {judge_res[:100]}... ===")
            
            # Conditional In-Loop Human Review
            # `judge_res` here is the feedback from the *previous* JudgeAgent call (or initial one)
            # `ml_dev_res` is the code associated with that `judge_res`
            if should_request_human_intervention(
                ml_agent_output=ml_dev_res, # Code that led to current judge_res
                judge_agent_feedback=judge_res, 
                current_iteration=current_loop_iteration, 
                max_iterations=MAX_ITER_TIMES,
                execution_success_history=context_variables["execution_success_history"],
                quality_score_history=context_variables["quality_score_history"]
            ):
                print(f"\n--- Human intervention triggered for Iteration {current_loop_iteration} (Idea Run). ---")
                review_completed_iterative = manage_code_review_session(
                    docker_env=self.code_env,
                    project_path_in_docker=project_path_in_docker,
                    host_review_dir_base=host_code_review_base_dir,
                    session_id_prefix=f"refinement_loop_idea_iter_{current_loop_iteration}_triggered" 
                )
                if not review_completed_iterative:
                    print(f"Error during triggered human code review (Iteration {current_loop_iteration}, Idea Run). Halting execution.")
                    raise SystemExit(f"Execution halted due to triggered code review error or cancellation at Iteration {current_loop_iteration}.")
                human_edit_note_for_ml_agent = "Note: Human has reviewed and potentially modified the code. Please consider these human modifications alongside the JudgeAgent's feedback when making your next set of changes. Prioritize human modifications if they conflict with the JudgeAgent's suggestions for the same code sections, but still address other valid points from JudgeAgent if applicable."
                print(f"--- Triggered human code review session (Iteration {current_loop_iteration}, Idea Run) concluded. ---")
            else:
                print(f"--- No human intervention triggered for Iteration {current_loop_iteration} (Idea Run). Proceeding with MLAgent refinement. ---")
                human_edit_note_for_ml_agent = "" # Clear if no review this iteration

            # MLAgent refinement call
            refinement_ml_query = f"""\
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
And your last implementation of the project:
{ml_dev_res}
The suggestion about your last implementation:
{judge_res}
Your task is to modify the project according to the suggestion. Note that you should MODIFY rather than create a new project! Take full advantage of the existing resources! Still use the SAME DATASET!

[IMPORTANT] You should modify the project in the directory `/{workplace_name}/project`, rather than create a new project!

[IMPORTANT] If you meet dataset missing problem, you should download the dataset from the reference codebases, and put the dataset in the directory `/{workplace_name}/project/data`. 

[IMPORTANT] You CANNOT stop util you 2 epochs of training and testing on your model with the ACTUAL dataset.

[IMPORTANT] You encounter ImportError while using `run_python()`, you should check whether every `__init__.py` file is correctly implemented in the directories in the `/{workplace_name}/project`!

[IMPORTANT] Carefully check whether model and its components are correctly implemented according to the model survey notes!

Remember: 
- Implementation MUST strictly follow model survey notes
- ALL components MUST be fully implemented
- Project MUST run end-to-end without placeholders
- MUST use actual dataset (no toy data)
- MUST complete 2 epochs of training and testing
"""
            if human_edit_note_for_ml_agent: # Prepend note if it exists
                refinement_ml_query_final = human_edit_note_for_ml_agent + "\n\n" + refinement_ml_query
            else:
                refinement_ml_query_final = refinement_ml_query
            
            # `loop_ml_agent_input_messages` holds the conversation that led to the current `judge_res`
            current_ml_history = [msg for msg in loop_ml_agent_input_messages]
            current_ml_history.append({"role": "user", "content": refinement_ml_query_final})
            
            ml_dev_messages, context_variables = await self.ml_agent(current_ml_history, context_variables, iter_times=current_loop_iteration)
            ml_dev_res = ml_dev_messages[-1]["content"] # New code from MLAgent

            # ---- In-Loop Code Quality Checks ----
            self.logger.info(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Validating project structure...")
            if not self.code_checker.validate_project_structure(project_path_in_docker):
                self.logger.warning(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Project structure validation (placeholder) indicated issues. Continuing with caution.")
            else:
                self.logger.info(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Project structure validation (placeholder) successful.")

            self.logger.info(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Performing pre-execution checks...")
            if not self.code_checker.pre_execution_check(project_path_in_docker, self.code_env):
                self.logger.warning(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Pre-execution checks (placeholder) indicated issues. Continuing with caution.")
            else:
                self.logger.info(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Pre-execution checks (placeholder) successful.")
            # ---- End In-Loop Code Quality Checks ----

            # JudgeAgent call to evaluate the refined code
            judge_refined_query = f"""\
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
The implementation of the project:
{ml_dev_res}
Please evaluate the implementation, and give a suggestion about the implementation.
"""
            # For JudgeAgent, we typically start a fresh evaluation of the new code
            judged_refined_code_messages_list = [{"role": "user", "content": judge_refined_query}]
            judged_refined_code_messages, context_variables = await self.judge_agent(judged_refined_code_messages_list, context_variables, iter_times=current_loop_iteration)
            judge_res = judged_refined_code_messages[-1]["content"] # This is the new feedback

            # Update histories for next iteration's intervention check AND intelligent stopping condition
            current_judge_feedback_str = judge_res # current_judge_feedback_str is judge_res
            current_ml_agent_output_for_quality = ml_dev_res # ml_dev_res is the code that was judged

            current_exec_success = get_execution_success(current_judge_feedback_str)
            # Pass ml_dev_res (code) and judge_res (feedback on that code)
            current_quality_score = get_estimated_code_quality_score(current_ml_agent_output_for_quality, current_judge_feedback_str)
            
            context_variables["quality_score_history"].append(current_quality_score)
            context_variables["execution_success_history"].append(current_exec_success)
            self.logger.info(f"[run_infer_idea.py] Iteration {current_loop_iteration}: Quality={current_quality_score}, ExecSuccess={current_exec_success}")
            print(f"--- [run_infer_idea.py] Iteration {current_loop_iteration}: Quality={current_quality_score}, ExecSuccess={current_exec_success} ---")
            
            # Update `loop_ml_agent_input_messages` for the next MLAgent call
            loop_ml_agent_input_messages = judged_refined_code_messages

            # Integrate intelligent_stopping_condition
            if intelligent_stopping_condition(
                judge_result_str=current_judge_feedback_str,
                current_loop_iter_idx=i, # loop variable i (0-indexed)
                max_loop_iters=MAX_ITER_TIMES,
                quality_score_history=context_variables.get("quality_score_history", []),
                logger=self.logger
            ):
                self.logger.info(f"[run_infer_idea.py] Intelligent stopping condition met at iteration {current_loop_iteration}. Breaking refinement loop.")
                print(f"--- [run_infer_idea.py] Intelligent stopping condition met at iteration {current_loop_iteration}. Breaking refinement loop. ---")
                break
            
            # Redundant '"fully_correct": true' check is removed as it's part of intelligent_stopping_condition.
            
            if current_loop_iteration >= MAX_ITER_TIMES:
                 print(f"--- Reached max refinement iterations ({MAX_ITER_TIMES}) for Idea Run. Loop will terminate. ---")


        # return judged_refined_code_messages[-1]["content"] # Or the last judge_res
        # submit the code to the environment -> get the result


        
        ml_submit_query = f"""\
You are given an innovative idea:
{survey_res}
And your last implementation of the project:
{ml_dev_res}
The suggestion about your last implementation:
{judge_res}
You have run out the maximum iteration times to implement the idea by running the script `run_training_testing.py` with TWO epochs of training and testing on ONE ACTUAL dataset.
Your task is to submit the code to the environment by running the script `run_training_testing.py` with APPROPRIATE epochs of training and testing on THIS ACTUAL dataset in order to get some stastical results. You must MODIFY the epochs in the script `run_training_testing.py` rather than use the 2 epochs.

[IMPORTANT] In this stage, you are NOT allowed to modify the existing code in the script `run_training_testing.py` except for the epochs!

Note that if your last implementation is not runable, you should finalize the submission with `case_not_resolved` function. But you can temporarily ignore the judgement of the `Judge Agent` which contains the suggestions about the implementation.
After you get the result, you should return the result with your analysis and suggestions about the implementation with `case_resolved` function.
"""
        judge_messages.append({"role": "user", "content": ml_submit_query})
        judge_messages, context_variables = await self.ml_agent(judge_messages, context_variables, iter_times="submit")
        submit_res = judge_messages[-1]["content"]

        EXP_ITER_TIMES = 2
        for i in range(EXP_ITER_TIMES):
            exp_planner_query = f"""\
You are given an innovative idea:
{survey_res}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{submit_res}
Your task is to: 
1. Analyze the experimental results and give a detailed analysis report about the results.
2. Analyze the reference codebases and papers, and give a further plan to let `Machine Learning Agent` to do more experiments based on the innovative idea. The further experiments could include but not limited to:
    - Modify the implementation to better fit the idea.
    - Add more experiments to prove the effectiveness and superiority of the idea. 
    - Visualize the experimental results and give a detailed analysis report about the results.
    - ANY other experiments that exsiting concurrent reference papers and codebases have done.
DO NOT use the `case_resolved` function before you have carefully and comprehensively analyzed the experimental results and the reference codebases and papers.
"""
            judge_messages.append({"role": "user", "content": exp_planner_query})
            judge_messages, context_variables = await self.exp_analyser(judge_messages, context_variables, iter_times=f"refine_{i+1}")
            analysis_report = judge_messages[-1]["content"]

            analysis_report = context_variables["experiment_report"][-1]["analysis_report"]
            further_plan = context_variables["experiment_report"][-1]["further_plan"]
            # print(analysis_report)
            refine_query = f"""\
You are given an innovative idea:
{survey_res}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{submit_res}
And a detailed analysis report about the results are given by the `Experiment Planner Agent`:
{analysis_report}
Your task is to refine the experimental results according to the analysis report by modifying existing code in the directory `/{workplace_name}/project`. You should NOT stop util every experiment is done with ACTUAL results. If you encounter Out of Memory problem, you should try another specific GPU device. If you encounter ANY other problems, you should try your best to solve the problem by yourself.

Note that you should fully utilize the existing code in the directory `/{workplace_name}/project` as much as possible. If you want to add more experiments, you should add the python script in the directory `/{workplace_name}/project/`, like `run_training_testing.py`. Select and output the important results during the experiments into the log files, do NOT output them all in the terminal.
"""
            judge_messages.append({"role": "user", "content": refine_query})
            judge_messages, context_variables = await self.ml_agent(judge_messages, context_variables, iter_times=f"refine_{i+1}")
            refine_res = judge_messages[-1]["content"]

        print(refine_res)
        
def main(args):
    """
    MAX_ATTEMPTS

    # load the eval instance

    # choose the code base

    # generate the detailed coding plan

    # coding and debuging -> fail to implement the plan

    -> success to implement the plan

    # submit the code to the environment -> get the result

    for attempt in range(MAX_ATTEMPTS): 
        # evaluate the result

        # coding and debuging

        # submit the code to the environment -> get the result
        if done:
            break
    """
    # load the eval instance
    with open(args.instance_path, "r", encoding="utf-8") as f:
        eval_instance = json.load(f)
    instance_id = eval_instance["instance_id"] + "_idea"
    local_root = os.path.join(os.getcwd(),"workplace_paper" , f"task_{instance_id}" + "_" + COMPLETION_MODEL.replace("/", "__"),  args.workplace_name)
    container_name = args.container_name + "_" + instance_id + "_" + COMPLETION_MODEL.replace("/", "__")
    os.makedirs(local_root, exist_ok=True)
    env_config = DockerConfig(container_name = container_name, 
                              workplace_name = args.workplace_name, 
                              communication_port = args.port, 
                              conda_path = "/home/user/micromamba", 
                              local_root = local_root,
                              )
    
    code_env = DockerEnv(env_config)
    code_env.init_container()
    setup_dataset(args.category, code_env.local_workplace)
    web_env = BrowserEnv(browsergym_eval_env = None, local_root=env_config.local_root, workplace_name=env_config.workplace_name)
    file_env = RequestsMarkdownBrowser(viewport_size=1024 * 4, local_root=env_config.local_root, workplace_name=env_config.workplace_name, downloads_folder=os.path.join(env_config.local_root, env_config.workplace_name, "downloads"))
    flow = InnoFlow(cache_path="cache_" + instance_id + "_" + COMPLETION_MODEL.replace("/", "__"), log_path="log_" + instance_id, code_env=code_env, web_env=web_env, file_env=file_env, model=args.model)
    # ml_result = await flow(instance_path=instance_path)
    asyncio.run(flow(instance_path=args.instance_path, task_level=args.task_level, local_root=local_root, workplace_name=args.workplace_name, max_iter_times=args.max_iter_times, category=args.category))
    # print(judge_result)




if __name__ == "__main__":
    args = get_args()
    main(args)





"""\
INPUT:
You are given an innovative idea:
Combine DDPM model with transformer model to generate the image.
And `Prepare Agent` has chosen the reference codebases:
{prepare_res}
And `Survey Agent` has given the model survey notes:
{survey_res}

REQUIREMENTS:
1. Model Organization
   - Break down the model into smaller, logical modules based on academic definitions
   - Each module should correspond to one or more academic concepts from the papers
   - Create a clear hierarchy of modules that can be assembled into the final model
   - Example structure:
     * Base modules (fundamental building blocks)
     * Intermediate modules (combining base modules)
     * Main model class (assembling all modules)

2. Module Implementation Guidelines
   - Each module should be in a separate file under `/{workplace_name}/project/model/`
   - Modules should have clear input/output interfaces
   - Include docstrings with academic references and mathematical formulations
   - Implement forward pass with complete mathematical operations

3. Complete Implementation Requirements
   - MUST implement EVERY component from model survey notes
   - NO placeholder code (no `pass`, `...`, `raise NotImplementedError`)
   - MUST include complete logic and mathematical operations
   - Each module MUST be fully functional and tested
   - Final model should inherit from nn.Module and combine all sub-modules

Remember: 
- Break down complex models into smaller, reusable modules
- Each module should map to specific academic concepts
- Implementation MUST strictly follow model survey notes
- ALL components MUST be fully implemented
- Project MUST run end-to-end without placeholders

Task: 
Carefully go through the model survey notes, break down the model into logical modules based on academic definitions, and implement each module in a realistic way. NO placeholder code. 
In this stage, you only care about the model implementation, and don't care about the dataset, training, testing.
"""