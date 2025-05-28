import json
import os # Ensure os is imported
from research_agent.inno.interaction_utils import present_text_to_user_for_review, manage_code_review_session # Import new utility
from research_agent.inno.workflow.flowcache import FlowModule, ToolModule, AgentModule
from research_agent.inno.tools.inno_tools.paper_search import get_arxiv_paper_meta, find_official_code_for_paper # Import new tool
from research_agent.inno.tools.inno_tools.code_search import search_github_repos, search_github_code
from research_agent.inno.agents.inno_agent.plan_agent import get_coding_plan_agent
from research_agent.inno.agents.inno_agent.prepare_agent import get_prepare_agent
from research_agent.inno.agents.inno_agent.ml_agent import get_ml_agent
from research_agent.inno.agents.inno_agent.judge_agent import get_judge_agent
from research_agent.inno.agents.inno_agent.survey_agent import get_survey_agent
from research_agent.inno.agents.inno_agent.exp_analyser import get_exp_analyser_agent
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
from research_agent.validation import validate_source_papers
from research_agent.intervention_utils import should_request_human_intervention, get_estimated_code_quality_score, get_execution_success, intelligent_stopping_condition # Added intelligent_stopping_condition
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
    # date = "2023-09-28"
    

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
        self.survey_agent = AgentModule(get_survey_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.exp_analyser = AgentModule(get_exp_analyser_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.code_checker = CodeQualityChecker(logger=self.logger) # Added
        self.code_env = code_env # Store code_env for pre_execution_check
        self.unified_logger = UnifiedLogger(main_logger=self.logger) # Added

    async def forward(self, instance_path: str, task_level: str, local_root: str, workplace_name: str, max_iter_times: int, category: str, *args, **kwargs):
        project_path_in_docker = f"/{workplace_name}/project"
        host_code_review_base_dir = os.path.abspath("./temp_human_code_review_sessions")
        
        metadata = self.load_ins({"instance_path": instance_path, "task_level": task_level})

        all_valid, validation_errors = validate_source_papers(metadata.get('source_papers', []))
        if not all_valid:
            error_str = "\n".join(validation_errors)
            self.logger.error(f"Source papers validation failed:\n{error_str}") # Assuming self.logger is available and has error method
            # Or print to console if logger is not set up at this point for this specific message
            print(f"CRITICAL: Source papers validation failed:\n{error_str}")
            # Decide on halting or warning. For now, halt on critical validation error.
            raise SystemExit("Execution halted due to invalid source paper data.")
        else:
            # self.logger.info("Source papers validated successfully.") # Optional: log success
            print("Source papers validated successfully.")
        
        context_variables = {
            "working_dir": workplace_name, # TODO: change to the codebase path
            "date_limit": metadata["date_limit"],
        }

        github_result = self.git_search({"metadata": metadata})

        # Gather official code links
        official_links_parts = []
        for paper_info in metadata["source_papers"]:
            title = paper_info.get("reference", "") # Assuming 'reference' is the title
            # Attempt to get a direct URL for the paper if available, otherwise None
            # This depends on the structure of `metadata["source_papers"]` items.
            # If items are just titles, arxiv_url might need to be found another way or be None.
            # For this example, let's assume 'url' might be a key in paper_info or we use the main eval_instance["url"]
            # as a fallback if it's relevant to all source_papers (which might not be true).
            # The prompt implies paper_info might have its own URL or we use the main one.
            # Let's assume for now we primarily use title and a general URL if specific one isn't in paper_info.
            paper_arxiv_url = paper_info.get("url", None) # If each source paper has a URL
            if not paper_arxiv_url and "url" in metadata: # Fallback to main URL if any
                 # This fallback is a bit of a guess, might need adjustment based on actual data structure.
                 # It's safer to rely on URLs within each paper_info if possible.
                 # For now, we'll assume 'url' is part of the main metadata, not per source paper.
                 # The original load_instance uses `eval_instance["url"]` for the main paper.
                 # Let's assume source_papers don't have individual URLs for now.
                 pass

            # For run_infer_plan, the main paper's URL is `metadata.get("url")` if `load_instance` adds it.
            # Let's refine this: get_arxiv_paper_meta is called with eval_instance["url"] in load_instance.
            # We need to ensure each source paper's URL is available or this step is skipped.
            # The prompt says "URL if available from the source_papers item or the main eval_instance["url"]"
            # This implies source_papers items might not have URLs.
            # We'll use the main eval_instance["url"] if the paper title matches the main paper, otherwise None.
            # This is still a bit tricky. Let's assume `paper_info` might have a specific `arxiv_id` or `url`.
            # For simplicity, if `paper_info` itself doesn't have a URL, we'll skip finding official code for it
            # unless it's the main paper of the benchmark. The prompt is ambiguous here.
            # Given the current structure, `metadata["source_papers"]` has 'reference' (title) and 'usage'.
            # It does not seem to have individual URLs for each source paper.
            # So, we can only reliably find official code for the *main* paper if its URL is in metadata.
            # For other source papers, we'd have to search by title, which find_official_code_for_paper doesn't support.
            #
            # Let's assume `find_official_code_for_paper` is robust enough to handle `arxiv_url=None` if only title is given,
            # or we only call it if an arxiv_url is present.
            # The tool `find_official_code_for_paper` uses `get_arxiv_paper_meta(arxiv_url)`
            # So an arxiv_url IS required.
            #
            # Re-thinking: The most straightforward is to iterate source_papers, and if a paper_info item
            # has a 'url' or 'arxiv_id', use that. Otherwise, we can't use the tool for that specific source paper.
            # The existing `metadata` from `load_instance` contains the main paper's URL via `eval_instance["url"]`.
            # Let's focus on the main paper's official links first, and then consider if source_papers can be processed.
            # The prompt asks to iterate `metadata["source_papers"]`.

            links = []
            # Assuming paper_info["reference"] is the title.
            # We need a URL for each source paper to use `find_official_code_for_paper`.
            # If `source_papers` items don't have URLs, we can't find official code for them this way.
            # For now, let's imagine `paper_info` might have a 'url' key.
            if paper_info.get("url"): # Check if a URL is provided for this source paper
                 links = find_official_code_for_paper(paper_title=title, arxiv_url=paper_info.get("url"))
            
            if links:
                official_links_parts.append(f"For paper '{title}':")
                for link_info in links:
                    official_links_parts.append(f"  - URL: {link_info['url']} (Source: {link_info['source_description']})")
            else:
                official_links_parts.append(f"For paper '{title}': No official code links found in arXiv metadata.")
        
        official_links_str = "\n".join(official_links_parts)
        if not official_links_parts:
            official_links_str = "No official code links were found for any of the source papers based on their arXiv metadata."

        task_description_for_prepare_agent = metadata["task_instructions"] # For run_infer_plan, this is the user's idea

        query_for_prepare_agent = f"""\
You are tasked with selecting reference codebases for a research project.
The project aims to implement the following innovative idea:
{task_description_for_prepare_agent}

Your selection should be based on the following information:

1. List of Source Papers:
{warp_source_papers(metadata["source_papers"])}

2. Official Code Links (extracted from arXiv metadata for the papers above):
{official_links_str}

3. General GitHub Search Results (based on paper titles):
{github_result}

Instructions for Selecting Code Repositories:
- Review all provided information: the source papers, the official code links, and the general GitHub search results.
- Prioritize official code links if they are relevant to the project's innovative idea.
- Also consider relevant repositories from the general GitHub search, especially if official links are missing or insufficient.
- You must choose at least 5 repositories in total.
- For each chosen repository, you MUST state its URL and its origin (e.g., "official_link from arXiv comment", "github_search").

Based on your review, provide a list of the chosen reference codebases.
"""
        messages = [{"role": "user", "content": query_for_prepare_agent}]
        self.unified_logger.log_agent_activity(agent_name="PrepareAgent", activity_type="start_call", metadata={"input_query_summary": query_for_prepare_agent[:200]}, message="Starting PrepareAgent execution.") # Unified Log
        prepare_messages, context_variables = await self.prepare_agent(messages, context_variables)
        prepare_res = prepare_messages[-1]["content"]
        self.unified_logger.log_agent_activity(agent_name="PrepareAgent", activity_type="end_call", metadata={"output_summary": prepare_res[:200]}, message="PrepareAgent execution finished.") # Unified Log
        
        # TeX paper downloading is now decoupled from PrepareAgent's output.
        # It will use the original list of source papers.
        # We need to extract just the titles for the download_papaer tool.
        source_paper_titles_for_tex = [p.get("reference", "") for p in metadata["source_papers"] if p.get("reference")]
        # Filter out any empty titles just in case.
        source_paper_titles_for_tex = [title for title in source_paper_titles_for_tex if title.strip()]

        download_res = self.download_papaer({
            "paper_list": source_paper_titles_for_tex, 
            "local_root": local_root, 
            "workplace_name": workplace_name
        })
        
        survey_query = f"""\
I have an innovative ideas related to machine learning:
{metadata["task_instructions"]}
And a list of papers for your reference:
{warp_source_papers(metadata["source_papers"])}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}
And I have also downloaded the corresponding paper in the Tex format, with the following information:
{download_res}

Your task is to do a comprehensive survey on the innovative ideas and the papers, and give me a detailed plan for the implementation.

Note that the math formula should be as complete as possible, and the code implementation should be as complete as possible. Don't use placeholder code.
"""
        messages = [{"role": "user", "content": survey_query}]
        context_variables["notes"] = []
        survey_messages, context_variables = await self.survey_agent(messages, context_variables)
        survey_res = survey_messages[-1]["content"]
        context_variables["model_survey"] = survey_res

        data_module = importlib.import_module(f"benchmark.process.dataset_candidate.{category}.metaprompt")

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

        plan_query = f"""\
I have an innovative ideas related to machine learning:
{metadata["task_instructions"]}
And a list of papers for your reference:
{warp_source_papers(metadata["source_papers"])}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}
I have also explored the innovative ideas and the papers, with the following notes:
{survey_res}

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
{metadata["task_instructions"]}. 
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
        self.unified_logger.log_agent_activity(agent_name="MLAgent", activity_type="start_call", metadata={"input_query_summary": ml_dev_query[:200], "stage": "initial_code_generation"}, message="Starting initial MLAgent execution.") # Unified Log
        ml_dev_messages, context_variables = await self.ml_agent(messages, context_variables)
        ml_dev_res = ml_dev_messages[-1]["content"]
        self.unified_logger.log_agent_activity(agent_name="MLAgent", activity_type="end_call", metadata={"output_summary": ml_dev_res[:200], "stage": "initial_code_generation"}, message="Initial MLAgent execution finished.") # Unified Log

        # Initial JudgeAgent call after first MLAgent code generation
        print("\n=== Initial code generated by MLAgent. Evaluating with JudgeAgent. ===")
        judge_query_after_initial_ml = f"""\
INPUT:
You are given an innovative idea:
{metadata["task_instructions"]}
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
        initial_judge_messages = [{"role": "user", "content": judge_query_after_initial_ml}]
        # Store the conversation with JudgeAgent regarding initial code
        context_variables["judge_feedback_history"] = [] 
        context_variables["quality_score_history"] = []
        context_variables["execution_success_history"] = []

        # ---- Initial Code Quality Checks ----
        self.logger.info("Validating project structure (initial)...")
        if not self.code_checker.validate_project_structure(project_path_in_docker):
            self.logger.warning("Initial project structure validation (placeholder) failed. Continuing with caution.")
        else:
            self.logger.info("Initial project structure validation (placeholder) successful.")

        self.logger.info("Performing pre-execution checks (initial)...")
        if not self.code_checker.pre_execution_check(project_path_in_docker, self.code_env): # self.code_env is DockerEnv
            self.logger.warning("Initial pre-execution checks (placeholder) indicated issues. Continuing with caution.")
        else:
            self.logger.info("Initial pre-execution checks (placeholder) successful.")
        # ---- End Initial Code Quality Checks ----

        judged_initial_code_messages, context_variables = await self.judge_agent(initial_judge_messages, context_variables)
        judge_res_initial = judged_initial_code_messages[-1]["content"] # This is initial_judge_res_str
        
        # Initialize and populate history lists after initial JudgeAgent call
        initial_execution_success = get_execution_success(judge_res_initial)
        # Note: get_estimated_code_quality_score takes (ml_agent_output, judge_agent_feedback)
        # ml_dev_res is the output from the first MLAgent call.
        initial_code_quality_score = get_estimated_code_quality_score(ml_dev_res, judge_res_initial) 
        
        context_variables["judge_feedback_history"].append(judge_res_initial) # Already present, ensure it's used consistently
        context_variables["quality_score_history"].append(initial_code_quality_score)
        context_variables["execution_success_history"].append(initial_execution_success)

        print(f"\n--- JudgeAgent initial evaluation complete. Initial Quality: {initial_code_quality_score}, Initial Exec Success: {initial_execution_success}. Intervention check (Iteration 0). ---")
        if should_request_human_intervention(
            ml_agent_output=ml_dev_res,
            judge_agent_feedback=judge_res_initial,
            current_iteration=0, # Iteration 0 for initial review
            max_iterations=max_iter_times, # Total iterations for the main loop
            execution_success_history=context_variables["execution_success_history"],
            quality_score_history=context_variables["quality_score_history"]
        ):
            print("\n=== Human intervention triggered for initial code. ===")
            review_completed_initial = manage_code_review_session(
                docker_env=self.code_env,
                project_path_in_docker=project_path_in_docker,
                host_review_dir_base=host_code_review_base_dir,
                session_id_prefix="initial_code_review_triggered"
            )
            if not review_completed_initial:
                print("Error during triggered initial human code review. Halting execution.")
                raise SystemExit("Execution halted due to triggered initial code review error or cancellation.")
            print("--- Triggered initial human code review session concluded. ---")
            # After human review, it's good to get fresh feedback from JudgeAgent if changes were made.
            # This part is optional but recommended. For now, we'll assume human edits are 'final' for this step.
            # Or, we could re-run judge and update judge_res_initial.
            # For simplicity, we'll proceed with the current judge_res_initial,
            # and the loop will handle further refinements.
        else:
            print("--- No human intervention triggered for initial code. Proceeding with JudgeAgent feedback. ---")

        # The main refinement loop will use judge_res_initial as the first feedback.
        judge_res = judge_res_initial 
        # ml_dev_messages is already primed from the first ml_agent call.
        # judge_messages should be primed with the initial judge feedback for the loop.
        # The judge_agent call below expects `input_messages` not a full history.
        # The `self.judge_agent` call below will create new `judge_messages`
        # Let's keep `judged_initial_code_messages` as the initial history for the loop's first judge call.
        # The variable `judge_messages` will be properly formed before the loop's first JudgeAgent call.

        # Initialize judge_messages for the loop. It should contain the conversation that led to judge_res.
        # The first iteration of the loop will use judge_res from the initial assessment.
        # The ml_agent will then be called with this judge_res.
        # Then the judge_agent is called again.
        # So, judge_messages should be `judged_initial_code_messages` before the loop.
        # This seems a bit off. The ML agent in the loop needs the *history* that led to judge_res.

        # Let's adjust: the loop's first ML Agent call needs `judged_initial_code_messages` (which led to `judge_res_initial`)
        # and the user query for refinement.
        # `ml_dev_messages` will be updated by the ml_agent call inside the loop.
        # `judge_messages` will be updated by the judge_agent call inside the loop.

        # The variable 'judge_messages' for the loop should be the sequence that led to the current 'judge_res'.
        # So, before the loop, 'judge_messages' should be 'judged_initial_code_messages'.
        # And 'ml_dev_messages' is the history that produced 'ml_dev_res'.

        # Current judge_messages to prime the loop (history that produced judge_res_initial)
        # This is the sequence that the ML agent will receive as history of suggestions
        loop_ml_agent_input_messages = judged_initial_code_messages 
        # loop_ml_agent_input_messages will have the user query for refinement appended in the loop.


        query = f"""\
INPUT:
You are given an innovative idea:
{metadata["task_instructions"]}
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
        # This initial query for JudgeAgent (which was run above) is not needed here anymore.
        # The variable judge_res now holds the feedback from that initial run.
        # input_messages = [{
        #     "role": "user",
        #     "content": query
        # }]
        # judge_messages, context_variables = await self.judge_agent(input_messages, context_variables)
        # judge_res = judge_messages[-1]["content"] # This is now judge_res_initial

        MAX_ITER_TIMES = max_iter_times
        human_edit_note = "" # Initialize human_edit_note

        for i in range(MAX_ITER_TIMES):
            current_loop_iteration = i + 1
            print(f"\n=== Refinement Iteration {current_loop_iteration}/{MAX_ITER_TIMES} ===")
            
            # Potentially request human intervention
            if should_request_human_intervention(
                ml_agent_output=ml_dev_res, # Current code from MLAgent (previous iteration or initial)
                judge_agent_feedback=judge_res, # Current feedback from JudgeAgent
                current_iteration=current_loop_iteration,
                max_iterations=MAX_ITER_TIMES,
                execution_success_history=context_variables["execution_success_history"],
                quality_score_history=context_variables["quality_score_history"]
            ):
                print(f"\n--- Human intervention triggered for Iteration {current_loop_iteration}. ---")
                review_completed_iterative = manage_code_review_session(
                    docker_env=self.code_env,
                    project_path_in_docker=project_path_in_docker,
                    host_review_dir_base=host_code_review_base_dir,
                    session_id_prefix=f"refinement_loop_iter_{current_loop_iteration}_triggered"
                )
                if not review_completed_iterative:
                    print(f"Error during triggered human code review (Iteration {current_loop_iteration}). Halting execution.")
                    raise SystemExit(f"Execution halted due to triggered code review error or cancellation at Iteration {current_loop_iteration}.")
                
                human_edit_note = "Note: Human has reviewed and potentially modified the code. Please consider these human modifications alongside the JudgeAgent's feedback when making your next set of changes. Prioritize human modifications if they conflict with the JudgeAgent's suggestions for the same code sections, but still address other valid points from JudgeAgent if applicable."
                print(f"--- Triggered human code review session (Iteration {current_loop_iteration}) concluded. ---")
                # Optional: After human edits, get fresh JudgeAgent feedback before MLAgent refines again.
                # This would mean running JudgeAgent here and updating judge_res.
                # For now, MLAgent will receive the human_edit_note and the *previous* judge_res.
            else:
                print(f"--- No human intervention triggered for Iteration {current_loop_iteration}. Proceeding with MLAgent refinement based on JudgeAgent feedback. ---")
                human_edit_note = "" # Reset if no intervention this time

            # MLAgent Refinement Call
            # The history for MLAgent comes from `loop_ml_agent_input_messages` which should be the conversation history that produced the *current* `judge_res`.
            # And ml_dev_res is the code associated with that judge_res.
            refinement_query = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
and the model survey notes you should carefully follow:
{survey_res}
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
            # Prepend human_edit_note if it's not empty
            if human_edit_note:
                 refinement_query_for_ml = human_edit_note + "\n\n" + refinement_query
            else:
                refinement_query_for_ml = refinement_query

            # The input to ml_agent should be the history that led to the current `judge_res`
            # `loop_ml_agent_input_messages` should represent this.
            # We append the new user query (refinement_query_for_ml) to this history.
            
            # Make a copy to avoid modifying the history list that might be used elsewhere unexpectedly.
            current_ml_agent_input_history = [msg for msg in loop_ml_agent_input_messages]
            current_ml_agent_input_history.append({"role": "user", "content": refinement_query_for_ml})
            
            # Call MLAgent for refinement
            ml_dev_messages, context_variables = await self.ml_agent(current_ml_agent_input_history, context_variables, iter_times=current_loop_iteration)
            ml_dev_res = ml_dev_messages[-1]["content"] # This is the new refined code

            # ---- In-Loop Code Quality Checks ----
            self.logger.info(f"Validating project structure (Iteration {current_loop_iteration})...")
            if not self.code_checker.validate_project_structure(project_path_in_docker):
                self.logger.warning(f"Project structure validation (placeholder) failed in Iteration {current_loop_iteration}. Continuing with caution.")
            else:
                self.logger.info(f"Project structure validation (placeholder) successful in Iteration {current_loop_iteration}.")

            self.logger.info(f"Performing pre-execution checks (Iteration {current_loop_iteration})...")
            if not self.code_checker.pre_execution_check(project_path_in_docker, self.code_env):
                self.logger.warning(f"Pre-execution checks (placeholder) indicated issues in Iteration {current_loop_iteration}. Continuing with caution.")
            else:
                self.logger.info(f"Pre-execution checks (placeholder) successful in Iteration {current_loop_iteration}.")
            # ---- End In-Loop Code Quality Checks ----

            # JudgeAgent Call to evaluate refined code
            judge_query_for_refined_code = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
and the model survey notes you should carefully follow:
{survey_res}
The implementation of the project:
{ml_dev_res}
Please evaluate the implementation, and give a suggestion about the implementation.
"""
            # The input to judge_agent should be a new sequence starting with the refined code.
            # Or, if judge_agent is conversational, it needs the history that led to ml_dev_res, then the new query.
            # Assuming judge_agent takes the current code and a query:
            # Let's assume judge_agent is NOT conversational in the same way as ml_agent regarding prior suggestions.
            # It evaluates the code passed in `ml_dev_res` based on the query.
            # So, its input message list is fresh for each evaluation.
            current_judge_agent_input_messages = [{"role": "user", "content": judge_query_for_refined_code}]
            
            judged_refined_code_messages, context_variables = await self.judge_agent(current_judge_agent_input_messages, context_variables, iter_times=current_loop_iteration)
            judge_res = judged_refined_code_messages[-1]["content"] # New feedback for the refined code

            # Update histories for intervention check and stopping condition
            current_execution_success = get_execution_success(judge_res)
            current_code_quality_score = get_estimated_code_quality_score(ml_dev_res, judge_res) # ml_dev_res is from current loop's MLAgent

            context_variables["judge_feedback_history"].append(judge_res)
            context_variables["quality_score_history"].append(current_code_quality_score)
            context_variables["execution_success_history"].append(current_execution_success)

            print(f"--- Iteration {current_loop_iteration} Judge Feedback Received. Quality: {current_code_quality_score}, Exec Success: {current_execution_success} ---")

            # Prepare for next MLAgent call: its history should be `judged_refined_code_messages`
            loop_ml_agent_input_messages = judged_refined_code_messages

            # Intelligent stopping condition check
            # current_loop_iter_idx is `i` (0-indexed)
            if intelligent_stopping_condition(
                judge_result_str=judge_res,
                current_loop_iter_idx=i, 
                max_loop_iters=MAX_ITER_TIMES, # This is the max number of loop iterations (e.g., if 3, loop is 0,1,2)
                quality_score_history=context_variables["quality_score_history"],
                logger=self.logger # Pass the FlowModule's logger
            ):
                print(f"--- Intelligent stopping condition met at Iteration {current_loop_iteration}. Exiting refinement loop. ---")
                break
            
            # The old 'fully_correct' check is now part of intelligent_stopping_condition.
            # if '"fully_correct": true' in judge_res:
            #     print(f"--- JudgeAgent reports 'fully_correct' in Iteration {current_loop_iteration}. Exiting refinement loop. ---")
            #     break
            
            if current_loop_iteration >= MAX_ITER_TIMES: # Check if loop should naturally end
                print(f"--- Reached max refinement iterations ({MAX_ITER_TIMES}). Exiting loop. ---")
                # No break needed here, loop condition i in range(MAX_ITER_TIMES) will handle termination

        # return judged_refined_code_messages[-1]["content"] # Or the last judge_res
        # submit the code to the environment -> get the result


        
        ml_submit_query = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
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
        # The `judge_messages` should be the ones that produced the final `judge_res`
        # If loop finished by "fully_correct", `judged_refined_code_messages` is correct.
        # If loop finished by max_iter, `judged_refined_code_messages` is also correct.
        # If loop was skipped (MAX_ITER_TIMES = 0), then `loop_ml_agent_input_messages` (which was `judged_initial_code_messages`) is correct.
        
        final_ml_agent_history_before_submit = loop_ml_agent_input_messages # This holds the message history that produced the last `judge_res`
        
        # Make a copy before appending
        submit_ml_agent_input_messages = [msg for msg in final_ml_agent_history_before_submit]
        submit_ml_agent_input_messages.append({"role": "user", "content": ml_submit_query})

        # The ml_agent call for submission should use the history that led to the final `judge_res`
        submit_ml_dev_messages, context_variables = await self.ml_agent(submit_ml_agent_input_messages, context_variables, iter_times="submit")
        submit_res = submit_ml_dev_messages[-1]["content"]

        EXP_ITER_TIMES = 2 # This seems to be a different loop for experimental analysis refinement
        
        # The `judge_messages` for this loop should probably start fresh or be based on `submit_res`
        # Let's assume it's a new sequence of interactions for exp_analyser and subsequent ml_agent calls.
        # For exp_analyser, it receives `submit_res`.
        current_exp_analysis_input_messages = [{"role": "user", "content": exp_planner_query}]


        for i in range(EXP_ITER_TIMES): # This is the experiment analysis loop
            # exp_planner_query is defined outside, let's ensure it's correctly used.
            # The query for exp_analyser should include the latest submit_res.
            # The definition of exp_planner_query already includes {submit_res}, so it's dynamic.

            # We need to pass the *current* `submit_res` to the `exp_planner_query`
            # This means `exp_planner_query` might need to be formatted inside the loop if `submit_res` changes,
            # or `exp_analyser` needs to be fed messages that build up.
            # The current `exp_planner_query` is static after definition.
            # Let's assume `exp_analyser` is called with a fresh query that includes the current `submit_res`.

            # The prompt for exp_analyser already contains {submit_res}
            # However, `judge_messages` is being appended to. This implies a conversational context for exp_analyser
            # Let's use `current_exp_analysis_input_messages`
            
            # Ensure `current_exp_analysis_input_messages` is correctly formed for each call if it's conversational
            if i > 0: # For subsequent iterations, the query might need to include prior analysis report or be part of a conversation
                # This part of the logic for EXP_ITER_TIMES needs clarification if it's conversational.
                # Assuming for now `exp_analyser` is called with a potentially growing history.
                # The original code appends to `judge_messages`. Let's try to follow that pattern for `exp_analyser` and subsequent `ml_agent`.
                # `submit_ml_dev_messages` was the last thing from ml_agent.
                # `current_exp_analysis_input_messages` will be the history for `exp_analyser`
                pass # `current_exp_analysis_input_messages` will grow.

            # Format the query with the current submit_res before passing to exp_analyser
            # This is important if submit_res can change or if the query needs to be fresh.
            # However, the current `exp_planner_query` definition is outside the loop and uses `submit_res` from *before* this loop.
            # This might be an issue if `submit_res` is expected to be updated by `ml_agent` *within* this `EXP_ITER_TIMES` loop,
            # which it is (the `refine_query` call to `ml_agent` updates `refine_res`, which then should feed into next `submit_res`).

            # This EXP_ITER_TIMES loop seems to be: ExpAnalyser -> MLAgent (refine) -> (submit_res should be updated for next ExpAnalyser call)
            # The current code does: ExpAnalyser -> MLAgent(refine) -> (submit_res is NOT updated, exp_planner_query uses old submit_res)
            # This needs correction. `submit_res` should be the result of the latest `ml_agent` refinement.
            
            # Let's redefine the exp_planner_query inside the loop to use the latest `refine_res` as `submit_res`
            # Initial `submit_res` comes from the "submit" stage ml_agent call.
            # In this loop, `refine_res` is the output of the ml_agent. This should be the input to the next `exp_analyser`
            
            current_submit_res_for_exp_analysis = submit_res # Initial value
            if i > 0: # After the first iteration, use the result of the previous refinement
                current_submit_res_for_exp_analysis = refine_res # refine_res is from previous iteration's ml_agent

            dynamic_exp_planner_query = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{current_submit_res_for_exp_analysis}
Your task is to: 
1. Analyze the experimental results and give a detailed analysis report about the results.
2. Analyze the reference codebases and papers, and give a further plan to let `Machine Learning Agent` to do more experiments based on the innovative idea. The further experiments could include but not limited to:
    - Modify the implementation to better fit the idea.
    - Add more experiments to prove the effectiveness and superiority of the idea, including but not limited to: ablation studies, sensitivity analysis, etc. ()
    - Visualize the experimental results and give a detailed analysis report about the results.
    - ANY other experiments that exsiting concurrent reference papers and codebases have done.
DO NOT use the `case_resolved` function before you have carefully and comprehensively analyzed the experimental results and the reference codebases and papers.
"""
            # For the first iteration of exp_analysis_loop, use a fresh start.
            if i == 0:
                current_exp_analysis_input_messages = [{"role": "user", "content": dynamic_exp_planner_query}]
            else:
                # Append to existing history for conversational context if exp_analyser supports it
                current_exp_analysis_input_messages.append({"role": "user", "content": dynamic_exp_planner_query})


            exp_analyser_messages, context_variables = await self.exp_analyser(current_exp_analysis_input_messages, context_variables, iter_times=f"refine_{i+1}")
            # `analysis_report` is now the last message from exp_analyser
            # The original code extracts from context_variables, let's stick to that if it's more reliable
            analysis_report = context_variables["experiment_report"][-1]["analysis_report"]
            # further_plan = context_variables["experiment_report"][-1]["further_plan"] # Not used in refine_query explicitly by name

            # Update current_exp_analysis_input_messages for the next iteration (if any)
            current_exp_analysis_input_messages = exp_analyser_messages


            # MLAgent call for refining based on experiment analysis
            dynamic_refine_query = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{current_submit_res_for_exp_analysis}
And a detailed analysis report about the results are given by the `Experiment Planner Agent`:
{analysis_report}
Your task is to refine the experimental results according to the analysis report by modifying existing code in the directory `/{workplace_name}/project`. You should NOT stop util every experiment is done with ACTUAL results. If you encounter Out of Memory problem, you should try another specific GPU device. If you encounter ANY other problems, you should try your best to solve the problem by yourself.

Note that you should fully utilize the existing code in the directory `/{workplace_name}/project` as much as possible. If you want to add more experiments, you should add the python script in the directory `/{workplace_name}/project/`, like `run_training_testing.py`. Select and output the important results during the experiments into the log files, do NOT output them all in the terminal.
"""
            # The input to this MLAgent call should be the history that led to `analysis_report`
            # which is `exp_analyser_messages`.
            current_ml_refine_input_messages = [msg for msg in exp_analyser_messages]
            current_ml_refine_input_messages.append({"role": "user", "content": dynamic_refine_query})
            
            ml_refined_exp_messages, context_variables = await self.ml_agent(current_ml_refine_input_messages, context_variables, iter_times=f"refine_{i+1}")
            refine_res = ml_refined_exp_messages[-1]["content"] # This is the new code/result after refinement.
            
            # This `refine_res` will be used as `current_submit_res_for_exp_analysis` in the next iteration of this loop.
            # The history for the next call to `exp_analyser` should be `ml_refined_exp_messages`.
            current_exp_analysis_input_messages = ml_refined_exp_messages


#         print(refine_res)
        
def main(args):
You are given an innovative idea:
{metadata["task_instructions"]}
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
    - Add more experiments to prove the effectiveness and superiority of the idea, including but not limited to: ablation studies, sensitivity analysis, etc. ()
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
{metadata["task_instructions"]}
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

#         print(refine_res)
        
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
    instance_id = eval_instance["instance_id"]
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