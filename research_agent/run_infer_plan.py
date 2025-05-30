import json
from research_agent.inno.workflow.flowcache import FlowModule, ToolModule, AgentModule
from research_agent.inno.tools.inno_tools.paper_search import get_arxiv_paper_meta
from research_agent.inno.tools.inno_tools.code_search import search_github_repos, search_github_code
from research_agent.inno.agents.inno_agent.plan_agent import get_coding_plan_agent
from research_agent.inno.agents.inno_agent.prepare_agent import get_prepare_agent
from research_agent.inno.agents.inno_agent.ml_agent import get_ml_agent
from research_agent.inno.agents.inno_agent.judge_agent import get_judge_agent
from research_agent.inno.agents.inno_agent.survey_agent import get_survey_agent
from research_agent.inno.agents.inno_agent.exp_analyser import get_exp_analyser_agent
from research_agent.inno.agents.inno_agent.quality_control_agents import get_plan_validator_agent, get_quality_assurance_agent
from research_agent.inno.tools.arxiv_source import download_arxiv_source_by_title
from research_agent.inno import MetaChain
from tqdm import tqdm
from pydantic import BaseModel, Field
from research_agent.constant import DOCKER_WORKPLACE_NAME, COMPLETION_MODEL, CHEEP_MODEL
from research_agent.inno.util import single_select_menu
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig
from research_agent.inno.environment.browser_env import BrowserEnv
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.config import SmartExecutionConfig
from research_agent.inno.quality_tracker import QualityTracker
import asyncio
import argparse
import os
from typing import List, Dict, Any, Union
from research_agent.inno.logger import MetaChainLogger
import importlib
from research_agent.inno.environment.utils import setup_dataset
# instance_path = "benchmark/gnn.json"
# task_level = "task1"

class QualityAssuranceRollbackRequired(Exception):
    """Custom exception to signal a rollback suggested by QualityAssuranceAgent."""
    pass

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
        
        self.plan_validator_agent = AgentModule(
            get_plan_validator_agent(model=CHEEP_MODEL, code_env=code_env), 
            self.client, 
            cache_path
        )
        self.quality_assurance_agent = AgentModule(
            get_quality_assurance_agent(model=CHEEP_MODEL, code_env=code_env), 
            self.client, 
            cache_path
        )

        self.smart_config = SmartExecutionConfig()
        self.code_env = code_env # Storing code_env for later use
        self.quality_tracker = QualityTracker()
        self.rollback_attempts_left = self.smart_config.max_rollback_attempts
        self.logger = self.client.logger # For easy access to logger
    async def forward(self, instance_path: str, task_level: str, local_root: str, workplace_name: str, max_iter_times: int, category: str, *args, **kwargs):
        self.logger.info("--- Starting AI Researcher Workflow ---")
        self.logger.info("--- Starting Preparation Phase ---")
        metadata = self.load_ins({"instance_path": instance_path, "task_level": task_level})
        context_variables = {
            "working_dir": workplace_name, # TODO: change to the codebase path
            "date_limit": metadata["date_limit"],
        }

        github_result = self.git_search({"metadata": metadata})
        
        
        query = f"""\
You are given a list of papers, searching results of the papers on GitHub, and innovative ideas according to the papers.
List of papers:
{warp_source_papers(metadata["source_papers"])}

Searching results of the papers on GitHub:
{github_result}

innovative ideas:
{metadata["task_instructions"]}

Your task is to choose at least 5 repositories as the reference codebases.
"""
        messages = [{"role": "user", "content": query}]
        prepare_messages, context_variables = await self.prepare_agent(messages, context_variables)
        prepare_res = prepare_messages[-1]["content"]
        prepare_dict = extract_json_from_output(prepare_res)
        paper_list = prepare_dict["reference_papers"]
        download_res = self.download_papaer({"paper_list": paper_list, "local_root": local_root, "workplace_name": workplace_name})
        self.logger.info("Reference materials and codebases prepared.")
        self.logger.info("--- Preparation Phase Completed ---")
        self.logger.info("--- Starting Survey & Planning Phase ---")
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

        self.logger.info("Plan generated. Validating plan...")
        context_variables["plan_content"] = plan_res # Pass the plan to the validator
        
        plan_validator_messages_history = [{"role": "user", "content": "Validate the plan provided in context_variables."}]
        validated_plan_messages, context_variables = await self.plan_validator_agent(
            plan_validator_messages_history,
            context_variables
        )
        plan_validation_res = validated_plan_messages[-1]["content"]
        self.logger.info(f"Plan Validation Result: {plan_validation_res}") # Existing log
        
        parsed_validation_output = None
        is_plan_valid = False
        try:
            # extract_json_from_output is a global helper in this file
            json_str_from_validator = extract_json_from_output(plan_validation_res) 
            if json_str_from_validator:
                parsed_validation_output = json.loads(json_str_from_validator)
                if isinstance(parsed_validation_output, dict):
                    validation_status = parsed_validation_output.get("status")
                    if validation_status == "PlanValid":
                        is_plan_valid = True
                        self.logger.info("Plan validation successful. Proceeding with the validated plan.")
                    elif validation_status == "PlanInvalid":
                        suggestions = parsed_validation_output.get("suggestions", "No specific suggestions provided.")
                        self.logger.error(f"Plan validation failed. Status: PlanInvalid. Suggestions: {suggestions}")
                        raise Exception(f"Plan validation failed by PlanValidatorAgent. Suggestions: {suggestions}")
                    else:
                        self.logger.warning(f"PlanValidatorAgent returned an unexpected status: {validation_status}. Treating as validation failure.")
                        raise Exception(f"PlanValidatorAgent returned an unexpected status: {validation_status}")
                else:
                    self.logger.error("Failed to parse PlanValidatorAgent output as a valid dictionary structure.")
                    raise Exception("Failed to parse PlanValidatorAgent output structure.")
            else:
                self.logger.error("Could not extract JSON from PlanValidatorAgent output.")
                raise Exception("Could not extract JSON from PlanValidatorAgent output.")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error for PlanValidatorAgent output: {e}. Raw output: {plan_validation_res}")
            raise Exception(f"JSON decoding error for PlanValidatorAgent output: {e}")
        except Exception as e: # Catch other exceptions, including the ones re-raised above
            self.logger.error(f"Error during plan validation processing: {e}")
            raise # Re-raise the exception to halt workflow or be caught by higher-level handler

        # If is_plan_valid is True, execution continues. Otherwise, an exception would have been raised.
        self.logger.info("--- Survey & Planning Phase Completed ---")
        self.logger.info("--- Starting Initial Implementation & Verification Phase ---")

        # Get GPU memory information (once before the main loop)
        gpu_memory_info = []
        if self.code_env: # Ensure code_env is available
            gpu_memory_info = self.code_env.get_gpu_memory_info()
            self.logger.info(f"Retrieved GPU Info: {gpu_memory_info}")

        # Update context_variables for all ml_agent calls within the loop
        context_variables["gpu_memory_info"] = gpu_memory_info
        context_variables["gpu_memory_threshold"] = self.smart_config.gpu_memory_threshold
        context_variables["auto_batch_size_adjustment"] = self.smart_config.auto_batch_size_adjustment
        
        # Main ML Development and Judging Loop
        current_ml_dev_res = "" # Stores the output of the last successful ml_agent call in the loop
        current_judge_res = ""  # Stores the output of the last successful judge_agent call in the loop
        
        ml_agent_initial_query = f"""INPUT:
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
IMPORTANT: After any attempt to compile or execute code using tools, you MUST update the `context_variables` dictionary by setting a key named `last_compilation_status` to `{"attempted": True, "successful": <True_or_False>, "details": "<output_or_error_summary>"}` before you finish your turn. If no compilation/execution was attempted, you do not need to set this key.
"""
        ml_agent_messages_history = [{"role": "user", "content": ml_agent_initial_query}]
        
        # MAX_ITER_TIMES is passed as an argument to forward method
        for i in range(max_iter_times):
            retry_attempt_for_current_iteration = 0
            max_retries_for_iteration = 1 # Max retries with rollback for a single iteration i

            while retry_attempt_for_current_iteration <= max_retries_for_iteration:
                try:
                    # --- ML Agent Execution ---
                    self.logger.info(f"Starting ML Agent iteration {i+1}, attempt {retry_attempt_for_current_iteration + 1}")
                    use_cache_interactively = (retry_attempt_for_current_iteration == 0)
                    iter_tag = i + 1 
        
                    ml_agent_response_messages, context_variables = await self.ml_agent(
                        list(ml_agent_messages_history), # Pass a copy to avoid modification issues if agent modifies input list
                        context_variables, 
                        iter_times=iter_tag,
                        interactive_cache_check=use_cache_interactively
                    )
                    # Assuming last message is the result and AgentModule appends to its own history for saving
                    # but returns the new messages part of the conversation for us to manage history.
                    # For this loop, we need to manage ml_agent_messages_history explicitly.
                    current_ml_dev_res = ml_agent_response_messages[-1]["content"] 
                    ml_agent_messages_history.extend(ml_agent_response_messages) # Add ML agent's new messages
                    self.quality_tracker.reset_consecutive_failures()

                    # Process compilation status if reported by ml_agent
                    compilation_status_info = context_variables.get("last_compilation_status")
                    if compilation_status_info and compilation_status_info.get("attempted"):
                        successful_compilation = compilation_status_info.get("successful", False)
                        details = compilation_status_info.get("details", "No details provided.")
                        
                        self.quality_tracker.record_compilation_attempt(successful=successful_compilation)
                        
                        if successful_compilation:
                            self.logger.info(f"Compilation attempt reported by ML Agent (Iter {i+1}): Successful. Details: {details}")
                        else:
                            self.logger.warning(f"Compilation attempt reported by ML Agent (Iter {i+1}): Failed. Details: {details}")
                    else:
                        # This might be normal if the agent's task didn't involve compilation in that turn.
                        self.logger.info(f"No explicit compilation attempt reported by ML Agent for this turn (Iter {i+1}).")
                    
                    # Clear the status from context_variables to avoid stale data for subsequent checks
                    if "last_compilation_status" in context_variables:
                        context_variables.pop("last_compilation_status")

                    # --- Judge Agent Execution ---
                    self.logger.info(f"Starting Judge Agent for iteration {i+1}")
                    judge_query = f"""INPUT:
You are given an innovative idea:
{metadata["task_instructions"]}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res} 
The implementation of the project:
{current_ml_dev_res}
Your task is to evaluate the implementation, and give a suggestion about the implementation. Note that you should carefully check whether the implementation meets the idea, especially the atomic academic concepts in the model survey notes one by one! If not, give comprehensive suggestions about the implementation.

[IMPORTANT] You should fully utilize the existing resources in the reference codebases as much as possible, including using the existing datasets, model components, and training process, but you should also implement the idea by creating new model components!
[IMPORTANT] You should recognize every key point in the innovative idea, and carefully check whether the implementation meets the idea one by one!
[IMPORTANT] Some tips about the evaluation:
1. The implementation should carefully follow the plan. Please check every component in the plan step by step.
2. The implementation should have the test process. All in all, you should train ONE dataset with TWO epochs, and finally test the model on the test dataset within one script. The test metrics should follow the plan.
3. The model should be train on GPU device. If you meet Out of Memory problem, you should try another specific GPU device.
"""
                    judge_agent_input_messages = [{"role": "user", "content": judge_query}]
                    
                    judge_agent_response_messages, context_variables = await self.judge_agent(
                        judge_agent_input_messages, 
                        context_variables, 
                        iter_times=iter_tag, 
                        interactive_cache_check=use_cache_interactively 
                    )
                    current_judge_res = judge_agent_response_messages[-1]["content"]
                    ml_agent_messages_history.extend(judge_agent_response_messages) # Add judge's response for next ML call

                    # --- Robust Parsing of JudgeAgent Output ---
                    parsed_judge_output = None
                    is_fully_correct = False
                    try:
                        json_str_from_judge = extract_json_from_output(current_judge_res) # Using global helper

                        if json_str_from_judge:
                            parsed_judge_output = json.loads(json_str_from_judge)
                            if isinstance(parsed_judge_output, dict) and "fully_correct" in parsed_judge_output:
                                is_fully_correct = parsed_judge_output.get("fully_correct", False)
                        else: # Fallback if JSON structure is not found
                            if '"fully_correct": true' in current_judge_res: # Less reliable check
                                is_fully_correct = True
                    except json.JSONDecodeError as json_e:
                        self.logger.error(f"Failed to parse JSON from JudgeAgent output: {json_e}. Raw output: {current_judge_res}")
                        is_fully_correct = False # Default to not fully correct

                    if is_fully_correct:
                        self.quality_tracker.add_quality_score(1.0)
                        self.logger.info(f"Iteration {i+1} (attempt {retry_attempt_for_current_iteration + 1}) successful and fully correct based on parsed judge output.")
                        # This break is for the inner retry loop (while loop)
                        break 
                    else:
                        # Using 0.3 as specified for "not fully correct but complete".
                        self.quality_tracker.add_quality_score(0.3) 
                        self.logger.warning(f"Iteration {i+1} (attempt {retry_attempt_for_current_iteration + 1}) completed but was not fully correct based on parsed judge output.")
                        if self.quality_tracker.is_review_triggered(self.smart_config):
                            self.logger.info("QualityTracker triggered an intelligent review due to low quality score. Invoking QualityAssuranceAgent.")
                            context_variables["ml_dev_res"] = current_ml_dev_res
                            context_variables["judge_res"] = current_judge_res
                            context_variables["quality_tracker_summary"] = {
                                "consecutive_failures": self.quality_tracker.consecutive_failures,
                                "latest_quality_score": self.quality_tracker.get_latest_quality_score(),
                                "compilation_failure_rate": self.quality_tracker.get_compilation_failure_rate()
                            }
                            qa_messages_history = [{"role": "user", "content": "An intelligent review has been triggered due to low quality score. Please assess."}]
                            qa_response_messages, context_variables = await self.quality_assurance_agent(
                                qa_messages_history, context_variables
                            )
                            qa_assessment_res = qa_response_messages[-1]["content"]
                            # self.logger.info(f"Quality Assurance Agent Assessment (low quality): {qa_assessment_res}") # Old logging

                            parsed_qa_output = None
                            try:
                                json_str_from_qa = extract_json_from_output(qa_assessment_res) # Use existing helper
                                if json_str_from_qa:
                                    parsed_qa_output = json.loads(json_str_from_qa)
                                    if isinstance(parsed_qa_output, dict):
                                        qa_status = parsed_qa_output.get("status")
                                        qa_assessment_details = parsed_qa_output.get("assessment", "No specific assessment.")
                                        qa_suggestions = parsed_qa_output.get("suggestions", [])

                                        self.logger.info(f"QualityAssuranceAgent assessment (low quality): Status='{qa_status}', Details='{qa_assessment_details}', Suggestions='{qa_suggestions}'")

                                        if qa_status == "QualityLowSuggestRollback":
                                            self.logger.warning("QualityAssuranceAgent suggested rollback due to low quality.")
                                            raise QualityAssuranceRollbackRequired(f"Rollback suggested by QA Agent (low quality). Assessment: {qa_assessment_details}, Suggestions: {qa_suggestions}")
                                        elif qa_status == "QualityLowNeedsMajorRevision":
                                            self.logger.error(f"QualityAssuranceAgent indicated major revisions are needed (low quality). Suggestions: {qa_suggestions}")
                                            raise Exception(f"Quality Assurance check failed (low quality): Major revision needed. Suggestions: {qa_suggestions}")
                                        elif qa_status == "QualityAcceptable":
                                            self.logger.info("QualityAssuranceAgent found the quality acceptable under review (low quality trigger).")
                                        else:
                                            self.logger.warning(f"QualityAssuranceAgent (low quality trigger) returned an unexpected status: {qa_status}.")
                                    else: # parsed_qa_output is not a dict
                                        self.logger.error("Could not parse QualityAssuranceAgent (low quality trigger) output as a valid dictionary structure.")
                                        # Potentially raise an exception here if QA output is critical and unparseable
                                else: # json_str_from_qa is None
                                    self.logger.error("Could not extract JSON from QualityAssuranceAgent output (low quality trigger).")
                                    # Potentially raise an exception here
                            except QualityAssuranceRollbackRequired:
                                raise # Re-raise to be caught by the outer exception handler for rollback
                            except Exception as qa_e: # Catch other exceptions during QA processing for low quality
                                self.logger.error(f"Error processing QualityAssuranceAgent response (low quality trigger): {qa_e}. Raw output: {qa_assessment_res}")
                                # Decide if this should halt everything. For now, let the original flow (outer try-except) handle it.
                                # If QA agent itself fails, it's a significant issue.
                                # If we raise here, it will be caught by the main 'except Exception as e'
                                raise qa_e 
                                
                    # This break is for the inner retry loop (while loop)
                    break 
                except Exception as e:
                    self.logger.error(f"Error during InnoFlow iteration {i+1} (attempt {retry_attempt_for_current_iteration + 1}): {e}")
                    self.quality_tracker.increment_consecutive_failures()
                    
                    if self.rollback_attempts_left > 0 and self.quality_tracker.is_review_triggered(self.smart_config):
                        self.rollback_attempts_left -= 1
                        retry_attempt_for_current_iteration += 1
                        self.logger.info(f"QualityTracker triggered an intelligent review due to error. Invoking QualityAssuranceAgent.")
                        # Prepare context for QualityAssuranceAgent
                        context_variables["ml_dev_res"] = current_ml_dev_res # Might be unset if ML agent itself errored early
                        context_variables["judge_res"] = current_judge_res # Might be unset if ML agent errored
                        context_variables["quality_tracker_summary"] = {
                            "consecutive_failures": self.quality_tracker.consecutive_failures,
                            "latest_quality_score": self.quality_tracker.get_latest_quality_score(),
                            "compilation_failure_rate": self.quality_tracker.get_compilation_failure_rate(),
                            "error_context": str(e) # Add error context for QA
                        }
                        qa_messages_history = [{"role": "user", "content": "An intelligent review has been triggered due to an error. Please assess."}]
                        
                        qa_response_messages, context_variables_after_qa = await self.quality_assurance_agent(
                            qa_messages_history,
                            context_variables 
                        )
                        qa_assessment_res = qa_response_messages[-1]["content"]
                        context_variables = context_variables_after_qa # Update context_variables
                        # self.logger.info(f"Quality Assurance Agent Assessment (error): {qa_assessment_res}") # Old logging

                        parsed_qa_output = None
                        try:
                            json_str_from_qa = extract_json_from_output(qa_assessment_res) # Use existing helper
                            if json_str_from_qa:
                                parsed_qa_output = json.loads(json_str_from_qa)
                                if isinstance(parsed_qa_output, dict):
                                    qa_status = parsed_qa_output.get("status")
                                    qa_assessment_details = parsed_qa_output.get("assessment", "No specific assessment.")
                                    qa_suggestions = parsed_qa_output.get("suggestions", [])

                                    self.logger.info(f"QualityAssuranceAgent assessment (error trigger): Status='{qa_status}', Details='{qa_assessment_details}', Suggestions='{qa_suggestions}'")

                                    if qa_status == "QualityLowSuggestRollback":
                                        self.logger.warning("QualityAssuranceAgent suggested rollback due to error trigger.")
                                        # This exception will be caught by the outer try-except block which handles rollbacks
                                        # No need to change 'e' here, the original 'e' is still the primary reason for being in this except block.
                                        # The rollback mechanism will proceed.
                                        # However, to make it explicit that QA confirmed rollback:
                                        # We are already in an exception handler. Raising QualityAssuranceRollbackRequired
                                        # here will replace the original exception 'e' if not handled carefully.
                                        # The existing logic is: if QA is called, then rollback is attempted.
                                        # This is fine. This new exception provides more context if needed higher up.
                                        # For now, logging is primary, rollback proceeds.
                                        # To ensure the rollback mechanism *knows* QA suggested it,
                                        # we could modify 'e' or rely on logs. Let's assume logs are sufficient for now.
                                        # If a specific action beyond normal rollback is needed due to this specific suggestion,
                                        # this is where it would go. If it's just to confirm rollback, existing flow is okay.
                                        # Let's make it more forceful by re-raising the specific exception
                                        # so it's clear QA was involved.
                                        raise QualityAssuranceRollbackRequired(f"Rollback confirmed by QA Agent (error trigger). Original Error: {str(e)}. QA Assessment: {qa_assessment_details}, Suggestions: {qa_suggestions}")
                                    elif qa_status == "QualityLowNeedsMajorRevision":
                                        self.logger.error(f"QualityAssuranceAgent indicated major revisions are needed (error trigger). Original Error: {str(e)}. Suggestions: {qa_suggestions}")
                                        # Halt further retries for this iteration by raising a new error.
                                        raise Exception(f"Quality Assurance check failed (error trigger): Major revision needed. Original Error: {str(e)}. Suggestions: {qa_suggestions}")
                                    elif qa_status == "QualityAcceptable":
                                        self.logger.info("QualityAssuranceAgent found the quality acceptable under review (error trigger), but rollback for original error will still proceed if attempts left.")
                                    else: # Unexpected status
                                        self.logger.warning(f"QualityAssuranceAgent (error trigger) returned an unexpected status: {qa_status}. Rollback for original error will proceed.")
                                else: # Not a dict
                                    self.logger.error("Could not parse QualityAssuranceAgent (error trigger) output as a valid dictionary structure. Rollback for original error will proceed.")
                            else: # No JSON
                                self.logger.error("Could not extract JSON from QualityAssuranceAgent output (error trigger). Rollback for original error will proceed.")
                        except QualityAssuranceRollbackRequired: # Re-raise if this specific exception was raised inside
                            raise
                        except Exception as qa_processing_e: # Catch errors from *processing* QA response
                            self.logger.error(f"Error processing QualityAssuranceAgent response (error trigger): {qa_processing_e}. Original error {str(e)} still applies. Rollback for original error will proceed.")
                            # Do not raise qa_processing_e as it would hide the original 'e' that triggered this path.
                            # The original exception 'e' will be handled by the outer rollback logic.

                        self.logger.info(f"Attempting rollback due to original error. Attempts left: {self.rollback_attempts_left}. Retrying iteration {i+1} (new attempt {retry_attempt_for_current_iteration + 1})")
                        
                        # Simplified rollback for ml_agent_messages_history:
                        # Remove the last two turns (failed ML agent + its preceding user query/judge response)
                        # This is an approximation. A robust solution would save/load history.
                        if len(ml_agent_messages_history) >= 2 : # If there was a judge response and then ML agent failed
                             # Remove judge response and the user query that was based on it
                            ml_agent_messages_history = ml_agent_messages_history[:-2]
                        elif len(ml_agent_messages_history) >=1 and ml_agent_messages_history[-1]["role"] == "user": # If ML agent failed on its first turn of this iteration
                            # This means the initial query for this iteration caused failure.
                            # We need to ensure the history is reset to before this user query.
                            # This depends on how history was built. If it's cumulative, this is tricky.
                            # The provided example query for "next_ml_query" is appended.
                            # If the very first ml_agent_initial_query fails, history is just that.
                            # This part needs careful state management of ml_agent_messages_history.
                            # For the provided example, the user message is added at the end of the loop for the *next* iteration.
                            # So if ml_agent fails, its input `ml_agent_messages_history` was from the *previous* iteration's end.
                            # Thus, we might not need to change ml_agent_messages_history here for retry,
                            # as the agent itself will try to load its *own* cached state from the previous successful step.
                            self.logger.warning("Simplified rollback: Retrying agent. Agent's own cache will be primary means of its state rollback.")
                        else: # Fallback if first iteration, first try
                             ml_agent_messages_history = [{"role": "user", "content": ml_agent_initial_query}]


                    else: 
                        self.logger.error(f"Iteration {i+1} failed. No more rollback attempts or rollback not triggered. Error: {e}")
                        raise e
            
            # Check if current iteration was successful (is_fully_correct would be set from the try block)
            # This check is for the outer loop (for i in range(max_iter_times))
            if is_fully_correct:
                 self.logger.info(f"Task deemed fully correct after iteration {i+1}. Breaking main iteration loop.")
                 break # Breaks the outer for i in range(max_iter_times) loop
            else:
                 # If the inner loop completed (meaning break was hit, either by success or by exhausting retries after an exception)
                 # but is_fully_correct is false, it means the attempt completed but wasn't "fully_correct".
                 # Or, if an exception was re-raised, this part won't be reached for this 'i'.
                 # The logger warning for "not fully correct" is already inside the try block.
                 pass # Continue to the next iteration of the outer loop or finish if max_iter_times is reached.


            if i < max_iter_times - 1:
                next_ml_query = f"""You are given an innovative idea:
{metadata["task_instructions"]}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
and the model survey notes you should carefully follow:
{survey_res}
And your last implementation of the project:
{current_ml_dev_res}
The suggestion about your last implementation:
{current_judge_res}
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
IMPORTANT: After any attempt to compile or execute code using tools, you MUST update the `context_variables` dictionary by setting a key named `last_compilation_status` to `{"attempted": True, "successful": <True_or_False>, "details": "<output_or_error_summary>"}` before you finish your turn. If no compilation/execution was attempted, you do not need to set this key.
"""
                # ml_agent_messages_history already contains current_judge_res from judge_agent_response_messages
                ml_agent_messages_history.append({"role": "user", "content": next_ml_query})
        
        self.logger.info("--- Initial Implementation & Verification Phase Completed ---")
        self.logger.info("--- Starting Final Submission Phase ---")
        
        # The rest of the forward method (ml_submit_query, exp_analyser calls) starts here.
        # These are outside the refactored loop for now.
        # Ensure current_ml_dev_res and current_judge_res have the latest values before this query
        # If the loop finished due to max_iter_times without full correctness, 
        # current_ml_dev_res and current_judge_res will hold the last attempt's results.
        # If it broke due to is_fully_correct, they hold the successful results.
        ml_submit_query = f"""\
You are given an innovative idea:
{metadata["task_instructions"]}
And your last implementation of the project:
{current_ml_dev_res} 
The suggestion about your last implementation:
{current_judge_res}
You have run out the maximum iteration times to implement the idea by running the script `run_training_testing.py` with TWO epochs of training and testing on ONE ACTUAL dataset.
Your task is to submit the code to the environment by running the script `run_training_testing.py` with APPROPRIATE epochs of training and testing on THIS ACTUAL dataset in order to get some stastical results. You must MODIFY the epochs in the script `run_training_testing.py` rather than use the 2 epochs.

[IMPORTANT] In this stage, you are NOT allowed to modify the existing code in the script `run_training_testing.py` except for the epochs!

Note that if your last implementation is not runable, you should finalize the submission with `case_not_resolved` function. But you can temporarily ignore the judgement of the `Judge Agent` which contains the suggestions about the implementation.
After you get the result, you should return the result with your analysis and suggestions about the implementation with `case_resolved` function.
IMPORTANT: After any attempt to compile or execute code using tools, you MUST update the `context_variables` dictionary by setting a key named `last_compilation_status` to `{"attempted": True, "successful": <True_or_False>, "details": "<output_or_error_summary>"}` before you finish your turn. If no compilation/execution was attempted, you do not need to set this key.
"""
        judge_messages.append({"role": "user", "content": ml_submit_query})

        # Update context_variables for ml_agent before this call too
        if self.code_env: # Potentially refresh GPU info
            gpu_memory_info = self.code_env.get_gpu_memory_info()
            # print(f"Retrieved GPU Info (submit): {gpu_memory_info}")
        context_variables["gpu_memory_info"] = gpu_memory_info
        context_variables["gpu_memory_threshold"] = self.smart_config.gpu_memory_threshold
        context_variables["auto_batch_size_adjustment"] = self.smart_config.auto_batch_size_adjustment
        
        judge_messages, context_variables = await self.ml_agent(judge_messages, context_variables, iter_times="submit")
        submit_res = judge_messages[-1]["content"]

        # Process compilation status for submit phase
        compilation_status_info_submit = context_variables.get("last_compilation_status")
        if compilation_status_info_submit and compilation_status_info_submit.get("attempted"):
            successful_compilation_submit = compilation_status_info_submit.get("successful", False)
            details_submit = compilation_status_info_submit.get("details", "No details provided.")
            self.quality_tracker.record_compilation_attempt(successful=successful_compilation_submit) # Track it
            if successful_compilation_submit:
                self.logger.info(f"Compilation attempt reported by ML Agent (Submit Phase): Successful. Details: {details_submit}")
            else:
                self.logger.warning(f"Compilation attempt reported by ML Agent (Submit Phase): Failed. Details: {details_submit}")
        else:
            self.logger.info("No explicit compilation attempt reported by ML Agent for submit phase.")
        if "last_compilation_status" in context_variables: # Clear the status
            context_variables.pop("last_compilation_status")
            
        self.logger.info(f"Submission Result: {submit_res}") # Existing or similar log
        self.logger.info("--- Final Submission Phase Completed ---")
        self.logger.info("--- Starting Experiment Refinement & Analysis Phase ---")

        EXP_ITER_TIMES = 2
        for i in range(EXP_ITER_TIMES):
            exp_planner_query = f"""\
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
IMPORTANT: After any attempt to compile or execute code using tools, you MUST update the `context_variables` dictionary by setting a key named `last_compilation_status` to `{"attempted": True, "successful": <True_or_False>, "details": "<output_or_error_summary>"}` before you finish your turn. If no compilation/execution was attempted, you do not need to set this key.
"""
            judge_messages.append({"role": "user", "content": refine_query})

            # Update context_variables for ml_agent before this call too
            if self.code_env: # Potentially refresh GPU info
                gpu_memory_info = self.code_env.get_gpu_memory_info()
                # print(f"Retrieved GPU Info (refine {i+1}): {gpu_memory_info}")
            context_variables["gpu_memory_info"] = gpu_memory_info
            context_variables["gpu_memory_threshold"] = self.smart_config.gpu_memory_threshold
            context_variables["auto_batch_size_adjustment"] = self.smart_config.auto_batch_size_adjustment

            judge_messages, context_variables = await self.ml_agent(judge_messages, context_variables, iter_times=f"refine_{i+1}")
            refine_res = judge_messages[-1]["content"]

            # Process compilation status for refine phase
            compilation_status_info_refine = context_variables.get("last_compilation_status")
            if compilation_status_info_refine and compilation_status_info_refine.get("attempted"):
                successful_compilation_refine = compilation_status_info_refine.get("successful", False)
                details_refine = compilation_status_info_refine.get("details", "No details provided.")
                self.quality_tracker.record_compilation_attempt(successful=successful_compilation_refine) # Track it
                if successful_compilation_refine:
                    self.logger.info(f"Compilation attempt reported by ML Agent (Refine Phase {i+1}): Successful. Details: {details_refine}")
                else:
                    self.logger.warning(f"Compilation attempt reported by ML Agent (Refine Phase {i+1}): Failed. Details: {details_refine}")
            else:
                self.logger.info(f"No explicit compilation attempt reported by ML Agent for refine phase {i+1}.")
            if "last_compilation_status" in context_variables: # Clear the status
                context_variables.pop("last_compilation_status")

        self.logger.info("--- Experiment Refinement & Analysis Phase Completed ---")
        self.logger.info("--- AI Researcher Workflow Finished ---")
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