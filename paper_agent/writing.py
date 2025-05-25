import os
from methodology_composing_using_template import methodology_composing
from related_work_composing_using_template import related_work_composing
from experiments_composing import experiments_composing
from introduction_composing import introduction_composing
from conclusion_composing import conclusion_composing
from abstract_composing import abstract_composing
import asyncio
import argparse

async def writing(
    research_field: str, 
    instance_id: str,
    base_input_dir: str,
    research_agent_workplace: str,
    research_agent_model_name: str,
    research_agent_run_workplace: str,
    is_idea_run: bool,
    paper_output_dir: str
):
    original_cwd = os.getcwd()
    try:
        os.makedirs(paper_output_dir, exist_ok=True)
        os.chdir(paper_output_dir)

        common_args = {
            "research_field": research_field,
            "instance_id": instance_id,
            "base_input_dir": base_input_dir,
            "research_agent_workplace": research_agent_workplace,
            "research_agent_model_name": research_agent_model_name,
            "research_agent_run_workplace": research_agent_run_workplace,
            "is_idea_run": is_idea_run,
        }

        await methodology_composing(**common_args)
        await related_work_composing(**common_args)
        await experiments_composing(**common_args)
        await introduction_composing(**common_args)
        await conclusion_composing(**common_args)
        await abstract_composing(**common_args)
    
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose a research paper from different sections.")
    parser.add_argument("--research_field", type=str, default="vq", help="The research field.")
    parser.add_argument("--instance_id", type=str, default="rotation_vq", help="The specific instance ID for the research.")
    
    parser.add_argument("--base_input_dir", type=str, default="../", help="Base directory for inputs relative to paper_agent execution path.")
    parser.add_argument("--research_agent_workplace", type=str, default="workplace_paper", help="Top-level directory for research_agent outputs.")
    parser.add_argument("--research_agent_model_name", type=str, required=True, help="Model name used by research_agent (e.g., gpt-4o-2024-08-06).")
    parser.add_argument("--research_agent_run_workplace", type=str, default="workplace", help="The --workplace_name argument passed to research_agent scripts.")
    parser.add_argument("--is_idea_run", action="store_true", help="Flag if this is an 'idea' run, affecting path naming.")
    
    parser.add_argument("--paper_output_dir", type=str, default="./paper_agent_output", help="Directory to store the final paper artifacts.")
    
    args = parser.parse_args()
    
    asyncio.run(writing(
        research_field=args.research_field,
        instance_id=args.instance_id,
        base_input_dir=args.base_input_dir,
        research_agent_workplace=args.research_agent_workplace,
        research_agent_model_name=args.research_agent_model_name,
        research_agent_run_workplace=args.research_agent_run_workplace,
        is_idea_run=args.is_idea_run,
        paper_output_dir=args.paper_output_dir
    ))