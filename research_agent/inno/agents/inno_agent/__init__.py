# research_agent/inno/agents/inno_agent/__init__.py

# Import agents from this package to make them easily accessible
# For example, if you have specific agent definitions in other files within this directory:
from .draft_agent import get_draft_agent
from .exp_analyser import get_exp_analyser_agent
from .idea_agent import get_idea_agent
from .judge_agent import get_judge_agent
from .ml_agent import get_ml_agent
from .plan_agent import get_coding_plan_agent
from .prepare_agent import get_prepare_agent
from .survey_agent import get_survey_agent

# Import the new quality control agents
from .quality_control_agents import get_plan_validator_agent, get_quality_assurance_agent

# You can also define an __all__ variable if you want to specify
# what is exported when using 'from .agents.inno_agent import *'
# For example:
# __all__ = [
#     "get_draft_agent", 
#     "get_exp_analyser_agent", 
#     "get_idea_agent",
#     "get_judge_agent", 
#     "get_ml_agent", 
#     "get_coding_plan_agent",
#     "get_prepare_agent", 
#     "get_survey_agent",
#     "get_plan_validator_agent",
#     "get_quality_assurance_agent"
# ]
