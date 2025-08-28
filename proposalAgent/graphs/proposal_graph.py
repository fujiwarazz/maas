
import os
from pathlib import Path
import json
from datetime import date
from re import T
from typing import Dict, Any, Tuple, List, Optional

from chromadb.api.types import D
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.tongyi import ChatTongyi
from google import genai
from google.genai import types
from langgraph.prebuilt import ToolNode

from proposalAgent.agents import *
from proposalAgent.model_config import TONGYI_CONFIG
from proposalAgent.agents.utils.memory import EmbeddingMemory
from proposalAgent.agents.utils.agent_states import (
    AgentState,

)
# from proposalAgent.agents.utils.tools_interface import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector


class ProposalAgentGraph:
    
    def __init__(
        self,
        config:Optional[Dict[str,Any]] = None
    ):
        self.config = config or TONGYI_CONFIG
        
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config['llm_provider'].lower() == "tongyi":
            self.deep_thinking_llm = ChatTongyi(model=self.config["deep_think_llm"], api_key=self.config["api_key"])
            self.quick_thinking_llm = ChatTongyi(model=self.config["quick_think_llm"], api_key=self.config["api_key"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        self.structure_llm = genai.Client().models

        self.toolkit = Toolkit(config = self.config)
        
        self.planning_memory = EmbeddingMemory(name = "planning analysis",config = self.config)
        self.impact_memory = EmbeddingMemory(name = "impact analysis",config = self.config)
        self.future_influence_memory = EmbeddingMemory(name="future_influence_memory",config=self.config)
        self.risk_memory = EmbeddingMemory(name="risk_memory",config=self.config)
        self.interdisciplinary_memory = EmbeddingMemory(name="interdisciplinary_memory",config=self.config)
        self.academic_memory = EmbeddingMemory(name="academic_memory",config=self.config)
        self.feasibility_memory = EmbeddingMemory(name="feasibility_memory",config=self.config)
        self.innovation_memory = EmbeddingMemory(name="innovation_memory",config=self.config)
        
        self.tool_nodes = self._create_tool_nodes()
        
        self.conditional_logic = ConditionalLogic()
        
        self.graph_setup = GraphSetup(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_think_llm=self.deep_thinking_llm,
            structure_llm=self.structure_llm,
            tool_nodes=self.tool_nodes,
            planning_memory=self.planning_memory,
            toolkit=self.toolkit,
            conditional_logic=self.conditional_logic,
            impact_memory=self.impact_memory,
            future_influence_memory=self.future_influence_memory,
            risk_memory=self.risk_memory,
            interdisciplinary_memory=self.interdisciplinary_memory,
            academic_memory=self.academic_memory,
            feasibility_memory=self.feasibility_memory,
            innovation_memory=self.innovation_memory,
        )
        self.curr_state = None

        self.reflector = Reflector(self.quick_thinking_llm)

        self.graph = self.graph_setup.setup_graph()
    
    def _create_tool_nodes(self):
        return{
            "structure":ToolNode(
                [self.toolkit.get_structure_infomation]
            ),
            "planning":ToolNode(
                [self.toolkit.get_planning,
                 self.toolkit.get_intention]
            ),
            "academic":ToolNode(
                [self.toolkit.get_academic_infomation]
            ),
            "feasibility":ToolNode(
                [self.toolkit.get_feasibility_infomation]
            ),
            "innovation":ToolNode(
                [self.toolkit.get_innovation_infomation]
            ),
            "risk":ToolNode(
                [self.toolkit.get_risk_infomation]
            ),
            "interdisciplinary":ToolNode(
                [self.toolkit.get_interdisciplinary_infomation]
            ),
            "future_influence":ToolNode(
                [self.toolkit.get_future_influence_infomation]
            ),
            "impact":ToolNode(
                [self.toolkit.get_impact_infomation]
            ),
            "collector":ToolNode(
                [self.toolkit.get_collector_infomation]
            ),
          
        }
    
    
    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        # self.reflector.reflect_bull_researcher(
        #     self.curr_state, returns_losses, self.bull_memory
        # )
        # self.reflector.reflect_bear_researcher(
        #     self.curr_state, returns_losses, self.bear_memory
        # )
        # self.reflector.reflect_trader(
        #     self.curr_state, returns_losses, self.trader_memory
        # )
        # self.reflector.reflect_invest_judge(
        #     self.curr_state, returns_losses, self.invest_judge_memory
        # )
        # self.reflector.reflect_risk_manager(
        #     self.curr_state, returns_losses, self.risk_manager_memory
        # )