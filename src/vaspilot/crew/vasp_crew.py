from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.utilities import I18N
from crewai.tools.agent_tools.agent_tools import AgentTools
import os
import copy
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from ..tools.wait_calc_tool import WaitCalcTool
from ..tools.ask_user_tool import AskUserTool
from ..tools.json_rag_tool import JsonApproxSearch, JsonStrictSearch
from crewai_tools import RagTool
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
import yaml
from typing import Dict, Any, Optional
from .embedding import LocalAPIEmbedder
from .local_llm import LocalLLM
from crewai_tools import MCPServerAdapter
from ..listener.message_listener import MessageListener

class VaspCrew():
	"""VASPilot crew"""

	def __init__(self, config: Dict[str, Any], message_listener: Optional[MessageListener] = None):
		self.config = copy.deepcopy(config)
		self.llm_mapper = config['llm_mapper']
		self.llm_config = {}
		for key, value in config['llm_config'].items():
			self.llm_config[key] = self.llm_mapper[value]
		self.embedder = LocalAPIEmbedder(url=config["embbeder"]["url"], model_id=config["embbeder"]["model_id"], api_key=config["embbeder"]["api_key"])
		self.persist_tools = {}
		self.message_listener = message_listener
		if self.config.get("tool_params", {}).get("json_approx_search_tool", None):
			self.persist_tools["json_approx_search_tool"] = JsonApproxSearch(embedding_function=self.embedder, source_files=self.config['tool_params']['json_approx_search_tool']['sources'], chroma_db_path=self.config['tool_params']['json_approx_search_tool']['chroma_db_path'])
		if self.config.get("tool_params", {}).get("json_strict_search_tool", None):
			self.persist_tools["json_strict_search_tool"] = JsonStrictSearch(source_files=self.config['tool_params']['json_strict_search_tool']['sources'])
		self.agent_tools_dict = {"ask_question_tool":[], "delegate_work_tool":[]}
		
	def _create_tools(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
		tool_dict = {}
		tool_dict.update(self.persist_tools)
		if self.config['mcp_server'] is not None:
			mcp_params = copy.deepcopy(self.config['mcp_server'])
			self.mcp_server = MCPServerAdapter(mcp_params)
		for tools in self.mcp_server.tools:
			tool_dict[tools.name] = tools
		tool_dict["wait_calc_tool"] = WaitCalcTool(mcp_url=self.config['mcp_server']['url'], message_listener=self.message_listener, conversation_id=conversation_id)
		tool_dict["ask_user_tool"] = AskUserTool(message_listener=self.message_listener, conversation_id=conversation_id)
		return tool_dict

	def _inject_agent_tools(self, agent_dict: dict[Agent]) -> Dict[str, Any]:
		asked_agents = []
		delegated_agents = []
		if self.config.get("tool_params", {}).get("ask_question_tool", None):
			for agents in self.config["tool_params"]["ask_question_tool"].get("agents", None):
				asked_agents.append(agent_dict[agents])
		if self.config.get("tool_params", {}).get("delegate_work_tool", None):
			for agents in self.config["tool_params"]["delegate_work_tool"].get("agents", None):
				delegated_agents.append(agent_dict[agents])
		_, ask_question_tool = AgentTools(agents=asked_agents).tools()
		delegate_work_tool,_  = AgentTools(agents=delegated_agents).tools()

		for agent_name in self.config['agents'].keys():
			if self.config['agents'][agent_name].get('tools', None):
				for tool_name in self.config['agents'][agent_name]['tools']:
					if tool_name == "ask_question_tool":
						agent_dict[agent_name].tools.append(ask_question_tool)
					elif tool_name == "delegate_work_tool":
						agent_dict[agent_name].tools.append(delegate_work_tool)
		return agent_dict

	def stop(self):
		self.mcp_server.stop()

	def _create_manager_agent(self) -> Agent:
		manager_goal = self.config['agents']['manager_agent']['goal']
		manager_backstory = self.config['agents']['manager_agent']['backstory']
		fn_call_llm = LLM(**self.llm_config.get('fn_call_llm', None)) if self.llm_config.get('fn_call_llm', None) else None
		manager = Agent(
			role="Project Manager",
			goal=manager_goal, #"Efficiently manage the crew and ensure high-quality task completion",
			backstory=manager_backstory, #"You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
			allow_delegation=True,
			llm=LLM(**self.llm_config["manager"]),
			function_calling_llm=fn_call_llm,
			verbose=True,
		)
		return manager

	def _create_agent(self, agent_name: str, tool_dict: dict) -> Agent:
		if not self.config['agents'][agent_name]:
			raise ValueError(f"Agent {agent_name} not found in config")
		
		tools = []
		if self.config['agents'][agent_name].get('tools', None):
			for tool_name in self.config['agents'][agent_name]['tools']:
				if not tool_name in ["ask_question_tool", "delegate_work_tool"]:
					tools.append(tool_dict[tool_name])
		
		fn_call_llm = LLM(**self.llm_config.get('fn_call_llm', None)) if self.llm_config.get('fn_call_llm', None) else None

		if self.config['agents'][agent_name].get('role', None):
			role = self.config['agents'][agent_name]['role']
		else:
			role = agent_name

		agent = Agent(
			role=role,
			goal=self.config['agents'][agent_name]['goal'],
			backstory=self.config['agents'][agent_name]['backstory'],
			llm = LLM(**self.llm_config[agent_name]),
			tools = tools,
			function_calling_llm= fn_call_llm,
			verbose=True,
			#use_system_prompt=False,
		)

		return agent

	def _create_working_agents(self,tool_dict: dict) -> dict[str, Agent]:
		agents_dict = {}
		for agent_name in self.config['agents'].keys():
			if agent_name == 'manager_agent':
				continue
			agents_dict[agent_name] = self._create_agent(agent_name, tool_dict)
		return agents_dict

	def crew(self, work_dir: str, conversation_id: Optional[str] = None) -> Crew:
		"""Creates the VASPilot crew"""
		if not os.path.exists(f"{work_dir}/memory/"):
			os.makedirs(f"{work_dir}/memory/")
		tool_dict = self._create_tools(conversation_id)
		agent_dict = self._create_working_agents(tool_dict)
		manager_agent = self._create_manager_agent()
		agent_dict = self._inject_agent_tools(agent_dict)
		working_agents = list(agent_dict.values())
		return Crew(
			agents=working_agents,
			tasks=[],
			process=Process.hierarchical,
			verbose=True,
			output_log_file=f"{work_dir}/output.log",
			manager_agent=manager_agent,
			memory=True,
			long_term_memory = LongTermMemory(
        		storage=LTMSQLiteStorage(
            		db_path=f"{work_dir}/memory/ltm_storage.db",
        		)
    		),
			short_term_memory = ShortTermMemory(
				storage = RAGStorage(
					type="short_term",
					path=f"{work_dir}/memory/stm",
					embedder_config={
						"provider": "custom",
						"config": {
							"embedder": self.embedder
						}
    				}
				),
			),
			entity_memory = EntityMemory(
				storage=RAGStorage(
					type="short_term",
					path=f"{work_dir}/memory/etm",
					embedder_config={
						"provider": "custom",
						"config": {
							"embedder": self.embedder
						}
    				}
				),
    		),
		)
