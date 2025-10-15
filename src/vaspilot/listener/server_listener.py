from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
    TaskEvaluationEvent,
)
from datetime import datetime
import json
from typing import Dict, Any, List

from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

class CrewServer(ABC):


    def __init__(self) -> None:
        super().__init__()
        self.history_log = []

    @abstractmethod
    def system_log(self, message: str, crew_fingerprint: str = None):
        pass
    
    @abstractmethod
    def agent_input(self, agent_role: str, message: str, crew_fingerprint: str = None):
        pass    
    
    @abstractmethod
    def agent_output(self, agent_role: str, message: str, crew_fingerprint: str = None):
        pass
    
    @abstractmethod
    def tool_input(self, tool_name: str, message: Dict[str, Any], crew_fingerprint: str = None):
        pass
    
    @abstractmethod
    def tool_output(self, tool_name: str, message: Dict[str, Any], crew_fingerprint: str = None):
        pass

    @abstractmethod
    def save_conversation_messages(self, conversation_id: str, role_name:str, message:str):
        pass

    def log_history(self, history: Dict[str, Any]):
        self.history_log.append(history)

class ServerListener(BaseEventListener):
    def __init__(self, server: CrewServer, exclude_agents: list = [], exclude_tools: list = []):
        super().__init__()
        self.exclude_agents = exclude_agents
        self.exclude_tools = exclude_tools
        self.server = server

    def _format_tool_output(self, tool_output: str) -> Dict[str, Any]:
        system_prompt = '\nYou ONLY have access to the following tools, and should NEVER make up tools that are not listed here:'
        tool_output = tool_output.split(system_prompt)[0]
        tool_output = tool_output.replace("'", '"')
        tool_output = tool_output.replace("None", "null")
        tool_output = tool_output.replace("True", "true")
        tool_output = tool_output.replace("False", "false")
        try:
            # 尝试解析JSON
            return json.loads(tool_output)
        except (json.JSONDecodeError, ValueError) as e:
            # 如果JSON解析失败，返回原始输出的字典格式
            print(f"[WARNING] 工具输出JSON解析失败: {str(e)}")
            print(f"[WARNING] 原始输出: {tool_output[:200]}...")
            return {
                "raw_output": tool_output,
                "parse_error": str(e),
                "error_type": "json_parse_failed"
            }

    def _format_agent_input(self, agent_input: str) -> str:
        system_prompt = "\n\n# Useful context: "
        task_input = agent_input.split(system_prompt)[0]
        return task_input

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            if event.crew is not None:
                self.server.system_log(f"Crew '{event.crew.fingerprint.uuid_str}' has started execution!", event.crew.fingerprint.uuid_str)
                self.server.log_history(
                    {
                        "type": "system",
                        "message": f"Crew '{event.crew.fingerprint.uuid_str}' has started execution!",
                        "timestamp": datetime.now().isoformat()
                    }
                )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            if event.crew is not None:
                self.server.system_log(f"Crew '{event.crew.fingerprint.uuid_str}' has finished execution!", event.crew.fingerprint.uuid_str)
                self.server.log_history(
                    {
                        "type": "system",
                        "message": f"Crew '{event.crew.fingerprint.uuid_str}' has finished execution!",
                        "timestamp": datetime.now().isoformat()
                    }
                )

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            if source.crew is not None:
                if not event.agent.role in self.exclude_agents:
                    task_input = self._format_agent_input(event.task_prompt)
                    self.server.agent_input(event.agent.role, task_input, source.crew.fingerprint.uuid_str)
                    self.server.log_history(
                        {
                            "type": "agent_input",
                            "name": event.agent.role,
                            "content": task_input,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            if source.crew is not None:
                self.server.agent_output(event.agent.role, event.output, source.crew.fingerprint.uuid_str)
                self.server.log_history(
                    {
                        "type": "agent_output",
                        "name": event.agent.role,
                        "content": event.output,
                        "timestamp": datetime.now().isoformat()
                    }
                )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            if source.agent is not None:
                if source.agent.crew is not None:
                    if not event.tool_name in self.exclude_tools:
                        tool_args = event.tool_args if isinstance(event.tool_args, dict) else json.loads(event.tool_args)
                        self.server.tool_input(event.tool_name, tool_args, source.agent.crew.fingerprint.uuid_str)
                        self.server.log_history(
                            {
                                "type": "tool_input",
                                "name": event.tool_name,
                                "content": tool_args,
                                "timestamp": datetime.now().isoformat()
                            }
                        )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            if source.agent is not None:
                if source.agent.crew is not None:
                    if not event.tool_name in self.exclude_tools:
                        try:
                            tool_output = self._format_tool_output(event.output)
                            self.server.tool_output(event.tool_name, tool_output, source.agent.crew.fingerprint.uuid_str)
                            self.server.log_history(
                                {
                                    "type": "tool_output",
                                    "name": event.tool_name,
                                    "content": tool_output,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                            print(f"[INFO] 成功记录工具输出: {event.tool_name}")
                        except Exception as e:
                            # 如果所有处理都失败，至少记录原始输出
                            print(f"[ERROR] 处理工具输出时发生错误: {str(e)}")
                            print(f"[ERROR] 工具名称: {event.tool_name}")
                            print(f"[ERROR] 原始输出: {str(event.output)[:200]}...")
                            
                            # 创建一个安全的错误输出记录
                            fallback_output = {
                                "error": "tool_output_processing_failed",
                                "tool_name": event.tool_name,
                                "raw_output": str(event.output),
                                "error_message": str(e)
                            }
                            
                            try:
                                self.server.tool_output(event.tool_name, fallback_output, source.agent.crew.fingerprint.uuid_str)
                                self.server.log_history(
                                    {
                                        "type": "tool_output",
                                        "name": event.tool_name,
                                        "content": fallback_output,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )
                                print(f"[INFO] 使用fallback方式记录了工具输出: {event.tool_name}")
                            except Exception as inner_e:
                                print(f"[FATAL] 连fallback记录都失败了: {str(inner_e)}")