import numpy as np
import time
import asyncio
from typing import Type, Optional, Dict, Any, List, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from fastmcp.client import Client
from ..listener.message_listener import MessageListener

class AskUserInput(BaseModel):
    """归档工具的输入模式"""
    message: str = Field(..., description="要向用户询问的问题")

class AskUserTool(BaseTool):
    message_listener: MessageListener = None
    conversation_id: str = None
    args_schema: Type[BaseModel] = AskUserInput
    
    def __init__(self, message_listener: MessageListener = None, conversation_id: str = None):
        super().__init__(
            name="ask_user",
            description="针对不确定的问题，向用户询问"
        )
        self.message_listener = message_listener
        self.conversation_id = conversation_id

    def _run(self,
             message: List[str]) -> Dict[str, Any]:
        """
            向用户询问问题并返回结果
            
            Args:
                message: 要向用户询问的问题
            
            Returns:
                包含用户回答的字典，格式为：
                {
                    status: 字符串，"success" 或 "error"
                    user_message: 字符串，用户的消息内容
                }
        """
        
        print(f"开始向用户询问问题，问题: {message}")
        self.message_listener.send_agent_message(message, self.conversation_id)
        while True:
            try:
                # 有用户消息时返回
                message = self.message_listener.read_user_message(self.conversation_id)
                if message:
                    return {"status": "success", "user_message": message}
                else:
                    time.sleep(5)
            except Exception as e:
                return {"status": "error", "error_message": str(e)}
