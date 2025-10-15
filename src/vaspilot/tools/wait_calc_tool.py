import numpy as np
import time
import asyncio
from typing import Type, Optional, Dict, Any, List, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from fastmcp.client import Client
from ..listener.message_listener import MessageListener

class WaitCalcInput(BaseModel):
    """归档工具的输入模式"""
    calculation_ids: List[str] = Field(..., description="要检查的计算ID列表")

class WaitCalcTool(BaseTool):
    mcp_url: str = "http://localhost:8933/mcp"
    message_listener: Optional[MessageListener] = None
    conversation_id: Optional[str] = None
    args_schema: Type[BaseModel] = WaitCalcInput
    
    def __init__(self, mcp_url: str, message_listener: Optional[MessageListener] = None, conversation_id: Optional[str] = None):
        super().__init__(
            name="wait_calculations",
            description="检查计算任务状态并返回结果"
        )
        self.mcp_url = mcp_url
        self.message_listener = message_listener
        self.conversation_id = conversation_id

    async def _check_status(self, calculation_ids: List[str]) -> Dict[str, Any]:
        async with Client(self.mcp_url) as client:
            # call tool
            tool_result = await client.call_tool("check_calculation_status", {"calculation_ids": calculation_ids})
        if tool_result.data is None:
            return {"error": "No result from check_calculation_status"}
        else:
            return tool_result.data

    async def _cancel_slurm_job(self, calculation_ids: List[str]) -> Dict[str, Any]:
        async with Client(self.mcp_url) as client:
            # call tool
            tool_result = await client.call_tool("cancel_slurm_job", {"calc_ids": calculation_ids})
        if tool_result.data is None:
            return {"error": "No result from cancel_slurm_job"}
        else:
            return tool_result.data

    def _run(self,
             calculation_ids: List[str]) -> Dict[str, Any]:
        """
        检查计算任务状态并返回结果
        
        Args:
            calculation_ids: 要检查的计算ID列表
        
        Returns:
            包含每个计算任务状态和结果的字典，格式为：
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... 其他结果数据
                }
            }
        """
        if not calculation_ids:
            return {}
        
        print(f"开始监控计算状态，计算ID列表: {calculation_ids}")
        
        # 维护已完成和最终结果的字典
        completed_results = {}
        pending_calc_ids = calculation_ids.copy()
        
        while pending_calc_ids:
            try:
                # 有用户消息时，取消计算任务并返回给agent
                if self.message_listener is not None:
                    message = self.message_listener.read_user_message(self.conversation_id)
                    if message:
                        asyncio.run(self._cancel_slurm_job(pending_calc_ids))
                        return {"status": "Interrupted by user", "user_message": message}
                
                # 只检查尚未完成的计算任务
                status_result = asyncio.run(self._check_status(pending_calc_ids))
                
                if "error" in status_result:
                    print(f"检查状态时出错: {status_result['error']}")
                    return status_result
                
                # 检查哪些计算已完成，移出待检查列表
                newly_completed = []
                for calc_id in pending_calc_ids.copy():
                    if calc_id in status_result:
                        status = status_result[calc_id].get("status", "unknown")
                        if status in ["completed", "failed", "cancelled", "unknown", "timeout"]:
                            # 任务已完成，保存结果并从待检查列表中移除
                            completed_results[calc_id] = status_result[calc_id]
                            newly_completed.append(calc_id)
                            pending_calc_ids.remove(calc_id)
                    else:
                        # 如果某个计算ID不在结果中，暂时保留在待检查列表中
                        pass
                
                # 统计当前状态
                running_count = len(pending_calc_ids)
                completed_count = len([r for r in completed_results.values() if r.get("status") == "completed"])
                failed_count = len([r for r in completed_results.values() if r.get("status") in ["failed", "unknown"]])
                
                if newly_completed:
                    print(f"新完成的任务: {newly_completed}")
                
                print(f"状态检查结果: 运行中 {running_count}, 已完成 {completed_count}, 失败 {failed_count}")
                
                # 如果所有计算都已完成，返回结果
                if not pending_calc_ids:
                    print("所有计算任务已完成")
                    return completed_results
                
                # 等待30秒后继续检查
                print(f"还有 {len(pending_calc_ids)} 个任务未完成，等待30秒后继续检查...")
                time.sleep(30)
                
            except Exception as e:
                print(f"监控过程中发生错误: {str(e)}")
                return {"error": f"监控过程中发生错误: {str(e)}"}
        
        # 理论上不应该到达这里，但为了满足linter要求添加默认返回值
        return completed_results
        