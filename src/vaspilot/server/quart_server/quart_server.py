#!/usr/bin/env python3
"""
CrewAI VASP Quart异步服务器
功能：任务提交、历史记录、详情查看、实时更新、并行任务队列管理
基于 CrewServer 基类实现，支持异步操作
"""

import os
import sys
import json
import uuid
import threading
import argparse
import re
import signal
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import zipfile
import tempfile

from quart import Quart, render_template, request, jsonify, g, send_file, abort, Response
import aiosqlite
from markdown import markdown
import ctypes
from werkzeug.utils import secure_filename

# 添加项目路径到sys.path
current_dir = Path(__file__).parent  # quart_server/

# 导入项目模块
from ...listener.server_listener import CrewServer, ServerListener
from ...listener.message_listener import MessageListener
from ...crew import VaspCrew
from crewai import Task
from fastmcp.client import Client


class TaskStatus(Enum):
    """任务状态枚举"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedTask:
    """队列中的任务"""
    conversation_id: str
    task_description: str
    created_at: datetime
    status: TaskStatus = TaskStatus.QUEUED


class QuartCrewServer(CrewServer):
    """基于Quart的异步CrewServer实现"""
    
    def __init__(self, crew_config: Dict[str, Any], title: str = "VASPilot Async Server", 
                 work_dir: str = ".", db_path: Optional[str] = None, 
                 allow_path: Optional[str] = None, max_concurrent_tasks: int = 3,
                 max_queue_size: int = 10):
        super().__init__()
        self.title = title
        self.config = crew_config
        self.work_dir = os.path.abspath(work_dir)
        self.allow_path = allow_path
        
        # 并发控制参数
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        
        # 任务管理
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: List[QueuedTask] = []
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._current_conversation_id: Optional[str] = None
        
        # 数据库路径
        if db_path is None:
            db_path = os.path.join(work_dir, 'crew_tasks.db')
        self.db_path = os.path.abspath(db_path)
        
        # 创建Quart应用
        template_folder = str(current_dir / "templates")
        self.app = Quart(__name__, template_folder=template_folder)
        self.app.secret_key = 'crew-ai-quart-server'
        
        # 上传目录
        self.upload_dir = os.path.join(self.work_dir, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self.message_listener = MessageListener(self)
        self.generator = VaspCrew(self.config, self.message_listener)
        self.current_logger = ServerListener(self)
        # 并行任务下映射关系：conversation_id <-> crew_fingerprint
        self._conversation_to_fingerprint: Dict[str, str] = {}
        self._fingerprint_to_conversation: Dict[str, str] = {}
        self._mapping_lock = threading.Lock()
        self._running_threads: Dict[str, threading.Thread] = {}
        self._crew_thread_ids: Dict[str, int] = {}
        
        # 日志基础设施（启动时在事件循环内初始化）
        self._log_queue = None
        self._log_worker_task = None
        self._event_loop = None
        # SSE 订阅者映射：conversation_id -> set(asyncio.Queue)
        self._sse_subscribers: Dict[str, set] = {}
        
        # 设置路由
        self._setup_routes()

    async def _init_db(self):
        """异步初始化数据库"""
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"📁 创建数据库目录: {db_dir}")
            
            print(f"🗄️ 初始化数据库: {self.db_path}")
            
            async with aiosqlite.connect(self.db_path) as conn:
                # 创建 task_executions 表
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT UNIQUE NOT NULL,
                        task_description TEXT NOT NULL,
                        status TEXT DEFAULT 'queued',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        result TEXT,
                        error_message TEXT
                    )
                ''')
                
                # 创建 activity_logs 表
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS activity_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        role_name TEXT,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES task_executions (conversation_id)
                    )
                ''')
                
                await conn.commit()
                
                # 验证表是否创建成功
                async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] async for row in cursor]
                    expected_tables = ['task_executions', 'activity_logs']
                    
                    for table in expected_tables:
                        if table in tables:
                            print(f"✅ 表 '{table}' 创建成功")
                        else:
                            raise Exception(f"表 '{table}' 创建失败")
                            
                print("🎉 数据库初始化完成")
                
        except Exception as e:
            print(f"❌ 数据库初始化失败: {str(e)}")
            print(f"数据库路径: {self.db_path}")
            print(f"工作目录: {self.work_dir}")
            raise

    async def _get_db(self):
        """获取数据库连接"""
        db = getattr(g, '_database', None)
        if db is None:
            try:
                db = g._database = await aiosqlite.connect(self.db_path)
                db.row_factory = aiosqlite.Row
                
                # 验证表是否存在
                async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'") as cursor:
                    result = await cursor.fetchone()
                    if not result:
                        # 如果表不存在，重新初始化数据库
                        print("⚠️ 检测到表不存在，重新初始化数据库...")
                        await db.close()
                        await self._init_db()
                        db = g._database = await aiosqlite.connect(self.db_path)
                        db.row_factory = aiosqlite.Row
                        
            except Exception as e:
                print(f"❌ 数据库连接失败: {str(e)}")
                raise
        return db

    async def _close_connection(self, exception):
        """关闭数据库连接"""
        db = getattr(g, '_database', None)
        if db is not None:
            await db.close()

    async def _get_recent_tasks(self, limit=10):
        """获取最近的任务"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM task_executions ORDER BY created_at DESC LIMIT ?',
                (limit,)
            ) as cursor:
                return await cursor.fetchall()

    async def _get_task_by_id(self, conversation_id):
        """根据ID获取任务"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM task_executions WHERE conversation_id = ?',
                (conversation_id,)
            ) as cursor:
                return await cursor.fetchone()

    async def _get_conversation_messages(self, conversation_id):
        """获取对话消息"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT role_name, content, timestamp FROM activity_logs WHERE conversation_id = ? AND type = ? ORDER BY id ASC',
                (conversation_id, 'conversation_messages')
            ) as cursor:
                messages = await cursor.fetchall()
        
        # 格式化消息
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'role': msg['role_name'] if msg['role_name'] else 'User',
                'content': msg['content'],
                'timestamp': msg['timestamp']
            })
        
        return formatted_messages

    async def _get_task_logs(self, conversation_id):
        """获取任务日志"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM activity_logs WHERE conversation_id = ? ORDER BY id ASC',
                (conversation_id,)
            ) as cursor:
                logs = await cursor.fetchall()
        
        # 格式化日志
        formatted_logs = []
        for log in logs:
            type_names = {
                'system': '系统',
                'agent_input': 'Agent输入',
                'agent_output': 'Agent输出',
                'tool_input': 'Tool输入',
                'tool_output': 'Tool输出'
            }
            
            # 安全地获取role_name字段（兼容旧数据）
            try:
                role_name = log['role_name'] if 'role_name' in log.keys() else None
            except (KeyError, TypeError):
                role_name = None
            
            formatted_logs.append({
                'type': log['type'],
                'type_name': type_names.get(log['type'], log['type']),
                'role_name': role_name,
                'content': log['content'],
                'timestamp': self._to_beijing_time_str(log['timestamp']),
                'preview': log['content'][:30] + '...' if len(log['content']) > 30 else log['content']
            })
        
        return formatted_logs

    async def _get_queue_status(self):
        """获取队列状态"""
        running_count = len(self.running_tasks)
        queued_count = len(self.task_queue)
        
        # 从数据库获取最新状态
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(f"SELECT COUNT(*) as count FROM task_executions WHERE status = '{TaskStatus.RUNNING.value}'") as cursor:
                db_running = await cursor.fetchone()
                db_running_count = db_running[0] if db_running else 0
            
            async with db.execute(f"SELECT COUNT(*) as count FROM task_executions WHERE status = '{TaskStatus.QUEUED.value}'") as cursor:
                db_queued = await cursor.fetchone()
                db_queued_count = db_queued[0] if db_queued else 0
        
        return {
            'running_count': running_count,
            'queued_count': queued_count,
            'db_running_count': db_running_count,
            'db_queued_count': db_queued_count,
            'max_concurrent': self.max_concurrent_tasks,
            'max_queue_size': self.max_queue_size
        }

    def _to_beijing_time_str(self, value: Any) -> Optional[str]:
        """将传入的 UTC/本地时间字符串或datetime转换为北京时间字符串。
        - 支持 str: 'YYYY-MM-DD HH:MM:SS[.ffffff]' 或 ISO 格式；
        - 支持 datetime: 有/无 tzinfo;
        返回格式: 'YYYY-MM-DD HH:MM:SS'
        """
        if value is None:
            return None
        dt: Optional[datetime] = None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            s = value.strip()
            # 先尝试常见格式
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except Exception:
                    dt = None
            if dt is None:
                # 退回ISO格式
                try:
                    # 支持末尾Z
                    if s.endswith('Z'):
                        s = s[:-1]
                        dt = datetime.fromisoformat(s)
                    else:
                        dt = datetime.fromisoformat(s)
                except Exception:
                    return value
        else:
            return str(value)

        # 将有时区信息的时间转换为UTC无tz的时间
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            dt_utc = dt - dt.utcoffset()
        else:
            # 假定数据库的 CURRENT_TIMESTAMP 为UTC
            dt_utc = dt

        bj_dt = dt_utc + timedelta(hours=8)
        return bj_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _format_task_row(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """将任务记录行转换为带北京时间字符串的字典。"""
        return {
            'conversation_id': row['conversation_id'],
            'task_description': row['task_description'],
            'status': row['status'],
            'created_at': self._to_beijing_time_str(row['created_at']),
            'started_at': self._to_beijing_time_str(row['started_at']) if 'started_at' in row.keys() else None,
            'completed_at': self._to_beijing_time_str(row['completed_at']) if 'completed_at' in row.keys() else None,
            'result': row['result'] if 'result' in row.keys() else None,
            'error_message': row['error_message'] if 'error_message' in row.keys() else None,
        }

    def _setup_routes(self):
        """设置Quart路由"""
        
        @self.app.teardown_appcontext
        async def close_connection(exception):
            await self._close_connection(exception)

        # 统一API错误为JSON，避免返回HTML导致前端解析错误
        @self.app.errorhandler(404)
        async def handle_404(error):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Not Found', 'path': request.path, 'status': 404}), 404
            return str(error), 404

        @self.app.errorhandler(405)
        async def handle_405(error):
            if request.path.startswith('/api/'):
                allowed = getattr(error, 'valid_methods', None)
                return jsonify({'error': 'Method Not Allowed', 'path': request.path, 'status': 405, 'allowed': allowed}), 405
            return str(error), 405

        @self.app.errorhandler(500)
        async def handle_500(error):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Internal Server Error', 'path': request.path, 'status': 500, 'message': str(error)}), 500
            return str(error), 500
        
        @self.app.route('/')
        async def index():
            """主页"""
            recent_tasks_rows = await self._get_recent_tasks()
            recent_tasks = [self._format_task_row(row) for row in recent_tasks_rows]
            queue_status = await self._get_queue_status()
            return await render_template('base.html', 
                                       title=self.title,
                                       recent_tasks=recent_tasks,
                                       queue_status=queue_status)

        @self.app.route('/upload', methods=['POST'])
        async def upload_structure():
            """上传晶体结构文件，返回保存后的绝对路径"""
            try:
                files = await request.files
                if 'file' not in files:
                    return jsonify({'error': 'File field not found'}), 400
                file = files['file']
                if not file or file.filename == '':
                    return jsonify({'error': 'No file uploaded'}), 400

                filename = secure_filename(file.filename)
                lower_name = filename.lower()
                if not (lower_name.endswith(('.vasp', '.cif', '.xyz')) or lower_name in ('poscar', 'contcar')):
                    return jsonify({'error': 'File type not supported. Only .vasp/.cif/.xyz or POSCAR/CONTCAR are allowed'}), 400

                unique_name = f"{uuid.uuid4().hex}_{filename}"
                save_path = os.path.join(self.upload_dir, unique_name)
                await file.save(save_path)
                abs_path = os.path.abspath(save_path)
                return jsonify({'success': True, 'path': abs_path, 'filename': filename})
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500

        @self.app.route('/submit', methods=['POST'])
        async def submit_task():
            """提交任务"""
            try:
                data = await request.get_json()
                task_description = data.get('task_description', '').strip()
                
                if not task_description:
                    return jsonify({'error': 'Please enter a valid task description'}), 400
                
                # 检查队列是否已满
                current_queue_size = len(self.task_queue)
                current_running = len(self.running_tasks)
                
                if current_queue_size + current_running >= self.max_queue_size + self.max_concurrent_tasks:
                    return jsonify({'error': f'队列已满，当前运行: {current_running}, 队列中: {current_queue_size}, 最大限制: {self.max_queue_size + self.max_concurrent_tasks}'}), 400
                
                # 创建任务记录
                conversation_id = str(uuid.uuid4())
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        'INSERT INTO task_executions (conversation_id, task_description, status) VALUES (?, ?, ?)',
                        (conversation_id, task_description, TaskStatus.QUEUED.value)
                    )
                    await db.commit()
                
                # 添加到队列并尝试处理
                queued_task = QueuedTask(
                    conversation_id=conversation_id,
                    task_description=task_description,
                    created_at=datetime.now()
                )
                self.task_queue.append(queued_task)
                
                # 异步处理队列
                asyncio.create_task(self._process_queue())
                
                return jsonify({
                    'success': True,
                    'conversation_id': conversation_id,
                    'message': 'Task submitted successfully',
                    'queue_position': len(self.task_queue)
                })
                
            except Exception as e:
                return jsonify({'error': f'Server error: {str(e)}'}), 500

        @self.app.route('/task/<conversation_id>')
        async def task_detail(conversation_id):
            """任务详情页面"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return "Task not found", 404
            
            logs = await self._get_task_logs(conversation_id)
            messages = await self._get_conversation_messages(conversation_id)
            recent_tasks_rows = await self._get_recent_tasks()
            recent_tasks = [self._format_task_row(row) for row in recent_tasks_rows]
            # 任务详情时间转为北京时间
            task_dict = {
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description'],
                'status': task['status'],
                'created_at': self._to_beijing_time_str(task['created_at']),
                'started_at': self._to_beijing_time_str(task['started_at']) if task['started_at'] else None,
                'completed_at': self._to_beijing_time_str(task['completed_at']) if task['completed_at'] else None,
                'result': task['result'],
                'error_message': task['error_message']
            }
            queue_status = await self._get_queue_status()
            
            return await render_template('task_detail.html',
                                       title=self.title,
                                       task=task_dict,
                                       logs=logs,
                                       messages=messages,
                                       recent_tasks=recent_tasks,
                                       queue_status=queue_status)

        @self.app.route('/api/task/<conversation_id>/events')
        async def task_events(conversation_id):
            """任务事件SSE接口：推送日志与状态变更。"""
            # 为当前会话创建订阅队列
            q: asyncio.Queue = asyncio.Queue()
            subs = self._sse_subscribers.setdefault(conversation_id, set())
            subs.add(q)

            async def gen():
                try:
                    # 初始推送当前状态
                    task = await self._get_task_by_id(conversation_id)
                    if task:
                        initial = json.dumps({'event': 'status', 'status': task['status']})
                        yield f"data: {initial}\n\n"
                    while True:
                        payload = await q.get()
                        data = json.dumps(payload, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    # 断开时移除订阅
                    subs = self._sse_subscribers.get(conversation_id)
                    if subs and q in subs:
                        subs.remove(q)
            return Response(gen(), mimetype='text/event-stream')

        @self.app.route('/api/task/<conversation_id>/status')
        async def get_task_status(conversation_id):
            """获取任务状态API"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            
            return jsonify({
                'status': task['status'],
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description']
            })

        @self.app.route('/api/task/<conversation_id>/logs')
        async def get_task_logs(conversation_id):
            """获取任务日志API"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            
            logs = await self._get_task_logs(conversation_id)
            messages = await self._get_conversation_messages(conversation_id)
            
            # 将日志转换为字典格式
            logs_data = []
            for log in logs:
                logs_data.append({
                    'type': log['type'],
                    'type_name': log['type_name'],
                    'role_name': log['role_name'],
                    'content': log['content'],
                    'timestamp': log['timestamp'],
                    'preview': log['preview']
                })
            
            return jsonify({
                'task': {
                    'status': task['status'],
                    'conversation_id': task['conversation_id'],
                    'task_description': task['task_description'],
                    'result': task['result'],
                    'error_message': task['error_message']
                },
                'logs': logs_data,
                'messages': messages
            })

        @self.app.route('/api/task/<conversation_id>/user_message', methods=['POST'])
        async def user_message(conversation_id):
            """接收人类反馈并转发给消息监听器"""
            try:
                # 校验任务是否存在
                task = await self._get_task_by_id(conversation_id)
                if not task:
                    return jsonify({'error': 'Task not found'}), 404
                data = await request.get_json()
                if not isinstance(data, dict):
                    return jsonify({'error': 'Invalid JSON payload'}), 400

                # 兼容字段名：message / content / text
                message = data.get('message') or data.get('content') or data.get('text')
                if not isinstance(message, str) or not message.strip():
                    return jsonify({'error': 'Please provide non-empty "message"'}), 400
                if task["status"] == TaskStatus.RUNNING.value:
                    self.message_listener.send_user_message(message.strip(), conversation_id)
                    return jsonify({'success': True, 'conversation_id': conversation_id})
                elif task["status"] == TaskStatus.COMPLETED.value:
                    if len(self.task_queue) + len(self.running_tasks) >= self.max_queue_size + self.max_concurrent_tasks:
                        return jsonify({'error': f'队列已满，当前运行: {len(self.running_tasks)}, 队列中: {len(self.task_queue)}, 最大限制: {self.max_queue_size + self.max_concurrent_tasks}'}), 400

                    # 添加到队列并尝试处理
                    queued_task = QueuedTask(
                        conversation_id=conversation_id,
                        task_description=message.strip(),
                        created_at=datetime.now()
                    )
                    self.task_queue.append(queued_task)
                    
                    # 异步处理队列
                    asyncio.create_task(self._process_queue())
                    
                    return jsonify({
                        'success': True,
                        'conversation_id': conversation_id,
                        'message': 'Task submitted successfully',
                        'queue_position': len(self.task_queue)
                    })
                else:
                    return jsonify({'error': 'Task is not running or completed'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to handle human message: {str(e)}'}), 500

        @self.app.route('/api/tasks')
        async def get_tasks():
            """获取任务列表API"""
            try:
                recent_tasks = await self._get_recent_tasks()
                tasks_data = []
                for task in recent_tasks:
                    tasks_data.append({
                        'conversation_id': task['conversation_id'],
                        'task_description': task['task_description'],
                        'status': task['status'],
                        'created_at': self._to_beijing_time_str(task['created_at']),
                        'started_at': self._to_beijing_time_str(task['started_at']) if task['started_at'] else None,
                        'completed_at': self._to_beijing_time_str(task['completed_at']) if task['completed_at'] else None
                    })
                return jsonify(tasks_data)
            except Exception as e:
                return jsonify({'error': f'Failed to get task list: {str(e)}'}), 500

        @self.app.route('/api/queue/status')
        async def get_queue_status_api():
            """获取队列状态API"""
            try:
                status = await self._get_queue_status()
                # 添加队列详情
                queue_details = []
                for i, queued_task in enumerate(self.task_queue):
                    queue_details.append({
                        'conversation_id': queued_task.conversation_id,
                        'task_description': queued_task.task_description,
                        'position': i + 1,
                        'created_at': self._to_beijing_time_str(queued_task.created_at)
                    })
                
                status['queue_details'] = queue_details
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': f'Failed to get queue status: {str(e)}'}), 500

        @self.app.route('/api/files/<conversation_id>/<path:filename>')
        async def serve_task_file(conversation_id, filename):
            """为特定任务提供文件访问"""
            from urllib.parse import unquote
            
            try:
                # 对路径进行分段解码
                path_segments = filename.split('/')
                decoded_segments = [unquote(segment) for segment in path_segments]
                decoded_filename = '/'.join(decoded_segments)
                
                # 检查是否有绝对路径标记
                is_absolute_path = False
                if decoded_filename.startswith('__ABS__'):
                    decoded_filename = decoded_filename[7:]
                    is_absolute_path = True
                
                # 构建文件路径
                task_dir = os.path.join(self.work_dir, conversation_id)
                
                if is_absolute_path or (decoded_filename.startswith('/') and self.allow_path):
                    file_path = decoded_filename
                else:
                    file_path = os.path.join(task_dir, decoded_filename)
                
                # 安全检查
                file_path = os.path.abspath(file_path)
                task_dir = os.path.abspath(task_dir)
                
                if not is_absolute_path and not self.allow_path:
                    if not file_path.startswith(task_dir) and not file_path.startswith(self.work_dir):
                        abort(403, description="Access denied: file path not in allowed range")
                
                if not os.path.exists(file_path):
                    abort(404, description=f"File not found: {decoded_filename}")
                
                # 根据文件扩展名设置MIME类型
                if decoded_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    mimetype = 'image/png' if decoded_filename.lower().endswith('.png') else 'image/jpeg'
                elif decoded_filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                    mimetype = 'text/plain'
                else:
                    mimetype = 'application/octet-stream'
                
                return await send_file(file_path, mimetype=mimetype)
                
            except Exception as e:
                abort(500, description=f"File service error: {str(e)}")

        @self.app.route('/api/files/<conversation_id>/zip/<path:dirpath>')
        async def zip_task_dir(conversation_id, dirpath):
            """将指定目录/文件打包为ZIP并下载"""
            from urllib.parse import unquote
            
            try:
                # 对路径进行分段解码
                path_segments = dirpath.split('/')
                decoded_segments = [unquote(segment) for segment in path_segments]
                decoded_dirpath = '/'.join(decoded_segments)
                
                # 处理绝对路径前缀
                is_absolute_path = False
                if decoded_dirpath.startswith('__ABS__'):
                    decoded_dirpath = decoded_dirpath[7:]
                    is_absolute_path = True
                
                task_dir = os.path.join(self.work_dir, conversation_id)
                
                if is_absolute_path or (decoded_dirpath.startswith('/') and self.allow_path):
                    src_path = decoded_dirpath
                else:
                    src_path = os.path.join(task_dir, decoded_dirpath)
                
                # 安全检查
                src_path = os.path.abspath(src_path)
                task_dir = os.path.abspath(task_dir)
                
                if not is_absolute_path and not self.allow_path:
                    if not src_path.startswith(task_dir) and not src_path.startswith(self.work_dir):
                        abort(403, description="Access denied: path not in allowed range")
                
                if not os.path.exists(src_path):
                    abort(404, description=f"Path not found: {decoded_dirpath}")
                
                # 生成临时zip文件（异步避免阻塞事件循环）
                def _create_zip(tmp_zip_path: str, source_path: str):
                    with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        if os.path.isdir(source_path):
                            base_name = os.path.basename(os.path.normpath(source_path)) or 'folder'
                            for root, dirs, files in os.walk(source_path):
                                for fname in files:
                                    abs_file = os.path.join(root, fname)
                                    rel_in_dir = os.path.relpath(abs_file, source_path)
                                    arcname = os.path.join(base_name, rel_in_dir)
                                    zf.write(abs_file, arcname=arcname)
                        else:
                            base_name = os.path.basename(source_path) or 'file'
                            zf.write(source_path, arcname=base_name)
                    return base_name
                
                # 创建临时文件路径
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                tmp_zip_path = tmp_file.name
                tmp_file.close()
                
                # 在线程中创建zip
                base_name = await asyncio.to_thread(_create_zip, tmp_zip_path, src_path)
                download_name = f"{base_name}.zip"
                
                # 发送文件，并在后台延迟清理临时文件（避免兼容性问题）
                response = await send_file(
                    tmp_zip_path,
                    mimetype='application/zip',
                    as_attachment=True,
                    attachment_filename=download_name
                )
                
                async def _cleanup_tmp(path: str):
                    try:
                        await asyncio.sleep(300)
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                
                try:
                    asyncio.create_task(_cleanup_tmp(tmp_zip_path))
                except Exception:
                    pass
                return response
            except Exception as e:
                abort(500, description=f"Zip service error: {str(e)}")

        @self.app.route('/api/files/<conversation_id>/list')
        async def list_task_files(conversation_id):
            """列出任务目录中的所有文件"""
            from urllib.parse import quote
            
            try:
                task_dir = os.path.join(self.work_dir, conversation_id)
                if not os.path.exists(task_dir):
                    return jsonify({'files': []})
                
                files = []
                for root, dirs, filenames in os.walk(task_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, task_dir)
                        file_size = os.path.getsize(file_path)
                        file_type = 'unknown'
                        
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            file_type = 'image'
                        elif filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                            file_type = 'structure'
                        elif filename.lower().endswith(('.txt', '.log', '.out')):
                            file_type = 'text'
                        
                        # 对路径进行分段编码
                        path_segments = relative_path.split('/')
                        encoded_segments = [quote(segment, safe='') for segment in path_segments]
                        encoded_path = '/'.join(encoded_segments)
                        
                        files.append({
                            'filename': filename,
                            'path': relative_path,
                            'size': file_size,
                            'type': file_type,
                            'url': f'/api/files/{conversation_id}/{encoded_path}'
                        })
                
                return jsonify({'files': files})
                
            except Exception as e:
                return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

        @self.app.route('/api/task/<conversation_id>/stop', methods=['POST'])
        async def stop_task(conversation_id):
            """取消任务API"""
            try:
                # 捕获当前已知的fingerprint（若存在）
                known_fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                # 检查任务是否存在
                task = await self._get_task_by_id(conversation_id)
                if not task:
                    return jsonify({'error': '任务未找到', 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 404
                
                # 若fingerprint暂不可用且任务处于运行态，短暂等待映射建立以缓解竞态
                if not known_fingerprint and task['status'] == TaskStatus.RUNNING.value:
                    for _ in range(10):
                        await asyncio.sleep(0.05)
                        known_fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                        if known_fingerprint:
                            break
                
                # 检查任务状态
                if task['status'] not in [TaskStatus.RUNNING.value, TaskStatus.QUEUED.value]:
                    return jsonify({'error': f'任务状态为 {task["status"]}，无法取消', 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 400
                
                success = False
                message = ""
                
                # 如果任务在队列中，直接从队列移除
                if task['status'] == TaskStatus.QUEUED.value:
                    self.task_queue = [t for t in self.task_queue if t.conversation_id != conversation_id]
                    success = True
                    message = f"任务已从队列中移除 (conversation_id={conversation_id}, fingerprint={known_fingerprint})"
                    # 移除可能存在的映射
                    self._unregister_mapping_by_conversation(conversation_id)
                
                # 如果任务正在运行，取消运行中的任务
                elif conversation_id in self.running_tasks:
                    running_task = self.running_tasks[conversation_id]
                    running_task.cancel()
                    try:
                        await running_task
                    except asyncio.CancelledError:
                        pass
                    self.running_tasks.pop(conversation_id, None)
                    success = True
                    message = f"运行中的任务已取消 (conversation_id={conversation_id}, fingerprint={known_fingerprint})"
                    
                    # 提取并取消SLURM任务
                    calc_ids = await self._extract_calc_ids_from_logs(conversation_id)
                    if calc_ids:
                        try:
                            cancel_results = await self._cancel_slurm_job(calc_ids)
                            # 优先使用已捕获的fingerprint进行日志记录
                            if known_fingerprint:
                                self.system_log(f"SLURM任务取消结果: {cancel_results}", known_fingerprint)
                            else:
                                # 无映射则直接按对话ID记录
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{timestamp}] SLURM任务取消结果: {cancel_results}"
                                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
                        except Exception as e:
                            if known_fingerprint:
                                self.system_log(f"取消SLURM任务时出错: {str(e)}", known_fingerprint)
                            else:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{timestamp}] 取消SLURM任务时出错: {str(e)}"
                                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

                    # 取消成功或移出队列后，解除映射
                    self._unregister_mapping_by_conversation(conversation_id)
                
                if success:
                    # 更新数据库状态
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute(
                            'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                            ('cancelled', 'Task cancelled', conversation_id)
                        )
                        await db.commit()
                
                return jsonify({
                    'success': success,
                    'message': message,
                    'conversation_id': conversation_id,
                    'fingerprint': known_fingerprint
                })
                
            except Exception as e:
                error_msg = f"取消任务失败: {str(e)} (conversation_id={conversation_id}, fingerprint={known_fingerprint})"
                if known_fingerprint:
                    self.system_log(error_msg, known_fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {error_msg}"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
                return jsonify({'error': error_msg, 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 500

    async def _process_queue(self):
        """处理任务队列"""
        while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
            # 获取信号量
            if self.task_semaphore.locked():
                break
                
            queued_task = self.task_queue.pop(0)
            
            # 创建并启动异步任务
            async_task = asyncio.create_task(
                self._execute_crew_task_async(queued_task.conversation_id, queued_task.task_description)
            )
            self.running_tasks[queued_task.conversation_id] = async_task
            
            # 不等待任务完成，继续处理队列
            asyncio.create_task(self._monitor_task(queued_task.conversation_id, async_task))

    async def _monitor_task(self, conversation_id: str, task: asyncio.Task):
        """监控任务完成"""
        try:
            await task
        except asyncio.CancelledError:
            fingerprint = self._conversation_to_fingerprint.get(conversation_id)
            if fingerprint:
                self.system_log(f"任务已取消", fingerprint)
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] 任务 {conversation_id} 被取消"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
        except Exception as e:
            fingerprint = self._conversation_to_fingerprint.get(conversation_id)
            if fingerprint:
                self.system_log(f"任务执行出错: {str(e)}", fingerprint)
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] 任务 {conversation_id} 执行出错: {str(e)}"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
        finally:
            # 清理运行中的任务记录
            if conversation_id in self.running_tasks:
                del self.running_tasks[conversation_id]
            # 解除指纹映射
            self._unregister_mapping_by_conversation(conversation_id)
            
            # 继续处理队列
            await self._process_queue()

    async def _extract_calc_ids_from_logs(self, conversation_id):
        """从任务日志中提取计算任务ID"""
        calc_ids = []
        try:
            logs = await self._get_task_logs(conversation_id)
            
            for log in logs:
                content = log['content']
                
                # 从tool_output中查找calculation_id
                if log['type'] == 'tool_output':
                    try:
                        # 尝试解析JSON内容
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            tool_data = json.loads(json_match.group())
                            if isinstance(tool_data, dict):
                                # 查找calculation_id字段
                                if 'calculation_id' in tool_data:
                                    calc_ids.append(tool_data['calculation_id'])
                                # 也检查嵌套结构中的calculation_id
                                elif isinstance(tool_data, dict):
                                    for key, value in tool_data.items():
                                        if isinstance(value, dict) and 'calculation_id' in value:
                                            calc_ids.append(value['calculation_id'])
                    except (json.JSONDecodeError, AttributeError):
                        # 如果JSON解析失败，使用正则表达式查找
                        calc_id_patterns = [
                            r'"calculation_id":\s*"([^"]+)"',
                            r"'calculation_id':\s*'([^']+)'",
                            r'calculation_id.*?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
                        ]
                        for pattern in calc_id_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            calc_ids.extend(matches)
                
                # 从其他日志类型中查找UUID格式的计算ID
                uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                uuid_matches = re.findall(uuid_pattern, content, re.IGNORECASE)
                
                # 过滤掉对话ID本身，只保留计算ID
                for match in uuid_matches:
                    if match != conversation_id and match not in calc_ids:
                        calc_ids.append(match)
            
            # 去重并返回
            return list(set(calc_ids))
            
        except Exception as e:
            self.system_log(f"提取计算ID时出错: {str(e)}")
            return []

    def _run_crew_kickoff_thread(self, local_dir, task_description, result_container: Dict[str, Any], conversation_id: str) -> None:
        """在独立线程中执行 crew.kickoff 并记录线程ID与结果。"""
        self._crew_thread_ids[conversation_id] = threading.get_ident()
        try:
            self.system_log("Initializing crew...")
            crew = self.generator.crew(local_dir, conversation_id)
            # 注册映射关系（先注册再记录日志，避免未映射时fingerprint为None）
            self._register_mapping(conversation_id, crew.fingerprint.uuid_str)
            self.system_log("Registered mapping", crew.fingerprint.uuid_str)
            self.system_log("Creating user task...", crew.fingerprint.uuid_str)
            final_task_description = task_description
            # 读取会话历史对话作为上下文（JSON）并拼接到 task_description
            try:
                import sqlite3
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT role_name, content FROM activity_logs WHERE conversation_id = ? AND type = 'conversation_messages' ORDER BY id ASC",
                        (conversation_id,)
                    )
                    rows = cursor.fetchall()
                    if rows:
                        history_items = []
                        for row in rows:
                            role = row["role_name"] if ("role_name" in row.keys() and row["role_name"]) else "User"
                            content = row["content"] if "content" in row.keys() else ""
                            if content:
                                history_items.append({"role": role, "content": content})
                        if history_items:
                            context_json = json.dumps({"conversation_context": history_items}, ensure_ascii=False)
                            prefix = "以下是与该会话相关的历史对话上下文（JSON）：\n" + context_json + "\n\n请结合以上上下文，完成本次任务：\n"
                            final_task_description = prefix + task_description
            except Exception as e:
                self.system_log(f"读取会话上下文失败: {str(e)}", crew.fingerprint.uuid_str)

            # 创建任务
            task = Task(
                description=final_task_description,
                expected_output="A detailed report, including the execution process, calculation results, and the location of the drawn charts.",
                output_file=f'crew_output_{uuid.uuid4().hex[:8]}.md',
            )
            
            crew.tasks = [task]
            
            self.system_log("Starting task execution...", crew.fingerprint.uuid_str)
            self.save_conversation_messages(conversation_id, 'User', task_description)
            result_container['result'] = crew.kickoff()
            self.save_conversation_messages(conversation_id, 'VASPilot', str(result_container['result']))

            self.system_log("Task completed!", crew.fingerprint.uuid_str)
            self.agent_output("FinalResult", str(result_container['result']), crew.fingerprint.uuid_str)
        except BaseException as e:
            result_container['error'] = e
        finally:
            try:
                self._crew_thread_ids.pop(conversation_id, None)
            except Exception:
                pass

    def _inject_exception_into_thread(self, thread_id: int, exc_type=SystemExit) -> bool:
        """向目标线程异步注入异常以尝试强制结束。
        返回 True 表示已注入，False 表示失败或回滚。
        """
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(exc_type))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
                return False
            return res == 1
        except Exception:
            return False

    async def _stop_and_join_crew_thread(self, conversation_id: str, timeout: float = 5.0) -> bool:
        """尝试强制结束并等待指定会话对应的 crew 执行线程退出。"""
        thread = self._running_threads.get(conversation_id)
        thread_id = self._crew_thread_ids.get(conversation_id)
        stopped = False
        if thread_id is not None:
            stopped = self._inject_exception_into_thread(thread_id, SystemExit)
            print(f"injected, result={stopped}")
        print(f"result={stopped}")
        if thread is not None and thread.is_alive():
            loop = asyncio.get_event_loop()
            start = loop.time()
            while thread.is_alive() and (loop.time() - start) < timeout:
                await asyncio.sleep(0.05)
        self._running_threads.pop(conversation_id, None)
        self._crew_thread_ids.pop(conversation_id, None)
        return stopped

    async def _execute_crew_task_async(self, conversation_id, task_description):
        """异步执行crew任务"""
        async with self.task_semaphore:
            try:
                # 更新任务状态
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, started_at = CURRENT_TIMESTAMP WHERE conversation_id = ?',
                        (TaskStatus.RUNNING.value, conversation_id)
                    )
                    await conn.commit()
                # 推送状态变更
                await self._publish_sse_async(conversation_id, {
                    'event': 'status',
                    'status': TaskStatus.RUNNING.value,
                    'ts': datetime.now().isoformat()
                })

                # 系统日志（直接按对话ID记录，fingerprint尚未生成）
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] 对话id:{conversation_id}"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
                
                # 创建工作目录
                local_dir = os.path.join(self.work_dir, conversation_id)
                os.makedirs(local_dir, exist_ok=True)
                old_cwd = os.getcwd()
                os.chdir(local_dir)
                
                try:
                    # 初始化并建立映射
                    # 在线程中执行同步 kickoff，便于后续强制停止
                    result_container: Dict[str, Any] = {}
                    thread = threading.Thread(
                        target=self._run_crew_kickoff_thread,
                        args=(local_dir, task_description, result_container, conversation_id),
                        daemon=True,
                        name=f"crew-kickoff-{conversation_id[:8]}"
                    )
                    self._running_threads[conversation_id] = thread
                    thread.start()
                    # 异步轮询等待线程结束
                    while thread.is_alive():
                        await asyncio.sleep(0.1)
                    # 线程结束后获取结果或异常
                    if 'error' in result_container:
                        raise result_container['error']
                    result = result_container.get('result')
                    
                    # 更新任务状态
                    async with aiosqlite.connect(self.db_path) as conn:
                        await conn.execute(
                            'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, result = ? WHERE conversation_id = ?',
                            ('completed', str(result), conversation_id)
                        )
                        await conn.commit()
                    await self._publish_sse_async(conversation_id, {
                        'event': 'status',
                        'status': TaskStatus.COMPLETED.value,
                        'ts': datetime.now().isoformat()
                    })
                finally:
                    os.chdir(old_cwd)
                         
            except asyncio.CancelledError:
                # 任务被取消
                # 强制停止后台 crew 执行线程并等待退出
                try:
                    await self._stop_and_join_crew_thread(conversation_id)
                except Exception as e:
                    print(f"Failed to stop crew thread: {e}")
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                        ('cancelled', 'Task cancelled', conversation_id)
                    )
                    await conn.commit()
                await self._publish_sse_async(conversation_id, {
                    'event': 'status',
                    'status': TaskStatus.CANCELLED.value,
                    'ts': datetime.now().isoformat()
                })
                raise
            except Exception as e:
                error_msg = f"Error occurred during execution: {str(e)}"
                
                # 记录错误
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                        ('failed', error_msg, conversation_id)
                    )
                    await conn.commit()
                await self._publish_sse_async(conversation_id, {
                    'event': 'status',
                    'status': TaskStatus.FAILED.value,
                    'ts': datetime.now().isoformat()
                })
                
                # 根据映射记录错误日志
                fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                if fingerprint:
                    self.system_log(error_msg, fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {error_msg}"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
            finally:
                # 完成日志
                fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                try:
                    self.generator.stop()
                except Exception as e:
                    print(f"Failed to stop mcp client: {e}")
                if fingerprint:
                    self.system_log("Mission ended", fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] Mission ended"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

    def save_conversation_messages(self, conversation_id: str, role_name:str, message:str):
        """实现同步对话消息方法（内部异步写库）"""
        self._schedule_log_to_db(conversation_id, 'conversation_messages', message, role_name=role_name)

    # CrewServer接口实现（同步版本）
    def system_log(self, message: str, crew_fingerprint: str = None):
        """实现同步系统日志方法（内部异步写库）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 通过指纹映射到会话ID
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

    def agent_input(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """实现同步Agent输入方法（内部异步写库）"""
        log_content = f"[{agent_role}] {message}"
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'agent_input', log_content, role_name=agent_role)

    def agent_output(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """实现同步Agent输出方法（内部异步写库）"""
        log_content = f"[{agent_role}] {message}"
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'agent_output', log_content, role_name=agent_role)

    def tool_input(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """实现同步Tool输入方法（内部异步写库）"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'tool_input', log_content, role_name=tool_name)

    def tool_output(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """实现同步Tool输出方法（内部异步写库）"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'tool_output', log_content, role_name=tool_name)

    def _schedule_log_to_db(self, conversation_id, log_type, content, role_name=None):
        """在事件循环中的单一日志队列排队，保证入库顺序与调用顺序一致。"""
        self._enqueue_log_event(conversation_id, log_type, content, role_name)

    async def _log_worker(self):
        """单线程异步日志写库工作协程，严格按队列顺序写入。"""
        if self._log_queue is None:
            self._log_queue = asyncio.Queue()
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                while True:
                    item = await self._log_queue.get()
                    try:
                        conversation_id, log_type, content, role_name = item
                        await conn.execute(
                            'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                            (conversation_id, log_type, role_name, content)
                        )
                        await conn.commit()
                        # 广播到SSE订阅者
                        await self._publish_sse_async(conversation_id, {
                            'event': 'log',
                            'type': log_type,
                            'role_name': role_name,
                            'content': content,
                            'ts': datetime.now().isoformat()
                        })
                    finally:
                        self._log_queue.task_done()
        except asyncio.CancelledError:
            pass

    def _enqueue_log_event(self, conversation_id, log_type, content, role_name=None):
        """将日志事件投递到主事件循环的日志队列。
        - 在同一事件循环内，直接 put_nowait 以保持调用顺序；
        - 从其他线程调用时，使用 run_coroutine_threadsafe 投递，尽可能保持先后顺序；
        - 如队列未初始化，兜底为同步直写（初始化之前很少发生）。
        """
        if self._log_queue is not None and self._event_loop is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop is self._event_loop:
                    self._log_queue.put_nowait((conversation_id, log_type, content, role_name))
                else:
                    asyncio.run_coroutine_threadsafe(
                        self._log_queue.put((conversation_id, log_type, content, role_name)),
                        self._event_loop
                    )
            except RuntimeError:
                asyncio.run_coroutine_threadsafe(
                    self._log_queue.put((conversation_id, log_type, content, role_name)),
                    self._event_loop
                )
        else:
            # 兜底：队列未就绪时同步直写，避免日志丢失
            import sqlite3
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                        (conversation_id, log_type, role_name, content)
                    )
                    conn.commit()
                # 兜底时也尝试推送SSE（若有事件循环）
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._publish_sse_async(conversation_id, {
                            'event': 'log',
                            'type': log_type,
                            'role_name': role_name,
                            'content': content,
                            'ts': datetime.now().isoformat()
                        }),
                        self._event_loop
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"Direct log fallback failed: {e}")

    async def _publish_sse_async(self, conversation_id: str, payload: Dict[str, Any]):
        """将事件推送给指定会话的所有SSE订阅者。"""
        subs = self._sse_subscribers.get(conversation_id)
        if not subs:
            return
        for q in list(subs):
            try:
                await q.put(payload)
            except Exception:
                try:
                    subs.remove(q)
                except Exception:
                    pass

    def _register_mapping(self, conversation_id: str, crew_fingerprint: str) -> None:
        """注册 conversation_id 与 crew_fingerprint 映射。"""
        print(f"[Mapping] register {conversation_id} -> {crew_fingerprint}")
        with self._mapping_lock:
            self._conversation_to_fingerprint[conversation_id] = crew_fingerprint
            self._fingerprint_to_conversation[crew_fingerprint] = conversation_id
        print(f"[Mapping] size conv2fp={len(self._conversation_to_fingerprint)}, fp2conv={len(self._fingerprint_to_conversation)}")

    def _unregister_mapping_by_conversation(self, conversation_id: str) -> None:
        """根据 conversation_id 解除映射。"""
        with self._mapping_lock:
            print(f"[Mapping] unregister by conversation {conversation_id}")
            crew_fingerprint = self._conversation_to_fingerprint.pop(conversation_id, None)
            if crew_fingerprint:
                self._fingerprint_to_conversation.pop(crew_fingerprint, None)
        print(f"[Mapping] size conv2fp={len(self._conversation_to_fingerprint)}, fp2conv={len(self._fingerprint_to_conversation)}")

    def _get_conversation_id_for_fingerprint(self, crew_fingerprint: Optional[str]) -> Optional[str]:
        """通过 crew_fingerprint 查找 conversation_id。"""
        if not crew_fingerprint:
            return None
        with self._mapping_lock:
            return self._fingerprint_to_conversation.get(crew_fingerprint)

    async def _log_to_db_async(self, conversation_id, log_type, content, role_name=None):
        """异步将日志记录到数据库"""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                (conversation_id, log_type, role_name, content)
            )
            await conn.commit()

    async def _cancel_slurm_job(self, calc_ids: list[str]):
        """异步取消SLURM任务"""
        async with Client(self.config["mcp_server"]["url"]) as client:
            tool_result = await client.call_tool("cancel_slurm_job", {"calc_ids": calc_ids})
        if tool_result.data is None:
            return {"error": "No result from cancel_slurm_job"}
        else:
            return tool_result.data

    async def launch_async(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """异步启动Quart应用"""
        print(f"🚀 启动 {self.title}...")
        print(f"💼 工作目录: {self.work_dir}")
        print(f"🗄️ 数据库: {self.db_path}")
        print(f"🌐 服务器地址: http://{host}:{port}")
        print(f"⚡ 最大并发任务数: {self.max_concurrent_tasks}")
        print(f"📋 最大队列大小: {self.max_queue_size}")
        print("=" * 50)
        print("✨ Quart Async Crew AI 服务器")
        print("📝 并行任务、📋 队列管理、🔍 实时更新")
        print("=" * 50)
        
        # 初始化数据库
        await self._init_db()
        
        # 初始化日志队列与工作协程（在主事件循环中）
        self._event_loop = asyncio.get_running_loop()
        self._log_queue = asyncio.Queue()
        self._log_worker_task = asyncio.create_task(self._log_worker())
        
        # 设置会话上下文
        async def set_conversation_context(conversation_id):
            old_id = getattr(self, '_current_conversation_id', None)
            self._current_conversation_id = conversation_id
            return old_id
        
        # 修改执行任务方法以设置上下文
        original_execute = self._execute_crew_task_async
        async def execute_with_context(conversation_id, task_description):
            old_id = await set_conversation_context(conversation_id)
            try:
                await original_execute(conversation_id, task_description)
            finally:
                self._current_conversation_id = old_id
        
        self._execute_crew_task_async = execute_with_context
        
        try:
            await self.app.run_task(host=host, port=port, debug=debug, **kwargs)
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止。")

    def get_app(self):
        """获取Quart应用对象"""
        return self.app

    # 同步启动方法（兼容性）
    def launch(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """启动服务器（同步包装）"""
        asyncio.run(self.launch_async(host, port, debug, **kwargs)) 