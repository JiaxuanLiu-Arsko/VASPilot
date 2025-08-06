#!/usr/bin/env python3
"""
CrewAI VASP Flask服务器
功能：任务提交、历史记录、详情查看、实时更新
基于 CrewServer 基类实现，模板分离
"""

import os
import sys
import json
import uuid
import sqlite3
import threading
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, g

from markdown import markdown

# 添加项目路径到sys.path
current_dir = Path(__file__).parent  # flask_server/

# 导入项目模块
from ...listener.server_listener import CrewServer, ServerListener
from ...crew import VaspCrew
from crewai import Task


class FlaskCrewServer(CrewServer):
    """基于Flask的CrewServer实现"""
    
    def __init__(self, crew_config: Dict[str, Any], title: str = "VASPilot Server", 
                 work_dir: str = ".", db_path: Optional[str] = None, allow_path: Optional[str] = None):
        super().__init__()
        self.title = title
        self.config = crew_config
        self.work_dir = os.path.abspath(work_dir)
        self.running_tasks = {}
        self._current_conversation_id: Optional[str] = None
        self.allow_path = allow_path
        
        # 数据库路径
        if db_path is None:
            db_path = os.path.join(work_dir, 'crew_tasks.db')
        self.db_path = os.path.abspath(db_path)
        
        # 创建Flask应用
        template_folder = str(current_dir / "templates")
        self.app = Flask(__name__, template_folder=template_folder)
        self.app.secret_key = 'crew-ai-flask-server'
        
        self.generator = VaspCrew(self.config)
        
        self.current_logger = ServerListener(self, None)
        # 初始化数据库
        self._init_db()
        
        # 设置路由
        self._setup_routes()

    def _init_db(self):
        """初始化数据库"""
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"📁 创建数据库目录: {db_dir}")
            
            print(f"🗄️ 初始化数据库: {self.db_path}")
            
            with sqlite3.connect(self.db_path) as conn:
                # 创建 task_executions 表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT UNIQUE NOT NULL,
                        task_description TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        result TEXT,
                        error_message TEXT
                    )
                ''')
                
                # 创建 activity_logs 表
                conn.execute('''
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
                
                # 检查是否需要添加role_name列（向后兼容）
                cursor = conn.execute("PRAGMA table_info(activity_logs)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'role_name' not in columns:
                    print("🔄 添加role_name列到activity_logs表")
                    conn.execute('ALTER TABLE activity_logs ADD COLUMN role_name TEXT')
                
                conn.commit()
                
                # 验证表是否创建成功
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
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

    def _get_db(self):
        """获取数据库连接"""
        db = getattr(g, '_database', None)
        if db is None:
            try:
                db = g._database = sqlite3.connect(self.db_path)
                db.row_factory = sqlite3.Row
                
                # 验证表是否存在
                cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'")
                if not cursor.fetchone():
                    # 如果表不存在，重新初始化数据库
                    print("⚠️ 检测到表不存在，重新初始化数据库...")
                    db.close()
                    self._init_db()
                    db = g._database = sqlite3.connect(self.db_path)
                    db.row_factory = sqlite3.Row
                    
            except Exception as e:
                print(f"❌ 数据库连接失败: {str(e)}")
                raise
        return db

    def _close_connection(self, exception):
        """关闭数据库连接"""
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    def _get_recent_tasks(self, limit=10):
        """获取最近的任务"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM task_executions ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        return cursor.fetchall()

    def _get_task_by_id(self, conversation_id):
        """根据ID获取任务"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM task_executions WHERE conversation_id = ?',
            (conversation_id,)
        )
        return cursor.fetchone()

    def _get_task_logs(self, conversation_id):
        """获取任务日志"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM activity_logs WHERE conversation_id = ? ORDER BY timestamp',
            (conversation_id,)
        )
        logs = cursor.fetchall()
        
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
                'timestamp': log['timestamp'],
                'preview': log['content'][:30] + '...' if len(log['content']) > 30 else log['content']
            })
        
        return formatted_logs

    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.teardown_appcontext
        def close_connection(exception):
            self._close_connection(exception)
        
        @self.app.route('/')
        def index():
            """主页"""
            recent_tasks = self._get_recent_tasks()
            return render_template('base.html', 
                                 title=self.title,
                                 recent_tasks=recent_tasks)

        @self.app.route('/submit', methods=['POST'])
        def submit_task():
            """提交任务"""
            try:
                data = request.get_json()
                task_description = data.get('task_description', '').strip()
                
                if not task_description:
                    return jsonify({'error': '请输入有效的任务描述'}), 400
                
                # 检查是否有任务在运行
                db = self._get_db()
                cursor = db.execute("SELECT COUNT(*) as count FROM task_executions WHERE status = 'running'")
                running_count = cursor.fetchone()['count']
                
                if running_count > 0:
                    return jsonify({'error': '当前已有任务在执行中，请等待完成后再提交新任务'}), 400
                
                # 创建任务记录
                conversation_id = str(uuid.uuid4())
                db.execute(
                    'INSERT INTO task_executions (conversation_id, task_description) VALUES (?, ?)',
                    (conversation_id, task_description)
                )
                db.commit()
                
                # 启动后台任务
                thread = threading.Thread(
                    target=self._execute_crew_task,
                    args=(conversation_id, task_description),
                    daemon=True
                )
                thread.start()
                self.running_tasks[conversation_id] = thread
                
                return jsonify({
                    'success': True,
                    'conversation_id': conversation_id,
                    'message': '任务已提交，开始执行'
                })
                
            except Exception as e:
                return jsonify({'error': f'服务器错误: {str(e)}'}), 500

        @self.app.route('/task/<conversation_id>')
        def task_detail(conversation_id):
            """任务详情页面"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return "任务未找到", 404
            
            logs = self._get_task_logs(conversation_id)
            recent_tasks = self._get_recent_tasks()
            
            return render_template('task_detail.html',
                                 title=self.title,
                                 task=task,
                                 logs=logs,
                                 recent_tasks=recent_tasks)

        @self.app.route('/api/task/<conversation_id>/status')
        def get_task_status(conversation_id):
            """获取任务状态API"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': '任务未找到'}), 404
            
            return jsonify({
                'status': task['status'],
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description']
            })

        @self.app.route('/api/task/<conversation_id>/logs')
        def get_task_logs(conversation_id):
            """获取任务日志API"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': '任务未找到'}), 404
            
            logs = self._get_task_logs(conversation_id)
            
            # 将日志转换为字典格式
            logs_data = []
            for log in logs:
                logs_data.append({
                    'type': log['type'],
                    'type_name': log['type_name'],
                    'role_name': log['role_name'],  # 这里log已经是formatted_logs中的dict了，可以直接访问
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
                'logs': logs_data
            })

        @self.app.route('/api/tasks')
        def get_tasks():
            """获取任务列表API"""
            try:
                recent_tasks = self._get_recent_tasks()
                tasks_data = []
                for task in recent_tasks:
                    tasks_data.append({
                        'conversation_id': task['conversation_id'],
                        'task_description': task['task_description'],
                        'status': task['status'],
                        'created_at': task['created_at'],
                        'started_at': task['started_at'],
                        'completed_at': task['completed_at']
                    })
                return jsonify(tasks_data)
            except Exception as e:
                return jsonify({'error': f'获取任务列表失败: {str(e)}'}), 500

        @self.app.route('/api/files/<conversation_id>/<path:filename>')
        def serve_task_file(conversation_id, filename):
            """为特定任务提供文件访问"""
            import os
            from flask import send_file, abort
            from urllib.parse import unquote
            
            try:
                
                # 对路径进行分段解码：将路径分段，逐段解码，然后重新组合
                path_segments = filename.split('/')
                decoded_segments = [unquote(segment) for segment in path_segments]
                decoded_filename = '/'.join(decoded_segments)
                
                
                # 检查是否有绝对路径标记
                is_absolute_path = False
                if decoded_filename.startswith('__ABS__'):
                    # 移除标记，恢复绝对路径
                    decoded_filename = decoded_filename[7:]  # 移除 '__ABS__'
                    is_absolute_path = True
                
                # 构建文件路径
                task_dir = os.path.join(self.work_dir, conversation_id)
                
                # 如果是绝对路径，直接使用绝对路径
                if is_absolute_path or (decoded_filename.startswith('/') and self.allow_path):
                    file_path = decoded_filename
                else:
                    file_path = os.path.join(task_dir, decoded_filename)
                
                # 安全检查：确保文件在任务目录内
                file_path = os.path.abspath(file_path)
                task_dir = os.path.abspath(task_dir)
                
                # 安全检查：对于绝对路径，如果没有明确禁止，则允许访问
                if not is_absolute_path and not self.allow_path:
                    if not file_path.startswith(task_dir) and not file_path.startswith(self.work_dir):
                        abort(403, description="访问被拒绝：文件路径不在允许范围内")
                elif is_absolute_path:
                    print(f"[DEBUG] 绝对路径访问被允许")
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    abort(404, description=f"文件未找到: {decoded_filename}")
                
                # 根据文件扩展名设置MIME类型
                if decoded_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    mimetype = 'image/png' if decoded_filename.lower().endswith('.png') else 'image/jpeg'
                elif decoded_filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                    mimetype = 'text/plain'
                else:
                    mimetype = 'application/octet-stream'
                
                
                return send_file(file_path, mimetype=mimetype)
                
            except Exception as e:
                import traceback
                abort(500, description=f"文件服务错误: {str(e)}")

        @self.app.route('/api/files/<conversation_id>/list')
        def list_task_files(conversation_id):
            """列出任务目录中的所有文件"""
            import os
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
                        
                        # 对路径进行分段编码：将路径分段，逐段编码，然后重新组合
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
                return jsonify({'error': f'列出文件失败: {str(e)}'}), 500

    def _execute_crew_task(self, conversation_id, task_description):
        """执行crew任务"""
        try:
            # 更新任务状态
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, started_at = CURRENT_TIMESTAMP WHERE conversation_id = ?',
                    ('running', conversation_id)
                )
                conn.commit()

            # 系统日志
            self.system_log(f"对话id:{conversation_id}")
            
            # 创建工作目录
            local_dir = os.path.join(self.work_dir, conversation_id)
            os.makedirs(local_dir, exist_ok=True)
            os.chdir(local_dir)
            
            self.system_log("初始化crew...")
            crew = self.generator.crew()
            self.system_log("设置监听器...")
            self.current_logger.crew_fingerprint = crew.fingerprint.uuid_str
            self.system_log("创建用户任务...")
            
            # 创建任务
            task = Task(
                description=task_description,
                expected_output="一份详尽的报告，报告内容包括任务执行过程、计算结果、画出的图表位置等。",
                output_file=f'crew_output_{uuid.uuid4().hex[:8]}.md',
            )
            
            crew.tasks = [task]
            
            self.system_log("开始执行任务...")
            
            # 执行crew
            result = crew.kickoff()
            
            self.system_log("任务完成!")
            self.agent_output("FinalResult", str(result))
            
            # 更新任务状态
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, result = ? WHERE conversation_id = ?',
                    ('completed', str(result), conversation_id)
                )
                conn.commit()
                
                    
        except Exception as e:
            error_msg = f"执行过程中出现错误: {str(e)}"
            
            # 记录错误
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                    ('failed', error_msg, conversation_id)
                )
                conn.commit()
            
            self.system_log(error_msg)
        finally:
            # 清理运行中的任务记录
            if conversation_id in self.running_tasks:
                del self.running_tasks[conversation_id]
            self.system_log("任务执行完成！")

    # CrewServer接口实现
    def system_log(self, message: str):
        """实现系统日志方法"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 获取当前会话ID（如果在任务执行中）
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'system', log_entry, role_name='system')

    def agent_input(self, agent_role: str, message: str):
        """实现Agent输入方法"""
        log_content = f"[{agent_role}] {message}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'agent_input', log_content, role_name=agent_role)

    def agent_output(self, agent_role: str, message: str):
        """实现Agent输出方法"""
        log_content = f"[{agent_role}] {message}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'agent_output', log_content, role_name=agent_role)

    def tool_input(self, tool_name: str, message: Dict[str, Any]):
        """实现Tool输入方法"""
        log_content = f"[{tool_name}] {json.dumps(message, ensure_ascii=False, indent=2)}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'tool_input', log_content, role_name=tool_name)

    def tool_output(self, tool_name: str, message: Dict[str, Any]):
        """实现Tool输出方法"""
        log_content = f"[{tool_name}] {json.dumps(message, ensure_ascii=False, indent=2)}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'tool_output', log_content, role_name=tool_name)

    def _log_to_db(self, conversation_id, log_type, content, role_name=None):
        """将日志记录到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                (conversation_id, log_type, role_name, content)
            )
            conn.commit()

    def launch(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """启动Flask应用"""
        print(f"🚀 启动 {self.title}...")
        print(f"💼 工作目录: {self.work_dir}")
        print(f"🗄️ 数据库: {self.db_path}")
        print(f"🌐 服务器地址: http://{host}:{port}")
        print("=" * 50)
        print("✨ Flask Crew AI 服务器")
        print("📝 提交任务、📋 查看历史、🔍 实时更新")
        print("=" * 50)
        
        # 在任务执行期间设置会话ID的上下文
        def set_conversation_context(conversation_id):
            def wrapper(func):
                def inner(*args, **kwargs):
                    old_id = getattr(self, '_current_conversation_id', None)
                    self._current_conversation_id = conversation_id
                    try:
                        return func(*args, **kwargs)
                    finally:
                        self._current_conversation_id = old_id
                return inner
            return wrapper
        
        # 修改执行任务方法以设置上下文
        original_execute = self._execute_crew_task
        def execute_with_context(conversation_id, task_description):
            self._current_conversation_id = conversation_id
            try:
                original_execute(conversation_id, task_description)
            finally:
                self._current_conversation_id = None
        
        self._execute_crew_task = execute_with_context
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True, **kwargs)
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止。")

    def get_app(self):
        """获取Flask应用对象"""
        return self.app
