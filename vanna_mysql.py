"""
Vanna MySQL 本地部署应用
基于 Ollama 的本地大语言模型，支持自然语言到SQL的转换
"""

import json
import sqlite3
import hashlib
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

import requests
import mysql.connector
from mysql.connector import Error
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, host: str = "http://127.0.0.1", port: int = 11434, model: str = "deepseek-r1:8b"):
        self.base_url = f"{host}:{port}"
        self.model = model
        
        # 创建一个持久的 session 对象
        self.session = requests.Session()
        
        # 设置代理为空，绕过系统代理设置
        self.session.proxies = {
            'http': None,
            'https': None
        }
        
        # 设置请求头，模拟 Postman 的行为
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        # 禁用 SSL 验证（如果需要）和设置连接池
        self.session.verify = False
        
        # 配置适配器以提高连接稳定性
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def test_connection(self) -> bool:
        """测试与 Ollama 服务的连接"""
        try:
            logger.info(f"测试连接到 Ollama 服务: {self.base_url}")
            
            # 首先尝试获取模型列表
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                logger.info(f"✅ 连接成功！可用模型: {model_names}")
                
                # 检查指定模型是否存在
                if any(self.model in name for name in model_names):
                    logger.info(f"✅ 找到指定模型: {self.model}")
                    return True
                else:
                    logger.warning(f"⚠️ 未找到指定模型 {self.model}，可用模型: {model_names}")
                    return False
            else:
                logger.error(f"❌ 连接失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 连接测试失败: {e}")
            return False
            
    def generate(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        """生成文本响应"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        # 如果有系统提示，添加到 payload 中
        if system:
            payload["system"] = system
        
        try:
            logger.debug(f"发送请求到: {url}")
            logger.debug(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # 使用 session 发送请求
            response = self.session.post(url, json=payload, timeout=120)
            
            logger.debug(f"响应状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                logger.info(f"✅ 成功生成文本，长度: {len(generated_text)}")
                return generated_text
            else:
                logger.error(f"❌ API 返回错误: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("❌ 请求超时")
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ 连接错误: {e}")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 请求异常: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解析错误: {e}")
            logger.error(f"响应内容: {response.text if 'response' in locals() else 'No response'}")
            return ""
        except Exception as e:
            logger.error(f"❌ 未知错误: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """聊天对话"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            logger.debug(f"发送聊天请求到: {url}")
            logger.debug(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = self.session.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                message_content = result.get("message", {}).get("content", "")
                logger.info(f"✅ 成功获取聊天响应，长度: {len(message_content)}")
                return message_content
            else:
                logger.error(f"❌ Chat API 返回错误: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Chat API 调用失败: {e}")
            return ""
    
    def close(self):
        """关闭 session"""
        if hasattr(self, 'session'):
            self.session.close()


class KnowledgeBase:
    """知识库管理类，使用SQLite存储训练数据和向量嵌入"""
    
    def __init__(self, db_path: str = "vanna_knowledge.db"):
        self.db_path = db_path
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conn.commit()
        conn.close()
    
    def _encode_embedding(self, embedding: np.ndarray) -> bytes:
        """将嵌入向量编码为二进制数据"""
        return embedding.tobytes()
    
    def _decode_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """将二进制数据解码为嵌入向量"""
        return np.frombuffer(embedding_bytes, dtype=np.float32)
    
    def add_ddl(self, ddl: str) -> bool:
        """添加DDL语句"""
        try:
            # 提取表名
            table_name = self._extract_table_name(ddl)
            
            # 生成嵌入
            embedding = self.sentence_model.encode(ddl)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ddl_statements (content, embedding, table_name)
                VALUES (?, ?, ?)
            ''', (ddl, self._encode_embedding(embedding), table_name))
            
            conn.commit()
            conn.close()
            
            logger.info(f"添加DDL成功: {table_name}")
            return True
        except Exception as e:
            logger.error(f"添加DDL失败: {e}")
            return False
    
    def add_documentation(self, doc: str, category: str = "general") -> bool:
        """添加文档"""
        try:
            embedding = self.sentence_model.encode(doc)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO documentation (content, embedding, category)
                VALUES (?, ?, ?)
            ''', (doc, self._encode_embedding(embedding), category))
            
            conn.commit()
            conn.close()
            
            logger.info(f"添加文档成功: {category}")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def add_question_sql(self, question: str, sql: str) -> bool:
        """添加问答对"""
        try:
            question_embedding = self.sentence_model.encode(question)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO question_sql_pairs (question, sql, question_embedding)
                VALUES (?, ?, ?)
            ''', (question, sql, self._encode_embedding(question_embedding)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"添加问答对成功: {question[:50]}...")
            return True
        except Exception as e:
            logger.error(f"添加问答对失败: {e}")
            return False
    
    def get_similar_ddl(self, text: str, limit: int = 3) -> List[str]:
        """获取相似的DDL语句"""
        try:
            query_embedding = self.sentence_model.encode(text)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT content, embedding FROM ddl_statements')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            similarities = []
            for content, embedding_bytes in results:
                stored_embedding = self._decode_embedding(embedding_bytes)
                similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                similarities.append((content, similarity))
            
            # 按相似度排序并返回前N个
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [content for content, _ in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"获取相似DDL失败: {e}")
            return []
    
    def get_similar_documentation(self, text: str, limit: int = 3) -> List[str]:
        """获取相似的文档"""
        try:
            query_embedding = self.sentence_model.encode(text)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT content, embedding FROM documentation')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            similarities = []
            for content, embedding_bytes in results:
                stored_embedding = self._decode_embedding(embedding_bytes)
                similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                similarities.append((content, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [content for content, _ in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"获取相似文档失败: {e}")
            return []
    
    def get_similar_questions(self, question: str, limit: int = 3) -> List[Tuple[str, str]]:
        """获取相似的问答对"""
        try:
            query_embedding = self.sentence_model.encode(question)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT question, sql, question_embedding FROM question_sql_pairs')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return []
            
            similarities = []
            for q, s, embedding_bytes in results:
                stored_embedding = self._decode_embedding(embedding_bytes)
                similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                similarities.append((q, s, similarity))
            
            similarities.sort(key=lambda x: x[2], reverse=True)
            return [(q, s) for q, s, _ in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"获取相似问答对失败: {e}")
            return []
    
    def _extract_table_name(self, ddl: str) -> str:
        """从DDL语句中提取表名"""
        match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', ddl, re.IGNORECASE)
        return match.group(1) if match else "unknown"


class VannaMySQLConnector:
    """MySQL数据库连接器"""
    
    def __init__(self, **kwargs):
        self.config = {
            'host': kwargs.get('host', '127.0.0.1'),
            'user': kwargs.get('user', 'root'),
            'password': kwargs.get('password', 'password'),
            'database': kwargs.get('database', 'test'),
            'charset': 'utf8mb4',
            'autocommit': True
        }
        self.connection = None
        
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info(f"成功连接到MySQL数据库: {self.config['database']}")
            return True
        except Error as e:
            logger.error(f"MySQL连接失败: {e}")
            return False
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """执行SQL查询"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                raise Exception("无法连接到数据库")
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                return pd.DataFrame(results)
            else:
                self.connection.commit()
                return pd.DataFrame({"affected_rows": [cursor.rowcount]})
                
        except Error as e:
            logger.error(f"SQL执行失败: {e}")
            raise
        finally:
            cursor.close()
    
    def get_table_info(self) -> Dict[str, Any]:
        """获取数据库表信息"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            cursor = self.connection.cursor()
            
            # 获取所有表名
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            table_info = {}
            for table in tables:
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                table_info[table] = {
                    'columns': [{'name': col[0], 'type': col[1], 'null': col[2], 'key': col[3]} for col in columns]
                }
            
            cursor.close()
            return table_info
            
        except Error as e:
            logger.error(f"获取表信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL连接已关闭")


class VannaMySQL:
    """Vanna MySQL 主类"""
    
    def __init__(self, ollama_config: Dict[str, Any] = None, mysql_config: Dict[str, Any] = None):
        # 初始化 Ollama 客户端
        if ollama_config is None:
            ollama_config = {
                "host": "http://127.0.0.1",
                "port": 11434,
                "model": "deepseek-r1:8b"
            }
        
        self.ollama = OllamaClient(**ollama_config)
        
        # 初始化知识库
        self.knowledge_base = KnowledgeBase()
        
        # 初始化MySQL连接器
        self.mysql_connector = None
        if mysql_config:
            self.mysql_connector = VannaMySQLConnector(**mysql_config)
            self.mysql_connector.connect()
    
    def connect_to_mysql(self, **kwargs):
        """连接到MySQL数据库"""
        self.mysql_connector = VannaMySQLConnector(**kwargs)
        return self.mysql_connector.connect()
    
    def train(self, ddl: str = None, documentation: str = None, question: str = None, sql: str = None):
        """训练模型"""
        if ddl:
            self.knowledge_base.add_ddl(ddl)
        
        if documentation:
            self.knowledge_base.add_documentation(documentation)
        
        if question and sql:
            self.knowledge_base.add_question_sql(question, sql)
    
    def generate_sql(self, question: str) -> str:
        """根据自然语言问题生成SQL"""
        try:
            # 获取相关的DDL和文档
            related_ddl = self.knowledge_base.get_similar_ddl(question)
            related_docs = self.knowledge_base.get_similar_documentation(question)
            similar_questions = self.knowledge_base.get_similar_questions(question)
            
            # 获取表信息
            table_info = {}
            if self.mysql_connector:
                table_info = self.mysql_connector.get_table_info()
            
            # 构建上下文
            context = self._build_context(related_ddl, related_docs, similar_questions, table_info)
            
            # 构建系统提示
            system_prompt = self._build_system_prompt()
            
            # 构建用户提示
            user_prompt = f"""
上下文信息：
{context}

用户问题：{question}

请根据上述信息生成准确的SQL查询。只返回SQL语句，不要包含任何解释或其他文本。
"""
            
            # 调用LLM生成SQL
            sql = self.ollama.generate(user_prompt, system_prompt)
            
            # 清理SQL
            sql = self._clean_sql(sql)
            
            logger.info(f"生成SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"生成SQL失败: {e}")
            return ""
    
    def ask(self, question: str) -> Dict[str, Any]:
        """询问问题并返回结果"""
        try:
            # 生成SQL
            sql = self.generate_sql(question)
            
            if not sql:
                return {
                    "question": question,
                    "sql": "",
                    "error": "无法生成SQL查询",
                    "data": None
                }
            
            # 执行SQL
            if self.mysql_connector:
                df = self.mysql_connector.execute_query(sql)
                
                return {
                    "question": question,
                    "sql": sql,
                    "data": df.to_dict('records') if not df.empty else [],
                    "columns": df.columns.tolist() if not df.empty else [],
                    "row_count": len(df)
                }
            else:
                return {
                    "question": question,
                    "sql": sql,
                    "error": "未连接到数据库",
                    "data": None
                }
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {
                "question": question,
                "sql": sql if 'sql' in locals() else "",
                "error": str(e),
                "data": None
            }
    
    def run_sql(self, sql: str) -> pd.DataFrame:
        """执行SQL查询"""
        if not self.mysql_connector:
            raise Exception("未连接到数据库")
        
        return self.mysql_connector.execute_query(sql)
    
    def _build_context(self, ddl_list: List[str], docs_list: List[str], 
                      questions_list: List[Tuple[str, str]], table_info: Dict[str, Any]) -> str:
        """构建上下文信息"""
        context_parts = []
        
        # DDL信息
        if ddl_list:
            context_parts.append("相关表结构：")
            for ddl in ddl_list:
                context_parts.append(ddl)
        
        # 表信息
        if table_info:
            context_parts.append("\n数据库表信息：")
            for table_name, info in table_info.items():
                context_parts.append(f"表名: {table_name}")
                for col in info['columns']:
                    context_parts.append(f"  - {col['name']} ({col['type']}) {'NOT NULL' if col['null'] == 'NO' else 'NULL'}")
        
        # 文档信息
        if docs_list:
            context_parts.append("\n相关文档：")
            for doc in docs_list:
                context_parts.append(f"- {doc}")
        
        # 相似问题
        if questions_list:
            context_parts.append("\n相似问题示例：")
            for q, s in questions_list:
                context_parts.append(f"问题: {q}")
                context_parts.append(f"SQL: {s}")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个专业的SQL生成助手。你的任务是根据用户的自然语言问题生成准确的SQL查询。

规则：
1. 只生成SQL查询语句，不要包含任何解释
2. 使用标准的MySQL语法
3. 确保SQL语法正确
4. 根据提供的表结构和上下文信息生成查询
5. 如果问题不明确，尽量生成最可能的查询
6. 使用适当的JOIN、WHERE、GROUP BY、ORDER BY等子句
7. 对于聚合查询，使用适当的聚合函数
8. 确保列名和表名正确

请根据上下文信息生成准确的SQL查询。"""
    
    def _clean_sql(self, sql: str) -> str:
        """清理SQL语句"""
        # 移除多余的空白和换行
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # 移除可能的markdown标记
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # 移除分号（如果在末尾）
        sql = sql.rstrip(';').strip()
        
        return sql


# Flask Web 应用
def create_flask_app(vanna_instance: VannaMySQL) -> Flask:
    """创建Flask Web应用"""
    app = Flask(__name__)
    
    # HTML模板
    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vanna MySQL - 自然语言转SQL</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        textarea {
            height: 80px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .sql-code {
            background-color: #f1f3f4;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            border-color: #f5c6cb;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .training-section {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #e9ecef;
            border: none;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Vanna MySQL</h1>
        <p>自然语言转SQL查询工具</p>
    </div>
    
    <div class="container">
        <div class="form-group">
            <label for="question">输入您的问题：</label>
            <input type="text" id="question" placeholder="例如：查询所有客户的数量" />
        </div>
        
        <button onclick="askQuestion()">🚀 生成SQL并查询</button>
        <button onclick="clearResults()">🗑️ 清空结果</button>
        
        <div id="result"></div>
        
        <div class="training-section">
            <h2>📚 模型训练</h2>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('ddl')">添加表结构</button>
                <button class="tab" onclick="showTab('doc')">添加文档</button>
                <button class="tab" onclick="showTab('qa')">添加问答对</button>
            </div>
            
            <div id="ddl-tab" class="tab-content active">
                <div class="form-group">
                    <label for="ddl">DDL 语句：</label>
                    <textarea id="ddl" placeholder="CREATE TABLE ..."></textarea>
                </div>
                <button onclick="trainDDL()">添加表结构</button>
            </div>
            
            <div id="doc-tab" class="tab-content">
                <div class="form-group">
                    <label for="documentation">文档内容：</label>
                    <textarea id="documentation" placeholder="描述表的用途、业务逻辑等"></textarea>
                </div>
                <button onclick="trainDoc()">添加文档</button>
            </div>
            
            <div id="qa-tab" class="tab-content">
                <div class="form-group">
                    <label for="train-question">问题：</label>
                    <input type="text" id="train-question" placeholder="例如：有多少个客户？" />
                </div>
                <div class="form-group">
                    <label for="train-sql">对应的SQL：</label>
                    <textarea id="train-sql" placeholder="SELECT COUNT(*) FROM customers"></textarea>
                </div>
                <button onclick="trainQA()">添加问答对</button>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // 隐藏所有标签内容
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            
            // 显示选中的标签
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question.trim()) {
                alert('请输入问题');
                return;
            }
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<p>🔄 正在处理您的问题...</p>';
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                resultDiv.innerHTML = '<div class="error">❌ 请求失败: ' + error + '</div>';
            });
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            let html = '<div class="result">';
            
            html += '<h3>📋 查询结果</h3>';
            html += '<p><strong>问题:</strong> ' + data.question + '</p>';
            
            if (data.sql) {
                html += '<p><strong>生成的SQL:</strong></p>';
                html += '<div class="sql-code">' + data.sql + '</div>';
            }
            
            if (data.error) {
                html += '<div class="error">❌ 错误: ' + data.error + '</div>';
            }
            
            if (data.data && data.data.length > 0) {
                html += '<p><strong>查询结果:</strong> (' + data.row_count + ' 行)</p>';
                html += '<table>';
                
                // 表头
                html += '<tr>';
                data.columns.forEach(col => {
                    html += '<th>' + col + '</th>';
                });
                html += '</tr>';
                
                // 数据行 (最多显示100行)
                const maxRows = Math.min(data.data.length, 100);
                for (let i = 0; i < maxRows; i++) {
                    html += '<tr>';
                    data.columns.forEach(col => {
                        const value = data.data[i][col];
                        html += '<td>' + (value !== null ? value : 'NULL') + '</td>';
                    });
                    html += '</tr>';
                }
                
                if (data.data.length > 100) {
                    html += '<tr><td colspan="' + data.columns.length + '">... 仅显示前100行</td></tr>';
                }
                
                html += '</table>';
            } else if (!data.error) {
                html += '<p>✅ 查询执行成功，无数据返回。</p>';
            }
            
            html += '</div>';
            resultDiv.innerHTML = html;
        }
        
        function trainDDL() {
            const ddl = document.getElementById('ddl').value;
            if (!ddl.trim()) {
                alert('请输入DDL语句');
                return;
            }
            
            train({ddl: ddl});
        }
        
        function trainDoc() {
            const doc = document.getElementById('documentation').value;
            if (!doc.trim()) {
                alert('请输入文档内容');
                return;
            }
            
            train({documentation: doc});
        }
        
        function trainQA() {
            const question = document.getElementById('train-question').value;
            const sql = document.getElementById('train-sql').value;
            
            if (!question.trim() || !sql.trim()) {
                alert('请输入问题和对应的SQL');
                return;
            }
            
            train({question: question, sql: sql});
        }
        
        function train(data) {
            fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    alert('✅ 训练数据添加成功！');
                    // 清空表单
                    if (data.ddl) document.getElementById('ddl').value = '';
                    if (data.documentation) document.getElementById('documentation').value = '';
                    if (data.question) {
                        document.getElementById('train-question').value = '';
                        document.getElementById('train-sql').value = '';
                    }
                } else {
                    alert('❌ 添加失败: ' + result.error);
                }
            })
            .catch(error => {
                alert('❌ 请求失败: ' + error);
            });
        }
        
        function clearResults() {
            document.getElementById('result').innerHTML = '';
            document.getElementById('question').value = '';
        }
        
        // 回车键提交
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
    '''
    
    @app.route('/')
    def index():
        return HTML_TEMPLATE
    
    @app.route('/ask', methods=['POST'])
    def ask():
        try:
            data = request.json
            question = data.get('question', '')
            
            if not question:
                return jsonify({'error': '问题不能为空'}), 400
            
            result = vanna_instance.ask(question)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"处理问题失败: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/train', methods=['POST'])
    def train():
        try:
            data = request.json
            
            ddl = data.get('ddl')
            documentation = data.get('documentation')
            question = data.get('question')
            sql = data.get('sql')
            
            vanna_instance.train(
                ddl=ddl,
                documentation=documentation,
                question=question,
                sql=sql
            )
            
            return jsonify({'success': True})
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'})
    
    return app


def main():
    """主函数"""
    print("🚀 正在启动 Vanna MySQL 应用...")
    
    # 创建Vanna实例
    print("🔧 正在初始化 Vanna 实例...")
    vn = VannaMySQL(
        ollama_config={
            "host": "http://127.0.0.1",
            "port": 11434,
            "model": "deepseek-r1:8b"
        }
    )
    print("✅ Vanna 实例初始化成功！")
    
    # 测试 Ollama 连接
    print("🔗 正在测试 Ollama 连接...")
    if vn.ollama.test_connection():
        print("✅ Ollama 连接测试成功！")
    else:
        print("❌ Ollama 连接测试失败，请检查服务是否运行")
        print("请确保:")
        print("  1. Ollama 服务正在运行")
        print("  2. 端口 11434 未被占用")
        print("  3. 模型 deepseek-r1:8b 已下载")
        return
    
    # 连接到MySQL数据库
    print("🔌 正在连接到 MySQL 数据库...")
    success = vn.connect_to_mysql(
        host='127.0.0.1',
        user='root',
        password='dcsz2024',
        database='dbgame_hdxyx_szdevgj_0001'
    )
    
    if success:
        print("✅ MySQL 数据库连接成功！")
    else:
        print("⚠️  MySQL 数据库连接失败，将在无数据库模式下运行")
    
    # 预训练一些基础数据
    print("📚 正在添加基础训练数据...")
    
    # 添加一些基础DDL示例
    vn.train(ddl="""
    CREATE TABLE customers (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE,
        phone VARCHAR(20),
        signup_date DATE DEFAULT CURRENT_DATE,
        status ENUM('active', 'inactive') DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    vn.train(ddl="""
    CREATE TABLE orders (
        id INT PRIMARY KEY AUTO_INCREMENT,
        customer_id INT,
        order_date DATE,
        total_amount DECIMAL(10,2),
        status VARCHAR(20),
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """)
    
    # 添加文档
    vn.train(documentation="customers表存储客户的基本信息，包括姓名、邮箱、电话、注册日期和状态")
    vn.train(documentation="orders表存储订单信息，通过customer_id与客户表关联")
    vn.train(documentation="客户状态包括active(活跃)和inactive(非活跃)两种")
    
    # 添加问答示例
    vn.train(question="有多少个客户？", sql="SELECT COUNT(*) as customer_count FROM customers")
    vn.train(question="列出所有活跃客户", sql="SELECT * FROM customers WHERE status = 'active'")
    vn.train(question="查询客户的订单数量", sql="SELECT c.name, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name")
    vn.train(question="今天的订单总金额", sql="SELECT SUM(total_amount) as daily_total FROM orders WHERE order_date = CURDATE()")
    
    print("✅ 基础训练数据添加完成！")
    
    # 创建并启动Flask应用
    print("🌐 正在启动 Web 服务...")
    app = create_flask_app(vn)
    
    print("🎉 Vanna MySQL 应用启动成功！")
    print("📱 Web界面地址: http://127.0.0.1:5000")
    print("🔧 健康检查: http://127.0.0.1:5000/health")
    print("📖 使用说明:")
    print("   1. 在Web界面输入自然语言问题")
    print("   2. 系统会自动生成SQL并执行查询")
    print("   3. 可以通过训练模块添加更多训练数据")
    print("   4. 支持DDL、文档和问答对训练")
    print("\n按 Ctrl+C 停止服务")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 正在关闭服务...")
        if vn.mysql_connector:
            vn.mysql_connector.close()
        if hasattr(vn.ollama, 'close'):
            vn.ollama.close()
        print("✅ 服务已关闭")


if __name__ == "__main__":
    main()
        