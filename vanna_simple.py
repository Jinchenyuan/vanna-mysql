#!/usr/bin/env python3
"""
简化版 Vanna MySQL 应用 - 不依赖 sentence_transformers
用于快速测试核心功能
"""

import json
import sqlite3
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
import mysql.connector
from mysql.connector import Error
import pandas as pd
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, host: str = "http://127.0.0.1", port: int = 11434, model: str = "deepseek-r1:8b"):
        self.base_url = f"{host}:{port}"
        self.model = model
        
        # 创建 session 并绕过代理
        self.session = requests.Session()
        self.session.proxies = {'http': None, 'https': None}
        
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
            
    def generate(self, prompt: str, system: str = None) -> str:
        """生成文本"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "system": system,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"生成失败: {e}")
        return ""


class SimpleKnowledgeBase:
    """简化的知识库 - 不使用向量嵌入"""
    
    def __init__(self, db_path: str = "simple_vanna.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ddl_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                table_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documentation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_sql_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_ddl(self, ddl: str) -> bool:
        """添加DDL"""
        try:
            table_name = self._extract_table_name(ddl)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ddl_statements (content, table_name)
                VALUES (?, ?)
            ''', (ddl, table_name))
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documentation (content, category)
                VALUES (?, ?)
            ''', (doc, category))
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO question_sql_pairs (question, sql)
                VALUES (?, ?)
            ''', (question, sql))
            conn.commit()
            conn.close()
            logger.info(f"添加问答对成功: {question[:50]}...")
            return True
        except Exception as e:
            logger.error(f"添加问答对失败: {e}")
            return False
    
    def get_all_ddl(self) -> List[str]:
        """获取所有DDL"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT content FROM ddl_statements ORDER BY created_at DESC')
            results = [row[0] for row in cursor.fetchall()]
            conn.close()
            return results
        except:
            return []
    
    def get_all_documentation(self) -> List[str]:
        """获取所有文档"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT content FROM documentation ORDER BY created_at DESC')
            results = [row[0] for row in cursor.fetchall()]
            conn.close()
            return results
        except:
            return []
    
    def get_all_questions(self) -> List[Tuple[str, str]]:
        """获取所有问答对"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT question, sql FROM question_sql_pairs ORDER BY created_at DESC')
            results = cursor.fetchall()
            conn.close()
            return results
        except:
            return []
    
    def _extract_table_name(self, ddl: str) -> str:
        """提取表名"""
        match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', ddl, re.IGNORECASE)
        return match.group(1) if match else "unknown"


class VannaMySQLConnector:
    """MySQL连接器"""
    
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
        """连接数据库"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info(f"连接成功: {self.config['database']}")
            return True
        except Error as e:
            logger.error(f"连接失败: {e}")
            return False
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """执行查询"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                raise Exception("无法连接数据库")
        
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
    
    def get_all_tables_ddl(self) -> Dict[str, str]:
        """获取所有表DDL"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            ddl_dict = {}
            for table in tables:
                cursor.execute(f"SHOW CREATE TABLE {table}")
                result = cursor.fetchone()
                if result:
                    ddl_dict[table] = result[1]
            
            cursor.close()
            logger.info(f"获取到 {len(ddl_dict)} 个表的DDL")
            return ddl_dict
        except Error as e:
            logger.error(f"获取DDL失败: {e}")
            return {}


class SimpleVannaMySQL:
    """简化版 Vanna MySQL"""
    
    def __init__(self, ollama_config: Dict[str, Any] = None, mysql_config: Dict[str, Any] = None):
        self.ollama = OllamaClient(**(ollama_config or {}))
        self.knowledge_base = SimpleKnowledgeBase()
        self.mysql_connector = None
        
        if mysql_config:
            self.mysql_connector = VannaMySQLConnector(**mysql_config)
            if self.mysql_connector.connect():
                self.sync_database_schema()
    
    def connect_to_mysql(self, **kwargs):
        """连接MySQL"""
        self.mysql_connector = VannaMySQLConnector(**kwargs)
        success = self.mysql_connector.connect()
        if success:
            self.sync_database_schema()
        return success
    
    def sync_database_schema(self):
        """同步数据库结构"""
        if not self.mysql_connector:
            return
        
        try:
            logger.info("🔄 同步数据库结构...")
            all_ddl = self.mysql_connector.get_all_tables_ddl()
            
            synced = 0
            for table_name, ddl in all_ddl.items():
                if self.knowledge_base.add_ddl(ddl):
                    synced += 1
                    # 添加表描述
                    desc = f"{table_name}表的结构定义"
                    self.knowledge_base.add_documentation(desc, f"table_{table_name}")
            
            logger.info(f"✅ 同步完成: {synced}/{len(all_ddl)} 个表")
        except Exception as e:
            logger.error(f"同步失败: {e}")
    
    def train(self, ddl: str = None, documentation: str = None, question: str = None, sql: str = None):
        """训练"""
        if ddl:
            self.knowledge_base.add_ddl(ddl)
        if documentation:
            self.knowledge_base.add_documentation(documentation)
        if question and sql:
            self.knowledge_base.add_question_sql(question, sql)
    
    def generate_sql(self, question: str) -> str:
        """生成SQL"""
        try:
            # 获取知识库内容
            ddl_list = self.knowledge_base.get_all_ddl()
            docs_list = self.knowledge_base.get_all_documentation()
            qa_list = self.knowledge_base.get_all_questions()
            
            # 构建上下文
            context_parts = []
            
            if ddl_list:
                context_parts.append("数据库表结构:")
                context_parts.extend(ddl_list[:5])  # 最多5个表
            
            if docs_list:
                context_parts.append("\n相关文档:")
                context_parts.extend(docs_list[:3])  # 最多3个文档
            
            if qa_list:
                context_parts.append("\n问答示例:")
                for q, s in qa_list[:3]:  # 最多3个示例
                    context_parts.append(f"问题: {q}")
                    context_parts.append(f"SQL: {s}")
            
            context = "\n".join(context_parts)
            
            # 构建提示
            system_prompt = """你是SQL生成专家。根据提供的表结构和示例，为用户问题生成准确的MySQL查询语句。
规则:
1. 只返回SQL语句，不要解释
2. 使用标准MySQL语法
3. 确保表名和列名正确
4. 根据上下文信息生成查询"""
            
            user_prompt = f"""上下文信息:
{context}

用户问题: {question}

请生成对应的SQL查询语句:"""
            
            sql = self.ollama.generate(user_prompt, system_prompt)
            
            # 清理SQL
            sql = re.sub(r'```sql\s*', '', sql)
            sql = re.sub(r'```\s*', '', sql)
            sql = sql.strip().rstrip(';')
            
            logger.info(f"生成SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"生成SQL失败: {e}")
            return ""
    
    def ask(self, question: str) -> Dict[str, Any]:
        """问答"""
        try:
            sql = self.generate_sql(question)
            
            if not sql:
                return {"question": question, "sql": "", "error": "无法生成SQL", "data": None}
            
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
                return {"question": question, "sql": sql, "error": "未连接数据库", "data": None}
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"question": question, "sql": sql if 'sql' in locals() else "", "error": str(e), "data": None}


def create_simple_app(vanna_instance: SimpleVannaMySQL) -> Flask:
    """创建简化的Flask应用"""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Vanna MySQL</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
        input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; cursor: pointer; }
        .result { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <h1>🔍 Simple Vanna MySQL</h1>
    
    <div>
        <input type="text" id="question" placeholder="输入您的问题..." />
        <button onclick="askQuestion()">生成SQL并查询</button>
        <button onclick="syncSchema()">同步数据库结构</button>
    </div>
    
    <div id="result"></div>
    
    <script>
        function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question.trim()) {
                alert('请输入问题');
                return;
            }
            
            document.getElementById('result').innerHTML = '<p>处理中...</p>';
            
            fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                let html = '<div class="result">';
                html += '<h3>查询结果</h3>';
                html += '<p><strong>问题:</strong> ' + data.question + '</p>';
                
                if (data.sql) {
                    html += '<p><strong>SQL:</strong> <code>' + data.sql + '</code></p>';
                }
                
                if (data.error) {
                    html += '<p style="color: red;"><strong>错误:</strong> ' + data.error + '</p>';
                }
                
                if (data.data && data.data.length > 0) {
                    html += '<p><strong>结果:</strong> ' + data.row_count + ' 行</p>';
                    html += '<table>';
                    html += '<tr>';
                    data.columns.forEach(col => html += '<th>' + col + '</th>');
                    html += '</tr>';
                    
                    data.data.slice(0, 50).forEach(row => {
                        html += '<tr>';
                        data.columns.forEach(col => {
                            const value = row[col];
                            html += '<td>' + (value !== null ? value : 'NULL') + '</td>';
                        });
                        html += '</tr>';
                    });
                    html += '</table>';
                } else if (!data.error) {
                    html += '<p>查询成功，无数据返回</p>';
                }
                
                html += '</div>';
                document.getElementById('result').innerHTML = html;
            })
            .catch(error => {
                document.getElementById('result').innerHTML = '<div class="result"><p style="color: red;">请求失败: ' + error + '</p></div>';
            });
        }
        
        function syncSchema() {
            fetch('/sync-schema', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(result => {
                alert(result.success ? '同步成功!' : '同步失败: ' + result.error);
            });
        }
    </script>
</body>
</html>
        """
    
    @app.route('/ask', methods=['POST'])
    def ask():
        try:
            data = request.json
            question = data.get('question', '')
            result = vanna_instance.ask(question)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/sync-schema', methods=['POST'])
    def sync_schema():
        try:
            vanna_instance.sync_database_schema()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return app


def main():
    """主函数"""
    print("🚀 启动简化版 Vanna MySQL...")
    
    # 创建实例
    vn = SimpleVannaMySQL(
        ollama_config={"host": "http://127.0.0.1", "port": 11434, "model": "deepseek-r1:8b"}
    )
    
    # 测试连接
    print("🔗 测试 Ollama 连接...")
    if not vn.ollama.test_connection():
        print("❌ Ollama 连接失败")
        return
    print("✅ Ollama 连接成功")
    
    # 连接数据库
    print("🔌 连接 MySQL...")
    success = vn.connect_to_mysql(
        host='127.0.0.1',
        user='root',
        password='dcsz2024',
        database='dbgame_hdxyx_szdevgj_0001'
    )
    
    if success:
        print("✅ MySQL 连接成功，已自动同步表结构")
    else:
        print("⚠️ MySQL 连接失败")
    
    # 添加基础训练数据
    vn.train(documentation="这是游戏数据库，包含游戏相关的表和数据")
    vn.train(question="查询表的记录数", sql="SELECT COUNT(*) FROM table_name")
    
    # 启动应用
    app = create_simple_app(vn)
    print("🌐 启动 Web 服务: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
