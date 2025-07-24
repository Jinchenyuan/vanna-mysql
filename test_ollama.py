#!/usr/bin/env python3
"""
测试 Ollama 连接的独立脚本
"""

import requests
import json
import time
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_basic_connection():
    """基础连接测试"""
    print("🔍 测试基础连接...")
    
    try:
        url = "http://127.0.0.1:11434/api/tags"
        # 绕过代理设置
        proxies = {
            'http': None,
            'https': None
        }
        response = requests.get(url, timeout=10, proxies=proxies)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 连接成功！")
            print(f"可用模型: {[m.get('name') for m in data.get('models', [])]}")
            return True
        else:
            print(f"❌ 连接失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False

def test_with_different_headers():
    """使用不同请求头测试"""
    print("\n🔧 测试不同请求头配置...")
    
    headers_configs = [
        {
            "name": "默认配置",
            "headers": {}
        },
        {
            "name": "Postman 模拟配置",
            "headers": {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'PostmanRuntime/7.32.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
        },
        {
            "name": "浏览器模拟配置",
            "headers": {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive'
            }
        },
        {
            "name": "简化配置",
            "headers": {
                'Content-Type': 'application/json',
                'User-Agent': 'Python-requests/2.31.0'
            }
        }
    ]
    
    for config in headers_configs:
        print(f"\n  📋 测试 {config['name']}...")
        try:
            url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": "deepseek-r1:8b",
                "prompt": "Hello, this is a test.",
                "stream": False
            }
            
            # 绕过代理设置
            proxies = {
                'http': None,
                'https': None
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers=config['headers'],
                proxies=proxies,
                timeout=30
            )
            
            print(f"    状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print(f"    ✅ 成功！响应长度: {len(response_text)}")
                if response_text:
                    print(f"    响应预览: {response_text[:100]}...")
                return config
            else:
                print(f"    ❌ 失败: {response.text[:200]}")
                
        except Exception as e:
            print(f"    ❌ 异常: {e}")
    
    return None

def test_direct_curl_equivalent():
    """测试直接等效于 curl 的请求"""
    print("\n🌐 测试 curl 等效请求...")
    
    try:
        # 创建一个新的 session
        session = requests.Session()
        
        # 绕过代理设置
        session.proxies = {
            'http': None,
            'https': None
        }
        
        # 设置 adapter
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "deepseek-r1:8b",
            "prompt": "Say hello in Chinese.",
            "stream": False
        }
        
        # 不设置任何特殊头部，使用最简单的请求
        response = session.post(url, json=payload, timeout=60)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print(f"✅ 成功！响应: {response_text}")
            return True
        else:
            print(f"❌ 失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def test_with_session():
    """使用 Session 对象测试"""
    print("\n🔄 测试 Session 连接...")
    
    try:
        session = requests.Session()
        
        # 绕过代理设置
        session.proxies = {
            'http': None,
            'https': None
        }
        
        # 配置重试策略
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        
        # 测试获取模型列表
        tags_url = "http://127.0.0.1:11434/api/tags"
        response = session.get(tags_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ 模型列表获取成功")
            
            # 测试生成请求
            generate_url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": "deepseek-r1:8b",
                "prompt": "What is 2+2?",
                "stream": False
            }
            
            response = session.post(generate_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 生成成功: {result.get('response', '')}")
                return True
            else:
                print(f"❌ 生成失败: {response.status_code} - {response.text}")
                return False
        else:
            print(f"❌ 模型列表获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Session 测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始 Ollama 连接测试...\n")
    
    # 1. 基础连接测试
    if not test_basic_connection():
        print("\n❌ 基础连接测试失败，请检查 Ollama 服务是否运行")
        return
    
    # 2. 不同请求头测试
    working_config = test_with_different_headers()
    if working_config:
        print(f"\n✅ 找到可用配置: {working_config['name']}")
    
    # 3. 直接 curl 等效测试
    test_direct_curl_equivalent()
    
    # 4. Session 测试
    test_with_session()
    
    print("\n🎉 测试完成！")
    
    # 给出建议
    print("\n💡 建议:")
    print("1. 如果所有测试都失败，请检查 Ollama 服务是否正常运行")
    print("2. 如果部分测试成功，使用成功的配置更新代码")
    print("3. 检查防火墙是否阻止了端口 11434")
    print("4. 尝试重启 Ollama 服务")

if __name__ == "__main__":
    main()
