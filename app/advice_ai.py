# app/advice_ai.py
import os
import json
import requests

# 手动加载 .env 文件（避免依赖 python-dotenv）
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是一位基于中医养生和现代医学的健康顾问。根据用户的健康检测结果，从5个维度给出具体、可操作的生活建议。

五个维度：饮食调理、作息建议、运动指导、中医调理、心理调节。

要求：
1. 每条建议30-80字，具体可操作，不说空话
2. 语气温和专业，有同理心
3. 以JSON格式返回，key为维度名称，value为建议内容
4. 只返回JSON，不要有其他文字"""


def build_user_prompt(label: str, probabilities: dict) -> str:
    return f"""用户健康检测结果：
- 综合判定：{label}
- 各维度概率：{json.dumps(probabilities, ensure_ascii=False)}

请根据上述结果，从饮食调理、作息建议、运动指导、中医调理、心理调节五个维度给出生活建议。"""


def generate_ai_advice(label: str, probabilities: dict) -> dict:
    """调用 DeepSeek API 生成 AI 生活建议"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return {"error": "未配置 DEEPSEEK_API_KEY 环境变量"}

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(label, probabilities)},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # 尝试解析 JSON
        advice = json.loads(content)
        # 验证维度完整性
        required = ["饮食调理", "作息建议", "运动指导", "中医调理", "心理调节"]
        for key in required:
            if key not in advice:
                advice[key] = f"建议咨询专业人士获取{key}方面的指导。"
        return advice
    except json.JSONDecodeError:
        return {"error": "AI 返回格式异常"}
    except requests.RequestException as e:
        return {"error": f"AI 服务请求失败: {str(e)}"}
