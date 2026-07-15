# utils_llm.py
import json, random
from openai import OpenAI
# from config import DEEPSEEK_KEY, DEEPSEEK_URL
import time

client = OpenAI(api_key="", base_url="https://api.deepseek.com",)  # Replace with your API key



def deepseek_paraphrase_batch(orig_sentences: list[str], n_each: int) -> list[str]:
    """
    输入：原始句子列表、每个句子需要生成的数量
    输出：所有生成句子的扁平列表
    """
    system = ('''You are a cybersecurity assistant. 
              Generate semantically equivalent and diverse paraphrases for each sentence sample''' 
              f"I will provide a list of original samples. Randomly select samples and generate until you have produced{n_each}semantically equivalent and diverse paraphrases. Please use English for all generated samples."
              f"Return valid JSON list of strings, length={n_each}."
              '''EXAMPLE JSON OUTPUT:
                {
                    "1": "paraphrases1",
                    "2": "paraphrases2",
                }
                ''')
    

    user_prompt = f"""
    Original samples:
    {orig_sentences}
    """
    
    results = []
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt}
    ]


    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format={
                    'type': 'json_object'
                }
            )
            # 解析JSON响应
            paragraphs = json.loads(response.choices[0].message.content)
            paragraphs = list(paragraphs.values())
            print(paragraphs)
            results.extend(paragraphs)
            return results
            
        except (json.JSONDecodeError, KeyError):
            print(f"解析失败，尝试 {attempt+1}/{max_retries}")
            time.sleep(2)
                
