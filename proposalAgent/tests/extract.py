import os
import json
import pathlib
from pathlib import Path

import time
from tqdm import tqdm
import google.genai as genai
from google.genai import types

# --- 1. 配置区域 ---

# 从环境变量中获取API密钥，如果找不到，请在此处直接赋值
# 推荐使用环境变量，例如: export GOOGLE_API_KEY="YOUR_API_KEY"
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # 如果您没有设置环境变量，请取消下一行的注释并填入您的密钥
    # API_KEY = "在这里填入您的Google API密钥"
    print("错误：未找到GOOGLE_API_KEY环境变量。请设置该变量或直接在脚本中赋值。")
    exit()

# 设置包含PDF的输入文件夹路径
# 例如: r"C:\Users\YourUser\Documents\PDFs" 或 "/home/user/pdfs"
INPUT_DIR = pathlib.Path("/Users/peelsannaw/Downloads/考试试题/test_data/aa") 

# 设置用于存放生成的JSON文件的输出文件夹路径
OUTPUT_DIR = pathlib.Path("/Users/peelsannaw/Downloads/考试试题/test_data/output")

# 发送给AI模型的指令
# 这个指令告诉AI需要做什么以及期望的输出格式
PROMPT = """
请从这个PDF文档中，抽取出所有的问题、问题选项和对应的正确答案。
生成一个JSON数组，其中每个对象代表一个问题。
请严格按照以下格式输出，不要在JSON前后添加任何多余的文字或解释：
[
  {
    "problem": "这里是问题描述",
    "answer": "这里是正确答案的标识，例如'A'或者'对'",
    "A": "选项A的内容",
    "B": "选项B的内容"
    "...": "如果有更多选项，继续添加"
  },
  {
    "problem": "这是第二个问题...",
    "answer": "...",
    "...": "..."
  }
]
对于判断题，如果没有A、B选项，则让A为：正确，B为错误，anwer为A或者B。
"""

# --- 脚本主体部分 ---

def clean_response_text(text: str) -> str:
    if not text: return ""
    """清理模型返回的文本，移除Markdown代码块标记。"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def process_all_pdfs():
    """
    遍历输入目录中的所有PDF文件，使用GenAI API处理并生成JSON文件。
    """
    print(f"--- 开始处理 ---")
    print(f"输入目录: {INPUT_DIR.resolve()}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")

    # 递归查找所有PDF文件
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"在目录 '{INPUT_DIR}' 中没有找到任何DOCX文件。")
        return

    print(f"共找到 {len(pdf_files)} 个PDF文件待处理。")

    # 使用tqdm创建进度条
    for pdf_path in tqdm(pdf_files, desc="处理docx文件"):
        try:
            # 计算输出文件的路径，并保持原始目录结构
            relative_path = pdf_path.relative_to(INPUT_DIR)
            output_path = OUTPUT_DIR / relative_path.with_suffix('.json')
            
            # 创建输出文件所在的目录
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果JSON文件已存在，则跳过，方便续传
            if output_path.exists():
                tqdm.write(f"文件已存在，跳过: {output_path.name}")
                continue

            # # 1. 上传文件
            tqdm.write(f"正在上传: {pdf_path.name}")
            # pdf_file_for_api = genai.upload_file(path=str(pdf_path))
            filepath = pathlib.Path(pdf_path)

            # # 2. 生成内容
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                PROMPT],
            )

            time.sleep(1) 
            cleaned_json_str = clean_response_text(response.text)
            json_data = json.loads(cleaned_json_str)

            # 4. 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            tqdm.write(f"成功生成JSON: {output_path}")

            # 5. 删除上传的文件以释放空间
            tqdm.write(f"已清理上传文件: {pdf_path.name}")

        except Exception as e:
            tqdm.write(f"处理文件 '{pdf_path.name}' 时发生错误: {e}")
            continue # 出错时继续处理下一个文件

    print("--- 所有文件处理完毕 ---")


if __name__ == "__main__":
    # 确保输入目录存在
    if not INPUT_DIR.is_dir():
        print(f"错误：输入目录 '{INPUT_DIR}' 不存在或不是一个文件夹。")
    else:
        # 确保输出目录存在
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        process_all_pdfs()