# app/classifier.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv
# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập prompt hệ thống
SYSTEM_MESSAGE = SystemMessage(
    content="""Bạn là một bộ phân loại cấp độ học vấn.
Dựa vào nội dung câu hỏi, bạn phải trả lời duy nhất một trong các nhãn sau:
- LOP_6
- LOP_9
- DAI_HOC"""
)

# Tạo đối tượng LLM
llm = ChatOpenAI(model="gpt-4.1", 
                 temperature=0,
                 openai_api_key=os.getenv("OPENAI_API_KEY"),)
def classify_level(text: str) -> str:
    # Tạo danh sách message đúng chuẩn
    messages = [
        SYSTEM_MESSAGE,
        HumanMessage(content=f"Văn bản: {text}\nNhãn:")
    ]
    # Gửi yêu cầu đến mô hình
    response = llm(messages)
    # Trả về chuỗi kết quả, loại bỏ khoảng trắng
    return response.content.strip()

# if __name__ == "__main__":
#     # Ví dụ sử dụng
#     example_text = "Hãy giải thích định lý Pythagore trong hình học."
#     level = classify_level(example_text)
#     print(f"Cấp độ phân loại: {level}")