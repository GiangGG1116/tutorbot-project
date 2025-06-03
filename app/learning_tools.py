# app/learning_tools.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- 1) Flashcard prompt ----------
FLASH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là giáo viên giỏi kỹ năng ghi nhớ. "
     "Hãy biến đoạn văn sau thành TỐI ĐA 5 flashcard (định dạng Q: ...\\nA: ...). "
     "Ngắn gọn, rõ ràng, chỉ kiến thức cốt lõi."
    ),
    ("human", "Nội dung:\n{content}\n---\nFlashcard:")
])

def gen_flashcards(content: str) -> list[dict]:
    """Trả về list flashcard [{'Q':..., 'A':...}, ...]"""
    raw = llm(FLASH_PROMPT.format(content=content)).content.strip()
    cards = []
    for block in raw.split("\n"):
        if block.startswith("Q:"):
            q = block[2:].strip()
            # tìm dòng tiếp theo bắt đầu bằng 'A:'
            # (giả định format đúng)
            continue
        if block.startswith("A:"):
            a = block[2:].strip()
            cards.append({"Q": q, "A": a})
    return cards

# ---------- 2) Mind-map prompt ----------
MINDMAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là chuyên gia tóm tắt ý chính. "
     "Chuyển nội dung sau thành mind-map dạng bullet:\n"
     "▸ cấp1\n ▹ cấp2\n  – cấp3."
    ),
    ("human", "Nội dung:\n{content}\n---\nMind-map:")
])

def gen_mindmap(content: str) -> str:
    """Trả về chuỗi bullet mind-map"""
    return llm(MINDMAP_PROMPT.format(content=content)).content.strip()
