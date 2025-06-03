# app/qa_chain.py
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from .classifier import classify_level
from .learning_tools import gen_flashcards, gen_mindmap
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vec = FAISS.load_local("data/faiss_index", OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vec.as_retriever(search_type="similarity", k=4),
        memory=memory,
        return_source_documents=True
)

def answer(question:str):
    level = classify_level(question)
    result = qa_chain({"question":question})
    answer_text = result["answer"]
    sources = [d.metadata["source"] for d in result["source_documents"]]

    # gợi ý học thêm
    flashcards = gen_flashcards(answer_text, level)
    mindmap   = gen_mindmap(answer_text, level)

    return {
        "grade_level": level,
        "answer": answer_text,
        "flashcards": flashcards,
        "mindmap": mindmap,
        "sources": sources
    }
