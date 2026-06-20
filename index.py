from langchain_community.vectorstores import FAISS
from embedding import embedding_function


def load_index(index_path: str = "m4_faiss_index"):
    return FAISS.load_local(
        index_path,
        embedding_function,
        allow_dangerous_deserialization=True
    )