from langchain_core.output_parsers import StrOutputParser
from llm import llm


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(vectorstore, prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = (
        {
            "context": (lambda x: x["retrieval_query"]) | retriever | format_docs,
            "model_outputs": lambda x: x["model_outputs"],
            "treatment_metadata": lambda x: x["treatment_metadata"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever