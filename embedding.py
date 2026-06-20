from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings"
)