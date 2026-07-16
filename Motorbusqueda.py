# =====================================================================
# 2. CREACIÓN DEL MOTOR DE BÚSQUEDA (VECTOR STORE)
# =====================================================================
def crear_base_conocimiento(documentos):
    # Dado que los registros CSV suelen ser cortos, dividimos con precaución
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = text_splitter.split_documents(documentos)
    
    # Creación de los embeddings, se almacenan en una base de datos vectorial FAISS en memoria
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs_divididos, embeddings)
    

    # Recuperación de los 3 fragmentos más relevantes
    return vector_store.as_retriever(search_kwargs={"k": 3}) 