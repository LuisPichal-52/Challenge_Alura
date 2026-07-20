# =====================================================================
# 3. FUNCIÓN AGREGADA: INICIALIZAR EL AGENTE EDUCATIVO
# =====================================================================
def inicializar_agente_educativo(retriever):
    # Instanciamos el modelo de lenguaje de manera óptima para el agente
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Definimos el prompt del sistema dándole el rol y restricciones
    system_prompt = (
        "Eres un agente de IA experto y servicial para nuestra plataforma educativa.\n"
        "Tu tarea es responder las preguntas de los estudiantes utilizando únicamente el contexto provisto.\n"
        "Si no sabes la respuesta o no se encuentra en la documentación, di de forma amable "
        "que no posees esa información y sugiéreles contactar al soporte.\n\n"
        "Contexto:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Creamos la cadena RAG combinando el recuperador y el LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain