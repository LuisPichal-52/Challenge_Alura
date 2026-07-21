# Este es un proyecto del curso Alura para crear un agente de IA que responda 
# cuestiones acerca de una documentación, en este caso, de una plataforma 
# educativa.

#Importamos la librería OpenAI para poder interactuar con el modelo de lenguaje
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Cambiamos la importación a la librería de Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# =====================================================================
# 1. PROCESAMIENTO DE ARCHIVOS CSV
# =====================================================================
def cargar_y_procesar_csvs():
    documentos = []

    # Mapeo de archivos y sus columnas para unificar la lectura
    config_archivos = {
        'faq_cursos_certificados.csv': {
            'texto_cols': ['Pregunta', 'Respuesta'],
            'metadata_cols': ['Categoria']
        },
        'guia_uso_plataforma.csv': {
            'texto_cols': ['Categoria', 'Problema_O_Componente', 'Solucion_O_Especificacion'],
            'metadata_cols': ['Categoria']
        },
        'politica_reembolso.csv': {
            'texto_cols': ['Seccion', 'Concepto', 'Regla_Criterio'],
            'metadata_cols': ['Seccion']
        },
        'reglamento_estudiante.csv': {
            'texto_cols': ['Seccion', 'Subcategoria', 'Regla_O_Norma', 'Detalle'],
            'metadata_cols': ['Seccion', 'Subcategoria']
        },
        'programa_becas_afiliados.csv': {
            'texto_cols': ['Tipo_Programa', 'Concepto', 'Detalles_Y_Condiciones'],
            'metadata_cols': ['Tipo_Programa']
        }
    }
    
    for archivo, config in config_archivos.items():
        if not os.path.exists(archivo):
            print(f"Archivo no encontrado: {archivo}. Se omitirá.")
            continue

        print(f"Procesando {archivo}...")
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip()
        
        for _, fila in df.iterrows():
            # Creación de un bloque de texto descriptivo y legible para la IA
            partes_texto = []
            for col in df.columns:
                if pd.notna(fila[col]):
                    partes_texto.append(f"{col}: {fila[col]}")
                    
            contenido_texto = "\n".join(partes_texto)
            
            # Extracción de metadata básica para el motor de búsqueda
            metadata = {"fuente": archivo}
            for meta_col in config['metadata_cols']:
                if meta_col in df.columns and pd.notna(fila[meta_col]):
                    metadata[meta_col] = str(fila[meta_col])

            # Creación del objeto Document de LangChain
            documentos.append(Document(page_content=contenido_texto, metadata=metadata))
    print(f"Total de registros cargados y convertidos en documentos: {len(documentos)}")
    return documentos

# =====================================================================
# 2. CREACIÓN DEL MOTOR DE BÚSQUEDA (VECTOR STORE)
# =====================================================================
from langchain_huggingface import HuggingFaceEmbeddings
def crear_base_conocimiento(documentos):
    # Dado que los registros CSV suelen ser cortos, dividimos con precaución
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = text_splitter.split_documents(documentos)
    
    # Usamos el modelo de embeddings gratuito de Google
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs_divididos, embeddings)
    
    # Recuperación de los 3 fragmentos más relevantes
    return vector_store.as_retriever(search_kwargs={"k": 3})

# =====================================================================
# 3. FUNCIÓN AGREGADA: INICIALIZAR EL AGENTE EDUCATIVO
# =====================================================================
def inicializar_agente_educativo(retriever):
    # Instanciamos el modelo de lenguaje de manera óptima para el agente
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)    

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
    
    # Formateador de documentos auxiliar
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Construcción de la cadena usando LCEL (operadores nativos)
    # Esto elimina la necesidad de importar 'create_retrieval_chain'

    #Se contrae el extracto del texto antes de buscar 
    def get_context(inputs):
        docs = retriever.invoke(inputs["input"])
        return format_docs(docs)
    
    rag_chain = (
        {
            "context": get_context, 
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
    )
    
    # Para mantener compatibilidad con tu bucle principal, devolvemos un objeto con 'invoke'
    class LegacyChainWrapper:
        def __init__(self, chain):
            self.chain = chain
        def invoke(self, inputs):
            res = self.chain.invoke(inputs)
            # Retorna el formato esperado por tu bloque principal
            return {"answer": res.content}
            
    return LegacyChainWrapper(rag_chain)

# =====================================================================
# 4. EJECUCIÓN / INTERFAZ DE CONSOLA
# =====================================================================
if __name__ == "__main__":
    # Validamos la API Key de Google
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: Debes configurar la variable de entorno 'GOOGLE_API_KEY'")
        exit(1)
        
    print("Iniciando configuración del agente educativo...")
    documentos = cargar_y_procesar_csvs()
    
    if not documentos:
        print("No se pudieron cargar documentos. Abortando.")
        exit(1)
        
    retriever = crear_base_conocimiento(documentos)
    agente = inicializar_agente_educativo(retriever)
    
    print("\n ¡Agente Educativo en línea! Pregúntame lo que quieras. Escribe 'salir' para terminar.\n")
    
    while True:
        pregunta = input("Estudiante: ")
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("Agente: ¡Hasta luego! Mucho éxito en tus estudios.")
            break
            
        if pregunta.strip() == "":
            continue
            
        respuesta = agente.invoke({"input": pregunta})
        
        print(f"\nAgente: {respuesta['answer']}")
        print("-" * 50)