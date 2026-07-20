# Este es un proyecto del curso Alura para crear un agente de IA que responda 
# cuestiones acerca de una documentación, en este caso, de una plataforma 
# educativa.

#Importamos la librería OpenAI para poder interactuar con el modelo de lenguaje
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
            'texto_cols': ['Categoria', 'Problema_O_Componente', 'Solucion_O_Especificacion'], # CORREGIDO: Especificacion
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
        
        # Sanitizar nombres de columnas por si acaso (quitar espacios en blanco)
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
def crear_base_conocimiento(documentos):
    # Dado que los registros CSV suelen ser cortos, dividimos con precaución
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = text_splitter.split_documents(documentos)
    
    # Creación de los embeddings, se almacenan en una base de datos vectorial FAISS en memoria
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs_divididos, embeddings)
    

    # Recuperación de los 3 fragmentos más relevantes
    return vector_store.as_retriever(search_kwargs={"k": 3}) 

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

# =====================================================================
# 4. EJECUCIÓN / INTERFAZ DE CONSOLA
# =====================================================================

if __name__ == "__main__":
    # Asegúrate de haber definido tu OPENAI_API_KEY en tu entorno antes de correr el script
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: Debes configurar la variable de entorno 'OPENAI_API_KEY'")
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
            
        # Ejecutar consulta
        respuesta = agente.invoke({"input": pregunta})
        
        print(f"\nAgente: {respuesta['answer']}")
        print("-" * 50)