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

# Configuración del API Key 
os.environ["OPENAI_API_KEY"] = "AQ.Ab8RN6JRarqcNeZRyxizf4Z7yaUCmceM84wfx7JmNMh7XCgD0Q"
llm = ChatOpenAI()

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