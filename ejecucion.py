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