import streamlit as st
import os
import sys
import threading
import queue
import time
from typing import List, Dict, Generator

# Add the project root directory to the Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.claude_transcript_assistant import ClaudeTranscriptAssistant

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Transcripciones",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #7857F7;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5F6368;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #7857F7;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-message.user {
        background-color: #F3F0FF;
        border-left: 5px solid #7857F7;
    }
    .chat-message.assistant {
        background-color: #F8F9FA;
        border-left: 5px solid #43A047;
    }
    .info-text {
        color: #5F6368;
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #FFF9C4;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .topic-pill {
        display: inline-block;
        background-color: #E8F0FE;
        color: #1967D2;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .file-uploader {
        padding: 1rem;
        border: 1px dashed #DADCE0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #7857F7;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 2s linear infinite;
        margin: 1rem auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Inicialización del estado de la sesión
def init_session_state():
    """Inicializa las variables de estado de la sesión."""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ClaudeTranscriptAssistant()
    if 'transcript_loaded' not in st.session_state:
        st.session_state.transcript_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pending_response' not in st.session_state:
        st.session_state.pending_response = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

# Función para transmitir la respuesta del asistente
def stream_assistant_response(assistant: ClaudeTranscriptAssistant, question: str) -> Generator[str, None, None]:
    """
    Transmite la respuesta del asistente usando un objeto generador.
    
    Args:
        assistant: Asistente de transcripciones
        question: Pregunta del usuario
        
    Returns:
        Generador que produce tokens de respuesta
    """
    # Convertir el historial de chat al formato esperado por el asistente
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] in ["user", "assistant"]:
            chat_history.append({"role": msg["role"], "content": msg["content"]})
    
    # Generar respuesta a través del stream del asistente
    response_stream = assistant.generate_response_stream(question, chat_history)
    
    # Variable para recolectar la respuesta completa
    full_response = ""
    
    # Generar tokens desde el stream
    for token in response_stream:
        full_response += token
        yield token
    
    # Almacenar la respuesta completa en el estado de la sesión
    st.session_state.pending_response = full_response

# Función para cargar el archivo de transcripción
def load_transcript_file(uploaded_file):
    """
    Carga un archivo de transcripción al asistente.
    
    Args:
        uploaded_file: Archivo subido por el usuario
    """
    try:
        # Marcar como procesando
        st.session_state.processing = True
        
        # Leer el contenido del archivo
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Establecer el contenido en el asistente
        st.session_state.assistant.set_transcript_content(content)
        
        # Actualizar estado
        st.session_state.transcript_loaded = True
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.messages = []  # Limpiar mensajes previos
        st.session_state.error_message = None
        
    except Exception as e:
        st.session_state.error_message = f"Error al cargar el archivo: {str(e)}"
        st.session_state.transcript_loaded = False
    finally:
        # Finalizar procesamiento
        st.session_state.processing = False

# Función principal de la aplicación
def run_app():
    """Función principal para la aplicación Streamlit."""
    # Inicializar estado de la sesión
    init_session_state()
    
    # Verificar si hay una respuesta pendiente para agregar al historial
    if st.session_state.pending_response is not None:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": st.session_state.pending_response
        })
        st.session_state.pending_response = None
        # Recargar para mostrar el nuevo mensaje
        st.rerun()
    
    # Título y descripción en el área de contenido principal
    st.markdown('<h1 class="main-header">Asistente de Transcripciones</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Carga la transcripción de una clase y haz preguntas sobre su contenido.</p>',
        unsafe_allow_html=True
    )
    
    # Barra lateral para cargar archivos y configuración
    with st.sidebar:
        st.markdown("### 📝 Cargar Transcripción")
        
        # Carga de archivo
        uploaded_file = st.file_uploader(
            "Sube un archivo de transcripción (.txt)",
            type=["txt"],
            help="Sube un archivo de texto con la transcripción de la clase"
        )
        
        # Botón para procesar archivo
        if uploaded_file is not None:
            if (st.session_state.current_file_name != uploaded_file.name or 
                not st.session_state.transcript_loaded):
                
                if st.button("Procesar Transcripción", use_container_width=True):
                    load_transcript_file(uploaded_file)
        
        # Mostrar información del archivo cargado
        if st.session_state.transcript_loaded and st.session_state.current_file_name:
            st.success(f"Archivo cargado: {st.session_state.current_file_name}")
            
            # Opción para reiniciar conversación
            if st.button("Reiniciar Conversación", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Configuración avanzada
        with st.expander("Configuración Avanzada"):
            model_name = st.selectbox(
                "Modelo de Claude",
                options=["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                index=0
            )
            
            # Actualizar el modelo si cambia
            if st.session_state.assistant.model_name != model_name:
                st.session_state.assistant.model_name = model_name
        
        # Créditos
        st.markdown("---")
        st.markdown("Desarrollado con Anthropic Claude")
        st.markdown("© 2025 - Asistente de Transcripciones")
    
    # Mostrar mensaje de error si existe
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        if st.button("Limpiar Error"):
            st.session_state.error_message = None
            st.rerun()
    
    # Mostrar indicador de procesamiento
    if st.session_state.processing:
        st.markdown('<div class="loader"></div>', unsafe_allow_html=True)
        st.info("Procesando la transcripción... Por favor espera.")
        return
    
    # Mostrar transcripción cargada y resumen
    if st.session_state.transcript_loaded:
        # Crear pestañas para la información y el chat
        tab1, tab2 = st.tabs(["💬 Chat", "ℹ️ Información de la Clase"])
        
        # Pestaña de chat
        with tab1:
            # Mostrar historial de chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Entrada de chat
            if prompt := st.chat_input("Haz una pregunta sobre esta clase..."):
                # Agregar mensaje del usuario al historial
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Mostrar mensaje del usuario
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Respuesta del asistente con transmisión
                with st.chat_message("assistant"):
                    try:
                        # Transmitir los tokens de respuesta
                        response_stream = stream_assistant_response(st.session_state.assistant, prompt)
                        st.write_stream(response_stream)
                    except Exception as e:
                        error_msg = f"Error al generar respuesta: {str(e)}"
                        st.error(error_msg)
                        st.session_state.pending_response = error_msg
        
        # Pestaña de información de la clase
        with tab2:
            # Mostrar resumen si está disponible
            if st.session_state.assistant.transcript_summary:
                st.markdown("### 📚 Resumen de la Clase")
                st.markdown(st.session_state.assistant.transcript_summary)
            
            # Mostrar temas principales si están disponibles
            if st.session_state.assistant.main_topics:
                st.markdown("### 🏷️ Temas Principales")
                topics_html = ""
                for topic in st.session_state.assistant.main_topics:
                    topics_html += f'<span class="topic-pill">{topic}</span>'
                st.markdown(topics_html, unsafe_allow_html=True)
            
            # Mostrar vista previa de la transcripción
            st.markdown("### 📝 Vista Previa de la Transcripción")
            
            # Mostrar los primeros 1000 caracteres
            preview_length = 1000
            transcript_preview = st.session_state.assistant.transcript_content[:preview_length]
            if len(st.session_state.assistant.transcript_content) > preview_length:
                transcript_preview += "..."
            
            st.text_area(
                "Primeros 1000 caracteres:",
                transcript_preview,
                height=200,
                disabled=True
            )
            
            # Estadísticas básicas
            word_count = len(st.session_state.assistant.transcript_content.split())
            char_count = len(st.session_state.assistant.transcript_content)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Palabras", f"{word_count:,}")
            with col2:
                st.metric("Caracteres", f"{char_count:,}")
    
    else:
        # Mostrar mensaje de bienvenida cuando no hay transcripción cargada
        st.info(
            """
            👋 Bienvenido al Asistente de Transcripciones
            
            Para comenzar:
            1. Sube un archivo de texto (.txt) con la transcripción de una clase en la barra lateral
            2. Haz clic en "Procesar Transcripción"
            3. Cuando la transcripción esté lista, podrás hacer preguntas sobre el contenido
            
            El asistente responderá basándose únicamente en la información presente en la transcripción.
            """
        )

if __name__ == "__main__":
    run_app()