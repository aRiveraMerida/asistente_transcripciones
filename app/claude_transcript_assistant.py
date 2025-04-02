import os
import json
import time
import random
import logging
import re
from functools import wraps
import anthropic
from anthropic.types import ContentBlock
from anthropic.types.message import Message
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict, Any, Optional, Callable, Generator, Union, TypeVar, Tuple, cast
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Type variable for retry decorator
T = TypeVar('T')
# Constants for retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1
MAX_TRANSCRIPT_CHUNK_SIZE = 50000  # Characters
DEFAULT_MAX_TOKENS = 1500
DEFAULT_SUMMARY_TEMPERATURE = 0.2
DEFAULT_RESPONSE_TEMPERATURE = 0.3

def retry_with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    exponential_base: float = 2,
    jitter: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff in seconds
        exponential_base: Base for the exponential backoff calculation
        jitter: Whether to add random jitter to the backoff time
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = str(e).lower()
                    # Only retry on certain errors
                    should_retry = (
                        "timeout" in error_message or
                        "overloaded" in error_message or
                        "rate_limit" in error_message or
                        "service_unavailable" in error_message or
                        "429" in error_message or
                        "503" in error_message or
                        "529" in error_message
                    )
                    
                    if not should_retry or retries >= max_retries:
                        logger.error(f"Error after {retries} retries: {e}")
                        raise
                    
                    # Calculate backoff time
                    backoff = initial_backoff * (exponential_base ** retries)
                    if jitter:
                        backoff = backoff + random.uniform(0, 1)
                    
                    logger.warning(f"Retry {retries + 1}/{max_retries} after error: {e}. Waiting {backoff:.2f}s...")
                    time.sleep(backoff)
                    retries += 1
        return wrapper
    return decorator

# Nombre de la clase cambiado para mayor claridad
class ClaudeTranscriptAssistant:
    """
    Asistente de transcripciones utilizando el modelo Claude para responder
    preguntas sobre el contenido de una clase grabada.
    
    Esta clase proporciona funcionalidades para cargar transcripciones, generar
    resúmenes automáticos, extraer temas principales y responder preguntas sobre
    el contenido de una clase grabada utilizando la API de Claude (Anthropic).
    
    Características:
    - Manejo de transcripciones largas mediante segmentación
    - Generación de respuestas sincrónicas y en streaming
    - Reintentos automáticos para errores de API
    - Manejo optimizado de tokens para reducir costos
    """
    
    def __init__(
        self, 
        model_name: str = "claude-3-5-sonnet-20240620", 
        api_key: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        summary_temperature: float = DEFAULT_SUMMARY_TEMPERATURE,
        response_temperature: float = DEFAULT_RESPONSE_TEMPERATURE
    ):
        """
        Inicializa el asistente de transcripciones.
        
        Args:
            model_name: Nombre del modelo de Claude a utilizar
            api_key: Clave API de Anthropic (opcional, por defecto se lee de variables de entorno)
            max_retries: Número máximo de reintentos para errores de API
            max_tokens: Número máximo de tokens a generar en respuestas
            summary_temperature: Temperatura para generación de resúmenes (0-1)
            response_temperature: Temperatura para respuestas a preguntas (0-1)
        
        Raises:
            ValueError: Si no se proporciona API key y no está en variables de entorno
        """
        # Obtener la API key de los argumentos o de las variables de entorno
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada en las variables de entorno ni se proporcionó como argumento")
        
        # Inicializar el cliente de Anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Configuración del modelo y parámetros
        self.model_name = model_name
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.summary_temperature = summary_temperature
        self.response_temperature = response_temperature
        
        # Estado del asistente
        self.transcript_content: Optional[str] = None
        self.transcript_summary: Optional[str] = None
        self.main_topics: Optional[List[str]] = None
        
        # Cache para almacenar resultados previos
        self._response_cache: Dict[str, str] = {}
    def load_transcript(self, file_path: str) -> bool:
        """
        Carga la transcripción desde un archivo de texto.
        
        Args:
            file_path: Ruta al archivo de transcripción
            
        Returns:
            True si la carga fue exitosa, False en caso contrario
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            PermissionError: Si no hay permisos para leer el archivo
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Archivo no encontrado: {file_path}")
                raise FileNotFoundError(f"El archivo {file_path} no existe")
                
            with open(file_path, 'r', encoding='utf-8') as file:
                self.transcript_content = file.read()
            
            logger.info(f"Transcripción cargada exitosamente desde {file_path} ({len(self.transcript_content)} caracteres)")
            
            # Limpiar la caché al cargar nueva transcripción
            self._response_cache = {}
            
            # Generar resumen y temas principales automáticamente
            self._generate_summary_and_topics()
            return True
        except FileNotFoundError as e:
            logger.error(f"Error al cargar la transcripción - archivo no encontrado: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Error al cargar la transcripción - permisos insuficientes: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al cargar la transcripción: {e}")
            return False
    def set_transcript_content(self, content: str) -> None:
        """
        Establece el contenido de la transcripción directamente.
        
        Args:
            content: Contenido de la transcripción
            
        Raises:
            ValueError: Si el contenido está vacío
        """
        if not content or not content.strip():
            logger.warning("Intento de establecer contenido de transcripción vacío")
            raise ValueError("El contenido de la transcripción no puede estar vacío")
            
        self.transcript_content = content
        logger.info(f"Contenido de transcripción establecido manualmente ({len(content)} caracteres)")
        
        # Limpiar la caché al establecer nueva transcripción
        self._response_cache = {}
        
        # Generar resumen y temas principales automáticamente
        self._generate_summary_and_topics()
        
    @retry_with_exponential_backoff(max_retries=DEFAULT_MAX_RETRIES)
    def _generate_summary_and_topics(self) -> None:
        """
        Genera automáticamente un resumen y lista de temas principales 
        a partir de la transcripción cargada.
        
        Esta función utiliza Claude para analizar la transcripción y:
        1. Crear un resumen conciso de los contenidos principales
        2. Extraer una lista de temas clave cubiertos en la transcripción
        
        Implementa manejo de transcripciones largas, reintentos automáticos
        para errores de API, y recuperación de fallos.
        
        Returns:
            None. Establece self.transcript_summary y self.main_topics directamente.
            
        Raises:
            Propaga excepciones después de agotar los reintentos.
        """
        if not self.transcript_content:
            logger.warning("No hay transcripción para generar resumen y temas")
            self.transcript_summary = "No hay transcripción cargada."
            self.main_topics = ["No hay transcripción cargada."]
            return
            
        logger.info("Generando resumen y temas principales...")
        
        try:
            # Determinar si necesitamos segmentar la transcripción
            if len(self.transcript_content) > MAX_TRANSCRIPT_CHUNK_SIZE:
                logger.info(f"Transcripción grande detectada ({len(self.transcript_content)} caracteres). Optimizando para procesamiento...")
                # Para resúmenes, es más efectivo usar un enfoque de "resumir el resumen"
                summary = self._process_large_transcript_for_summary()
            else:
                # Para transcripciones más pequeñas, usar el enfoque directo
                summary = self._process_standard_transcript_for_summary()
                
            # Establecer los resultados
            self.transcript_summary = summary.get("summary", "No se pudo generar un resumen.")
            self.main_topics = summary.get("main_topics", ["No se pudieron identificar temas principales."])
            
            logger.info(f"Resumen generado: {len(self.transcript_summary)} caracteres, {len(self.main_topics)} temas identificados")
            
        except Exception as e:
            logger.error(f"Error al generar resumen y temas: {str(e)}")
            self.transcript_summary = "No se pudo generar un resumen automático."
            self.main_topics = ["No se pudieron identificar temas principales."]
            
    def _process_standard_transcript_for_summary(self) -> Dict[str, Any]:
        """
        Procesa una transcripción estándar (no demasiado larga) para obtener resumen y temas.
        
        Returns:
            Dict con "summary" y "main_topics"
        """
        # Prompt para análisis de contenido
        system_prompt = """
        Eres un asistente especializado en analizar transcripciones de clases para extraer información clave.
        Tu tarea es analizar la transcripción y proporcionar:
        1. Un resumen conciso y bien estructurado (máximo 300 palabras)
        2. Una lista de los temas principales tratados (5-10 temas)
        
        Debes devolver tu análisis en formato JSON con las claves "summary" y "main_topics".
        """
        
        # Construir el mensaje para la API
        messages = [{
            "role": "user", 
            "content": f"Analiza esta transcripción de clase y proporciona un resumen y lista de temas principales en formato JSON:\n\n{self.transcript_content}"
        }]
        
        # Realizar la llamada a la API
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.summary_temperature,
            system=system_prompt,
            messages=messages
        )
        
        # Extraer el contenido
        response_content = response.content[0].text
        
        # Intentar extraer el JSON de la respuesta
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Si no hay marcadores de código, intentar usar toda la respuesta
            json_str = response_content
        
        # Limpiar caracteres no JSON
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            # Si no podemos parsear como JSON, extraer manualmente
            return self._extract_summary_and_topics_fallback(response_content)
            
    def _process_large_transcript_for_summary(self) -> Dict[str, Any]:
        """
        Procesa una transcripción larga dividiéndola en segmentos y luego
        combinando los resultados.
        
        Returns:
            Dict con "summary" y "main_topics"
        """
        logger.info("Procesando transcripción larga mediante fragmentación...")
        
        # Dividir la transcripción en segmentos manejables
        segments = self._split_transcript_into_segments(self.transcript_content)
        segment_summaries = []
        all_topics = []
        
        # Procesar cada segmento
        for i, segment in enumerate(segments):
            logger.info(f"Procesando segmento {i+1}/{len(segments)}...")
            
            system_prompt = """
            Eres un asistente especializado en analizar transcripciones de clases.
            Este es un segmento de una transcripción más larga.
            Proporciona:
            1. Un resumen conciso de este segmento (máximo 150 palabras)
            2. Una lista de los temas tratados en este segmento (3-5 temas)
            
            Devuelve tu análisis en formato JSON con las claves "summary" y "main_topics".
            """
            
            messages = [{
                "role": "user", 
                "content": f"Analiza este segmento de transcripción y proporciona un resumen y lista de temas en formato JSON:\n\n{segment}"
            }]
            
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.summary_temperature,
                    system=system_prompt,
                    messages=messages
                )
                
                response_content = response.content[0].text
                
                # Extraer JSON
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_content
                
                # Limpiar caracteres no JSON
                json_str = re.sub(r'^[^{]*', '', json_str)
                json_str = re.sub(r'[^}]*$', '', json_str)
                
                try:
                    result = json.loads(json_str)
                    segment_summaries.append(result.get("summary", ""))
                    all_topics.extend(result.get("main_topics", []))
                except json.JSONDecodeError:
                    # Fallback
                    fallback = self._extract_summary_and_topics_fallback(response_content)
                    segment_summaries.append(fallback.get("summary", ""))
                    all_topics.extend(fallback.get("main_topics", []))
                    
            except Exception as e:
                logger.error(f"Error procesando segmento {i+1}: {str(e)}")
                segment_summaries.append(f"[Error procesando segmento {i+1}]")
        
        # Generar un resumen final combinando los resúmenes de segmentos
        combined_summary = "\n\n".join(segment_summaries)
        
        system_prompt = """
        Eres un asistente especializado en sintetizar información.
        Se te proporcionarán varios resúmenes de segmentos de una misma transcripción.
        Tu tarea es:
        1. Combinar estos resúmenes en un único resumen coherente y bien estructurado (máximo 350 palabras)
        2. Identificar los temas principales globales (5-10 temas)
        
        Devuelve tu análisis en formato JSON con las claves "summary" y "main_topics".
        """
        
        messages = [{
            "role": "user", 
            "content": f"Combina estos resúmenes de segmentos en un único resumen coherente y lista de temas principales en formato JSON:\n\n{combined_summary}\n\nTemas identificados en los segmentos: {', '.join(all_topics)}"
        }]
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.summary_temperature,
                system=system_prompt,
                messages=messages
            )
            
            response_content = response.content[0].text
            
            # Extraer JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_content
            
            # Limpiar caracteres no JSON
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                return self._extract_summary_and_topics_fallback(response_content)
                
        except Exception as e:
            logger.error(f"Error al combinar resúmenes: {str(e)}")
            # Fallback: usar el primer resumen de segmento y los temas principales más frecuentes
            return {
                "summary": segment_summaries[0] if segment_summaries else "No se pudo generar un resumen.",
                "main_topics": list(set(all_topics))[:10] if all_topics else ["No se pudieron identificar temas principales."]
            }
    
    def _split_transcript_into_segments(self, transcript: str) -> List[str]:
        """
        Divide una transcripción larga en segmentos más pequeños y manejables.
        
        Args:
            transcript: La transcripción completa
            
        Returns:
            Lista de segmentos de texto
        """
        # Dividir por párrafos o líneas naturales
        paragraphs = [p for p in re.split(r'\n\s*\n', transcript) if p.strip()]
        
        segments = []
        current_segment = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # Si añadir este párrafo excedería el tamaño máximo, guardamos el segmento actual
            if current_length + paragraph_length > MAX_TRANSCRIPT_CHUNK_SIZE and current_segment:
                segments.append("\n\n".join(current_segment))
                current_segment = []
                current_length = 0
            
            # Añadir el párrafo al segmento actual
            current_segment.append(paragraph)
            current_length += paragraph_length
        
        # Añadir el último segmento si existe
        if current_segment:
            segments.append("\n\n".join(current_segment))
        
        return segments
        
    def _extract_summary_and_topics_fallback(self, text: str) -> Dict[str, Any]:
        """
        Método de respaldo para extraer resumen y temas cuando falla el parsing JSON.
        
        Args:
            text: Texto completo de la respuesta
            
        Returns:
            Dict con "summary" y "main_topics"
        """
        result = {"summary": "", "main_topics": []}
        
        # Buscar secciones de resumen
        summary_patterns = [
            r'(?:resumen|summary)[:]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'(?:resumen|summary)[^:]*?:\s*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["summary"] = match.group(1).strip()
                break
        
        # Buscar listas de temas
        topics = []
        
        # Buscar listas numeradas (1. Tema)
        numbered_topics = re.findall(r'\n\s*\d+\.\s*(.*?)(?:\n|$)', text)
        if numbered_topics:
            topics.extend(numbered_topics)
        
        # Buscar listas con viñetas (-, *, •)
        bullet_patterns = [
            r'\n\s*[-*•]\s*(.*?)(?:\n|$)',  # Simple bullet points
            r'\n\s*[-*•][ \t]+(.*?)(?:\n|$)',  # Bullet points with spacing
            r'(?<=\n)[ \t]*[-*•][ \t]+(.*?)(?:\n|$)'  # Bullet points with indentation
        ]
        
        for pattern in bullet_patterns:
            bullet_topics = re.findall(pattern, text, re.MULTILINE)
            if bullet_topics:
                topics.extend([topic.strip() for topic in bullet_topics if topic.strip()])
        
        # Buscar temas con formato "Tema X:" o "X. Tema:"
        labeled_topics = re.findall(r'(?:\n|^)\s*(?:tema|topic)\s+\d+[:.]\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if labeled_topics:
            topics.extend([topic.strip() for topic in labeled_topics if topic.strip()])
        
        # Si no se encontraron temas, intentar detectar párrafos cortos separados que podrían ser temas
        if not topics:
            # Buscar frases cortas o líneas que podrían ser temas
            possible_topics = re.findall(r'(?:\n|^)\s*([A-Z][^.!?]{5,50}[.!?])(?:\s*\n|$)', text)
            if possible_topics:
                topics.extend(possible_topics[:10])  # Limitar a 10 temas
        
        # Eliminar duplicados y normalizar los temas
        cleaned_topics = []
        for topic in topics:
            # Limpieza básica
            cleaned = topic.strip()
            # Eliminar comillas o caracteres no deseados
            cleaned = re.sub(r'^["\'`]|["\'`]$', '', cleaned)
            # Solo añadir temas significativos y no duplicados
            if cleaned and cleaned not in cleaned_topics and len(cleaned) > 3:
                cleaned_topics.append(cleaned)
        
        result["main_topics"] = cleaned_topics[:10]  # Limitar a 10 temas
        
        # Si no hay resumen, intentar usar el primer párrafo del texto como resumen
        if not result["summary"]:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if paragraphs:
                result["summary"] = paragraphs[0]
        
        return result
    
    def ask_question(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Responde a una pregunta sobre la transcripción.
        
        Args:
            question: Pregunta del usuario
            chat_history: Historial de conversación opcional
            
        Returns:
            Respuesta a la pregunta
        """
        if not self.transcript_content:
            return "No hay transcripción cargada. Por favor, carga una transcripción primero."
        
        # Preparar el historial de chat si existe
        messages = []
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Construir el prompt con el sistema y la transcripción
        system_prompt = """
        Eres un asistente educativo especializado en explicar el contenido de clases grabadas.
        
        INSTRUCCIONES IMPORTANTES:
        1. SOLO responde basándote en la información presente en la transcripción proporcionada.
        2. Si la información para responder no está en la transcripción, indica claramente: "Lo siento, esa información no está presente en la transcripción de la clase".
        3. No inventes ni agregues información que no esté explícitamente en la transcripción.
        4. Si se te pide explicar algo de otro modo, puedes reformular pero manteniendo el mismo significado y sin agregar información extra.
        5. Cita partes relevantes de la transcripción cuando sea útil.
        6. Si te piden información específica (como fechas, nombres, fórmulas) que no aparece en la transcripción, indícalo claramente.
        7. Cuando sea apropiado, puedes mencionar en qué parte aproximada de la clase se discutió un tema.
        8. Mantén un tono educativo, claro y servicial.
        """
        
        # Añadir la pregunta actual
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                temperature=0.3,
                system=system_prompt,
                messages=messages,
                context=[
                    {
                        "text": f"TRANSCRIPCIÓN DE LA CLASE:\n\n{self.transcript_content}"
                    }
                ]
            )
            
            return response.content[0].text
        
        except Exception as e:
            return f"Error al procesar tu pregunta: {str(e)}"
    
    def generate_response_stream(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """
        Genera una respuesta en streaming a una pregunta sobre la transcripción.
        
        Args:
            question: Pregunta del usuario
            chat_history: Historial de conversación opcional
            
        Returns:
            Stream de tokens de respuesta
        """
        if not self.transcript_content:
            yield "No hay transcripción cargada. Por favor, carga una transcripción primero."
            return
        
        # Preparar el historial de chat si existe
        messages = []
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Construir el prompt con el sistema y la transcripción
        system_prompt = f"""
        Eres un asistente educativo especializado en explicar el contenido de clases grabadas.
        
        TRANSCRIPCIÓN DE LA CLASE:
        {self.transcript_content}
        
        INSTRUCCIONES IMPORTANTES:
        1. SOLO responde basándote en la información presente en la transcripción proporcionada.
        2. Si la información para responder no está en la transcripción, indica claramente: "Lo siento, esa información no está presente en la transcripción de la clase".
        3. No inventes ni agregues información que no esté explícitamente en la transcripción.
        4. Si se te pide explicar algo de otro modo, puedes reformular pero manteniendo el mismo significado y sin agregar información extra.
        5. Cita partes relevantes de la transcripción cuando sea útil.
        6. Si te piden información específica (como fechas, nombres, fórmulas) que no aparece en la transcripción, indícalo claramente.
        7. Cuando sea apropiado, puedes mencionar en qué parte aproximada de la clase se discutió un tema.
        8. Mantén un tono educativo, claro y servicial.
        """
        
        # Añadir la pregunta actual
        messages.append({"role": "user", "content": question})
        
        try:
            # Crear una respuesta en streaming
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=1500,
                temperature=0.3,
                system=system_prompt,
                messages=messages
            ) as stream:
                # Procesar el stream
                for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            yield f"Error al procesar tu pregunta: {str(e)}"

# Ejemplo de uso directo
if __name__ == "__main__":
    # Configurar API key desde variables de entorno (.env)
    # ANTHROPIC_API_KEY=tu_clave_aqui
    
    assistant = ClaudeTranscriptAssistant()
    
    # Ejemplo: cargar una transcripción
    transcript_path = "ruta/a/tu/transcripcion.txt"
    if os.path.exists(transcript_path):
        assistant.load_transcript(transcript_path)
        
        # Hacer una pregunta
        question = "¿Cuáles fueron los principales temas discutidos en esta clase?"
        answer = assistant.ask_question(question)
        
        print(f"Pregunta: {question}")
        print(f"Respuesta: {answer}")
    else:
        print(f"El archivo {transcript_path} no existe.")