import re
import time
from typing import Dict, Any, Optional

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrae un objeto JSON de un texto.
    
    Args:
        text: Texto que puede contener un objeto JSON
        
    Returns:
        Diccionario con el objeto JSON extraído o None si no se encuentra
    """
    try:
        # Buscar contenido JSON entre triple comillas
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Si no hay marcadores de código, intentar usar todo el texto
            json_str = text
        
        # Limpiar caracteres no JSON
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        import json
        return json.loads(json_str)
    except Exception as e:
        print(f"Error al extraer JSON: {str(e)}")
        return None

def format_time(seconds: int) -> str:
    """
    Formatea segundos en formato HH:MM:SS.
    
    Args:
        seconds: Número de segundos
        
    Returns:
        Tiempo formateado como HH:MM:SS
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def calculate_read_time(text: str, wpm: int = 200) -> int:
    """
    Calcula el tiempo estimado de lectura en minutos.
    
    Args:
        text: Texto para calcular el tiempo de lectura
        wpm: Palabras por minuto promedio (por defecto: 200)
        
    Returns:
        Tiempo estimado de lectura en minutos
    """
    word_count = len(text.split())
    minutes = max(1, round(word_count / wpm))
    return minutes