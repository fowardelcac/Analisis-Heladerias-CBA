# Importación de librerías necesarias
import pandas as pd  # Para manipulación de datos en DataFrames
from dateutil.relativedelta import relativedelta  # Para cálculos relativos de fechas
from datetime import datetime, timedelta  # Para manejo de fechas y horas
import re  # Para expresiones regulares
import string  # Para operaciones con cadenas de texto
import nltk  # Biblioteca de procesamiento de lenguaje natural
from nltk.tokenize import word_tokenize  # Para tokenización de texto
from nltk.corpus import stopwords  # Para obtener palabras vacías (stopwords)
import stanza  # Para lematización y análisis de sentimientos en español

# Descarga de recursos necesarios de NLTK
nltk.download("punkt")       # Modelo de tokenización
nltk.download("punkt_tab")   # Tablas adicionales para tokenización
nltk.download("stopwords")   # Lista de palabras vacías
stop_words = set(stopwords.words("spanish"))  # Conjunto de stopwords en español

# Descarga e inicialización del modelo de Stanza para español
stanza.download("es")  # Descarga el modelo en español
nlp = stanza.Pipeline("es")  # Pipeline de procesamiento para español

# Función para limpiar texto
def clean_text(text):
    text = text.lower()  # Convierte todo a minúsculas
    text = re.sub(r"\d+", "", text)  # Elimina todos los números
    text = text.translate(
        str.maketrans("", "", string.punctuation)  # Elimina signos de puntuación
    )
    text = re.sub(r"\W", " ", text)  # Reemplaza caracteres no alfanuméricos por espacios
    return text

# Función para procesar fechas relativas desde texto
def process_date(txt: str):
    today = datetime.today()  # Obtiene la fecha actual
    final_date = None  # Inicializa la fecha final como None
    
    # Determina la fecha según el período especificado en el texto
    if ("año" or "años") in txt:
        cantidad = 1 if "un" in txt else int(txt.split()[1])  # Extrae la cantidad
        final_date = today - relativedelta(years=cantidad)    # Resta años
    elif "mes" in txt:
        cantidad = 1 if "un" in txt else int(txt.split()[1])  # Extrae la cantidad
        final_date = today - relativedelta(months=cantidad)   # Resta meses
    elif "semana" in txt:
        cantidad = 1 if "una" in txt else int(txt.split()[0]) # Extrae la cantidad
        final_date = today - timedelta(weeks=cantidad)        # Resta semanas
    elif "día" in txt:
        cantidad = 1 if "un" in txt else int(txt.split()[1])  # Extrae la cantidad
        final_date = today - timedelta(days=cantidad)         # Resta días
    elif "hora" in txt:
        cantidad = int(txt.split()[1])                        # Extrae la cantidad
        final_date = today - timedelta(hours=cantidad)        # Resta horas
    else:
        return None  # Devuelve None si no se reconoce el formato

    return final_date.strftime("%d/%m/%Y")  # Devuelve la fecha en formato dd/mm/yyyy

# Función para lematizar tokens
def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))  # Procesa los tokens con Stanza
    # Devuelve una lista de lemas para cada palabra en el texto
    return [word.lemma for sentence in doc.sentences for word in sentence.words]

# Función para obtener el sentimiento del texto
def get_sentiment(text):
    if text is None:
        return "Neutral"  # Si no hay texto, devuelve Neutral
    try:
        doc = nlp(text)  # Procesa el texto con Stanza
        # Verifica si hay oraciones para analizar
        if not doc.sentences:
            return "Neutral"
        return doc.sentences[0].sentiment  # Devuelve el sentimiento de la primera oración
    except Exception as e:
        print(f"Error processing text: {e}")  # Imprime error si ocurre
        return "Neutral"  # Devuelve Neutral en caso de error

# Función para limpiar y procesar un DataFrame de reseñas
def clean_maps(df: pd.DataFrame):
    # Elimina duplicados basados en usuario, rating y fecha
    df.drop_duplicates(subset=["Usuario", "Rating", "Fecha"], inplace=True)
    
    # Convierte el rating a entero (toma el primer número del texto)
    df["Rating"] = df["Rating"].apply(lambda x: int(x.split()[0]))
    
    # Procesa las fechas relativas a formato estándar
    df["Fecha"] = df["Fecha"].apply(process_date)
    
    # Rellena descripciones vacías con una cadena vacía
    df["Descripcion"] = df["Descripcion"].fillna("")
    
    # Aplica la limpieza de texto a las descripciones
    df["Descripcion_Procesada"] = df["Descripcion"].apply(clean_text)
    
    # Asegura que no haya valores nulos en descripciones procesadas
    df["Descripcion_Procesada"] = df["Descripcion_Procesada"].fillna("")
    df["Descripcion_Procesada"] = df["Descripcion_Procesada"].astype(str)
    
    # Obtiene el sentimiento de las descripciones procesadas
    df["Sentimiento"] = df["Descripcion_Procesada"].apply(get_sentiment)
    
    # Tokeniza las descripciones procesadas
    df["Tokens"] = df["Descripcion_Procesada"].apply(word_tokenize)
    
    # Filtra los tokens eliminando stopwords
    df["Tokens"] = df["Tokens"].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )
    
    # Lematiza los tokens filtrados
    df["Lematizer"] = df["Tokens"].apply(lemmatize_tokens)
    
    return df  # Devuelve el DataFrame procesado
