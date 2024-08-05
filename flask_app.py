import sys
import os

# Añadir el directorio que contiene tu aplicación al path
sys.path.insert(0, '/home/AlbaMRM/PythonAnywhere_TC')

# Configurar el entorno
os.environ['FLASK_ENV'] = 'production'

# Importar la aplicación Flask desde tu archivo principal
from app_model import app

# Crear la aplicación
application = app