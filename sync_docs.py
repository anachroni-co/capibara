#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path

def sync_documentation() -> None:
    """
    Sincroniza la documentación entre el repositorio principal y el repositorio de documentación.
    """
    # Configuración de rutas
    current_dir = Path(__file__).parent.parent
    docs_repo_path = os.getenv('DOCS_REPO_PATH', '')
    
    if not docs_repo_path:
        print("Error: Por favor, configura la variable de entorno DOCS_REPO_PATH")
        print("Ejemplo: export DOCS_REPO_PATH=/ruta/al/repo/docs")
        return
    
    # Verificar que el repositorio de docs existe
    if not os.path.exists(docs_repo_path):
        print(f"Error: El directorio {docs_repo_path} no existe")
        return
    
    # Construir la documentación
    try:
        subprocess.run(['sphinx-build', '-b', 'html', 'docs', 'docs/_build/html'], check=True)
        print("Documentación construida exitosamente")
    except subprocess.CalledProcessError as e:
        print(f"Error al construir la documentación: {e}")
        return
    
    # Copiar la documentación construida al repositorio de docs
    try:
        # Crear directorio de destino si no existe
        os.makedirs(os.path.join(docs_repo_path, 'docs'), exist_ok=True)
        
        # Copiar archivos
        shutil.copytree(
            'docs/_build/html',
            os.path.join(docs_repo_path, 'docs'),
            dirs_exist_ok=True
        )
        print("Documentación copiada exitosamente")
        
        # Actualizar el repositorio de docs
        os.chdir(docs_repo_path)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Actualización automática de documentación'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Documentación actualizada en el repositorio remoto")
        
    except Exception as e:
        print(f"Error al sincronizar la documentación: {e}")
        return

if __name__ == '__main__':
    sync_documentation() 