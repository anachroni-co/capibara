"""
Capibara - Modelo de lenguaje basado en State Space Models (SSM)
"""

__version__ = "2.1.5"
__author__ = "Anachroni s.coop"

# Importaciones principales
try:
    from .model import CapibaraModel
    from .config import CapibaraConfig

    __all__ = [
        "CapibaraModel",
        "CapibaraConfig",
    ]
except ImportError:
    # Para cuando se está construyendo la documentación
    pass 