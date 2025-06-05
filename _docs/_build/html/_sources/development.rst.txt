Desarrollo
==========

Esta sección proporciona información para desarrolladores que deseen contribuir al proyecto Capibara.

Configuración del Entorno
------------------------

1. Clonar el repositorio:

.. code-block:: bash

    git clone https://github.com/tu-usuario/capibara.git
    cd capibara

2. Crear un entorno virtual:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate  # Windows

3. Instalar dependencias de desarrollo:

.. code-block:: bash

    pip install -e ".[dev]"

Estructura del Proyecto
----------------------

::

    capibara/
    ├── core/               # Componentes principales
    │   ├── model.py       # Implementación del modelo
    │   ├── config.py      # Configuración
    │   ├── optimizer.py   # Optimizadores
    │   └── tokenizer.py   # Tokenizadores
    ├── utils/             # Utilidades
    │   ├── monitoring.py  # Monitoreo
    │   ├── logging.py     # Logging
    │   └── checkpointing.py # Checkpointing
    ├── tests/             # Tests
    ├── docs/              # Documentación
    └── setup.py           # Configuración del paquete

Guía de Estilo
-------------

- Seguir PEP 8 para el estilo de código
- Usar type hints
- Documentar todas las funciones y clases
- Escribir tests para nuevas funcionalidades

Ejemplo de documentación:

.. code-block:: python

    def train_model(
        model: DynamicCapibaraModel,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int
    ) -> Dict[str, float]:
        """Entrena el modelo con el dataset proporcionado.

        Args:
            model: Modelo a entrenar
            dataset: Dataset de entrenamiento
            num_epochs: Número de épocas
            batch_size: Tamaño del batch

        Returns:
            Dict con métricas de entrenamiento
        """
        pass

Tests
-----

Ejecutar los tests:

.. code-block:: bash

    pytest tests/

Cobertura de tests:

.. code-block:: bash

    pytest --cov=capibara tests/

Documentación
------------

1. Instalar Sphinx:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme

2. Generar documentación:

.. code-block:: bash

    cd docs
    make html

Contribuciones
-------------

1. Crear un fork del repositorio
2. Crear una rama para tu feature
3. Hacer commit de tus cambios
4. Hacer push a la rama
5. Crear un Pull Request

Requisitos para Pull Requests:

- Tests pasando
- Documentación actualizada
- Código siguiendo la guía de estilo
- Descripción clara de los cambios

Release
-------

1. Actualizar versión en `setup.py`
2. Actualizar CHANGELOG.md
3. Crear tag de versión
4. Publicar en PyPI:

.. code-block:: bash

    python setup.py sdist bdist_wheel
    twine upload dist/*

Soporte
-------

- Issues en GitHub
- Discord: #capibara-dev
- Email: dev@capibara.ai 