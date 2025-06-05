Guía de Contribución
===================

¡Gracias por tu interés en contribuir a CapibaraModel! Esta guía te ayudará a entender cómo puedes contribuir al proyecto.

Cómo Contribuir
--------------

1. Fork del Repositorio
~~~~~~~~~~~~~~~~~~~~~~

1. Haz fork del repositorio en GitHub
2. Clona tu fork localmente
3. Configura el upstream remoto

.. code-block:: bash

    git clone https://github.com/tu-usuario/capibara.git
    cd capibara
    git remote add upstream https://github.com/anachroni-io/capibara.git

2. Configuración del Entorno
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Crea un entorno virtual
2. Instala las dependencias de desarrollo
3. Instala el paquete en modo desarrollo

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

3. Creación de una Rama
~~~~~~~~~~~~~~~~~~~~~~

Crea una rama para tu contribución:

.. code-block:: bash

    git checkout -b feature/nombre-de-tu-caracteristica

4. Desarrollo
~~~~~~~~~~~~

1. Sigue las convenciones de código
2. Escribe tests para tu código
3. Actualiza la documentación

5. Envío de Cambios
~~~~~~~~~~~~~~~~~~

1. Haz commit de tus cambios
2. Sube tu rama
3. Crea un Pull Request

.. code-block:: bash

    git add .
    git commit -m "Descripción de tus cambios"
    git push origin feature/nombre-de-tu-caracteristica

Convenciones de Código
---------------------

Estilo de Código
~~~~~~~~~~~~~~~

- Usa black para formatear el código
- Usa isort para ordenar imports
- Sigue PEP 8

.. code-block:: bash

    black .
    isort .

Documentación
~~~~~~~~~~~~

- Documenta todas las funciones y clases
- Usa docstrings en formato Google
- Actualiza la documentación cuando sea necesario

Tests
~~~~~

- Escribe tests para todo el código nuevo
- Mantén la cobertura de tests alta
- Usa pytest para ejecutar los tests

.. code-block:: bash

    pytest tests/
    pytest --cov=capibara tests/

Tipos de Contribuciones
----------------------

Reporte de Errores
~~~~~~~~~~~~~~~~~

1. Usa el template de issue
2. Proporciona información detallada
3. Incluye código para reproducir el error

Nuevas Características
~~~~~~~~~~~~~~~~~~~~~

1. Discute la característica en un issue primero
2. Implementa la característica
3. Añade tests y documentación

Mejoras de Código
~~~~~~~~~~~~~~~~

1. Identifica áreas de mejora
2. Propón soluciones
3. Implementa los cambios

Documentación
~~~~~~~~~~~~

1. Identifica áreas que necesitan documentación
2. Escribe documentación clara y concisa
3. Actualiza ejemplos si es necesario

Proceso de Revisión
------------------

1. Los PRs son revisados por el equipo
2. Se pueden solicitar cambios
3. Una vez aprobado, se mergea al main

Contacto
--------

Para preguntas o dudas:

- Abre un issue en GitHub
- Únete a nuestro Discord
- Envía un email a marco@anachroni.co 