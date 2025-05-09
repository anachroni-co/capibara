Guía de Contribución
=================

¡Gracias por tu interés en contribuir a CapibaraGPT! Esta guía te ayudará a participar en el desarrollo del proyecto.

Código de Conducta
---------------

Por favor, lee y sigue nuestro `Código de Conducta <CODE_OF_CONDUCT.md>`_ antes de contribuir.

Proceso de Contribución
--------------------

1. Fork del Repositorio
~~~~~~~~~~~~~~~~~~~~

* Haz fork del repositorio en GitHub
* Clona tu fork localmente
* Configura el upstream remoto

.. code-block:: bash

   git clone https://github.com/tu-usuario/capibara-gpt.git
   cd capibara-gpt
   git remote add upstream https://github.com/capibara-ai/capibara-gpt.git

2. Configuración del Entorno
~~~~~~~~~~~~~~~~~~~~~~~~~

* Crea un entorno virtual
* Instala dependencias de desarrollo

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -e ".[dev]"

3. Desarrollo
~~~~~~~~~~

* Crea una rama para tu feature
* Implementa tus cambios
* Sigue las guías de estilo

.. code-block:: bash

   git checkout -b feature/nueva-caracteristica
   # Realiza tus cambios
   git add .
   git commit -m "feat: añade nueva característica"

4. Pruebas
~~~~~~~~

* Ejecuta las pruebas unitarias
* Verifica la cobertura
* Asegura que todas las pruebas pasen

.. code-block:: bash

   pytest
   pytest --cov=capibara_model
   pytest --cov=capibara_model --cov-report=html

5. Documentación
~~~~~~~~~~~~

* Actualiza la documentación
* Verifica que se construya correctamente

.. code-block:: bash

   cd docs
   make html
   # Verifica _build/html/index.html

6. Pull Request
~~~~~~~~~~~~

* Actualiza tu fork
* Crea un pull request
* Describe tus cambios

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main
   git push origin feature/nueva-caracteristica

Guías de Estilo
------------

Código Python
~~~~~~~~~~~

* Sigue PEP 8
* Usa type hints
* Documenta con docstrings
* Mantén las líneas bajo 88 caracteres

Ejemplo:

.. code-block:: python

   from typing import List, Optional

   def procesar_texto(
       texto: str,
       max_length: Optional[int] = None
   ) -> List[str]:
       """Procesa el texto de entrada.

       Args:
           texto: Texto a procesar
           max_length: Longitud máxima opcional

       Returns:
           Lista de tokens procesados
       """
       # Implementación
       pass

Documentación
~~~~~~~~~~~

* Usa reStructuredText
* Incluye ejemplos
* Mantén la documentación actualizada

Ejemplo:

.. code-block:: rst

   Función de Procesamiento
   ----------------------

   Esta función procesa el texto de entrada.

   .. code-block:: python

       from capibara_model import procesar_texto

       resultado = procesar_texto("ejemplo")
       print(resultado)

Pruebas
~~~~~~

* Escribe pruebas unitarias
* Incluye casos de borde
* Verifica la cobertura

Ejemplo:

.. code-block:: python

   def test_procesar_texto():
       # Caso básico
       assert procesar_texto("test") == ["t", "e", "s", "t"]
       
       # Caso vacío
       assert procesar_texto("") == []
       
       # Caso con longitud máxima
       assert len(procesar_texto("test", max_length=2)) == 2

Estructura del Proyecto
--------------------

::

   capibara-gpt/
   ├── capibara_model/
   │   ├── __init__.py
   │   ├── core/
   │   ├── models/
   │   └── utils/
   ├── tests/
   │   ├── __init__.py
   │   ├── test_core/
   │   └── test_models/
   ├── docs/
   │   ├── conf.py
   │   └── *.rst
   ├── setup.py
   └── README.md

Flujo de Trabajo
--------------

1. **Planificación**
   * Revisa los issues
   * Propón cambios
   * Discute implementación

2. **Desarrollo**
   * Implementa cambios
   * Sigue guías de estilo
   * Escribe pruebas

3. **Revisión**
   * Actualiza documentación
   * Ejecuta pruebas
   * Prepara PR

4. **Integración**
   * Responde a feedback
   * Actualiza cambios
   * Espera merge

Recursos
-------

* `Documentación <https://capibara-gpt.readthedocs.io/>`_
* `Issues <https://github.com/capibara-ai/capibara-gpt/issues>`_
* `Discusiones <https://github.com/capibara-ai/capibara-gpt/discussions>`_

Contacto
-------

* Email: contribuciones@capibara.ai
* Discord: https://discord.gg/capibara-gpt
* Twitter: @capibara_gpt 