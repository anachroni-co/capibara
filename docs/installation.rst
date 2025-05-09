Instalación
==========

Esta guía te ayudará a instalar y configurar CapibaraGPT en tu sistema.

Requisitos del Sistema
--------------------

* Python 3.8 o superior
* CUDA 11.7+ (para GPU)
* TPU v4 (opcional, para optimizaciones TPU)
* 16GB+ RAM
* 50GB+ espacio en disco

Instalación Básica
----------------

Instalación con pip
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install capibara-model

Instalación desde fuente
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/capibara-ai/capibara-gpt.git
   cd capibara-gpt
   pip install -e .

Dependencias Opcionales
---------------------

Para funcionalidades específicas, puedes instalar dependencias adicionales:

.. code-block:: bash

   # Para soporte TPU
   pip install capibara-model[tpu]

   # Para desarrollo
   pip install capibara-model[dev]

   # Para documentación
   pip install capibara-model[docs]

Configuración
-----------

Variables de Entorno
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Configuración de GPU
   export CUDA_VISIBLE_DEVICES=0

   # Configuración de TPU
   export TPU_NAME=your-tpu-name

   # Configuración de memoria
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

Verificación de la Instalación
----------------------------

Para verificar que la instalación fue exitosa:

.. code-block:: python

   from capibara_model import CapibaraGPT

   # Inicializar el modelo
   model = CapibaraGPT()

   # Realizar una prueba simple
   response = model.generate("Hola, ¿cómo estás?")
   print(response)

Solución de Problemas Comunes
---------------------------

Problemas de GPU
~~~~~~~~~~~~~~

* Verificar instalación de CUDA
* Comprobar drivers actualizados
* Verificar compatibilidad de versiones

Problemas de TPU
~~~~~~~~~~~~~~

* Verificar configuración de TPU
* Comprobar permisos de acceso
* Verificar versión de JAX

Problemas de Memoria
~~~~~~~~~~~~~~~~~~

* Ajustar XLA_PYTHON_CLIENT_MEM_FRACTION
* Reducir batch size
* Usar gradiente checkpointing

Soporte
------

Para obtener ayuda adicional:

* GitHub Issues: https://github.com/capibara-ai/capibara-gpt/issues
* Documentación: https://capibara-gpt.readthedocs.io/
* Email: soporte@capibara.ai 