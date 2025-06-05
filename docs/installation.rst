Instalación
==========

Requisitos del Sistema
--------------------

* Python 3.8+
* TPU v4-32 (32 chips)
* 32GB HBM por chip
* CUDA 11.7+ (para desarrollo local)
* 256GB RAM (mínimo)

Dependencias
-----------

.. code-block:: bash

    # Instalar dependencias base
    pip install -r requirements.txt

    # Instalar JAX con soporte TPU
    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

    # Instalar dependencias adicionales
    pip install -r requirements-dev.txt

Configuración del Entorno
-----------------------

Variables de Entorno
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Configurar variables de entorno
    export TPU_NAME="your-tpu-name"
    export TPU_ZONE="us-central1-a"
    export PROJECT_ID="your-project-id"
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

Configuración TPU
~~~~~~~~~~~~~~

.. code-block:: python

    from capibara.config.tpu_config import TPUConfig

    # Configuración básica
    tpu_config = TPUConfig(
        num_chips=32,
        topology='4x8',
        memory_per_chip=32 * 1024**3,
        flops_per_chip=275e12
    )

    # Configuración avanzada
    tpu_config.set_mixed_precision(True)
    tpu_config.set_memory_fraction(0.8)
    tpu_config.set_gradient_checkpointing(True)

Instalación desde Fuente
---------------------

.. code-block:: bash

    # Clonar repositorio
    git clone https://github.com/your-org/capibara-gpt.git
    cd capibara-gpt

    # Instalar en modo desarrollo
    pip install -e .

Verificación de la Instalación
---------------------------

.. code-block:: python

    import capibara
    from capibara.config.tpu_config import TPUConfig

    # Verificar instalación
    print(f"Versión: {capibara.__version__}")

    # Verificar configuración TPU
    tpu_config = TPUConfig()
    print(f"TPU disponible: {tpu_config.is_tpu_available()}")
    print(f"Configuración: {tpu_config.get_config()}")

Solución de Problemas
------------------

Problemas Comunes
~~~~~~~~~~~~~~

1. Error de memoria insuficiente:
   * Reducir batch_size
   * Habilitar gradient checkpointing
   * Ajustar memory_fraction

2. Errores de sharding:
   * Verificar topología TPU
   * Ajustar PartitionSpec
   * Revisar dimensiones del tensor

3. Problemas de precisión mixta:
   * Verificar compatibilidad de operaciones
   * Ajustar dtype de parámetros
   * Revisar configuración de JAX

Soporte
------

* GitHub Issues: https://github.com/your-org/capibara-gpt/issues
* Documentación: https://capibara-gpt.readthedocs.io
* Email: support@capibara-gpt.org 