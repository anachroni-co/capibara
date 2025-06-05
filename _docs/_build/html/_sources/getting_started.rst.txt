Getting Started
==============

Instalación
-----------

Capibara puede ser instalado usando pip:

.. code-block:: bash

    pip install capibara

Para soporte de TPU:

.. code-block:: bash

    pip install capibara[tpu]

Para soporte de GPU:

.. code-block:: bash

    pip install capibara[gpu]

Uso Básico
----------

Aquí hay un ejemplo básico de cómo usar Capibara:

.. code-block:: python

    from capibara import DynamicCapibaraModel, ModelConfig

    # Configuración del modelo
    config = ModelConfig(
        model_type="mamba",
        hidden_size=768,
        num_layers=12
    )

    # Inicialización del modelo
    model = DynamicCapibaraModel(config)

    # Generación de texto
    output = model.generate("Tu texto de entrada aquí")

Módulo Semiótico
--------------

El módulo semiótico es un componente fundamental que permite el análisis e interpretación de contenido a múltiples niveles. Para comenzar a usarlo:

1. **Instalación**
   ```bash
   pip install capibara[semio]
   ```

2. **Uso Básico**
   ```python
   from capibara.sub_models.experimental.semio import SemioModule
   
   # Configuración básica
   config = {
       'hidden_size': 256,
       'num_heads': 8,
       'dropout_rate': 0.1
   }
   
   # Inicializar módulo
   semio = SemioModule(**config)
   
   # Procesar entrada
   output = semio(x)
   ```

3. **Integración con Otros Módulos**
   ```python
   from capibara.modules.shared_attention import SharedAttention
   
   # Crear capa de atención con análisis semiótico
   attention = SharedAttention(config)
   output = attention(x, context)
   ```

Para más detalles sobre el módulo semiótico, consulta la :doc:`documentación completa <semio>`.

Requisitos del Sistema
---------------------

- Python >= 3.9
- JAX >= 0.4.23
- Flax >= 0.8.2
- TensorFlow >= 2.16.1 (opcional, para TPU)
- PyTorch >= 2.2.2 (opcional, para GPU)

Para TPU:
    - Acceso a Google Cloud TPU
    - libtpu instalado

Para GPU:
    - NVIDIA GPU con CUDA 11.8+
    - Drivers NVIDIA actualizados

Siguientes Pasos
---------------

- Ver la :doc:`user_guide` para más detalles sobre el uso del modelo
- Explorar los :doc:`examples` para ver casos de uso avanzados
- Consultar la :doc:`api_reference` para la documentación completa de la API 