Welcome to CapibaraGPT v2.0's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   installation
   modules/quantum
   modules/semiotic
   modules/tpu
   api_reference
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

CapibaraGPT v2.0 es un modelo de lenguaje avanzado optimizado para TPU v4-32, que integra capacidades cuánticas y semióticas para un procesamiento más eficiente y preciso.

Características Principales
-------------------------

* 🚀 Optimizaciones TPU v4-32
* 🧮 Módulos Cuánticos
* 🎨 Análisis Semiótico
* 🔄 Pipeline Asíncrono
* 📊 Métricas Avanzadas

Instalación Rápida
-----------------

.. code-block:: bash

   pip install -r requirements.txt
   export TPU_CONFIG="capibara/config/tpu_config.py"

Uso Básico
---------

.. code-block:: python

   from capibara.config.tpu_config import TPUConfig
   from capibara.quantum.vqbit.quantum import QuantumSubmodel
   from capibara.sub_models.semiotic.mnemosyne_semio_module import MnemosyneSemioModule

   # Configuración TPU
   tpu_config = TPUConfig(
       num_chips=32,
       topology='4x8',
       memory_per_chip=32 * 1024**3,
       flops_per_chip=275e12
   )

   # Inicializar módulos
   quantum_model = QuantumSubmodel(config=quantum_config)
   semio_module = MnemosyneSemioModule()

Documentación Detallada
----------------------

* :doc:`getting_started` - Guía de inicio rápido
* :doc:`installation` - Instrucciones de instalación detalladas
* :doc:`modules/quantum` - Documentación del módulo cuántico
* :doc:`modules/semiotic` - Documentación del módulo semiótico
* :doc:`modules/tpu` - Configuración y optimizaciones TPU
* :doc:`api_reference` - Referencia completa de la API
* :doc:`contributing` - Guías para contribuir
* :doc:`changelog` - Registro de cambios 