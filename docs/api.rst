Referencia de API
===============

Esta sección documenta la API pública de CapibaraGPT.

Modelo Principal
--------------

.. autoclass:: capibara_model.CapibaraGPT
   :members:
   :undoc-members:
   :show-inheritance:

Arquitectura
-----------

.. autoclass:: capibara_model.architecture.HybridSSMTransformer
   :members:
   :undoc-members:
   :show-inheritance:

Sistema Semiótico
---------------

.. autoclass:: capibara_model.semio.SemioticSystem
   :members:
   :undoc-members:
   :show-inheritance:

Router Cuántico
-------------

.. autoclass:: capibara_model.quantum.QuantumRouter
   :members:
   :undoc-members:
   :show-inheritance:

Optimizaciones
------------

.. autoclass:: capibara_model.optimization.TPUOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Multimodal
----------------

.. autoclass:: capibara_model.multimodal.MultimodalPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Utilidades
---------

.. autofunction:: capibara_model.utils.preprocess_text
.. autofunction:: capibara_model.utils.postprocess_output
.. autofunction:: capibara_model.utils.load_checkpoint
.. autofunction:: capibara_model.utils.save_checkpoint

Configuración
-----------

.. autoclass:: capibara_model.config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara_model.config.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara_model.config.QuantumConfig
   :members:
   :undoc-members:
   :show-inheritance:

Tipos de Datos
------------

.. autoclass:: capibara_model.types.InputData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara_model.types.OutputData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara_model.types.QuantumState
   :members:
   :undoc-members:
   :show-inheritance:

Excepciones
---------

.. autoexception:: capibara_model.exceptions.ModelError
.. autoexception:: capibara_model.exceptions.QuantumError
.. autoexception:: capibara_model.exceptions.OptimizationError 