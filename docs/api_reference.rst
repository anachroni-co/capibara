Referencia de API
================

Submodelos
----------

.. automodule:: sub_models.Byte_TPU
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sub_models.SSM_TPU
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sub_models.deep_dialog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sub_models.aleph_Tilde
   :members:
   :undoc-members:
   :show-inheritance:

Capas (Layers)
--------------

.. automodule:: layers.self_attention
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: layers.quantumL
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: layers.capibara_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: layers.conv1d_block
   :members:
   :undoc-members:
   :show-inheritance:

Capas de Esparsidad
-------------------

.. automodule:: layers.sparsity.sparse_capibara
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: layers.sparsity.mixture_of_rookies
   :members:
   :undoc-members:
   :show-inheritance:

Módulos
-------

.. automodule:: modules.capivision
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: modules.shared_attention
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: modules.contextual_router
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: modules.capibara_quantum_router
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: modules.contextual_activation
   :members:
   :undoc-members:
   :show-inheritance:

Modelo Principal
---------------

.. autoclass:: core.model.DynamicCapibaraModel
   :members:
   :undoc-members:
   :show-inheritance:

Configuración
------------

.. autoclass:: core.config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

Interfaces
---------

.. autoclass:: core.interfaces.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: core.interfaces.BaseLayer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: core.interfaces.ContentFilter
   :members:
   :undoc-members:
   :show-inheritance:

Optimización
-----------

.. autoclass:: core.optimizer.Optimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: core.optimizer.OptimizerType
   :members:
   :undoc-members:
   :show-inheritance:

Tokenización
-----------

.. autoclass:: core.tokenizer.Tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: core.tokenizer.AutoTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

Utilidades
---------

Monitoreo
~~~~~~~~

.. autoclass:: utils.monitoring.RealTimeMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.monitoring.ResourceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.system_info.SystemMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Checkpointing
~~~~~~~~~~~~

.. autoclass:: utils.checkpointing.CheckpointManager
   :members:
   :undoc-members:
   :show-inheritance:

Logging
~~~~~~~

.. autofunction:: utils.logging.setup_logger

.. autofunction:: utils.logging.setup_logging

Validación
~~~~~~~~~

.. autoclass:: utils.validator.Validator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.schemas.ModelSchema
   :members:
   :undoc-members:
   :show-inheritance:

Configuración
~~~~~~~~~~~~

.. autoclass:: utils.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.config_schema.ConfigSchema
   :members:
   :undoc-members:
   :show-inheritance:

Métricas
~~~~~~~~

.. autoclass:: utils.metrics.Metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.metrics.MetricLogger
   :members:
   :undoc-members:
   :show-inheritance:

Procesamiento de Datos
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: utils.data_processing.DataProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.data_processing.DatasetBuilder
   :members:
   :undoc-members:
   :show-inheritance:

Manejo de Errores
~~~~~~~~~~~~~~~~

.. autoclass:: utils.error_handling.CapibaraError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: utils.error_handling.ErrorHandler
   :members:
   :undoc-members:
   :show-inheritance:

Inicialización de Módulos
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: utils.module_initializer.ModuleInitializer
   :members:
   :undoc-members:
   :show-inheritance:

Interfaces
----------

.. automodule:: interfaces.ilayers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: interfaces.imodules
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: interfaces.isub_models
   :members:
   :undoc-members:
   :show-inheritance:

Interfaces Semióticas
===================

IModule
-------

.. autoclass:: capibara.interfaces.imodules.IModule
   :members:
   :undoc-members:
   :show-inheritance:

ISemioModule
-----------

.. autoclass:: capibara.interfaces.imodules.ISemioModule
   :members:
   :undoc-members:
   :show-inheritance:

ILayer
------

.. autoclass:: capibara.interfaces.ilayer.ILayer
   :members:
   :undoc-members:
   :show-inheritance:

ISemioLayer
----------

.. autoclass:: capibara.interfaces.ilayer.ISemioLayer
   :members:
   :undoc-members:
   :show-inheritance:

Módulos Semióticos
================

SemioModule
----------

.. autoclass:: capibara.sub_models.experimental.semio.SemioModule
   :members:
   :undoc-members:
   :show-inheritance:

ContextualActivation
------------------

.. autoclass:: capibara.modules.contextual_activation.ContextualActivation
   :members:
   :undoc-members:
   :show-inheritance:

SharedAttention
-------------

.. autoclass:: capibara.modules.shared_attention.SharedAttention
   :members:
   :undoc-members:
   :show-inheritance:

ContextualRouter
--------------

.. autoclass:: capibara.modules.contextual_router.ContextualRouter
   :members:
   :undoc-members:
   :show-inheritance: 