Guía del Usuario
===============

Esta guía proporciona información detallada sobre cómo usar Capibara para diferentes tareas.

Configuración del Modelo
-----------------------

Capibara ofrece varias opciones de configuración a través de la clase ``ModelConfig``:

.. code-block:: python

    from capibara import ModelConfig

    config = ModelConfig(
        model_type="mamba",  # o "transformer", "hybrid"
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        activation="gelu",
        use_bias=True,
        layer_norm_eps=1e-6,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        gradient_checkpointing=True,
        use_cache=True
    )

Entrenamiento
------------

Para entrenar el modelo:

.. code-block:: python

    from capibara import DynamicCapibaraModel, TrainingPipeline
    from capibara.utils.monitoring import RealTimeMonitor

    # Inicializar modelo y pipeline
    model = DynamicCapibaraModel(config)
    pipeline = TrainingPipeline(model)

    # Configurar monitoreo
    monitor = RealTimeMonitor()

    # Entrenar
    pipeline.train(
        train_dataset,
        eval_dataset,
        num_epochs=10,
        batch_size=32,
        monitor=monitor
    )

Inferencia
----------

Para usar el modelo en inferencia:

.. code-block:: python

    from capibara import DynamicCapibaraModel
    from capibara.utils.checkpointing import CheckpointManager

    # Cargar modelo y checkpoint
    model = DynamicCapibaraModel(config)
    checkpoint_manager = CheckpointManager()
    model = checkpoint_manager.load_checkpoint(model, "ruta/al/checkpoint")

    # Generar texto
    output = model.generate(
        "Tu texto de entrada aquí",
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )

Optimización para TPU/GPU
------------------------

Para optimizar el rendimiento:

.. code-block:: python

    # Configuración para TPU
    config.tpu.use_tpu = True
    config.tpu.num_cores = 8
    config.tpu.dtype = "bfloat16"

    # Configuración para GPU
    config.use_mixed_precision = True
    config.gradient_accumulation_steps = 4

Monitoreo y Logging
------------------

Capibara proporciona herramientas de monitoreo:

.. code-block:: python

    from capibara.utils.monitoring import (
        RealTimeMonitor,
        ResourceMonitor,
        SystemMonitor
    )

    # Monitoreo en tiempo real
    realtime_monitor = RealTimeMonitor()
    realtime_monitor.log_metrics(metrics)

    # Monitoreo de recursos
    resource_monitor = ResourceMonitor()
    resource_monitor.log_tpu_metrics()

    # Monitoreo del sistema
    system_monitor = SystemMonitor()
    system_info = system_monitor.get_system_info()

Checkpointing
------------

Para guardar y cargar checkpoints:

.. code-block:: python

    from capibara.utils.checkpointing import CheckpointManager

    # Guardar checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save_checkpoint(model, "ruta/checkpoint")

    # Cargar checkpoint
    model = checkpoint_manager.load_checkpoint(model, "ruta/checkpoint")

Siguientes Pasos
---------------

- Ver los :doc:`examples` para más casos de uso
- Consultar la :doc:`api_reference` para detalles técnicos
- Explorar :doc:`development` para contribuir al proyecto 