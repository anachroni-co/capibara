Ejemplos
========

Esta sección proporciona ejemplos prácticos de cómo usar Capibara.

Entrenamiento Básico
-------------------

Ejemplo de entrenamiento básico con un dataset personalizado:

.. code-block:: python

    from capibara import DynamicCapibaraModel, ModelConfig
    from capibara.utils.monitoring import RealTimeMonitor
    from capibara.utils.checkpointing import CheckpointManager

    # Configuración del modelo
    config = ModelConfig(
        model_type="mamba",
        hidden_size=768,
        num_layers=12,
        num_heads=12
    )

    # Inicialización
    model = DynamicCapibaraModel(config)
    monitor = RealTimeMonitor()
    checkpoint_manager = CheckpointManager()

    # Entrenamiento
    model.train(
        train_dataset,
        eval_dataset,
        num_epochs=10,
        batch_size=32,
        monitor=monitor,
        checkpoint_manager=checkpoint_manager
    )

Fine-tuning
----------

Ejemplo de fine-tuning con un dataset específico:

.. code-block:: python

    from capibara import DynamicCapibaraModel
    from capibara.utils.checkpointing import CheckpointManager

    # Cargar modelo pre-entrenado
    checkpoint_manager = CheckpointManager()
    model = checkpoint_manager.load_checkpoint(
        DynamicCapibaraModel(config),
        "ruta/al/modelo/pre-entrenado"
    )

    # Fine-tuning
    model.fine_tune(
        fine_tuning_dataset,
        num_epochs=5,
        learning_rate=1e-5,
        batch_size=16
    )

Inferencia
----------

Ejemplo de generación de texto:

.. code-block:: python

    from capibara import DynamicCapibaraModel

    # Inicializar modelo
    model = DynamicCapibaraModel(config)

    # Generar texto
    output = model.generate(
        "¿Cuál es la capital de Francia?",
        max_length=100,
        temperature=0.7,
        top_p=0.9
    )

    print(output)

Optimización para TPU
--------------------

Ejemplo de configuración para TPU:

.. code-block:: python

    from capibara import ModelConfig

    # Configuración específica para TPU
    config = ModelConfig(
        model_type="mamba",
        hidden_size=768,
        num_layers=12,
        tpu={
            "use_tpu": True,
            "num_cores": 8,
            "dtype": "bfloat16"
        }
    )

Optimización para GPU
--------------------

Ejemplo de configuración para GPU:

.. code-block:: python

    from capibara import ModelConfig

    # Configuración específica para GPU
    config = ModelConfig(
        model_type="mamba",
        hidden_size=768,
        num_layers=12,
        gpu={
            "use_mixed_precision": True,
            "gradient_accumulation_steps": 4
        }
    )

Monitoreo Avanzado
-----------------

Ejemplo de monitoreo detallado:

.. code-block:: python

    from capibara.utils.monitoring import (
        RealTimeMonitor,
        ResourceMonitor,
        SystemMonitor
    )

    # Configurar monitores
    realtime_monitor = RealTimeMonitor()
    resource_monitor = ResourceMonitor()
    system_monitor = SystemMonitor()

    # Monitorear durante el entrenamiento
    while training:
        metrics = model.get_metrics()
        realtime_monitor.log_metrics(metrics)
        
        if config.tpu.use_tpu:
            resource_monitor.log_tpu_metrics()
        
        system_info = system_monitor.get_system_info()
        realtime_monitor.log_system_info(system_info)

Checkpointing Avanzado
---------------------

Ejemplo de manejo de checkpoints:

.. code-block:: python

    from capibara.utils.checkpointing import CheckpointManager

    checkpoint_manager = CheckpointManager()

    # Guardar checkpoint con metadatos
    checkpoint_manager.save_checkpoint(
        model,
        "ruta/checkpoint",
        metadata={
            "epoch": current_epoch,
            "loss": current_loss,
            "accuracy": current_accuracy
        }
    )

    # Cargar checkpoint específico
    model = checkpoint_manager.load_checkpoint(
        model,
        "ruta/checkpoint",
        checkpoint_id="mejor_modelo"
    )

Procesamiento de Datos
---------------------

Ejemplo de preparación de datos:

.. code-block:: python

    from capibara.utils.data_processing import (
        DataProcessor,
        DatasetBuilder
    )

    # Procesar datos
    processor = DataProcessor()
    processed_data = processor.process(
        raw_data,
        tokenizer=tokenizer,
        max_length=512
    )

    # Construir dataset
    builder = DatasetBuilder()
    dataset = builder.build(
        processed_data,
        batch_size=32,
        shuffle=True
    )

Uso del Módulo Semiótico
=======================

Ejemplo Básico
------------

.. code-block:: python

    from capibara.sub_models.experimental.semio import SemioModule
    import jax.numpy as jnp
    
    # Configuración
    config = {
        'hidden_size': 256,
        'num_heads': 8,
        'dropout_rate': 0.1
    }
    
    # Inicializar módulo
    semio = SemioModule(**config)
    
    # Procesar entrada
    x = jnp.random.normal(size=(32, 128, 256))  # (batch, seq_len, hidden)
    output = semio(x)
    
    # Acceder a interpretaciones
    literal = output['literal_interpretation']
    cultural = output['cultural_interpretation']
    symbolic = output['symbolic_interpretation']
    
    # Obtener métricas
    confidence = output['confidence']
    diversity = output['diversity']

Integración con Atención
----------------------

.. code-block:: python

    from capibara.modules.shared_attention import SharedAttention
    from capibara.interfaces.ilayer import ISemioLayer
    
    class SemioAttention(SharedAttention, ISemioLayer):
        def __init__(self, config):
            super().__init__(config)
            
        def get_confidence_scores(self):
            return self.module_state['metrics']['semio']['confidence']
            
        def get_interpretation_weights(self):
            return self.module_state['metrics']['semio']['interpretation_weights']
            
        def get_attention_metrics(self):
            return self.module_state['metrics']['attention']
    
    # Uso
    attention = SemioAttention(config)
    output = attention(x, context)
    
    # Acceder a métricas
    confidence = attention.get_confidence_scores()
    weights = attention.get_interpretation_weights()
    attn_metrics = attention.get_attention_metrics()

Enrutamiento Contextual
---------------------

.. code-block:: python

    from capibara.modules.contextual_router import ContextualRouter
    
    # Configuración
    router_config = {
        'hidden_size': 256,
        'num_heads': 8,
        'dropout_rate': 0.1,
        'semio_enabled': True,
        'semio_threshold': 0.7
    }
    
    # Inicializar router
    router = ContextualRouter(router_config)
    
    # Procesar entrada
    x = jnp.random.normal(size=(32, 128, 256))
    context = jnp.random.normal(size=(32, 128, 256))
    
    output = router(x, context)
    
    # Acceder a resultados
    routed = output['output']
    interpretations = output['interpretations']
    metrics = output['metrics']

Integración con Meta-Loop
------------------------

.. code-block:: python

    from capibara.core.meta_loop import MetaLoop
    
    # Configurar meta-loop
    meta_config = {
        'hidden_size': 256,
        'num_heads': 8
    }
    
    meta_loop = MetaLoop(**meta_config)
    
    # Validar salida del módulo semiótico
    validation = meta_loop(
        x=output['output'],
        context=context,
        metrics=output['metrics']
    )
    
    # Ajustar si es necesario
    if validation['needs_adjustment']:
        output['output'] *= validation['adjustment_factor'] 