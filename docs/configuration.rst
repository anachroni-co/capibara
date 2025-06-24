Configuración
=============

CapibaraGPT-v2 utiliza un **sistema de configuración TOML unificado** que permite optimizaciones automáticas, detección de hardware, y personalización avanzada. Esta sección cubre todas las opciones de configuración disponibles en el sistema completamente funcional.

Sistema de Configuración TOML
------------------------------

El sistema `ModularModelConfig` centraliza todas las configuraciones usando archivos TOML optimizados:

.. code-block:: python

    from capibara.config import ModularModelConfig
    
    # Configuración desde archivo TOML (recomendado)
    config = ModularModelConfig.from_toml(
        "capibara/config/configs_toml/production/tpu_v4.toml"
    )
    
    # Configuración programática personalizada
    config = ModularModelConfig(
        model_name="capibara_enterprise",
        hidden_size=768,
        use_jax_native=True,
        use_vq=True,
        vq_codes=64,
        use_tpu_v4_optimizations=True
    )

Archivos TOML Predefinidos
--------------------------

La configuración está organizada jerárquicamente por propósito:

**Producción (production/)**

.. code-block:: toml

    # capibara/config/configs_toml/production/tpu_v4.toml
    [model]
    model_name = "capibara_tpu_v4"
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    vocab_size = 32000
    max_seq_length = 2048
    
    [jax_config]
    use_jax_native = true
    backend = "capibara_jax"
    fallback_to_standard = true
    
    [tpu_v4_optimizations]
    enabled = true
    mesh_shape = [4, 8]
    use_bfloat16 = true
    async_collective = true
    memory_limit_gb = 32.0
    
    [vector_quantization]
    enabled = true
    codes = 64
    embedding_dim = 768
    adaptive_threshold = 0.5
    use_tpu_optimizations = true
    
    [sparsity]
    enabled = true
    ratio = 0.65
    method = "mixture_of_rookies"
    
    [performance]
    mixed_precision = true
    compile_mode = "aggressive"
    cache_optimized = true

**Desarrollo (development/)**

.. code-block:: toml

    # capibara/config/configs_toml/development/development.toml
    [model]
    model_name = "capibara_dev"
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    
    [jax_config]
    use_jax_native = true
    backend = "auto"
    debug_mode = true
    
    [hardware]
    device = "auto"
    fallback_cpu = true
    memory_limit_gb = 8.0
    
    [testing]
    enable_comprehensive_tests = true
    mock_expensive_ops = true
    fast_startup = true

**Especializadas (specialized/)**

.. code-block:: toml

    # ARM Axion optimizado
    # capibara/config/configs_toml/specialized/arm_axion_inference.toml
    [arm_optimizations]
    enabled = true
    sve_vectorization = true
    neon_fallback = true
    uma_memory_optimized = true
    
    [vector_quantization]
    codes = 64  # Optimizado para ARM
    precision = "int8"
    
    # TPU v6 enterprise
    # capibara/config/configs_toml/specialized/tpu_v6_vq_v33.toml
    [tpu_v6_features]
    enabled = true
    vq_codes = 128  # Enterprise premium
    adaptive_ml = true
    cost_management = true

Parámetros de Configuración Detallados
--------------------------------------

Modelo Base
~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # Identificación
        model_name="capibara_custom",      # Nombre del modelo
        version="v3.0.0",                 # Versión del modelo
        
        # Arquitectura Transformer
        hidden_size=768,                  # Dimensión embeddings (512, 768, 1024, 2048)
        num_layers=12,                    # Capas transformer (6, 12, 24, 36)
        num_heads=12,                     # Cabezas atención (8, 12, 16, 32)
        vocab_size=32000,                 # Tamaño vocabulario
        max_seq_length=2048,              # Longitud máxima secuencia
        
        # Capacidades multimodales
        multimodal_enabled=False,         # Procesamiento multimodal
        vision_encoder_type=None,         # "clip", "vit", None
        audio_encoder_type=None,          # "wav2vec2", "whisper", None
        
        # Características avanzadas
        use_chain_of_thought=True,        # CoT reasoning
        use_dual_process=True,            # Sistema dual de pensamiento
        personality_enabled=True,         # Sistema de personalidad
    )

JAX Nativo y Backend
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # JAX Configuration
        use_jax_native=True,              # Usar JAX nativo capibara.jax
        jax_backend="capibara_jax",       # "capibara_jax", "standard", "auto"
        fallback_to_standard=True,        # Fallback a JAX estándar
        
        # Compilación y Optimización
        compile_mode="default",           # "default", "aggressive", "conservative"
        enable_jit=True,                  # Compilación JIT
        cache_compiled_functions=True,    # Cache de funciones compiladas
        
        # Debug y Desarrollo
        jax_debug_mode=False,             # Modo debug JAX
        trace_execution=False,            # Tracing ejecución
        profile_performance=False,        # Profiling automático
    )

Vector Quantization (VQ)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # VQ Base
        use_vq=True,                      # Activar Vector Quantization
        vq_codes=64,                      # Códigos cuantización (64, 128)
        vq_embedding_dim=768,             # Dimensión embeddings VQ
        
        # VQ Avanzado
        vq_adaptive_threshold=0.5,        # Threshold adaptativo (0.0-1.0)
        vq_diversity_regularization=True, # Regularización diversidad
        vq_commitment_weight=0.25,        # Peso commitment loss
        
        # Optimizaciones Hardware
        vq_use_tpu_optimizations=True,    # Optimizaciones TPU
        vq_use_simd=True,                 # Vectorización SIMD
        vq_cache_codebooks=True,          # Cache codebooks en memoria
        
        # VQ Enterprise (TPU v6)
        vq_enterprise_mode=False,         # 128 códigos + ML adaptativo
        vq_cost_management=True,          # Gestión costos automática
        vq_fallback_strategy="tpu_v4",    # Estrategia fallback
    )

Optimizaciones TPU v4-32
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # TPU v4 Core
        use_tpu_v4_optimizations=True,    # Optimizaciones TPU v4-32
        tpu_mesh_shape=[4, 8],            # Configuración mesh
        tpu_memory_limit_gb=32.0,         # Límite memoria HBM
        
        # Precisión y Rendimiento
        use_bfloat16=True,                # Precisión bfloat16 nativa
        mixed_precision=True,             # Precisión mixta automática
        async_collective=True,            # Operaciones colectivas async
        
        # Sharding y Paralelización
        sharding_strategy="auto",         # "auto", "manual", "disabled"
        data_parallel=True,               # Paralelismo datos
        model_parallel=False,             # Paralelismo modelo
        
        # Memoria y Cache
        gradient_checkpointing=True,      # Checkpointing gradientes
        activation_offloading=True,       # Offloading activaciones
        optimizer_sharding=True,          # Sharding optimizer states
        
        # Kernels Especializados
        use_custom_kernels=True,          # Kernels TPU v4 optimizados
        kernel_categories=[               # Categorías kernels activas
            "linalg", "attention", "scan", 
            "collective", "conv", "fft"
        ],
    )

Sparsity y Quantización
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # Sparsity General
        use_sparse=True,                  # Activar sparsity
        sparsity_ratio=0.65,              # Ratio sparsity (0.0-0.95)
        sparsity_method="mixture_of_rookies",  # "mor", "topk", "magnitude"
        
        # Mixture of Rookies (MoR)
        mor_num_experts=8,                # Número expertos MoR
        mor_top_k=2,                      # Top-K expertos activos
        mor_load_balancing=True,          # Load balancing automático
        
        # BitNet Quantization
        use_bitnet=False,                 # BitNet 1.58b quantización
        bitnet_precision="int8",          # "int8", "int4", "1.58bit"
        
        # Affine Quantization
        use_affine_quantization=True,     # Quantización afín
        affine_bits=8,                    # Bits quantización (4, 8, 16)
        affine_symmetric=True,            # Quantización simétrica
        
        # Sparsity Adaptativa
        adaptive_sparsity=True,           # Sparsity adaptativa
        sparsity_schedule="cosine",       # "linear", "cosine", "exponential"
        min_sparsity=0.3,                 # Sparsity mínima
        max_sparsity=0.8,                 # Sparsity máxima
    )

Hardware y Deployment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # Detección Hardware
        device="auto",                    # "auto", "tpu_v4", "tpu_v6", "gpu", "cpu"
        auto_device_selection=True,       # Selección automática dispositivo
        hardware_fallback=True,           # Fallback automático hardware
        
        # Configuración Multi-GPU
        num_gpus=1,                       # Número GPUs (si GPU disponible)
        gpu_memory_fraction=0.8,          # Fracción memoria GPU
        gpu_memory_growth=True,           # Crecimiento memoria dinámico
        
        # ARM Axion Optimizations
        use_arm_optimizations=False,      # Optimizaciones ARM Axion
        arm_sve_enabled=True,             # SVE vectorization
        arm_neon_fallback=True,           # NEON fallback
        arm_uma_memory=True,              # UMA memory optimizada
        
        # Distribuido
        distributed_training=False,       # Entrenamiento distribuido
        world_size=1,                     # Número procesos distribuidos
        local_rank=0,                     # Rank local proceso
        
        # Batch y Memoria
        batch_size="auto",                # Batch size automático
        max_batch_size=32,                # Batch size máximo
        adaptive_batching=True,           # Batching adaptativo
        memory_optimization_level=2,      # Nivel optimización memoria (0-3)
    )

Módulos Enterprise y Avanzados
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # Model Control Protocol (MCP)
        mcp_enabled=True,                 # Activar MCP
        mcp_port=8080,                    # Puerto MCP server
        mcp_auth_required=True,           # Autenticación requerida
        mcp_rate_limiting=True,           # Rate limiting
        
        # Meta Loop (Elixir/OTP)
        meta_loop_enabled=True,           # Activar Meta Loop
        ethics_guard_enabled=True,        # Control ético activado
        rag_hub_enabled=True,             # RAG Hub activo
        elixir_bridge_port=9090,          # Puerto bridge Python-Elixir
        
        # Monitoring y Observabilidad
        monitoring_enabled=True,          # Monitoreo tiempo real
        metrics_endpoint="/metrics",      # Endpoint métricas
        health_check_interval=30,         # Intervalo health check (seg)
        performance_profiling=True,       # Profiling automático
        
        # Logging Avanzado
        structured_logging=True,          # Logging estructurado
        log_level="INFO",                 # Nivel logging
        audit_logging=True,               # Audit logs
        sensitive_data_masking=True,      # Enmascaramiento datos sensibles
        
        # Security
        api_key_required=False,           # API key requerida
        request_encryption=False,         # Encriptación requests
        output_filtering=True,            # Filtrado outputs
        content_safety_check=True,        # Verificación seguridad contenido
    )

Entrenamiento y Fine-tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModularModelConfig(
        # Entrenamiento Base
        learning_rate=1e-4,               # Learning rate
        weight_decay=0.01,                # Weight decay
        gradient_clip_norm=1.0,           # Gradient clipping
        
        # Consensus Distilling
        use_consensus_distilling=True,    # Consensus distilling automático
        consensus_num_teachers=5,         # Número teachers consensus
        consensus_num_critics=3,          # Número critics consensus
        consensus_temperature=3.0,        # Temperatura distillation
        
        # Optimizador
        optimizer="adamw",                # "adamw", "sgd", "rmsprop"
        beta1=0.9,                        # Beta1 Adam
        beta2=0.999,                      # Beta2 Adam
        epsilon=1e-8,                     # Epsilon optimizer
        
        # Scheduling
        lr_scheduler="cosine",            # "linear", "cosine", "polynomial"
        warmup_steps=1000,                # Pasos warmup
        decay_steps=10000,                # Pasos decay
        
        # Checkpointing
        save_checkpoint_steps=1000,       # Pasos entre checkpoints
        keep_checkpoint_max=5,            # Checkpoints máximos a mantener
        checkpoint_compression=True,      # Compresión checkpoints
        async_checkpointing=True,         # Checkpointing asíncrono
    )

Configuración Avanzada por Variables de Entorno
-----------------------------------------------

.. code-block:: bash

    # JAX y Backend
    export CAPIBARA_USE_JAX_NATIVE=true
    export CAPIBARA_JAX_BACKEND="capibara_jax"
    export CAPIBARA_JAX_DEBUG=false
    
    # Hardware
    export CAPIBARA_DEVICE="auto"
    export CAPIBARA_TPU_MESH="4,8"
    export CAPIBARA_GPU_MEMORY_FRACTION=0.8
    export CAPIBARA_ARM_OPTIMIZATIONS=false
    
    # Vector Quantization
    export CAPIBARA_USE_VQ=true
    export CAPIBARA_VQ_CODES=64
    export CAPIBARA_VQ_ADAPTIVE_THRESHOLD=0.5
    
    # Optimizaciones
    export CAPIBARA_USE_SPARSITY=true
    export CAPIBARA_SPARSITY_RATIO=0.65
    export CAPIBARA_MIXED_PRECISION=true
    export CAPIBARA_COMPILE_MODE="default"
    
    # Enterprise Features
    export CAPIBARA_MCP_ENABLED=true
    export CAPIBARA_META_LOOP_ENABLED=true
    export CAPIBARA_MONITORING_ENABLED=true
    
    # Desarrollo
    export CAPIBARA_LOG_LEVEL="INFO"
    export CAPIBARA_DEBUG_MODE=false
    export CAPIBARA_FAST_STARTUP=false

Configuración Dinámica en Tiempo de Ejecución
----------------------------------------------

.. code-block:: python

    # Actualizar configuración en runtime
    config.update_runtime_settings({
        "batch_size": 16,
        "temperature": 0.8,
        "use_vq": True,
        "vq_adaptive_threshold": 0.6
    })
    
    # Aplicar configuración al modelo
    model.apply_config_updates(config)
    
    # Obtener configuración actual
    current_config = model.get_current_config()
    print(f"VQ enabled: {current_config.use_vq}")
    print(f"TPU optimizations: {current_config.use_tpu_v4_optimizations}")

Validación y Debugging de Configuración
---------------------------------------

.. code-block:: python

    from capibara.config import ConfigValidator
    
    # Validar configuración
    validator = ConfigValidator()
    
    # Validación completa
    validation_result = validator.validate_full_config(config)
    
    if validation_result.is_valid:
        print("✅ Configuración válida")
    else:
        print("❌ Errores en configuración:")
        for error in validation_result.errors:
            print(f"   - {error}")
    
    # Verificar compatibilidad hardware
    hardware_check = validator.check_hardware_compatibility(config)
    print(f"Hardware compatible: {hardware_check.compatible}")
    print(f"Recomendaciones: {hardware_check.recommendations}")

Ejemplos de Configuración Específica
------------------------------------

**Configuración para Investigación**

.. code-block:: python

    research_config = ModularModelConfig(
        model_name="capibara_research",
        hidden_size=1024,
        num_layers=24,
        use_vq=True,
        vq_codes=128,  # Máxima capacidad VQ
        use_tpu_v4_optimizations=True,
        sparsity_ratio=0.5,  # Sparsity moderada
        monitoring_enabled=True,
        performance_profiling=True
    )

**Configuración para Producción Cost-Effective**

.. code-block:: python

    production_config = ModularModelConfig(
        model_name="capibara_production",
        hidden_size=768,
        num_layers=12,
        use_vq=True,
        vq_codes=64,  # Optimizado costo-rendimiento
        use_arm_optimizations=True,
        sparsity_ratio=0.65,  # Alta sparsity
        batch_size=32,
        mixed_precision=True
    )

**Configuración Enterprise Premium**

.. code-block:: python

    enterprise_config = ModularModelConfig(
        model_name="capibara_enterprise",
        hidden_size=2048,
        num_layers=36,
        use_vq=True,
        vq_codes=128,
        vq_enterprise_mode=True,
        use_tpu_v4_optimizations=True,
        mcp_enabled=True,
        meta_loop_enabled=True,
        monitoring_enabled=True,
        security_level="high"
    )

Migración de Configuraciones Anteriores
---------------------------------------

.. code-block:: python

    from capibara.config import ConfigMigrator
    
    # Migrar configuración v2.x a v3.0
    migrator = ConfigMigrator()
    
    # Desde configuración legacy
    legacy_config = {
        "model_size": "600m",
        "use_quantum": True,  # Será migrado a use_vq
        "device": "tpu_v4"
    }
    
    new_config = migrator.migrate_from_legacy(legacy_config)
    print(f"Migrated: {new_config.model_name}")
    print(f"VQ enabled: {new_config.use_vq}")  # quantum -> vq

Recursos de Configuración
-------------------------

- **Archivos TOML**: ``capibara/config/configs_toml/``
- **Validación**: ``capibara.config.ConfigValidator``
- **Migración**: ``capibara.config.ConfigMigrator``
- **Documentación**: :doc:`api/config_api`
- **Ejemplos**: ``examples/configuration/`` 