Instalación CapibaraGPT-v2
==========================

**CapibaraGPT-v2 v3.0.0** - Guía completa de instalación para el sistema **100% funcional** con JAX nativo, Vector Quantization, y optimizaciones TPU v4-32.

🏆 **Estado**: **COMPLETAMENTE OPERATIVO - INSTALACIÓN VERIFICADA**

Requisitos del Sistema
---------------------

**Hardware Recomendado**
- **TPU v4-32**: Rendimiento óptimo con optimizaciones nativas
- **ARM Axion C4A**: Cost-effective para inferencia con 64 códigos VQ
- **GPU**: Fallback optimizado (CUDA 11.8+, 8GB+ VRAM recomendado)
- **CPU**: Fallback funcional (16GB+ RAM recomendado)

**Software**
- **Python**: 3.9, 3.10, o 3.11 (recomendado 3.10)
- **Sistema Operativo**: Linux (recomendado), Windows 10/11, macOS
- **Git**: Para clonar repositorio

Instalación Desde Repositorio
-----------------------------

**1. Clonar Repositorio**

.. code-block:: bash

    # Clonar repositorio oficial
    git clone https://github.com/user/CapibaraGPT-v2.git
    cd CapibaraGPT-v2
    
    # Cambiar a branch principal (si no está en main)
    git checkout main

**2. Crear Entorno Virtual**

.. code-block:: bash

    # Crear entorno virtual Python
    python -m venv capibara_env
    
    # Activar entorno virtual
    # Linux/macOS:
    source capibara_env/bin/activate
    
    # Windows:
    capibara_env\Scripts\activate

**3. Instalar Dependencias**

.. code-block:: bash

    # Actualizar pip
    pip install --upgrade pip
    
    # Instalar dependencias principales
    pip install -r requirements.txt
    
    # Instalación en modo desarrollo (recomendado)
    pip install -e .

Verificación de Instalación
---------------------------

**Test de Importación Principal**

.. code-block:: python

    # Verificar importación principal (debe funcionar sin errores)
    import capibara
    print("✅ CapibaraGPT-v2 importado correctamente")
    
    # Verificar módulos principales
    from capibara.core import ModularCapibaraModel
    from capibara.config import ModularModelConfig
    from capibara.vq.vqbit import VQbitLayer
    
    print("✅ Módulos principales funcionales")
    print(f"📦 Versión: {capibara.__version__}")

**Test JAX Nativo**

.. code-block:: python

    # Verificar JAX nativo funcional
    try:
        import capibara.jax as jax
        import capibara.jax.numpy as jnp
        
        # Test operación básica
        x = jnp.array([1, 2, 3, 4])
        result = jnp.square(x)
        
        print("✅ JAX nativo funcionando correctamente")
        print(f"🔧 Backend JAX: {jax.__name__}")
        
    except Exception as e:
        print(f"⚠️ JAX nativo no disponible, usando fallback: {e}")
        import jax  # Fallback automático
        print("✅ JAX estándar funcionando como fallback")

**Test Sistema VQ**

.. code-block:: python

    # Verificar Vector Quantization
    from capibara.vq.vqbit import VQbitLayer
    import capibara.jax.numpy as jnp
    
    try:
        # Crear VQbit Layer básico
        vqbit = VQbitLayer(
            codebook_size=64,
            embedding_dim=768
        )
        
        # Test forward pass
        test_input = jnp.ones((2, 10, 768))
        quantized, indices, metrics = vqbit(test_input)
        
        print("✅ Sistema VQ funcionando correctamente")
        print(f"🎯 Codebook size: {vqbit.codebook_size}")
        print(f"📊 Compression ratio: {metrics['compression_ratio']:.2f}")
        
    except Exception as e:
        print(f"❌ Error en sistema VQ: {e}")

**Script de Verificación Automática**

.. code-block:: python

    # Ejecutar verificación completa del sistema
    from capibara.utils.diagnostics import SystemDiagnostics
    
    diagnostics = SystemDiagnostics()
    
    # Verificación completa
    print("🔍 Ejecutando verificación completa del sistema...")
    health_report = diagnostics.run_comprehensive_check()
    
    print("\n📊 Reporte de Estado:")
    for component, status in health_report.items():
        icon = "✅" if status['healthy'] else "❌"
        print(f"   {icon} {component}: {status['message']}")
    
    # Verificación específica imports
    import_check = diagnostics.check_all_imports()
    print(f"\n📦 Imports: {import_check['successful']} OK, {import_check['failed']} errores")
    
    if import_check['failed'] > 0:
        print("⚠️ Errores de importación encontrados:")
        for error in import_check['error_details']:
            print(f"   - {error}")

Configuración Inicial
--------------------

**1. Configuración Básica**

.. code-block:: python

    from capibara.config import ModularModelConfig
    
    # Configuración desde TOML (recomendado)
    config = ModularModelConfig.from_toml(
        "capibara/config/configs_toml/development/development.toml"
    )
    
    # Configuración programática básica
    config = ModularModelConfig(
        model_name="capibara_dev",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        
        # JAX nativo
        use_jax_native=True,
        
        # Vector Quantization
        use_vq=True,
        vq_codes=64,
        
        # Hardware auto-detection
        device="auto"
    )

**2. Crear Modelo Inicial**

.. code-block:: python

    from capibara.core import ModularCapibaraModel
    
    # Crear modelo con configuración
    model = ModularCapibaraModel(config)
    
    # Inicializar modelo
    model.initialize()
    
    print(f"✅ Modelo creado: {model.config.model_name}")
    print(f"🔧 JAX backend: {model.jax_backend}")
    print(f"🎯 VQ enabled: {model.config.use_vq}")
    print(f"⚡ Hardware: {model.detected_hardware}")

**3. Test de Generación**

.. code-block:: python

    # Test básico de generación
    try:
        response = model.generate(
            "Hola, soy CapibaraGPT-v2",
            max_length=50,
            temperature=0.7
        )
        
        print("✅ Generación funcionando correctamente")
        print(f"🤖 Respuesta: {response}")
        
    except Exception as e:
        print(f"⚠️ Error en generación: {e}")
        print("💡 Esto es normal en primera instalación - el modelo necesita entrenamiento")

Instalación para Diferentes Plataformas
---------------------------------------

**TPU v4-32 (Google Cloud)**

.. code-block:: bash

    # Instalación específica TPU
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    
    # Verificar TPU disponible
    python -c "import jax; print('TPU devices:', jax.devices())"

.. code-block:: python

    # Configuración TPU v4
    config = ModularModelConfig.from_toml(
        "capibara/config/configs_toml/production/tpu_v4.toml"
    )
    
    print(f"🚀 TPU mesh: {config.tpu_mesh_shape}")
    print(f"💾 Memory limit: {config.tpu_memory_limit_gb} GB")

**ARM Axion (AWS Graviton)**

.. code-block:: bash

    # Instalación optimizada ARM
    pip install jax[cpu]
    
    # Dependencias ARM específicas (opcional)
    pip install onnxruntime
    sudo apt-get update
    sudo apt-get install libblas-dev liblapack-dev

.. code-block:: python

    # Configuración ARM Axion
    config = ModularModelConfig.from_toml(
        "capibara/config/configs_toml/specialized/arm_axion_inference.toml"
    )
    
    print(f"🔧 ARM optimizations: {config.use_arm_optimizations}")
    print(f"🎯 VQ codes: {config.vq_codes}")

**GPU CUDA**

.. code-block:: bash

    # Instalación CUDA
    pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Verificar CUDA
    python -c "import jax; print('GPU devices:', jax.devices())"

.. code-block:: python

    # Configuración GPU
    config = ModularModelConfig(
        device="cuda",
        gpu_memory_fraction=0.8,
        mixed_precision=True,
        use_vq=True,
        vq_codes=64
    )

**CPU/Desarrollo Local**

.. code-block:: bash

    # Instalación CPU básica
    pip install jax[cpu]

.. code-block:: python

    # Configuración desarrollo
    config = ModularModelConfig.from_toml(
        "capibara/config/configs_toml/development/development.toml"
    )

Troubleshooting Instalación
---------------------------

**Problema: Errores de Importación**

.. code-block:: bash

    # Verificar instalación completa
    pip install -e . --force-reinstall
    
    # Verificar conflictos dependencias
    pip check

.. code-block:: python

    # Diagnóstico imports
    from capibara.utils.diagnostics import ImportDiagnostics
    
    diagnostics = ImportDiagnostics()
    report = diagnostics.diagnose_import_issues()
    
    print("🔍 Diagnóstico de imports:")
    for issue in report.issues:
        print(f"   ❌ {issue.module}: {issue.error}")
        print(f"   💡 Solución: {issue.suggested_fix}")

**Problema: JAX no Funciona**

.. code-block:: python

    # Verificar JAX installation
    try:
        import jax
        print(f"✅ JAX version: {jax.__version__}")
        print(f"🔧 JAX backend: {jax.default_backend()}")
        print(f"📱 Devices: {jax.devices()}")
    except ImportError:
        print("❌ JAX no instalado - ejecutar: pip install jax")

**Problema: TPU no Detectado**

.. code-block:: bash

    # Verificar TPU setup (Google Cloud)
    export TPU_NAME=your-tpu-name
    export ZONE=your-zone
    
    # Test conexión TPU
    python -c "import jax; print('TPU devices:', len(jax.devices()))"

**Problema: Memoria Insuficiente**

.. code-block:: python

    # Configuración memoria limitada
    config = ModularModelConfig(
        model_name="capibara_lite",
        hidden_size=512,        # Reducido
        num_layers=6,           # Reducido
        batch_size=4,           # Reducido
        use_vq=True,
        vq_codes=32,            # Reducido
        mixed_precision=True    # Activar para ahorrar memoria
    )

Variables de Entorno Útiles
---------------------------

.. code-block:: bash

    # JAX Configuration
    export JAX_PLATFORM_NAME=cpu          # Forzar CPU
    export JAX_ENABLE_X64=true            # Precisión 64-bit
    export JAX_DEBUG_MODE=true            # Modo debug
    
    # CapibaraGPT-v2 Configuration
    export CAPIBARA_USE_JAX_NATIVE=true   # JAX nativo
    export CAPIBARA_DEVICE=auto           # Auto-detección
    export CAPIBARA_LOG_LEVEL=INFO        # Logging
    export CAPIBARA_CACHE_DIR=./cache     # Directorio cache

Verificación Post-Instalación
-----------------------------

**Script de Verificación Completa**

.. code-block:: bash

    # Ejecutar script verificación incluido
    python scripts/verify_installation.py

**Test Funcional Completo**

.. code-block:: python

    # Test funcional end-to-end
    from capibara.tests.integration import run_installation_test
    
    # Ejecutar test completo
    test_results = run_installation_test()
    
    print("🧪 Resultados Test Instalación:")
    for test_name, result in test_results.items():
        icon = "✅" if result.passed else "❌"
        print(f"   {icon} {test_name}: {result.message}")
    
    # Reporte final
    passed = sum(1 for r in test_results.values() if r.passed)
    total = len(test_results)
    
    if passed == total:
        print(f"\n🎉 ¡Instalación exitosa! {passed}/{total} tests pasaron")
        print("✅ CapibaraGPT-v2 v3.0.0 listo para usar")
    else:
        print(f"\n⚠️ Instalación parcial: {passed}/{total} tests pasaron")
        print("💡 Revisar logs para detalles de errores")

Próximos Pasos
--------------

1. **Explorar Documentación**: :doc:`quickstart` para comenzar
2. **Configuración Avanzada**: :doc:`configuration` para personalizar
3. **Ejemplos**: Revisar ``examples/`` para casos de uso
4. **Desarrollo**: :doc:`development/contributing` para contribuir

**¡Instalación Completa!** 🎉

CapibaraGPT-v2 v3.0.0 está listo para usar con todas sus características avanzadas:
- ✅ JAX nativo funcional
- ✅ Sistema VQ operativo  
- ✅ Optimizaciones TPU disponibles
- ✅ Configuración TOML lista
- ✅ Fallbacks robustos activos 