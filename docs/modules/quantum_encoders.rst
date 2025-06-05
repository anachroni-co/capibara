Codificadores Cuánticos
====================

Este módulo implementa los codificadores cuánticos especializados de CapibaraGPT v2.0.

QuantumEncoder
-------------

.. code-block:: python

    from capibara.quantum_encoders import QuantumEncoder

    # Inicializar codificador cuántico
    encoder = QuantumEncoder(
        input_dim=512,
        hidden_dim=256,
        num_qubits=8,
        use_mixed_precision=True
    )

    # Codificar datos
    encoded_data = encoder.encode(
        input_data,
        batch_size=32
    )

Características:
* Codificación cuántica optimizada
* VQbit con caché inteligente
* Pipeline asíncrono
* Métricas de coherencia

QuantumDecoder
-------------

.. code-block:: python

    from capibara.quantum_encoders import QuantumDecoder

    # Inicializar decodificador cuántico
    decoder = QuantumDecoder(
        input_dim=256,
        output_dim=512,
        num_qubits=8,
        use_mixed_precision=True
    )

    # Decodificar datos
    decoded_data = decoder.decode(
        encoded_data,
        batch_size=32
    )

Características:
* Decodificación cuántica optimizada
* Reconstrucción de estados
* Pipeline asíncrono
* Métricas de fidelidad

QuantumAutoencoder
----------------

.. code-block:: python

    from capibara.quantum_encoders import QuantumAutoencoder

    # Inicializar autoencoder cuántico
    autoencoder = QuantumAutoencoder(
        input_dim=512,
        hidden_dim=256,
        num_qubits=8,
        use_mixed_precision=True
    )

    # Entrenar autoencoder
    autoencoder.train(
        train_data,
        num_epochs=100,
        batch_size=32
    )

Características:
* Autoencoder cuántico optimizado
* Compresión cuántica
* Pipeline asíncrono
* Métricas de reconstrucción

Optimizaciones
------------

Gestión de Memoria
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configurar memoria
    encoder.set_memory_fraction(0.8)
    encoder.set_gradient_checkpointing(True)

    # Optimizar buffers
    encoder.optimize_buffers()

Sharding
~~~~~~~~

.. code-block:: python

    # Configurar sharding
    devices = mesh_utils.create_device_mesh((4, 8))
    mesh = Mesh(devices, axis_names=('data', 'model'))

    # Aplicar sharding
    encoder.set_mesh(mesh)

Métricas y Monitoreo
------------------

.. code-block:: python

    # Obtener métricas
    metrics = encoder.get_metrics()
    
    # Monitorear rendimiento
    stats = encoder.profile_performance()
    
    # Detectar hotspots
    hotspots = encoder.detect_hotspots()

API Reference
------------

.. autoclass:: capibara.quantum_encoders.QuantumEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara.quantum_encoders.QuantumDecoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara.quantum_encoders.QuantumAutoencoder
   :members:
   :undoc-members:
   :show-inheritance: 