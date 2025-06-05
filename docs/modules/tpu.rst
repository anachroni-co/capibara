Configuración TPU
==============

La configuración TPU de CapibaraGPT está optimizada para TPU v4-32, implementando sharding dinámico, precisión mixta y gestión de memoria avanzada.

Características
-------------

Hardware
~~~~~~~

* TPU v4-32 (32 chips)
* 32GB HBM por chip
* Topología 4x8
* 275 TFLOPS por chip

Optimizaciones
~~~~~~~~~~~~

* Sharding dinámico
* Precisión mixta (bfloat16/float32)
* Gestión de memoria avanzada
* Profiling integrado

Uso
---

Configuración Básica
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from capibara.config.tpu_config import TPUConfig

    # Configuración TPU v4-32
    tpu_config = TPUConfig(
        num_chips=32,
        topology='4x8',
        memory_per_chip=32 * 1024**3,  # 32GB HBM
        flops_per_chip=275e12,  # 275 TFLOPS
        compute_dtype='bfloat16',
        param_dtype='float32'
    )

Sharding
~~~~~~~~

.. code-block:: python

    # Crear mesh para sharding
    devices = mesh_utils.create_device_mesh((4, 8))
    mesh = Mesh(devices, axis_names=('data', 'model'))

    # Obtener especificación de sharding
    sharding_spec = tpu_config.get_sharding_spec(tensor_shape)

Precisión Mixta
~~~~~~~~~~~~~

.. code-block:: python

    # Configuración de precisión
    if tpu_config.use_mixed_precision:
        policy = jax.pjit(
            in_axis_resources=PartitionSpec('data', 'model'),
            out_axis_resources=PartitionSpec('data', 'model')
        )

Optimizaciones Implementadas
-------------------------

Gestión de Memoria
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Configuración de memoria
    memory_config = tpu_config.get_memory_config()
    
    # Aplicar configuración
    if memory_config['use_mixed_precision']:
        compute_dtype = jnp.bfloat16
        param_dtype = jnp.float32

Sharding Dinámico
~~~~~~~~~~~~~~~

.. code-block:: python

    def get_sharding_spec(self, tensor_shape: tuple) -> PartitionSpec:
        """Obtiene la especificación de sharding óptima para un tensor."""
        if len(tensor_shape) == 2:
            return PartitionSpec('data', 'model')
        elif len(tensor_shape) == 3:
            return PartitionSpec('data', None, 'model')
        else:
            return PartitionSpec('data', *[None] * (len(tensor_shape) - 2), 'model')

Métricas y Monitoreo
------------------

Profiling
~~~~~~~~

.. code-block:: python

    def profile_tpu_usage(self) -> Dict[str, Any]:
        with jax.profiler.trace("/tmp/tpu_trace"):
            memory_stats = jax.device_memory_allocated()
            memory_peak = jax.device_memory_peak()
            device_count = len(jax.devices('tpu'))
            device_utilization = jax.device_get(jax.device_count())
            
            return {
                'memory_allocated': memory_stats,
                'memory_peak': memory_peak,
                'device_count': device_count,
                'device_utilization': device_utilization
            }

Configuración Avanzada
-------------------

Hotspot Detection
~~~~~~~~~~~~~~~

.. code-block:: python

    def detect_hotspots(self, load: jnp.ndarray) -> List[Tuple[int, int]]:
        threshold = self.scaling_config.load_threshold
        hotspots = []
        
        for i in range(load.shape[0]):
            for j in range(load.shape[1]):
                if load[i,j] > threshold:
                    hotspots.append((i,j))
                    
        return hotspots

Optimización de Sharding
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def optimize_sharding(self) -> PartitionSpec:
        devices = mesh_utils.create_device_mesh((4, 8))
        mesh = Mesh(devices, axis_names=('data', 'model'))
        
        if len(batch_hotspots) > len(hidden_hotspots):
            return PartitionSpec(
                data=('data', 'model'),
                model=None
            )
        else:
            return PartitionSpec(
                data=None,
                model=('data', 'model')
            )

API Reference
------------

.. autoclass:: capibara.config.tpu_config.TPUConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: capibara.config.tpu_config.TPUError
   :members:
   :undoc-members:
   :show-inheritance: 