Registro de Cambios
=================

Este documento registra los cambios importantes en CapibaraModel.

v2.0.0 (2025-05-04)
-------------------

Nuevas Características
~~~~~~~~~~~~~~~~~~~~~

- Implementación completa del modelo SSM
- Soporte para TPU y GPU
- Entrenamiento distribuido
- Sistema de monitoreo mejorado
- Nuevo sistema de tokenización
- Optimizaciones de rendimiento

Mejoras
~~~~~~~

- Mejor manejo de memoria
- Optimización de inferencia
- Mejoras en la documentación
- Nuevos ejemplos y tutoriales

Cambios en la API
~~~~~~~~~~~~~~~~

- Nueva interfaz de configuración
- Mejoras en la API de entrenamiento
- Nuevos métodos de inferencia
- Mejor manejo de errores

Correcciones de Errores
~~~~~~~~~~~~~~~~~~~~~~

- Corrección de problemas de memoria
- Solución de problemas de TPU
- Mejoras en la estabilidad
- Correcciones en el manejo de datos

v1.0.0 (2024-01-01)
-------------------

Primera versión estable
~~~~~~~~~~~~~~~~~~~~~

- Implementación inicial del modelo
- Soporte básico para GPU
- Sistema de entrenamiento básico
- Documentación inicial

Notas de Actualización
---------------------

Para actualizar de v1.0.0 a v2.0.0:

1. Actualiza las dependencias:

.. code-block:: bash

    pip install --upgrade capibara_model

2. Actualiza tu código:

- Revisa los cambios en la API
- Actualiza las configuraciones
- Verifica la compatibilidad

3. Verifica la instalación:

.. code-block:: python

    import capibara_model
    print(capibara_model.__version__)  # Debería mostrar 2.0.0

Próximas Versiones
-----------------

v2.1.0 (Planeado)
~~~~~~~~~~~~~~~~

- Mejoras en el rendimiento
- Nuevas características de optimización
- Mejor soporte para TPU
- Nuevas capacidades de inferencia

v2.2.0 (Planeado)
~~~~~~~~~~~~~~~~

- Integración con más frameworks
- Mejoras en el entrenamiento distribuido
- Nuevas capacidades de fine-tuning
- Mejoras en la documentación 