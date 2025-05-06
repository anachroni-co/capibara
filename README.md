# 🦫 CapibaraGPT-v2

Modelo de lenguaje avanzado basado en arquitecturas Transformer y Capibara SSM.

## Estructura de Interfaces

El proyecto utiliza un sistema de interfaces jerárquico para garantizar consistencia y tipado fuerte:

### Jerarquía de Interfaces

```
IModule (Protocol)
├── ILayer (Protocol + nn.Module)
└── ISubModel (Protocol)
```

### IModule (Base)
Interfaz base para todos los módulos:
```python
class MiModulo(IModule):
    def __call__(self, x: jnp.ndarray, **kwargs) -> Dict[str, Any]:
        return {
            "output": x,
            "metrics": {"loss": 0.0}
        }
```

### ILayer
Interfaz para capas con soporte de entrenamiento:
```python
class MiCapa(ILayer):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False, **kwargs) -> Dict[str, Any]:
        return {
            "output": x,
            "metrics": {"loss": 0.0},
            "training": training
        }
```

### ISubModel
Interfaz para submodelos con configuración:
```python
class MiSubModel(ISubModel):
    def __call__(self, x: jnp.ndarray, config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return {
            "output": x,
            "metrics": {"loss": 0.0},
            "config": config
        }
```

## Características Principales

- Arquitectura basada en Transformer y Capibara SSM
- Soporte para TPU con validaciones robustas
- Sistema de interfaces tipado y consistente
- Optimizaciones de memoria avanzadas
- Validación cruzada integrada

## ✨ Características

- **🧠 Arquitectura**:
  - 🔄 Multi-head attention con BitNet 1.58
  - 🎯 Activación contextual dinámica
  - 🔍 Detección de coherencia mejorada
  - 🎭 Gestión de personalidad adaptativa
  - ⚡ Capas de embedding cuánticas
  - 🧩 Submodelo dinámico (CapibaraQuantum)

- **⚡ Optimizaciones**:
  - 🚀 Soporte nativo para TPU v4
  - ⚙️ Procesamiento eficiente con JAX
  - 📦 Batching optimizado
  - 🕸️ Esparcidad integrada
  - 🔄 Cuantización BitNet
  - 🎯 Mezcla dinámica de modelos

## 📋 Requisitos

- Python >= 3.8
- JAX >= 0.4.28
- Flax >= 0.8.1
- Optax >= 0.1.9
- TensorFlow >= 2.15.0

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Anachroni-co/CapibaraGPT-v2.git
cd CapibaraGPT-v2

# Instalar dependencias
pip install -r capibara_model/requirements.txt

# Configurar variables de entorno
cp capibara_model/_.env.example capibara_model/.env
# Editar .env con tus configuraciones
```

## 💻 Uso

### Ejemplo Básico

```python
from capibara_model.core.model import DynamicCapibaraModel
from capibara_model.core.inference import CapibaraInference

# Crear modelo
model = DynamicCapibaraModel(
    hidden_size=768,
    num_heads=8,
    num_layers=12,
    dropout_rate=0.1
)

# Crear inferencia
inference = CapibaraInference(model)

# Generar respuesta
response = inference("¿Cómo estás?")
print(response)
```

### Ejemplo Avanzado con TPU

```python
from capibara_model.core.model import DynamicCapibaraModel
from capibara_model.training.train_unified import train_model
import yaml

# Cargar configuración
with open('config_300m.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Crear y entrenar modelo
model = train_model(
    config=config,
    output_dir='checkpoints/300m',
    use_tpu=True
)
```

## ⚙️ Configuración

El sistema utiliza un archivo YAML para configuración. Ejemplo:
```yaml
security:
  api_key: "tu-api-key"
  rate_limit: 100

database:
  host: "localhost"
  port: 5432
  name: "capibara"

models:
  health_advisor:
    min_confidence: 0.7
    cache_ttl: 3600

api:
  host: "0.0.0.0"
  port: 8000
```

## 🛠️ Módulos

### HealthAdvisor

Módulo para proporcionar recomendaciones de salud personalizadas basadas en datos biométricos y hábitos.

#### Características
- Análisis de datos de salud
- Recomendaciones personalizadas
- Seguimiento de hábitos
- Cálculo de puntaje de salud

#### Uso
```python
from capibara_model.mcp.HealthAdvisor import HealthAdvisor

# Inicializar el módulo
advisor = HealthAdvisor()

# Datos de ejemplo
data = {
    "age": 30,
    "weight": 70,
    "height": 175,
    "sleep_hours": 6,
    "water_intake": 1.5,
    "exercise_hours": 0.3,
    "stress_level": 8,
    "last_checkup": "2023-01-01T00:00:00"
}

# Obtener recomendaciones
result = advisor.process({"data": data})
```

### DocRetriever

Módulo para recuperar documentos relevantes usando búsqueda semántica.

#### Características
- Búsqueda semántica
- Recuperación de documentos
- Ranking de relevancia
- Caché de embeddings

#### Uso
```python
from capibara_model.mcp.DocRetriever import DocRetriever

# Inicializar el módulo
retriever = DocRetriever()

# Realizar búsqueda
result = retriever.process({
    "query": "ejemplo de búsqueda",
    "max_results": 5
})
```

### VeracityVerifier

Módulo para verificar la veracidad de afirmaciones.

#### Características
- Verificación de afirmaciones
- Búsqueda de evidencia
- Análisis de confianza
- Caché de resultados

#### Uso
```python
from capibara_model.mcp.VeracityVerifier import VeracityVerifier

# Inicializar el módulo
verifier = VeracityVerifier()

# Verificar afirmación
result = verifier.process({
    "claim": "La afirmación a verificar"
})
```

## 📊 Métricas y Monitoreo

- WandB integrado para seguimiento de métricas
- Monitoreo de recursos en tiempo real
- Validación de precisión mixta
- Backup automático de checkpoints

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📫 Contacto

- 👨‍💻 Developer: Marco Durán
- 📧 Email: [marco@anachroni.co](mailto:marco@anachroni.co)
- 🌐 Web: anachroni.co

## 🔗 Enlaces Útiles

- 📚 [Documentación JAX](https://jax.readthedocs.io/)
- 🔧 [Documentación TPU](https://cloud.google.com/tpu/docs)
- 🐍 [Paper Capibara SSM](https://arxiv.org/abs/2312.00752)

## 📜 License

This project is under MIT License. See [LICENSE.md](LICENSE.md) for details.

## 🔄 Latest Updates

- ✨ TPU v4-32 Support
- 🚀 Capibara SSM Optimization
- 📦 Dependencies Update
- 🐛 Distributed Training Fixes

## Módulos

### HealthAdvisor

Módulo para proporcionar recomendaciones de salud personalizadas basadas en datos biométricos y hábitos.

#### Características
- Análisis de datos de salud
- Recomendaciones personalizadas
- Seguimiento de hábitos
- Cálculo de puntaje de salud

#### Uso
```python
from capibara_model.mcp.HealthAdvisor import HealthAdvisor

# Inicializar el módulo
advisor = HealthAdvisor()

# Datos de ejemplo
data = {
    "age": 30,
    "weight": 70,
    "height": 175,
    "sleep_hours": 6,
    "water_intake": 1.5,
    "exercise_hours": 0.3,
    "stress_level": 8,
    "last_checkup": "2023-01-01T00:00:00"
}

# Obtener recomendaciones
result = advisor.process({"data": data})
```

#### Respuesta
```json
{
    "recommendations": [
        {
            "category": "Sueño",
            "message": "Intenta dormir al menos 7 horas diarias",
            "priority": 1,
            "score": 0.2
        },
        {
            "category": "Hidratación",
            "message": "Aumenta tu ingesta de agua a al menos 2 litros diarios",
            "priority": 1,
            "score": 0.15
        }
    ],
    "health_score": 0.75,
    "analysis_time": 0.123
}
```

#### Validación de Datos
- Edad: 0-120 años
- Peso: >0 kg
- Altura: >0 cm
- Horas de sueño: 0-24 horas
- Ingesta de agua: ≥0 litros
- Horas de ejercicio: 0-24 horas
- Nivel de estrés: 1-10

### DocRetriever

Módulo para recuperar documentos relevantes usando búsqueda semántica.

#### Características
- Búsqueda semántica
- Recuperación de documentos
- Ranking de relevancia
- Caché de embeddings

#### Uso
```python
from capibara_model.mcp.DocRetriever import DocRetriever

# Inicializar el módulo
retriever = DocRetriever()

# Realizar búsqueda
result = retriever.process({
    "query": "ejemplo de búsqueda",
    "max_results": 5
})
```

### VeracityVerifier

Módulo para verificar la veracidad de afirmaciones.

#### Características
- Verificación de afirmaciones
- Búsqueda de evidencia
- Análisis de confianza
- Caché de resultados

#### Uso
```python
from capibara_model.mcp.VeracityVerifier import VeracityVerifier

# Inicializar el módulo
verifier = VeracityVerifier()

# Verificar afirmación
result = verifier.process({
    "claim": "La afirmación a verificar"
})
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/capibaraGPT-v2.git
cd capibaraGPT-v2
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Configuración

El sistema utiliza un archivo YAML para configuración. Ejemplo:
```yaml
security:
  api_key: "tu-api-key"
  rate_limit: 100

database:
  host: "localhost"
  port: 5432
  name: "capibara"

models:
  health_advisor:
    min_confidence: 0.7
    cache_ttl: 3600

api:
  host: "0.0.0.0"
  port: 8000
```

## Desarrollo

### Estructura del Proyecto
```
capibaraGPT-v2/
├── capibara_model/
│   ├── mcp/
│   │   ├── HealthAdvisor/
│   │   ├── DocRetriever/
│   │   └── VeracityVerifier/
│   ├── utils/
│   └── config/
├── tests/
├── requirements.txt
└── README.md
```

### Tests
```bash
pytest tests/
```

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.
