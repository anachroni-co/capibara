from setuptools import setup, find_packages

setup(
    name="capibara_model",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.28",
        "flax>=0.8.1",
        "optax>=0.1.9",
        "tensorflow>=2.15.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.23.0",
            "myst-parser>=2.0.0",
        ],
        "quantum": [
            "qiskit>=0.45.0",
            "cirq>=1.3.0",
            "pennylane>=0.32.0",
        ],
    },
    author="Marco Durán",
    author_email="marco@anachroni.co",
    description="Modelo de lenguaje avanzado basado en arquitecturas Transformer y Capibara SSM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Anachroni-co/CapibaraGPT-v2",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
) 