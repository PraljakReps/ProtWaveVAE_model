from setuptools import setup, find_packages



setup(
    name="ProtWave-VAE",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here, e.g.:
        "torch",
        "numpy",
    ],
    extras_require={
        # Add optional dependencies and their versions, e.g.:
        "dev": ["pytest", "black", "isort"],
    },
    classifiers=[
        # Add classifiers describing your package, e.g.:
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Deep learning framework :: Torch :: >= 1.9.0"
    ],
)
