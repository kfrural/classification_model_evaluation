from setuptools import setup, find_packages

setup(
    name="classification_model_evaluation",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn>=0.24.2",
        "numpy>=1.21.2",
        "matplotlib>=3.4.3",
    ],
)
