from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="agentsight",
    version="1.0.0",
    author="Minato Namikaze",
    description="Step-level hallucination detection and root-cause analysis for autonomous AI agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Minato-sudo/agentsight",
    packages=find_packages(),
    py_modules=["agentsight_sdk"],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "scikit-learn>=1.0",
        "huggingface_hub>=0.19.0",
        "pydantic",
        "fastapi",
        "uvicorn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
