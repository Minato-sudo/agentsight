from setuptools import setup, find_packages

setup(
    name="agentsight",
    version="1.0.0",
    author="AgentSight Project",
    description="A step-level hallucination detection and risk propagation tool for agentic AI.",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "pydantic",
        "fastapi",
        "uvicorn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
