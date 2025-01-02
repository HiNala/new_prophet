from setuptools import setup, find_packages

setup(
    name="prophet_forecaster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "yfinance>=0.2.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
        "aiohttp>=3.8.0",
        "pyarrow>=12.0.0",
        "python-dateutil>=2.8.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
    ],
    python_requires=">=3.8",
) 