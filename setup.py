from setuptools import setup, find_packages

__version__ = '2.0'

setup(
    name='sgcharts-pointer-generator',
    version=__version__,
    python_requires='>=3.5.0',
    install_requires=[
        'tensorflow==1.10.0',
        'pyrouge==0.1.3',
        'spacy==2.0.12',
        'en_core_web_sm==2.0.0',
        'sgcharts-stringx==1.1.1'
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    description='News Summarizer'
)
