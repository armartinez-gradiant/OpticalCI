from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pytorch-onn',
    version='0.1.0',
    description='A PyTorch Library for Photonic Integrated Circuit Simulation and Photonic AI Computing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='JeremieMelo et al.',
    packages=find_packages(exclude=['tests', 'examples', 'unitest']),
    python_requires='>=3.7,<=3.13',
    install_requires=[
        'torch>=2.0.0',
        'tensorflow>=2.12.0',
        'pyutils>=0.0.2',
        'numpy>=1.24.0',
    ],
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)