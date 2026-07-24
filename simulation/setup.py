from setuptools import find_packages, setup

setup(
    name='legged_panguin',
    version='1.0.0',
    author='Nikita Rudin',
    license="BSD-3-Clause",
    packages=find_packages(exclude=("tests",)),
    author_email='rudinn@ethz.ch',
    description='Isaac Gym environments for Legged Robots',
    install_requires=['matplotlib', 'numpy'],
    extras_require={
        'training': ['isaacgym', 'rsl-rl'],
        'deployment': ['onnxruntime'],
    },
)
