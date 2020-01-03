from setuptools import setup

setup(
    name='e2epipeline',
    version='1.0.0',
    description='Generalisation of sklearn Pipeline to allow for more flexible mapping of input and output parameters',
    author='Filip Trojan',
    author_email='f.trojan@centrum.cz',
    packages=['e2epipeline'],
    package_dir={'e2epipeline': 'src'},
    install_requires=['numpy', 'pandas'],
    data_files=[('', ['LICENSE'])]
)
