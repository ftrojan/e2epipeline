from setuptools import setup

setup(
    name='e2epipeline',
    version='1.0.1',
    description='Generalisation of sklearn Pipeline to allow for more flexible mapping of input and output parameters',
    author='Filip Trojan',
    author_email='f.trojan@centrum.cz',
    packages=['e2epipeline'],
    package_dir={'e2epipeline': 'src'},
    install_requires=['numpy', 'pandas'],
    license='BSD 2-Clause License',
    url='https://github.com/ftrojan/e2epipeline',
)
