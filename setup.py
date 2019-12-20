from setuptools import setup

setup(
   name='e2epipeline',
   version='1.0.0',
   description='A useful module',
   author='Filip Trojan',
   author_email='f.trojan@centrum.cz',
   packages=['e2epipeline'],  #same as name
   install_requires=['numpy', 'pandas'], #external packages as dependencies
   scripts=[
            'examples/example1',
            'examples/example2',
            'examples/example3',
            'examples/example_rename'
           ]
)
