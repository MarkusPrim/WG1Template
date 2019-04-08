from setuptools import setup

setup(
    name='WG1Template',
    version='0.0.1',
    packages=['wg1template'],
    url='',
    license='MIT',
    author='Max Welsch, Markus Prim, Peter Lewis',
    author_email='markus.prim@kit.edu',
    description='Plotting template for the Belle II working group 1.',
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'uncertainties', 'scikit-learn'],
)
