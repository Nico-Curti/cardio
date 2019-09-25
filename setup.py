from setuptools import setup

setup(
    name='cardio',
    version='0.1.0',
    author='Nico Curti, Lorenzo Dall\'Olio, Valentino Recaldini',
    author_email='nico.curti2@unibo.it, lorenzo.dallolio@studio.unibo.it, valentino.recaldini@studio.unibo.it',
    description='Tool for PPG signal features extraction and analysis',
    long_description=open('README.md').read(),
    url='https://github.com/Nico-Curti/cardio',
    license='Proprietary',  # NO LICENSEs available. All rights reserved.

    packages=['cardio'],
    install_requires=['argparse>=1.1', 'pandas>=0.23.3', 'numpy>=1.14.3',
                      'scipy>=1.1.0', 'scikit-learn>=0.19.1',
                      'statsmodels>=0.9.0', 'jsonschema>=2.6.0',
                      'matplotlib>=2.2.2'],

    scripts=['pipeline/data_analysis.py', 'pipeline/feature_extraction.py'],

    classifiers=['License :: Other/Proprietary License']
)
