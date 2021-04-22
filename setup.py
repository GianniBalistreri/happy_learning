import setuptools
import subprocess
import sys

from happy_learning.text_miner import LANG_MODELS

# Install complete dask library for handling big data sets using parallel computing:
subprocess.run(['python{} -m pip install "dask[distributed]"'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)
subprocess.run(['python{} -m pip install "dask[complete]"'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)

# Install jupyter notebook extensions for using EasyExplore_examples.ipynb more conveniently:
subprocess.run(['python{} -m pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)

# Install spacy language models:
#subprocess.run('python{} -m pip install spacy'.format('3' if sys.platform.find('win') != 0 else ''), shell=True)
#for lang in LANG_MODELS.keys():
#    for model in LANG_MODELS[lang]['model']['spacy'].keys():
#        subprocess.run('python{} -m spacy download {}'.format('3' if sys.platform.find('win') != 0 else '',
#                                                              LANG_MODELS[lang]['model']['spacy'][model]
#                                                              ),
#                       shell=True)

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='happy_learning',
    version='0.3.5',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Toolbox for reinforced developing of machine learning models (as proof-of-concept)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='feature engineering feature selection genetic algorithm machine learning automl reinforcement shapley clustering',
    license='GNU',
    url='https://github.com/GianniBalistreri/happy_learning',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'happy_learning': ['LICENSE',
                                     'README.md',
                                     'requirements.txt',
                                     'setup.py',
                                     'Happy Learnings.ipynb',
                                     'Happy Learning - Methodology.pdf'
                                     ]
                  },
    data_file=[('test', ['test/test_chaid_decision_tree.py',
                         'test/test_data_miner.py',
                         'test/test_evaluate_machine_learning.py',
                         'test/test_feature_engineer.py',
                         'test/test_feature_learning.py',
                         'test/test_feature_selector.py',
                         'test/test_feature_tournament.py',
                         'test/test_genetic_algorithm.py',
                         'test/test_missing_data_analysis.py',
                         'test/test_multiple_imputation.py',
                         'test/test_neural_network_generator_torch.py',
                         'test/test_neural_network_torch.py',
                         'test/test_sampler.py',
                         'test/test_self_taught_short_clustering.py',
                         'test/test_supervised_machine_learning.py',
                         'test/test_swarm_intelligence.py',
                         'test/test_text_clustering.py',
                         'test/test_text_clustering_generator.py',
                         'test/test_text_miner.py',
                         'test/test_utils.py'
                         ]
                )],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requires
)
