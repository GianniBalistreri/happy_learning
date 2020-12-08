import setuptools

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='happy_learning',
    version='0.0.1',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Toolbox for easy and effective developing of supervised machine learning models as proof-of-concept',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='feature engineering feature selection genetic algorithm machine learning automl reinforcement shapley',
    license='GNU',
    url='https://github.com/GianniBalistreri/happy_learning',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'happy_learning': ['LICENSE',
                                     'README.md',
                                     'requirements.txt',
                                     'setup.py',
                                     #'Happily Applied Learning -> Avocado.ipynb',
                                     'Happy Learnings.ipynb'
                                     ]
                  },
    data_file=[('test', ['test/test_chaid_decision_tree.py',
                         'test/test_data_miner.py',
                         'test/test_feature_engineer.py',
                         'test/test_feature_learning.py',
                         'test/test_feature_selector.py',
                         'test/test_feature_tournament.py',
                         'test/test_genetic_algorithm.py',
                         'test/test_missing_data_analysis.py',
                         'test/test_multiple_imputation.py',
                         #'test/test_neural_network.py',
                         'test/test_sampler.py',
                         'test/test_supervised_machine_learning.py',
                         'test/test_utils.py'
                         ]
                )],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requires
)
