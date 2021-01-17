# Happy ;) Learning

## Description:
Toolbox for reinforced developing of supervised learning models as proof-of-concept in python. 
It is specially designed to breed and optimize supervised machine learning models using genetic algorithm (GA) both on the feature engineering side and on the hyper parameter tuning side.

## Table of Content:
1. Installation
2. Requirements
3. Introduction
    - Practical Usage
    - FeatureEngineer
    - FeatureTournament
    - FeatureSelector
    - FeatureLearning
    - Genetic
    - DataMiner


## 1. Installation:
You can easily install Happy Learning via pip install happy_learning on every operating system.

## 2. Requirements:
 - ...

## 3. Introduction:
 - Practical Usage:

HappyLearning is designed for reinforced developing of supervised machine learning prototypes using structured (tabular) data especially.
It covers all aspects of the developing process, such as feature engineering, feature and model selection as well as model optimization.
To handle big data sets it has dask implemented under the hood.

- Feature Engineer:

Process your tabular data smartly. The Feature Engineer module is equipped with all necessary (tabular) feature processing methods. Moreover, it is able to capture the meta data about the data set such as scaling measurement types of the features, taken processing steps, etc.

 - Feature Learning:
 
It combines both the feature engineering module and the genetic algorithm module to create a reinforcement learning environment to smartly generate new features.
The module creates separate learning environments for categorical and continuous features. The categorical features are one-hot encoded and then unified (one-hot merging).
Whereas the (semi-) continuous features are systematically processed by using several transformation and interaction methods.

 - Feature Tournament:
 
Feature tournament is a process to evaluate the importance of each feature regarding to a specific target feature. It uses the concept of (Additive) Shapley Values to calculate the importance score.

    -- Data Typing:

        Check whether represented data types of Pandas is equal to the real data types occuring in the data

- Feature Selector:

The Feature Selector module applies the feature tournament to calculate feature importance scores and select automatically the best n features based on the scoring.

- Genetic:

Reinforcement learning module either to evaluate the fittest model / hyper parameter configuration or to engineer (tabular) features. 
It captures several evaluation statistics regarding the evolution process as well as the model performance metrics.
More over, it is able to transfer knowledge across re-trainings.

    -- Model / Hyper Parameter Optimization:

        Optimize model / hyper parameter selection ...
            -> Sklearn models
            -> Popular "stand alone" models like XGBoost, CatBoost, etc.
            -> Deep Learning models (using PyTorch only)

    -- Feature Engineering / Selection:

        Optimize feature engineering / selection using processing methods from Feature Engineer module ...
            -> Choose only features of fittest models to apply feature engineering based on the action space of the Feature Engineer module

- DataMiner:

Combines all modules above in such a way, that it becomes an ai for reinforced prototyping itself. 
Therefore it uses the ...
   -> Feauture Engineer module to pre-process data in general (imputation, label encoding, date feature processing, etc.)
   -> Feature Learning module to smartly engineer tabular features
   -> Feature Selector module to select the most important features
   -> Genetic module to find a proper model all by its self.

- TextMiner

Use text data (natural language) by generating various numerical features describing the text

    -- Segmentation:

        Categorize potential text features into following segments ...
            -> Web features
                1) URL
                2) EMail
            -> Enumerated features
            -> Natural language (original text features)
            -> Identifier (original id features)
            -> Unknown

    -- Simple text processing:
        Apply simple processing methods to text features
            -> Merge two text features by given separator
            -> Replace occurances
            -> Subset data set or feature list by given string

    -- Language methods:
        Apply methods to ...
            -> ... detect language in text
            -> ... translate using Google Translate under the hood

    -- Generate linguistic features:
        Apply semantic text processing to generate numeric features
            -> Clean text counter (text after removing stop words, punctuation and special character and lemmatizing)
            -> Part-of-Speech Tagging counter & labels
            -> Named Entity Recognition counter & labels
            -> Dependencies counter & labels (Tree based / Noun Chunks)
            -> Emoji counter & labels

    -- Generate similarity / clustering features:
        Apply similarity methods to generate continuous features using word embeddings
            -> TF-IDF

## 4. Documentation & Examples:

Check the jupyter notebooks for the documentation and examples. Happy ;) Learning
