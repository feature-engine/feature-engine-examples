# Jupyter notebooks with Demos of Feature-engine's functionality

![PythonVersion](https://img.shields.io/badge/python-3.6%20|3.7%20|%203.8%20|%203.9-success)
[![License https://github.com/feature-engine/feature_engine/blob/master/LICENSE.md](https://img.shields.io/badge/license-BSD-success.svg)](https://github.com/feature-engine/feature-engine-examples/blob/master/LICENSE.md)
[![Sponsorship https://www.trainindata.com/](https://img.shields.io/badge/Powered%20By-TrainInData-orange.svg)](https://www.trainindata.com/)


Feature-engine is a Python library with multiple transformers to engineer and select features for use in machine learning models. 
Feature-engine's transformers follow scikit-learn's functionality with fit() and transform() methods to first learn the 
transforming parameters from data and then transform the data.

In this repo, you will find a lot of examples on how to use Feature-engine's transformers on various datasets. The notebooks are sorted in the following folders and include examples for the following transformers:


## creation
 * MathFeatures
 * RelativeFeatures
 * CyclicalFeatures

## discretisation
* EqualFrequencyDiscretiser
* EqualFrequencyDiscretiser plus WoEEncoder
* EqualWidthDiscretiser
* EqualWidthDiscretiser plus OrdinalEncoder
* DecisionTreeDiscretiser
* ArbitraryDiscreriser
* ArbitraryDiscreriser plus MeanEncoder

## encoding
* OneHotEncoder
* OrdinalEncoder
* CountFrequencyEncoder
* MeanEncoder
* WoEEncoder
* PRatioEncoder
* RareLabelEncoder
* DecisionTreeEncoder

## imputation
* MeanMedianImputer
* RandomSampleImputer
* EndTailImputer
* AddMissingIndicator
* CategoricalImputer
* ArbitraryNumberImputer
* DropMissingData -- notebook wanted, please contribute

## outliers
* Winsorizer
* ArbitraryOutlierCapper
* OutlierTrimmer

## pipelines
* create new features - wine data
* regression pipeline - house prices data
* more notebooks wanted, please constribute

## transformation
* LogTransformer
* LogCpTransformer
* ReciprocalTransformer
* PowerTransformer
* BoxCoxTransformer
* YeoJohnsonTransformer

### wrappers
 * SklearnTransformerWrapper plus Scikit-learn's OneHotEncoder
 * SklearnTransformerWrapper plus Scikit-learn's feature selection classes
 * SklearnTransformerWrapper plus Scikit-learn's KBinsDiscretizer
 * SklearnTransformerWrapper plus Scikit-learn's Scalers
 * SklearnTransformerWrapper plus Scikit-learn's SimpleImputer

## selection
 * notebooks wanted, please contribute



# Contributing

We welcome notebooks from users of the package. If you want to create one of the missing notebooks, or want to add a notebook of your own, provided that the data set is free to share, make a pull request with the code.

How to contribute:

### Local Setup Steps
- Fork the repo
- Clone your fork into your local computer: ``git clone https://github.com/<YOURUSERNAME>/feature-engine-examples.git``
- cd into the repo ``cd feature-engine-examples``
- If you haven't done so yet, install feature-engine ``pip install feature_engine``
- Create a feature branch with a meaningful name for your notebook: ``git checkout -b mynotebookbranch``
- Develop your notebook
- Add the changes to your copy of the fork: ``git add .``, ``git commit -m "a meaningful commit message"``, ``git pull origin mynotebookbranch:mynotebookbranch``
- Go to your fork on Github and make a PR to this repo
- Done

Thank you!!

## Feature-engine features in the following resources:

* [Feature Engineering for Machine Learning, Online Course](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

* [Feature Selection for Machine Learning, Online Course](https://www.trainindata.com/p/feature-selection-for-machine-learning)

* [Python Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)


## Blogs about Feature-engine:

* [Feature-engine: A new open-source Python package for feature engineering](https://trainindata.medium.com/feature-engine-a-new-open-source-python-package-for-feature-engineering-29a0ab88ea7c)

* [Practical Code Implementations of Feature Engineering for Machine Learning with Python](https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd)

## Documentation

* [Documentation](http://feature-engine.trainindata.com)

## En Español:

* [Ingeniería de variables, MachinLenin, charla online](https://www.youtube.com/watch?v=NhCxOOoFXds)
