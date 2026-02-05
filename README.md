# Single-task and multi-task learning based on examples in medicine
> Implementing TabNet in both STL and MTL versions to see how well they will work on classification task. Predicting diabetes and heart disease separately and collectively.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup and usage](#setup-and-usage)
* [Results](#results)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)


## General Information
This project focuses on comparing single-task and multi-task learning, using disease prediction based on survey data as an example. Multi-task learning is an approach in which a single model simultaneously solves several related but not identical tasks. In this project, it is the prediction of diabetes and heart disease. The implemented models achieved satisfactory results, especially in the context of medical data and unbalanced classes. 


## Technologies Used
- Python - version 3.11
- numpy
- scikit-learn
- torch
- pytorch_tabnet
- scipy



## Setup and usage
In order to use the project you have to download BRFSS data for both 2024 and 2023 (here is a link for 2024) https://www.cdc.gov/brfss/annual_data/annual_2024.html, then create data folder and put it here.
This project consists of a few steps. 
1. Loading and preprocessing data - run scripts/prepare_data.py.py
2. Training single task model - run scripts/train_single diabetes and scripts/train_single heart 
3. Training multi task model - run scripts/train_multi.py
4. Comparing results - run scripts/compare_results.py
Requirements are listed in requirements.txt file. 


## Results
In the experiments conducted, the multi-task approach proved to be more effective than the single-task approach, but only to a small extent. MTL was better than STL in 9/10 metrics. Detailed metrics are visible in results folder as json files. 


## Room for Improvement
This project is complete, however classical machine learning methods can be added as a comparison. Also model and training parameters could be improved.


## Acknowledgements
- data provided by BRFSS 

