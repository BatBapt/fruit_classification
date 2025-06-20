# 🍎🍌🍊 Fruit Classification Project

## Description
This project aims to classify images of fruits using a deep learning model. The goal is to predict the type of fruit from a given image. 🍓🍍

## Context
The project uses the fruit classification dataset available on [Kaggle](https://www.kaggle.com/datasets/icebearogo/fruit-classification-dataset). The model is based on a Convolutional Neural Network (CNN) architecture. For more details on the model, refer to the research paper [here](https://arxiv.org/abs/1512.03385). 📚

## Project Structure

### Main Files

1. **custom_dataset.py** : This file contains the code to load and preprocess the fruit dataset. It includes the definition of image transformations and the creation of training and test datasets. 📂🔄

2. **model.py** : This file defines the architecture of the classification model. It includes the definition of neural network layers and functions to train and evaluate the model. 🏗️⚙️

3. **train.py** : This file contains the script to train the model. It includes the training loop, device management (CPU/GPU), and saving results. 🏋️‍♂️💾

4. **configuration.py** : This file contains configuration parameters such as paths to data. ⚙️📝


## Dataset and Research Paper Links

- Dataset: [Fruit Classification Dataset](https://www.kaggle.com/datasets/icebearogo/fruit-classification-dataset) 📂🍎
- Research Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 📄🔍

## Images and Graphs

Images and graphs will be added to better illustrate the results and project structure. 🖼️📊

## F1 Score Table

| Fruit        | F1 Score |
|--------------|----------|
| Apple        | -        |
| Banana       | -        |
| Orange       | -        |
| ...          | ...      |


## Future Work

- Automate training with a YAML file. ✅📄
- Test multiple backbones to improve model performance. 🏋️‍♂️🔁
- Add images and graphs to illustrate results. 📊📈
- Incorporate an F1 score table to evaluate model performance. 📋🏆

<small>this file was enhanced through my Mistral AI Agent</small>