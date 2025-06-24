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
### Stone Fruits

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| apricot | 0.85 | 0.88 | 0.86 | 50 |
| apricot | 0.81 | 0.86 | 0.83 | 50 |
| apricot | 0.65 | 0.66 | 0.65 | 50 |
| apricot | 0.76 | 0.76 | 0.76 | 50 |
| apricot | 0.98 | 0.88 | 0.93 | 50 |
| plumcot | 0.78 | 0.90 | 0.83 | 50 |
| plumcot | 0.71 | 0.84 | 0.77 | 50 |

### Berries

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| black_berry | 0.45 | 0.48 | 0.47 | 50 |
| black_berry | 0.52 | 0.62 | 0.56 | 50 |
| black_berry | 0.80 | 0.86 | 0.83 | 50 |
| chokeberry | 0.64 | 0.54 | 0.59 | 50 |
| chokeberry | 0.85 | 0.78 | 0.81 | 50 |
| chokeberry | 0.82 | 0.90 | 0.86 | 50 |
| chokeberry | 0.75 | 0.90 | 0.82 | 50 |
| dewberry | 0.81 | 0.68 | 0.74 | 50 |
| dewberry | 0.72 | 0.68 | 0.70 | 50 |
| rose_hip | 0.84 | 0.74 | 0.79 | 50 |
| strawberry_guava | 0.69 | 0.62 | 0.65 | 50 |
| redcurrant | 0.67 | 0.70 | 0.69 | 50 |
| redcurrant | 0.86 | 0.84 | 0.85 | 50 |

### Exotic Fruits

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| barbadine | 0.74 | 0.70 | 0.72 | 50 |
| barbadine | 0.58 | 0.56 | 0.57 | 50 |
| camu_camu | 0.74 | 0.64 | 0.69 | 50 |
| camu_camu | 0.65 | 0.64 | 0.65 | 50 |
| custard_apple | 0.86 | 0.84 | 0.85 | 50 |
| custard_apple | 0.87 | 0.92 | 0.89 | 50 |
| custard_apple | 0.60 | 0.58 | 0.59 | 50 |
| custard_apple | 0.77 | 0.72 | 0.74 | 50 |
| dragonfruit | 0.47 | 0.50 | 0.49 | 50 |
| durian | 0.55 | 0.46 | 0.50 | 50 |
| feijoa | 0.60 | 0.60 | 0.60 | 50 |
| feijoa | 0.76 | 0.70 | 0.73 | 50 |
| feijoa | 0.84 | 0.82 | 0.83 | 50 |
| gooseberry | 0.77 | 0.92 | 0.84 | 50 |
| grenadilla | 0.83 | 0.78 | 0.80 | 50 |
| guava | 0.68 | 0.72 | 0.70 | 50 |
| guava | 0.81 | 0.88 | 0.85 | 50 |
| hard_kiwi | 0.81 | 0.70 | 0.75 | 50 |
| hawthorn | 0.66 | 0.74 | 0.70 | 50 |
| hawthorn | 0.70 | 0.64 | 0.67 | 50 |
| hawthorn | 0.83 | 0.96 | 0.89 | 50 |
| hawthorn | 0.90 | 0.86 | 0.88 | 50 |
| hawthorn | 0.92 | 0.90 | 0.91 | 50 |
| jaboticaba | 0.74 | 0.74 | 0.74 | 50 |
| jackfruit | 0.62 | 0.60 | 0.61 | 50 |
| jalapeno | 0.67 | 0.82 | 0.74 | 50 |
| jalapeno | 0.79 | 0.74 | 0.76 | 50 |
| jujube | 0.75 | 0.72 | 0.73 | 50 |
| jujube | 0.66 | 0.66 | 0.66 | 50 |
| kaffir_lime | 0.75 | 0.54 | 0.63 | 50 |
| longan | 0.83 | 0.90 | 0.87 | 50 |
| mabolo | 0.76 | 0.82 | 0.79 | 50 |
| malay_apple | 0.74 | 0.62 | 0.67 | 50 |
| malay_apple | 0.82 | 0.80 | 0.81 | 50 |
| mandarine | 0.91 | 0.86 | 0.89 | 50 |
| mandarine | 0.76 | 0.78 | 0.77 | 50 |
| mandarine | 0.68 | 0.68 | 0.68 | 50 |
| mountain_soursop | 0.60 | 0.66 | 0.63 | 50 |
| olive | 0.86 | 0.86 | 0.86 | 50 |
| oil_palm | 0.60 | 0.62 | 0.61 | 50 |
| pawpaw | 0.82 | 0.90 | 0.86 | 50 |
| pawpaw | 0.87 | 0.78 | 0.82 | 50 |
| pineapple | 0.70 | 0.78 | 0.74 | 50 |
| pomegranate | 0.84 | 0.94 | 0.89 | 50 |
| prikly_pear | 0.52 | 0.54 | 0.53 | 50 |
| rambutan | 0.68 | 0.56 | 0.62 | 50 |
| rose_leaf_bramble | 0.94 | 0.90 | 0.92 | 50 |
| rose_leaf_bramble | 0.81 | 0.84 | 0.82 | 50 |
| rose_leaf_bramble | 0.66 | 0.62 | 0.64 | 50 |
| salak | 0.41 | 0.42 | 0.42 | 50 |
| salak | 0.83 | 0.90 | 0.87 | 50 |
| santol | 0.72 | 0.68 | 0.70 | 50 |
| santol | 0.80 | 0.78 | 0.79 | 50 |
| santol | 0.88 | 0.88 | 0.88 | 50 |
| santol | 0.83 | 0.76 | 0.79 | 50 |
| sapodilla | 0.70 | 0.66 | 0.68 | 50 |
| sapodilla | 0.64 | 0.56 | 0.60 | 50 |
| sugar_apple | 0.91 | 0.96 | 0.93 | 50 |
| taxus_baccata | 0.83 | 0.70 | 0.76 | 50 |

### Others

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| apple | 0.73 | 0.70 | 0.71 | 50 |
| avocado | 0.91 | 0.96 | 0.93 | 50 |
| brazil_nut | 0.63 | 0.76 | 0.69 | 50 |
| cashew | 0.54 | 0.60 | 0.57 | 50 |
| cashew | 0.77 | 0.74 | 0.76 | 50 |
| cashew | 0.63 | 0.72 | 0.67 | 50 |
| chenet | 0.92 | 0.88 | 0.90 | 50 |
| chenet | 0.75 | 0.92 | 0.83 | 50 |
| corn_kernel | 0.94 | 0.94 | 0.94 | 50 |
| eggplant | 0.92 | 0.90 | 0.91 | 50 |
| greengage | 0.67 | 0.60 | 0.63 | 50 |
| greengage | 0.48 | 0.48 | 0.48 | 50 |
| greengage | 0.84 | 0.84 | 0.84 | 50 |
| greengage | 0.44 | 0.38 | 0.41 | 50 |
| greengage | 0.83 | 0.86 | 0.84 | 50 |
| grape | 0.90 | 0.90 | 0.90 | 50 |
| grape | 0.76 | 0.70 | 0.73 | 50 |
| hog_plum | 0.98 | 0.92 | 0.95 | 50 |
| hog_plum | 0.90 | 0.94 | 0.92 | 50 |
| passion_fruit | 0.92 | 0.90 | 0.91 | 50 |
| passion_fruit | 0.75 | 0.78 | 0.76 | 50 |

### Global Metrics

| Metric | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **Accuracy** | | | 0.75 | 5000 |
| **Macro Avg** | 0.75 | 0.75 | 0.75 | 5000 |
| **Weighted Avg** | 0.75 | 0.75 | 0.75 | 5000 |



## Future Work

- Automate training with a YAML file. ✅📄
- Test multiple backbones to improve model performance. 🏋️‍♂️🔁
- Add images and graphs to illustrate results. 📊📈
- Incorporate an F1 score table to evaluate model performance. 📋🏆

<small>this file was enhanced through my Mistral AI Agent</small>