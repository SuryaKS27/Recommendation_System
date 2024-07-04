# Movie Recommendation Systems

Welcome to the ultimate guide for movie recommendation systems! This repository showcases diverse machine learning techniques to help you find the perfect movie. From classic collaborative filtering to cutting-edge deep learning, explore how different methods can enhance your viewing experience.

## Table of Contents
1. [Introduction](#introduction)
2. [Recommendation Systems Implemented](#recommendation-systems-implemented)
    - [TensorFlow Recommenders: Collaborative Filtering](#1-tensorflow-recommenders-collaborative-filtering)
    - [Keras: Matrix Factorization](#2-keras-matrix-factorization)
    - [TF-IDF & Collaborative Filtering: Hybrid Approach](#3-tf-idf--collaborative-filtering-hybrid-approach)
    - [Deep Learning: Neural Collaborative Filtering](#4-deep-learning-neural-collaborative-filtering)
3. [Setup and Usage](#setup-and-usage)
4. [Datasets](#datasets)
5. [Dependencies](#dependencies)
6. [Author](#author)

## Introduction

In this repository, we implement multiple recommendation systems to personalize your movie-watching experience. These systems range from traditional methods to advanced neural networks, each with its unique approach to predicting what you'll enjoy next.

## Recommendation Systems Implemented

### 1. TensorFlow Recommenders: Collaborative Filtering

**Overview:**
Harness the power of TensorFlow Recommenders for collaborative filtering. This system focuses on learning from user interactions to make accurate movie suggestions.

**Key Components:**
- **Embeddings:** Learns dense representations for users and movies.
- **Retrieval Task:** Uses embeddings to retrieve the top-K recommendations for users.
- **Evaluation:** Measures the model’s performance to ensure high-quality suggestions.
- **Custom Functions:** Provides tailored recommendations for users and finds movies similar to those they like.

**Dataset:** Movielens 100k

### 2. Keras: Matrix Factorization

**Overview:**
Matrix factorization with Keras offers a powerful way to decompose the user-item interaction matrix, revealing latent features that predict user preferences.

**Key Components:**
- **Data Preparation:** Encodes user and movie IDs and splits the dataset for training and testing.
- **Neural Network:** Builds a neural network with embedding layers for users and movies.
- **Training:** Uses stochastic gradient descent (SGD) and sparse categorical cross-entropy loss for optimization.
- **Recommendations:** Generates movie suggestions based on learned user preferences.

**Dataset:** Movielens 100k

### 3. TF-IDF & Collaborative Filtering: Hybrid Approach

**Overview:**
Combine the strengths of content-based filtering and collaborative filtering with a hybrid model. This method integrates TF-IDF vectorization for movie titles with user rating patterns to provide well-rounded recommendations.

**Key Components:**
- **TF-IDF Vectorization:** Converts movie titles into numerical vectors based on term frequency-inverse document frequency.
- **Cosine Similarity:** Calculates the similarity between movies using TF-IDF vectors.
- **User Ratings:** Enhances recommendations by factoring in user rating patterns and preferences.

**Dataset:** Movielens 25m

### 4. Deep Learning: Neural Collaborative Filtering (just tried)

**Overview:**
Explore advanced neural collaborative filtering techniques using deep learning. This approach models complex interactions between users and movies through deep neural networks.

**Key Components:**
- **Embedding Layers:** Creates dense vector representations for users and movies.
- **Dense Layers:** Stacks fully connected layers to capture intricate user-movie interactions.
- **Training:** Optimizes the model using the Adam optimizer and mean squared error loss.
- **Interactive Widgets:** Provides an interactive interface for users to select movies and get recommendations, or find similar movies.

**Dataset:** Movielens 25m

## Setup and Usage

### Running the Recommendation Systems

Each recommendation system is encapsulated in a Colab notebook: **movie_recommendation_system.ipynb**. Open the notebook, follow the instructions, and start exploring the models.

### Datasets

We use the Movielens datasets (ml-100k and ml-25m). You can download them from the [Movielens website](https://grouplens.org/datasets/movielens/).

### Dependencies

Ensure you have the following dependencies installed:
- TensorFlow
- TensorFlow Recommenders
- TensorFlow Datasets
- Scikit-learn
- Pandas
- NumPy
- Keras

## Author

- [Surya K S](https://github.com/SuryaKS27)
