# Recommendation_System

This repository contains implementations of multiple recommendation systems for movies using different approaches and datasets.

## Recommendation Systems Implemented

### Recommendation System 1: Collaborative Filtering with TensorFlow Recommenders

**Description:** Uses TensorFlow and TensorFlow Recommenders to build a collaborative filtering recommendation system.

**Dataset:** Movielens 100k dataset.

**Features:**

1. Loads and preprocesses user ratings and movie details.

2. Builds embeddings for users and movies.

3. Defines a retrieval task to predict top-k recommendations.

4. Trains a model and evaluates its performance.

5. Provides functions to recommend movies for specific users and find similar movies based on user preferences.

### Recommendation System 2: Matrix Factorization with Keras

**Description:** Implements matrix factorization using Keras for collaborative filtering.

**Dataset:** Movielens 100k dataset.

**Features:**

1. Loads user ratings and movie details.

2. Preprocesses data by encoding user and movie IDs.

3. Constructs a neural network model with embeddings for users and movies.

4. Trains the model using SGD optimizer and sparse categorical cross-entropy loss.

5. Provides functions to recommend movies based on user preferences.

### Recommendation System 3: Content-Based Filtering with TF-IDF and Collaborative Filtering

**Description:** Integrates TF-IDF for content-based filtering and collaborative filtering for recommendations.

**Dataset:** Movielens 25m dataset.

**Features:**

1. Cleans and vectorizes movie titles using TF-IDF.

2. Implements a function to find movies similar to a given title based on cosine similarity of TF-IDF vectors.

3. Utilizes user ratings to recommend movies that similar users have rated highly.

## Setup and Usage

#### Running the Recommendation Systems:

Each recommendation system is contained in one colab notebook **Movie_recommendation_system.ipynb**.

Open and execute the notebook to explore each recommendation system.

#### Datasets:

The Movielens datasets (ml-100k and ml-25m) are used. You can download them from the Movielens website.

[Dataset](https://grouplens.org/datasets/movielens/)

#### Dependencies:

TensorFlow, TensorFlow Recommenders, TensorFlow Datasets, Scikit-learn, Pandas, NumPy, Keras

## Note

Added a new Recommendation System 4 and updated the ipynb file.

## Author

- [@Surya K S ](https://github.com/SuryaKS27/)
