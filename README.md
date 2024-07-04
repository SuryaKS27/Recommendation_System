# Movie Recommendation Systems

Welcome to the Movie Recommendation Systems repository! Here, you'll find multiple implementations of recommendation systems that help you discover movies you’ll love. Whether you're a data science enthusiast, a machine learning expert, or just a movie buff, there's something here for you.

## Recommendation Systems Implemented

### **1. Collaborative Filtering with TensorFlow Recommenders**

**Description:**
Dive into the world of collaborative filtering with TensorFlow Recommenders. This system uses user interactions to find patterns and suggest movies.

**Dataset:** Movielens 100k dataset.

**Key Features:**

- **User and Movie Embeddings:** Learns and uses embeddings for users and movies to provide recommendations.
- **Top-K Recommendations:** Defines a retrieval task to predict the top-K recommendations tailored to each user.
- **Performance Evaluation:** Trains the model and evaluates its performance for reliable recommendations.
- **Custom Functions:** Includes functions to recommend movies for specific users and find similar movies based on user preferences.

### **2. Matrix Factorization with Keras**

**Description:**
Implement matrix factorization using Keras to provide collaborative filtering recommendations. This method factors the user-item interaction matrix to identify latent features.

**Dataset:** Movielens 100k dataset.

**Key Features:**

- **Data Preprocessing:** Loads user ratings and movie details, and preprocesses data by encoding user and movie IDs.
- **Neural Network Model:** Constructs a neural network with embeddings for users and movies.
- **Training with SGD:** Trains the model using the SGD optimizer and sparse categorical cross-entropy loss.
- **Personalized Recommendations:** Provides functions to recommend movies based on user preferences.

### **3. Content-Based Filtering with TF-IDF and Collaborative Filtering**

**Description:**
Combines the power of TF-IDF for content-based filtering with collaborative filtering for recommendations, offering a hybrid approach.

**Dataset:** Movielens 25m dataset.

**Key Features:**

- **TF-IDF Vectorization:** Cleans and vectorizes movie titles using TF-IDF.
- **Cosine Similarity:** Implements functions to find movies similar to a given title based on cosine similarity of TF-IDF vectors.
- **User Preferences:** Utilizes user ratings to recommend movies that similar users have rated highly.

### **4. Neural Collaborative Filtering with Deep Learning**

**Description:**
Explore deep learning for recommendation systems with neural collaborative filtering, integrating embeddings and neural networks for advanced movie recommendations.

**Dataset:** Movielens 25m dataset.

**Key Features:**

- **Embedding Layers:** Builds user and movie embeddings using TensorFlow and Keras.
- **Dense Layers:** Uses dense layers for learning non-linear interactions between user and movie embeddings.
- **Training and Evaluation:** Trains the model and evaluates its performance on a test set.
- **Interactive Widgets:** Provides interactive widgets to select users and movies, and get recommendations or find similar movies.

## Setup and Usage

### Running the Recommendation Systems:

Each recommendation system is contained in one Colab notebook: **movie_recommendation_system.ipynb**. Open and execute the notebook to explore each recommendation system.

### Datasets:

The Movielens datasets (ml-100k and ml-25m) are used. You can download them from the Movielens website.

[Dataset](https://grouplens.org/datasets/movielens/)

### Dependencies:

- TensorFlow
- TensorFlow Recommenders
- TensorFlow Datasets
- Scikit-learn
- Pandas
- NumPy
- Keras

## Author

- [@Surya K S](https://github.com/SuryaKS27/)

Dive into the world of movie recommendations and start exploring now! Happy coding and happy watching!
