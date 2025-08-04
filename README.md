# FimlFeel: Movie Recommendation & Sentiment Analysis Engine

![FimlFeel Logo](https://github.com/kshvyadav/FilmFeel/blob/40ee4308f439c792be923329a3c8846ee31d3a54/static/logo.jpg)

FimlFeel is a web application that provides movie recommendations and performs sentiment analysis on movie reviews. Built with Python and the Flask framework, it leverages a combination of machine learning techniques to offer a personalized and insightful experience for movie lovers.

## ✨ Features

- **Personalized Movie Recommendations:** Discover new movies based on your preferences.
- **Content-Based Filtering:** The recommendation engine uses **Cosine Similarity** to suggest movies similar to the ones you like, based on their genre, cast, crew, and keywords.
- **Sentiment Analysis:** Gain quick insights into public opinion by analyzing movie reviews. The application uses a machine learning model (**LSTM**) trained on a large corpus of movie reviews to classify sentiments as positive or negative.
- **Intelligent Text Processing:** The system uses **Word2Vec** to convert text data into meaningful numerical representations, improving the accuracy of both the recommendation and sentiment analysis models.
- **Intuitive User Interface:** A clean and easy-to-use web interface built with Flask and Jinja2 templates, making it simple for users to search for movies and explore recommendations.

## 🚀 Technologies Used

- **Backend:** Python, Flask
- **Machine Learning Libraries:** Scikit-learn, NLTK, Gensim, TensorFlow/Keras (for LSTM)
- **Data Processing:** Pandas, NumPy
- **Styling:** HTML, CSS, JavaScript (optional)

## 🧠 How it Works

The FimlFeel engine operates on two core machine learning models:

1.  **Recommendation Engine:**
    - The system ingests a comprehensive movie dataset, including metadata like genres, cast, crew, and plot keywords.
    - It uses **Word2Vec** to create vector representations of the text features.
    - **Cosine Similarity** is then calculated between movies based on these vector representations. When a user selects a movie, the application finds other movies with the highest similarity scores and presents them as recommendations.

2.  **Sentiment Analysis Model:**
    - This model is a deep learning-based **LSTM** network.
    - It is trained on a dataset of movie reviews, where each review is labeled with a sentiment (positive, negative).
    - The text of the reviews is pre-processed and converted into word vectors using a **Word2Vec** model.
    - When a user requests a sentiment analysis for a movie, the model processes the available reviews and predicts the overall sentiment, providing a clear and concise summary.


---
