import re
import requests
import pandas as pd
import pickle
import random
import numpy as np
import ast
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import pad_sequences # type: ignore
from flask import Flask, render_template, request
app = Flask(__name__)


TMDB_API_KEY = 'cd61242a4b31538d8b9f20959ce388d2'
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

similarity = pickle.load(open('models/similarity.pkl', 'rb'))

data = pd.read_csv('datasets/cleaned_data.csv')


def recommend(movie):
    if movie not in data['title'].values:
        return []

    index = data.loc[(data['title'] == movie)].index.to_list()[0]
    distances = sorted(list(enumerate(similarity[:][index])), reverse=True, key=lambda x: x[::-1][0])

    recommended_movies_data = []
    for i in distances[:7][1:]:
        movie_title = data.iloc[[i][0][0]].title
        movie_overview = data.iloc[[i][0][0]].tags
        movie_poster = fetch_poster(movie_title)
        # movie_backdrop = fetch_backdrop(movie_title)
        recommended_movies_data.append({'title': movie_title, 'overview': movie_overview, 'poster': movie_poster})
    return recommended_movies_data

def predict_sentiment(review_text):
    try:
        loaded_model = load_model('models\sentiment_model.h5')
        with open('models/tokenizer.pkl', 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading model or tokenizer: {e}")
        return "Error: Could not load necessary files."

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
        return text

    cleaned_review = clean_text(review_text)

    sequence = loaded_tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction = loaded_model.predict(padded_sequence)
    sentiment_score = prediction[0][0]

    if sentiment_score >= 0.5:
        tu = '/static/Thumbs up.svg'
        return tu
    else:
        td = '/static/Thumbs down.svg'
        return td

def fetch_poster(movie_title):
    cleaned_title = requests.utils.quote(movie_title)
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={cleaned_title}"

    try:
        response = requests.get(search_url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return TMDB_IMAGE_BASE_URL + poster_path
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {movie_title}: {e}")
    except ValueError:
        print(f"Failed to parse JSON for {movie_title}")

    return "https://via.placeholder.com/280x420?text=No+Image+Available"


def fetch_backdrop(movie_title):
    cleaned_title = requests.utils.quote(movie_title)
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={cleaned_title}"

    try:
        response = requests.get(search_url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data['results']:
            poster_path = data['results'][0].get('backdrop_path')
            if poster_path:
                return TMDB_IMAGE_BASE_URL + poster_path
    except requests.exceptions.RequestException as e:
        print(f"API request failed for {movie_title}: {e}")
    except ValueError:
        print(f"Failed to parse JSON for {movie_title}")

    return "https://via.placeholder.com/280x420?text=No+Image+Available"


def fetch_movie_video_key(movie_id):
    video_key = None
    if movie_id:
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        try:
            videos_response = requests.get(videos_url, timeout=5)
            videos_response.raise_for_status()
            videos_data = videos_response.json()

            if 'results' in videos_data and videos_data['results']:
                for video in videos_data['results']:
                    if video['site'] == 'YouTube' and (video['type'] == 'Trailer' or video['type'] == 'Teaser'):
                        video_key = video['key']
                        break
                if not video_key:
                    for video in videos_data['results']:
                        if video['site'] == 'YouTube':
                            video_key = video['key']
                            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching video for movie ID {movie_id}: {e}")
        except ValueError:
            print(f"Failed to parse JSON for video data for movie ID {movie_id}")
    return video_key

FEEDBACK_FILE = 'feedback.csv'
try:
    feedback_df = pd.read_csv(FEEDBACK_FILE)
except FileNotFoundError:
    feedback_df = pd.DataFrame(columns=['movie_title', 'name', 'review','SentimentScore'])

def save_feedback(movie_title, name, review, sentiment):
    global feedback_df
    new_feedback = pd.DataFrame([{'movie_title': movie_title, 'name': name, 'review': review, 'SentimentScore':sentiment}])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

def get_movie_reviews(movie_title):
    return feedback_df[feedback_df['movie_title'] == movie_title][['name', 'review', 'SentimentScore']].tail(3).to_dict('records')

@app.route('/movie_details', methods=['POST','GET'])
def movie_details():
    movie_title = request.args.get('title')
    movie_data = None
    video_key = None

    if request.method == 'POST':
        name = request.form.get('name')
        review = request.form.get('feedback')
        
        sentiment= predict_sentiment(review)
        
        if movie_title and name and review:
            save_feedback(movie_title, name, review,sentiment )
    

    if movie_title:
        movie_row = data[data['title'] == movie_title]
        if not movie_row.empty:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(movie_title)}"
            try:
                search_response = requests.get(search_url, timeout=5)
                search_response.raise_for_status()
                search_data = search_response.json()
                movie_id = None
                if search_data['results']:
                    movie_id = search_data['results'][0]['id']

                if movie_id:
                    video_key = fetch_movie_video_key(movie_id)

            except requests.exceptions.RequestException as e:
                print(f"Error searching for movie ID for video: {e}")
            except ValueError:
                print(f"Failed to parse JSON for movie ID search.")

            movie_data = {
                'title': movie_row['title'].iloc[0],
                'overview': movie_row['tags'].iloc[0],
                'poster': fetch_poster(movie_row['title'].iloc[0]),
                'video_key': video_key
            }
        else:
            return render_template('moviedata.html', movie=None)

    reviews = get_movie_reviews(movie_title)
    return render_template('moviedata.html', movie=movie_data, reviews=reviews)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        recommend_movies = recommend(name)
        return render_template('output.html', recommend_movies=recommend_movies)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
