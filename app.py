from flask import Flask, render_template, request
import pickle
import requests
from surprise import SVD



app = Flask(__name__)

def utility_processor():
    def zipped(a, b):
        return zip(a, b)
    return dict(zip=zipped)


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def hybrid(userId, title, indices, cosine_sim, algo, smd, indices_map):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'movieId']]
    movies['est'] = movies['movieId'].apply(lambda x: algo.predict(userId, indices_map.get(x, 'id')).est)


    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    selected_movie = request.form['movie']
    userId = request.form['userId']  # Assuming you have a form field for userId
    # Load necessary model files
    indices = pickle.load(open('model/indices.pkl', 'rb'))
    id_map = pickle.load(open('model/id_map.pkl', 'rb'))
    cosine_sim = pickle.load(open('model/cosine_sim.pkl', 'rb'))
    algo = pickle.load(open('model/algo.pkl', 'rb'))
    smd = pickle.load(open('model/smd.pkl', 'rb'))
    indices_map = pickle.load(open('model/indices_map.pkl', 'rb'))
    recommended_movies = hybrid(userId, selected_movie, indices, cosine_sim, algo, smd, indices_map)
    recommended_movie_data = zip(recommended_movies['title'], [fetch_poster(movie_id) for movie_id in recommended_movies['movieId'].tolist()])
    return render_template('index.html', movie_data=recommended_movie_data)


if __name__ == '__main__':
    app.run(debug=True)