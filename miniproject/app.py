import os
from dotenv import load_dotenv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

load_dotenv()

app = Flask(__name__)

api_key = os.getenv('GOOGLE_BOOKS_API_KEY')

# Function to fetch book data with pagination support
def get_books(query, page=1, max_results=8):
    start_index = (page - 1) * max_results  # Calculate start index for pagination
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        'q': query,
        'maxResults': max_results,
        'startIndex': start_index,
        'key': api_key 
    }
    response = requests.get(url, params=params)
    data = response.json()

    books = []
    for item in data.get('items', []):
        book_info = item.get('volumeInfo', {})
        book = {
            'title': book_info.get('title', 'No Title'),
            'authors': book_info.get('authors', ['No Author']),
            'description': book_info.get('description', 'No Description'),
            'categories': book_info.get('categories', ['No Categories']),
            'preview_link': book_info.get('previewLink', ''),
            'cover_image': book_info.get('imageLinks', {}).get('thumbnail', None)
        }
        
        if not book['cover_image']:
            book['cover_image'] = 'https://via.placeholder.com/150x200?text=No+Cover'  # Default placeholder image

        books.append(book)

    return books

# Content-based book recommendation function using TF-IDF
def recommend_books(book_data, book_idx, num_recommendations=5):
    descriptions = [book['description'] for book in book_data]
    
    # Vectorize descriptions with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix)
    
    # Get indices of most similar books
    similar_books = cosine_sim.argsort()[0, -num_recommendations-1:-1][::-1]
    
    recommendations = []
    for idx in similar_books:
        recommendations.append(book_data[idx])
    
    return recommendations

# Route for rendering the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for getting book recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query', default="fiction")
    page = int(request.args.get('page', default=1))  # Get current page from request
    
    books = get_books(query, page=page, max_results=8)
    
    if books:
        recommended_books = recommend_books(books, 0, num_recommendations=3)
        response = {
            'books': books,
            'recommended': recommended_books
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No books found'})

if __name__ == '__main__':
    app.run(debug=True)
