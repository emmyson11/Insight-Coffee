
from flask import Flask, render_template, request, jsonify
from main_code import rag_chain, vectorstore, retriever
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_cafes():
    try:
        user_query = request.json.get('query', '')
        if not user_query:
            return jsonify({'error': 'Please provide a query'}), 400

        # Use the existing RAG chain from main_code.py
        response = rag_chain.invoke(user_query)

        return jsonify({
            'success': True,
            'recommendations': response.content
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
