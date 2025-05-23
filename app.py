from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
import asyncio
import logging
from dietary_recommendation_system import DietaryRecommendationAPI

app = Flask(__name__)

CORS(app) 


recommendation_api = DietaryRecommendationAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint to get dietary recommendations"""
    try:
        data = request.get_json()
        
        condition = data.get('condition', '').strip()
        allergies_text = data.get('allergies', '').strip()
        
        # Parse allergies
        allergies = []
        if allergies_text:
            allergies = [allergy.strip().lower() for allergy in allergies_text.split(',')]
        
        if not condition:
            return jsonify({
                'status': 'error',
                'message': 'Please provide a health condition'
            }), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                recommendation_api.get_recommendations(condition, allergies)
            )
        finally:
            loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing your request'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)