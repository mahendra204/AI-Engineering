from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .infer import SentimentModel

app = FastAPI(title="Sentiment API", version="0.1.0")
model = SentimentModel()

class Query(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Welcome To Sentiment Analysis Web Application </title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'calibri', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 1000px;
                width: 100%;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
                font-size: 50px;
            }
            .subtitle {
                color: #777;
                text-align: center;
                margin-bottom: 30px;
                font-size: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                color: #333;
                margin-bottom: 10px;
                font-weight: 600;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 1000px;
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
                min-height: 120px;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 100px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result.positive {
                background: #d4edda;
                border: 2px solid #28a745;
            }
            .result.negative {
                background: #f8d7da;
                border: 2px solid #dc3545;
            }
            .result-label {
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 10px;
            }
            .result.positive .result-label {
                color: #155724;
            }
            .result.negative .result-label {
                color: #721c24;
            }
            .scores {
                display: flex;
                gap: 20px;
                margin-top: 15px;
            }
            .score-item {
                flex: 1;
            }
            .score-label {
                font-size: 12px;
                font-weight: 600;
                margin-bottom: 5px;
                color: #333;
            }
            .score-bar {
                width: 100%;
                height: 8px;
                background: rgba(0,0,0,0.1);
                border-radius: 4px;
                overflow: hidden;
            }
            .score-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s;
            }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                margin-top: 10px;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #667eea;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #dc3545;
                padding: 10px;
                background: #f8d7da;
                border-radius: 5px;
                margin-top: 10px;
                display: none;
            }
            .error.show {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>😊 Do Analysis your text sentiment </h1>
            <p class="subtitle">Discover the sentiment of your text</p>
            
            <form id="sentimentForm">
                <div class="form-group">
                    <label for="textInput">Enter your text:</label>
                    <textarea id="textInput" placeholder="Type something here to do sentimenti analysis..." required></textarea>
                </div>
                <button type="submit">Analyze</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div> Analyzing...
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <div class="result-label" id="resultLabel"></div>
                <div class="scores" id="scores"></div>
            </div>
        </div>

        <script>
            const form = document.getElementById('sentimentForm');
            const textInput = document.getElementById('textInput');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const result = document.getElementById('result');
            const resultLabel = document.getElementById('resultLabel');
            const scoresDiv = document.getElementById('scores');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const text = textInput.value.trim();
                
                if (!text) return;
                
                loading.classList.add('show');
                error.classList.remove('show');
                result.classList.remove('show');
                
                try {
                    const response = await fetch(`/predict?text=${encodeURIComponent(text)}`);
                    if (!response.ok) throw new Error('API error');
                    
                    const data = await response.json();
                    
                    const label = data.label.toUpperCase();
                    const emoji = data.label === 'positive' ? '😊' : '😞';
                    resultLabel.textContent = `${emoji} ${label}`;
                    
                    scoresDiv.innerHTML = '';
                    for (const [key, value] of Object.entries(data.scores)) {
                        const scoreItem = document.createElement('div');
                        scoreItem.className = 'score-item';
                        scoreItem.innerHTML = `
                            <div class="score-label">${key.toUpperCase()}</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${value * 100}%"></div>
                            </div>
                            <div class="score-label">${(value * 100).toFixed(1)}%</div>
                        `;
                        scoresDiv.appendChild(scoreItem);
                    }
                    
                    result.classList.add(data.label);
                    result.classList.add('show');
                } catch (err) {
                    error.textContent = 'Error analyzing text. Please try again.';
                    error.classList.add('show');
                } finally {
                    loading.classList.remove('show');
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/predict")
def predict_get(text: str = "This product is amazing!"):
    result = model.predict(text)
    return result

@app.post("/predict")
def predict_post(q: Query):
    result = model.predict(q.text)
    return result
