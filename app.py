import joblib
import re
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pydantic import BaseModel
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the model and vectorizer from the .pkl files
model = joblib.load(r'models/logistic_regression_model.pkl')
vectorizer = joblib.load(r'models/tfidf_vectorizer.pkl')

# Initialize FastAPI app
app = FastAPI()

# Initialize the stemmer
port_stem = PorterStemmer()

# Mount the static directory for CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates directory for rendering HTML
templates = Jinja2Templates(directory="templates")

def preprocess_text(text):
    """Preprocess the input text by stemming and removing stopwords."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Define the request body schema using Pydantic
class NewsInput(BaseModel):
    title: str
    paragraph: str

# Root endpoint to render the form
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict(input_data: NewsInput):
    title = input_data.title
    paragraph = input_data.paragraph

    # Combine title and paragraph
    content = title + ' ' + paragraph

    # Preprocess the text
    preprocessed_text = preprocess_text(content)

    # Transform the text using the vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])

    # Make a prediction using the model
    prediction = model.predict(transformed_text)

    # Return prediction as a response
    return {"prediction": "Fake News" if prediction[0] == 1 else "Real News"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
