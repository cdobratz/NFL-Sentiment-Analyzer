from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pymongo
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = FastAPI(title="NFL Sentiment Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection with error handling
try:
    client = pymongo.MongoClient(os.getenv("MONGODB_URL"))
    # Verify connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas!")
    db = client[os.getenv("DATABASE_NAME", "nfl_sentiment")]
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    raise

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

@app.get("/")
async def root():
    return {"message": "NFL Sentiment Analysis API", "status": "connected to MongoDB Atlas"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    try:
        result = sentiment_analyzer(input_data.text)[0]
        sentiment_data = {
            "text": input_data.text,
            "sentiment": result["label"],
            "score": float(result["score"]),
            "timestamp": datetime.utcnow()
        }
        
        # Store in MongoDB
        db.sentiments.insert_one(sentiment_data)
        
        return sentiment_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent", response_model=List[SentimentResponse])
async def get_recent_sentiments():
    try:
        recent = list(db.sentiments.find(
            {}, 
            {"_id": 0, "timestamp": 0}
        ).sort([("timestamp", -1)]).limit(10))
        return recent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
