from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import whisper
import openai
import os
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Global model references
whisper_model = None
sentiment_analyzer = None
sentence_model = None
openai.api_key = os.getenv("OPENAI_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up models"""
    global whisper_model, sentiment_analyzer, sentence_model
    
    print("Loading AI models...")
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model("base")
        
        # Load sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        
        # Load sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("All models loaded successfully!")
        yield
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise
    finally:
        # Cleanup resources if needed
        print("Cleaning up resources...")
        torch.cuda.empty_cache()

# Initialize FastAPI with lifespan
app = FastAPI(
    title="AI Interview Simulator", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InterviewQuestion(BaseModel):
    question: str
    context: str = ""

class AnalysisResponse(BaseModel):
    confidence_score: float
    clarity_score: float
    relevance_score: float
    feedback: str
    next_question: str

class TranscriptionRequest(BaseModel):
    audio_data: str

@app.get("/")
async def root():
    return {"message": "AI Interview Simulator API is running!"}

@app.post("/start-interview")
async def start_interview():
    """Start a new interview session"""
    initial_question = "Hello! Welcome to your mock interview. Let's start with a simple question: Can you tell me about yourself and your background?"
    
    return {
        "question": initial_question,
        "session_id": "interview_001",
        "status": "started"
    }

@app.post("/analyze-response")
async def analyze_response(audio_file: UploadFile = File(...), question: str = ""):
    """Analyze user's audio response and provide feedback"""
    try:
        # Save uploaded audio file temporarily
        audio_path = f"temp_audio_{audio_file.filename}"
        with open(audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Transcribe audio using Whisper
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]
        
        # Clean up temp file
        os.remove(audio_path)
        
        # Analyze the response
        analysis = await analyze_text_response(transcript, question)
        
        return {
            "transcript": transcript,
            "analysis": analysis,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

async def analyze_text_response(transcript: str, question: str) -> Dict:
    """Analyze transcript for confidence, clarity, and relevance"""
    # 1. Confidence Analysis
    confidence_score = analyze_confidence(transcript)
    
    # 2. Clarity Analysis
    clarity_score = analyze_clarity(transcript)
    
    # 3. Relevance Analysis
    relevance_score = analyze_relevance(transcript, question)
    
    # Generate feedback
    feedback = generate_feedback(confidence_score, clarity_score, relevance_score)
    
    # Generate next question
    next_question = await generate_next_question(transcript, question)
    
    return {
        "confidence_score": confidence_score,
        "clarity_score": clarity_score,
        "relevance_score": relevance_score,
        "feedback": feedback,
        "next_question": next_question
    }

def analyze_confidence(transcript: str) -> float:
    """Analyze confidence based on sentiment and language patterns"""
    try:
        sentiment_scores = sentiment_analyzer(transcript)
        positive_score = next(
            (score['score'] for score in sentiment_scores[0] 
            if score['label'] == 'LABEL_2'), 
            0.5
        )
        
        confidence_words = ['confident', 'sure', 'definitely', 'absolutely']
        uncertainty_words = ['maybe', 'perhaps', 'might', 'possibly']
        
        words = transcript.lower().split()
        confidence_count = sum(1 for word in words if word in confidence_words)
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        
        word_confidence = max(0, (confidence_count - uncertainty_count) / len(words)) if words else 0
        final_score = (positive_score * 0.7 + word_confidence * 0.3) * 100
        
        return min(100, max(0, final_score))
        
    except Exception as e:
        print(f"Confidence analysis error: {e}")
        return 50.0

def analyze_clarity(transcript: str) -> float:
    """Analyze clarity based on grammar and structure"""
    try:
        if not transcript.strip():
            return 0.0
        
        sentences = sent_tokenize(transcript)
        words = word_tokenize(transcript)
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        optimal_length = 15
        length_score = 1 - abs(avg_sentence_length - optimal_length) / optimal_length
        
        filler_words = ['um', 'uh', 'like', 'you know']
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_penalty = min(0.3, filler_count / len(words)) if words else 0
        
        clarity_score = (max(0, length_score) * 0.6 + (1 - filler_penalty) * 0.4) * 100
        
        return min(100, max(0, clarity_score))
        
    except Exception as e:
        print(f"Clarity analysis error: {e}")
        return 50.0

def analyze_relevance(transcript: str, question: str) -> float:
    """Analyze relevance using semantic similarity"""
    try:
        if not transcript.strip() or not question.strip():
            return 0.0
        
        question_embedding = sentence_model.encode([question])
        answer_embedding = sentence_model.encode([transcript])
        
        similarity = np.dot(question_embedding[0], answer_embedding[0]) / (
            np.linalg.norm(question_embedding[0]) * np.linalg.norm(answer_embedding[0])
        )
        
        return min(100, max(0, similarity * 100))
        
    except Exception as e:
        print(f"Relevance analysis error: {e}")
        return 50.0

def generate_feedback(confidence: float, clarity: float, relevance: float) -> str:
    """Generate personalized feedback"""
    feedback = []
    
    if confidence >= 80:
        feedback.append("Excellent confidence!")
    elif confidence >= 60:
        feedback.append("Good confidence, but could be stronger.")
    else:
        feedback.append("Try to speak with more confidence.")
    
    if clarity >= 80:
        feedback.append("Very clear communication.")
    elif clarity >= 60:
        feedback.append("Clear but could reduce filler words.")
    else:
        feedback.append("Work on organizing your thoughts more clearly.")
    
    if relevance >= 80:
        feedback.append("Perfectly relevant response.")
    elif relevance >= 60:
        feedback.append("Mostly relevant but could be more focused.")
    else:
        feedback.append("Try to stay more focused on the question.")
    
    return " ".join(feedback)

async def generate_next_question(transcript: str, previous_question: str) -> str:
    """Generate follow-up question using OpenAI"""
    try:
        if not openai.api_key:
            return get_default_follow_up_question()
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are an interview coach. Generate a relevant follow-up question."
            }, {
                "role": "user",
                "content": f"Previous Q: {previous_question}\nResponse: {transcript}"
            }],
            max_tokens=100,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return get_default_follow_up_question()

def get_default_follow_up_question() -> str:
    """Fallback questions"""
    questions = [
        "Can you elaborate on that experience?",
        "How did that situation affect your approach?",
        "What did you learn from that experience?",
        "How would you apply that in this role?"
    ]
    import random
    return random.choice(questions)

@app.post("/text-analysis")
async def analyze_text_only(transcript: str, question: str = ""):
    """Text-only analysis endpoint"""
    try:
        analysis = await analyze_text_response(transcript, question)
        return {
            "transcript": transcript,
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)