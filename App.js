import React, { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);

  const {
    transcript: speechTranscript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
    if (speechTranscript) {
      setTranscript(speechTranscript);
    }
  }, [speechTranscript]);

  const startInterview = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/start-interview`);
      setCurrentQuestion(response.data.question);
      setSessionId(response.data.session_id);
      setInterviewStarted(true);
      setConversationHistory([{
        type: 'question',
        content: response.data.question,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (error) {
      console.error('Error starting interview:', error);
      alert('Failed to start interview. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const startRecording = () => {
    resetTranscript();
    setTranscript('');
    setIsRecording(true);
    SpeechRecognition.startListening({ continuous: true });
  };

  const stopRecording = async () => {
    SpeechRecognition.stopListening();
    setIsRecording(false);
    
    if (transcript.trim()) {
      await analyzeResponse(transcript);
    }
  };

  const analyzeResponse = async (responseText) => {
    try {
      setLoading(true);
      
      // For now, we'll use text analysis since audio upload is more complex
      const response = await axios.post(`${API_BASE_URL}/text-analysis`, null, {
        params: {
          transcript: responseText,
          question: currentQuestion
        }
      });

      const analysis = response.data.analysis;
      setFeedback(analysis);
      
      // Add to conversation history
      setConversationHistory(prev => [
        ...prev,
        {
          type: 'answer',
          content: responseText,
          timestamp: new Date().toLocaleTimeString(),
          feedback: analysis
        },
        {
          type: 'question',
          content: analysis.next_question,
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
      
      // Set next question
      setCurrentQuestion(analysis.next_question);
      setTranscript('');
      
    } catch (error) {
      console.error('Error analyzing response:', error);
      alert('Failed to analyze response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetInterview = () => {
    setInterviewStarted(false);
    setCurrentQuestion('');
    setTranscript('');
    setFeedback(null);
    setSessionId('');
    setConversationHistory([]);
    resetTranscript();
  };

  if (!browserSupportsSpeechRecognition) {
    return (
      <div className="app">
        <div className="error-message">
          <h2>Browser doesn't support speech recognition</h2>
          <p>Please use Chrome, Edge, or Safari for the best experience.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚀 AI Interview Simulator</h1>
        <p>Practice interviews with real-time AI feedback</p>
      </header>

      <main className="app-main">
        {!interviewStarted ? (
          <div className="start-screen">
            <div className="welcome-card">
              <h2>Welcome to Your Mock Interview</h2>
              <p>
                This AI-powered interview simulator will ask you questions and provide 
                real-time feedback on your confidence, clarity, and relevance.
              </p>
              <ul className="features-list">
                <li>🎤 Voice recognition technology</li>
                <li>🧠 Real-time NLP analysis</li>
                <li>📊 Instant feedback on your responses</li>
                <li>🔄 Dynamic follow-up questions</li>
              </ul>
              <button 
                className="start-button"
                onClick={startInterview}
                disabled={loading}
              >
                {loading ? 'Starting...' : 'Start Interview'}
              </button>
            </div>
          </div>
        ) : (
          <div className="interview-screen">
            <div className="interview-container">
              
              {/* Current Question */}
              <div className="question-section">
                <h3>Current Question:</h3>
                <div className="question-card">
                  <p>{currentQuestion}</p>
                </div>
              </div>

              {/* Recording Controls */}
              <div className="recording-section">
                <div className="recording-controls">
                  {!isRecording ? (
                    <button 
                      className="record-button"
                      onClick={startRecording}
                      disabled={loading}
                    >
                      🎤 Start Recording
                    </button>
                  ) : (
                    <button 
                      className="stop-button"
                      onClick={stopRecording}
                      disabled={loading}
                    >
                      ⏹️ Stop & Analyze
                    </button>
                  )}
                </div>
                
                {listening && (
                  <div className="recording-indicator">
                    <div className="pulse"></div>
                    <span>Listening...</span>
                  </div>
                )}
              </div>

              {/* Live Transcript */}
              {transcript && (
                <div className="transcript-section">
                  <h4>Your Response:</h4>
                  <div className="transcript-box">
                    <p>{transcript}</p>
                  </div>
                </div>
              )}

              {/* Feedback */}
              {feedback && (
                <div className="feedback-section">
                  <h4>AI Feedback:</h4>
                  <div className="feedback-card">
                    <div className="scores">
                      <div className="score-item">
                        <span className="score-label">Confidence:</span>
                        <span className="score-value">{Math.round(feedback.confidence_score)}%</span>
                        <div className="score-bar">
                          <div 
                            className="score-fill confidence"
                            style={{ width: `${feedback.confidence_score}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className="score-item">
                        <span className="score-label">Clarity:</span>
                        <span className="score-value">{Math.round(feedback.clarity_score)}%</span>
                        <div className="score-bar">
                          <div 
                            className="score-fill clarity"
                            style={{ width: `${feedback.clarity_score}%` }}
                          ></div>
                        </div>
                      </div>
                      <div className="score-item">
                        <span className="score-label">Relevance:</span>
                        <span className="score-value">{Math.round(feedback.relevance_score)}%</span>
                        <div className="score-bar">
                          <div 
                            className="score-fill relevance"
                            style={{ width: `${feedback.relevance_score}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <div className="feedback-text">
                      <p>{feedback.feedback}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Loading Indicator */}
              {loading && (
                <div className="loading-section">
                  <div className="loading-spinner"></div>
                  <p>Analyzing your response...</p>
                </div>
              )}

              {/* Interview Controls */}
              <div className="interview-controls">
                <button 
                  className="reset-button"
                  onClick={resetInterview}
                >
                  🔄 Reset Interview
                </button>
              </div>
            </div>

            {/* Conversation History */}
            <div className="history-section">
              <h4>Interview History:</h4>
              <div className="conversation-history">
                {conversationHistory.map((item, index) => (
                  <div key={index} className={`history-item ${item.type}`}>
                    <div className="history-header">
                      <span className="history-type">
                        {item.type === 'question' ? '🤖 AI' : '👤 You'}
                      </span>
                      <span className="history-time">{item.timestamp}</span>
                    </div>
                    <div className="history-content">
                      <p>{item.content}</p>
                      {item.feedback && (
                        <div className="mini-scores">
                          <span>C: {Math.round(item.feedback.confidence_score)}%</span>
                          <span>Cl: {Math.round(item.feedback.clarity_score)}%</span>
                          <span>R: {Math.round(item.feedback.relevance_score)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
