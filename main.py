from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime

from app.auth.token_auth import TokenAuth
from app.models.pytorch_model import LabQAModel
from app.utils.logger import RequestLogger

app = FastAPI(title="Laboratory Quality Assurance System", version="1.0.0")

# Initialize components
token_auth = TokenAuth()
model = LabQAModel()
logger = RequestLogger()

# Mount static files
app.mount("/static", StaticFiles(directory="./static"), name="static")

class PatientData(BaseModel):
    sex: str
    age: int
    timeBetweenDraw: float
    
    # Current values (required)
    sodium_current: float
    potassium_current: float
    creatinine_current: float
    urea_current: float
    totalProtein_current: float
    albumin_current: float
    alkalinePhosphatase_current: float
    alanineTransaminase_current: float
    totalBilirubin_current: float
    
    # Previous values (optional)
    sodium_previous: Optional[float] = None
    potassium_previous: Optional[float] = None
    creatinine_previous: Optional[float] = None
    urea_previous: Optional[float] = None
    totalProtein_previous: Optional[float] = None
    albumin_previous: Optional[float] = None
    alkalinePhosphatase_previous: Optional[float] = None
    alanineTransaminase_previous: Optional[float] = None
    totalBilirubin_previous: Optional[float] = None

@app.get("/")
async def root(request: Request, token: str = None):
    """Serve the main interface with token authentication"""
    # if not token or not token_auth.validate_token(token):
        # return {"error": "Invalid or missing token", "message": "Please provide a valid token parameter"}
    
    # Log the access
    client_ip = request.client.host
    logger.log_access(token, client_ip, "main_interface")
    
    return FileResponse("app/static/index.html")

@app.post("/api/analyze-quality")
async def analyze_quality(
    patient_data: PatientData,
    request: Request,
    # token: str = Depends(token_auth.get_token_from_header)
):
    """Analyze laboratory quality with PyTorch model"""
    try:
        token = 'test123456'
        # Log the request
        client_ip = request.client.host
        logger.log_request(token, client_ip, "analyze_quality", patient_data.dict())
        
        # Process the data and get predictions
        results = model.predict(patient_data)
        
        # Log the response
        logger.log_response(token, "analyze_quality", "success")
        
        return results
        
    except Exception as e:
        logger.log_response(token, "analyze_quality", f"error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/health")
async def health_check(token: str = Depends(token_auth.get_token_from_query)):
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model.is_loaded()}

@app.get("/api/logs")
async def get_logs(token: str = Depends(token_auth.get_admin_token)):
    """Get access logs (admin only)"""
    return logger.get_recent_logs()

@app.post("/api/tokens")
async def create_token(
    request: Dict[str, Any],
    admin_token: str = Depends(token_auth.get_admin_token)
):
    """Create new access token (admin only)"""
    user_id = request.get("user_id")
    description = request.get("description", "")
    expires_days = request.get("expires_days", 30)
    
    token = token_auth.create_token(user_id, description, expires_days)
    return {"token": token, "expires_days": expires_days}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
