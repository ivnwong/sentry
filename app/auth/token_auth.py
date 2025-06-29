import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Query, Header
import hashlib

class TokenAuth:
    def __init__(self):
        self.tokens_file = "tokens.json"
        self.admin_token = os.getenv("ADMIN_TOKEN", "admin_demo_token_2024")
        self.tokens = self._load_tokens()
    
    def _load_tokens(self) -> Dict[str, Any]:
        """Load tokens from file"""
        if os.path.exists(self.tokens_file):
            try:
                with open(self.tokens_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_tokens(self):
        """Save tokens to file"""
        with open(self.tokens_file, 'w') as f:
            json.dump(self.tokens, f, indent=2, default=str)
    
    def create_token(self, user_id: str, description: str = "", expires_days: int = 30) -> str:
        """Create a new access token"""
        token = secrets.token_urlsafe(32)
        
        self.tokens[token] = {
            "user_id": user_id,
            "description": description,
            "created": datetime.utcnow(),
            "expires": datetime.utcnow() + timedelta(days=expires_days),
            "access_count": 0,
            "last_access": None
        }
        
        self._save_tokens()
        return token
    
    def validate_token(self, token: str) -> bool:
        """Validate if token exists and is not expired"""
        if not token or token not in self.tokens:
            return False
        
        token_data = self.tokens[token]
        if datetime.fromisoformat(token_data["expires"]) < datetime.utcnow():
            return False
        
        # Update access info
        self.tokens[token]["access_count"] += 1
        self.tokens[token]["last_access"] = datetime.utcnow()
        self._save_tokens()
        
        return True
    
    def get_token_from_query(self, token: Optional[str] = Query(None)) -> str:
        """Get and validate token from query parameter"""
        if not token or not self.validate_token(token):
            raise HTTPException(status_code=401, detail="Invalid or missing token")
        return token
    
    def get_token_from_header(self, authorization: Optional[str] = Header(None)) -> str:
        """Get and validate token from Authorization header"""
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(status_code=401, detail="Invalid authorization scheme")
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        if not self.validate_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return token
    
    def get_admin_token(self, admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> str:
        """Validate admin token"""
        if admin_token != self.admin_token:
            raise HTTPException(status_code=403, detail="Admin access required")
        return admin_token
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token information"""
        return self.tokens.get(token)
