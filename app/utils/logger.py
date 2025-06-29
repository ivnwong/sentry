import json
import os
from datetime import datetime
from typing import Dict, Any, List
import threading

class RequestLogger:
    def __init__(self):
        self.log_file = "logs/access.log"
        self.request_log_file = "logs/requests.log"
        self._ensure_log_directory()
        self.lock = threading.Lock()
    
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        os.makedirs("logs", exist_ok=True)
    
    def _write_log(self, log_file: str, data: Dict[str, Any]):
        """Write log entry to file"""
        with self.lock:
            with open(log_file, 'a') as f:
                f.write(json.dumps(data, default=str) + '\n')
    
    def log_access(self, token: str, client_ip: str, endpoint: str):
        """Log access attempt"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "access",
            "token": token[:8] + "...",  # Masked token
            "client_ip": client_ip,
            "endpoint": endpoint
        }
        self._write_log(self.log_file, log_entry)
    
    def log_request(self, token: str, client_ip: str, endpoint: str, request_data: Dict[str, Any]):
        """Log API request"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "token": token[:8] + "...",
            "client_ip": client_ip,
            "endpoint": endpoint,
            "request_data": request_data
        }
        self._write_log(self.request_log_file, log_entry)
    
    def log_response(self, token: str, endpoint: str, status: str):
        """Log API response"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response", 
            "token": token[:8] + "...",
            "endpoint": endpoint,
            "status": status
        }
        self._write_log(self.request_log_file, log_entry)
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        logs = []
        
        # Read access logs
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                for line in f.readlines()[-limit:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except:
                        pass
        
        # Read request logs
        if os.path.exists(self.request_log_file):
            with open(self.request_log_file, 'r') as f:
                for line in f.readlines()[-limit:]:
                    try:
                        logs.append(json.loads(line.strip()))
                    except:
                        pass
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return logs[:limit]
