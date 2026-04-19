#!/usr/bin/env python3
"""run.py — NeuroLearn-GEN v2 entry point"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvicorn
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║       NeuroLearn-GEN  v2.0.0                 ║
║  Adaptive: Teach → Quiz → Reteach → Test     ║
╠══════════════════════════════════════════════╣
║  Flow:                                       ║
║   POST /learn/start        ← begin topic     ║
║   POST /learn/quiz/start   ← mini-quiz       ║
║   POST /learn/quiz/submit  ← check answers   ║
║   POST /learn/reteach      ← fix weak areas  ║
║   POST /learn/test/start   ← final test      ║
║   POST /learn/test/submit  ← get report      ║
║                                              ║
║  Docs: http://localhost:8000/docs            ║
╚══════════════════════════════════════════════╝
    """)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000,
                reload=os.getenv("DEBUG","true").lower()=="true", log_level="info")
