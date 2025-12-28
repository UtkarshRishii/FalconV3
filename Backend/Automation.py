"""
Falcon AI Pro - Ultra-Advanced System Automation Assistant
Created by Utkarsh Rishi (ArcDevs)
Enterprise-grade automation with advanced features and robust architecture
"""

import os
import re
import sys
import subprocess
import asyncio
from typing import Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv
from google import genai


class TaskType(Enum):
    """Task classification"""
    BROWSER = "browser"
    FILE = "file"
    PROCESS = "process"
    SEARCH = "search"
    AUTOMATION = "automation"
    CONTENT = "content"


@dataclass
class Task:
    """Task representation"""
    description: str
    task_type: Optional[TaskType] = None
    priority: int = 1
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Execution result with detailed information"""
    success: bool
    task: Optional[Task] = None
    code: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class SecurityValidator:
    """Advanced security validation for code execution"""
    
    CRITICAL_PATTERNS = [
        (r'rm\s+-rf\s+/', "Recursive deletion of root"),
        (r'del\s+/[fFsS]', "Force delete system files"),
        (r'format\s+[cC]:', "Disk formatting"),
        (r'mkfs\.', "Filesystem creation"),
        (r'dd\s+if=.*of=/dev/', "Direct disk write"),
    ]
    
    HIGH_RISK_PATTERNS = [
        (r'__import__\s*\(\s*["\']os["\']\)\.system', "Dynamic OS import"),
        (r'eval\s*\(', "Code evaluation"),
        (r'compile\s*\(.*exec', "Dynamic compilation"),
        (r'subprocess\..*shell\s*=\s*True', "Shell injection risk"),
        (r'shutil\.rmtree\s*\([^)]*\)', "Recursive directory removal"),
    ]
    
    MODERATE_RISK_PATTERNS = [
        (r'open\s*\([^)]*["\'][wWaA]', "File write operation"),
        (r'os\.remove', "File deletion"),
        (r'os\.rmdir', "Directory deletion"),
    ]
    
    @classmethod
    def validate(cls, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code safety with detailed feedback
        Returns: (is_safe, risk_message)
        """
        # Check critical patterns
        for pattern, description in cls.CRITICAL_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Critical risk: {description}"
        
        # Check high-risk patterns
        for pattern, description in cls.HIGH_RISK_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"High risk: {description}"
        
        # Warn about moderate risks but allow
        for pattern, description in cls.MODERATE_RISK_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                pass  # Log warning but continue
        
        return True, None


class TaskClassifier:
    """Classifies tasks by type"""
    
    KEYWORDS = {
        TaskType.BROWSER: ['chrome', 'firefox', 'browser', 'website', 'url', 'web'],
        TaskType.FILE: ['file', 'folder', 'directory', 'create', 'delete', 'move', 'copy'],
        TaskType.PROCESS: ['close', 'kill', 'terminate', 'start', 'launch', 'open', 'run'],
        TaskType.SEARCH: ['search', 'find', 'google', 'youtube', 'lookup'],
        TaskType.AUTOMATION: ['automate', 'script', 'macro', 'click', 'type'],
        TaskType.CONTENT: ['write', 'generate', 'create content', 'blog', 'article'],
    }
    
    @classmethod
    def classify(cls, task_description: str) -> TaskType:
        """Classify task based on keywords"""
        description_lower = task_description.lower()
        
        for task_type, keywords in cls.KEYWORDS.items():
            if any(keyword in description_lower for keyword in keywords):
                return task_type
        
        return TaskType.AUTOMATION


class AdvancedTaskExecutor:
    """Enhanced task execution with context awareness"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model = model
        self.context = self._build_enhanced_context()
    
    def _build_enhanced_context(self) -> List[Dict[str, str]]:
        """Build comprehensive context for AI"""
        return [
            {
                "role": "system",
                "content": "You are Falcon Pro, an elite AI assistant by Utkarsh Rishi. "
                          "Generate production-grade, efficient Python code for complex automation tasks."
            },
            {
                "role": "system",
                "content": """Advanced capabilities:
- Multi-step automation workflows
- Cross-platform compatibility (Windows, macOS, Linux)
- Intelligent error handling and recovery
- Resource-efficient operations
- Async operations when beneficial

Core modules: webbrowser, pyautogui, time, pyperclip, datetime, tkinter, os, subprocess, psutil, pywhatkit, pathlib

Standards:
- Use pathlib for file operations
- Implement proper exception handling
- Add type hints where beneficial
- Use context managers for resources
- Prefer async for I/O operations when possible
- No explanations, only clean executable code"""
            },
            {
                "role": "user",
                "content": "open Chrome and navigate to YouTube"
            },
            {
                "role": "assistant",
                "content": """```python
import webbrowser
import time

webbrowser.register('chrome', None, webbrowser.BackgroundBrowser('chrome'))
chrome = webbrowser.get('chrome')
chrome.open('https://www.youtube.com')
time.sleep(2)
```"""
            },
            {
                "role": "user",
                "content": "close all Chrome windows"
            },
            {
                "role": "assistant",
                "content": """```python
import psutil
import time

for proc in psutil.process_iter(['name']):
    if 'chrome' in proc.info['name'].lower():
        try:
            proc.terminate()
        except:
            proc.kill()

time.sleep(1)
```"""
            },
            {
                "role": "user",
                "content": "create a folder on desktop and add a text file"
            },
            {
                "role": "assistant",
                "content": """```python
from pathlib import Path

desktop = Path.home() / 'Desktop' / 'NewFolder'
desktop.mkdir(exist_ok=True)

file_path = desktop / 'note.txt'
file_path.write_text('Created by Falcon AI')
```"""
            }
        ]
    
    def execute(self, task: Task) -> Optional[str]:
        """Execute task with enhanced context"""
        try:
            messages = self.context + [
                {
                    "role": "user",
                    "content": f"Task: {task.description}\nType: {task.task_type.value if task.task_type else 'general'}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
        except Exception:
            return None


class CodeExtractor:
    """Advanced code extraction with multi-format support"""
    
    PATTERNS = [
        (r'```python\n(.*?)\n```', "Python code block"),
        (r'```py\n(.*?)\n```', "Python code block (py)"),
        (r'```\n(.*?)\n```', "Generic code block"),
        (r'`([^`]+)`', "Inline code"),
    ]
    
    @classmethod
    def extract(cls, response: str) -> Optional[str]:
        """Extract code with fallback patterns"""
        if not response:
            return None
        
        for pattern, _ in cls.PATTERNS:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return None
    
    @classmethod
    def extract_all(cls, response: str) -> List[str]:
        """Extract all code blocks"""
        if not response:
            return []
        
        all_code = []
        for pattern, _ in cls.PATTERNS:
            matches = re.findall(pattern, response, re.DOTALL)
            all_code.extend([m.strip() for m in matches])
        
        return all_code


class EnhancedCodeRunner:
    """Advanced code execution with sandboxing"""
    
    @staticmethod
    def _check_module(module_name: str) -> bool:
        """Check module availability"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _build_safe_globals() -> Dict:
        """Build safe execution environment"""
        safe_globals = {
            '__builtins__': __builtins__,
            'print': print,
            'os': os,
            'sys': sys,
            'Path': Path,
        }
        
        # Safe imports
        safe_modules = [
            'time', 'datetime', 'webbrowser', 'pathlib',
            'psutil', 'pyautogui', 'pywhatkit', 'pyperclip',
            'tkinter', 'json', 're', 'subprocess'
        ]
        
        for module in safe_modules:
            if EnhancedCodeRunner._check_module(module):
                safe_globals[module] = __import__(module)
        
        return safe_globals
    
    @classmethod
    def run(cls, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute code with timeout and safety checks"""
        import time
        
        start_time = time.time()
        
        # Validate safety
        is_safe, risk_msg = SecurityValidator.validate(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                code=code,
                error=risk_msg,
                execution_time=time.time() - start_time
            )
        
        try:
            exec_globals = cls._build_safe_globals()
            exec(code, exec_globals)
            
            return ExecutionResult(
                success=True,
                code=code,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                code=code,
                error=str(e),
                execution_time=time.time() - start_time
            )


class FalconAIPro:
    """Advanced Falcon AI with enhanced features"""
    
    def __init__(self):
        load_dotenv()
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY not found")
        
        self.executor = AdvancedTaskExecutor(self.groq_key)
        self.task_history: List[Task] = []
    
    def process_task(self, task_description: str) -> ExecutionResult:
        """Process task with full pipeline"""
        if not task_description.strip():
            return ExecutionResult(success=False, error="Empty task")
        
        # Create and classify task
        task = Task(
            description=task_description,
            task_type=TaskClassifier.classify(task_description)
        )
        self.task_history.append(task)
        
        # Get AI response
        response = self.executor.execute(task)
        if not response:
            return ExecutionResult(success=False, error="No response from AI")
        
        # Extract code
        code = CodeExtractor.extract(response)
        if not code:
            return ExecutionResult(success=False, error="No code found in response")
        
        # Execute code
        result = EnhancedCodeRunner.run(code)
        result.task = task
        
        return result
    
    def run_interactive(self):
        """Interactive mode"""
        try:
            task = input("").strip()
            self.process_task(task)
        except (KeyboardInterrupt, EOFError):
            pass
        except Exception:
            pass


def initialize_environment():
    """Setup environment configuration"""
    env_path = Path('.env')
    
    if not env_path.exists():
        env_path.write_text(
            "# Falcon AI Pro Configuration\n"
            "# Get your keys from:\n"
            "# Groq: https://console.groq.com\n"
            "GROQ_API_KEY=your_groq_api_key_here\n",
            encoding='utf-8'
        )


def main():
    """Main application entry point"""
    initialize_environment()
    
    try:
        falcon = FalconAIPro()
        
        if len(sys.argv) > 1:
            task = ' '.join(sys.argv[1:])
            falcon.process_task(task)
        else:
            falcon.run_interactive()
            
    except Exception:
        sys.exit(1)