"""
FALCON AI Assistant - Advanced Personal AI Companion
With Sophisticated Long-Term Memory System
"""

import os
import sys
import json
import sqlite3
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Backend.Automation import FalconAIPro
from Backend.RealTimeAI import search_floxai
from Backend.ImageGen import Main as ImageGenMain

# Environment setup
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY)


class MemoryType(Enum):
    """Types of memories stored"""
    EPISODIC = "episodic"  # Specific conversations and events
    SEMANTIC = "semantic"  # Facts, preferences, knowledge about user
    PROCEDURAL = "procedural"  # How to do tasks, user's workflows
    EMOTIONAL = "emotional"  # Emotional context and sentiment


class ToolType(Enum):
    """Available tool types"""
    SYSTEM_TASK = "execute_system_task"
    IMAGE_GEN = "generate_image"
    CONTENT_WRITER = "write_content"
    REALTIME_SEARCH = "realtime_information_search"


@dataclass
class Memory:
    """Structured memory entry"""
    content: str
    memory_type: str
    importance: float  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: Optional[datetime] = None
    embedding: Optional[str] = None  # JSON string of embedding vector
    tags: Optional[List[str]] = None
    related_memories: Optional[List[int]] = None


@dataclass
class UserProfile:
    """Dynamic user profile built from memories"""
    name: str = "Utkarsh Rishi"
    preferences: Dict[str, Any] = None
    interests: List[str] = None
    work_context: Dict[str, Any] = None
    communication_style: Dict[str, Any] = None
    important_facts: List[str] = None
    
    def __post_init__(self):
        self.preferences = self.preferences or {}
        self.interests = self.interests or []
        self.work_context = self.work_context or {}
        self.communication_style = self.communication_style or {}
        self.important_facts = self.important_facts or []


class AdvancedMemorySystem:
    """
    Sophisticated long-term memory system with:
    - Semantic memory storage
    - Importance-based retention
    - Memory consolidation
    - Intelligent retrieval
    - User profile building
    """
    
    def __init__(self, db_path: str = 'Database/FALCON_MEMORY.db'):
        self.db_path = db_path
        self._ensure_database_exists()
        self._initialize_schema()
        self.user_profile = self._load_user_profile()
    
    def _ensure_database_exists(self) -> None:
        """Create database directory if needed"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get optimized database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    def _initialize_schema(self) -> None:
        """Initialize comprehensive memory database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    importance REAL DEFAULT 0.5,
                    sentiment REAL,
                    embedding TEXT
                )
            ''')
            
            # Semantic memories table (extracted facts and knowledge)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    embedding TEXT,
                    metadata TEXT
                )
            ''')
            
            # User profile and preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )
            ''')
            
            # Memory connections (for related memories)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 INTEGER,
                    memory_id_2 INTEGER,
                    relation_type TEXT,
                    strength REAL DEFAULT 0.5,
                    FOREIGN KEY (memory_id_1) REFERENCES semantic_memories(id),
                    FOREIGN KEY (memory_id_2) REFERENCES semantic_memories(id)
                )
            ''')
            
            # Memory tags for categorical organization
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER,
                    tag TEXT,
                    FOREIGN KEY (memory_id) REFERENCES semantic_memories(id)
                )
            ''')
            
            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_importance ON conversations(importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON semantic_memories(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_importance ON semantic_memories(importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_access ON semantic_memories(access_count DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON memory_tags(tag)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_profile_category ON user_profile(category)')
            
            conn.commit()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate simple text embedding using hash-based approach"""
        # Simple embedding using character n-gram hashing
        # For production, use OpenAI embeddings or sentence-transformers
        words = text.lower().split()
        embedding = [0.0] * 128
        
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(128):
                embedding[i] += ((hash_val >> i) & 1) * (1.0 / len(words))
        
        # Normalize
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, dot_product))
    
    def _calculate_importance(self, user_message: str, assistant_message: str) -> float:
        """Calculate conversation importance based on heuristics"""
        importance = 0.5  # Base importance
        
        # Importance indicators
        important_keywords = [
            'remember', 'important', 'always', 'never', 'prefer', 'like', 'dislike',
            'my name', 'i am', 'i work', 'my job', 'my project', 'password', 'email',
            'birthday', 'anniversary', 'family', 'friend', 'colleague'
        ]
        
        text = (user_message + " " + (assistant_message or "")).lower()
        
        # Check for important keywords
        for keyword in important_keywords:
            if keyword in text:
                importance += 0.1
        
        # Length indicates detail
        if len(text) > 200:
            importance += 0.1
        
        # Questions about user preferences
        if any(q in text for q in ['what do you', 'how do you', 'do you prefer']):
            importance += 0.15
        
        return min(1.0, importance)
    
    def _extract_semantic_memories(self, conversation_id: int, user_message: str, 
                                   assistant_message: str) -> List[Dict]:
        """Extract semantic memories from conversation using LLM"""
        try:
            extraction_prompt = f"""Analyze this conversation and extract important information about the user that should be remembered long-term.

User: {user_message}
Assistant: {assistant_message}

Extract and return ONLY the following in JSON format (if found):
{{
    "facts": ["factual information about the user"],
    "preferences": ["user preferences and likes/dislikes"],
    "work_context": ["work-related information, projects, tools used"],
    "interests": ["user interests and hobbies"],
    "important_details": ["names, dates, important events, commitments"]
}}

Return ONLY valid JSON. If nothing important to remember, return empty arrays."""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.3,
                max_tokens=512
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            if extracted_text.startswith('```json'):
                extracted_text = extracted_text.split('```json')[1].split('```')[0].strip()
            elif extracted_text.startswith('```'):
                extracted_text = extracted_text.split('```')[1].split('```')[0].strip()
            
            extracted = json.loads(extracted_text)
            
            memories = []
            
            # Process facts
            for fact in extracted.get('facts', []):
                if fact.strip():
                    memories.append({
                        'content': fact,
                        'memory_type': MemoryType.SEMANTIC.value,
                        'importance': 0.8,
                        'tags': ['fact', 'user_info']
                    })
            
            # Process preferences
            for pref in extracted.get('preferences', []):
                if pref.strip():
                    memories.append({
                        'content': pref,
                        'memory_type': MemoryType.SEMANTIC.value,
                        'importance': 0.7,
                        'tags': ['preference']
                    })
            
            # Process work context
            for work in extracted.get('work_context', []):
                if work.strip():
                    memories.append({
                        'content': work,
                        'memory_type': MemoryType.PROCEDURAL.value,
                        'importance': 0.75,
                        'tags': ['work', 'professional']
                    })
            
            # Process interests
            for interest in extracted.get('interests', []):
                if interest.strip():
                    memories.append({
                        'content': interest,
                        'memory_type': MemoryType.SEMANTIC.value,
                        'importance': 0.6,
                        'tags': ['interest', 'hobby']
                    })
            
            # Process important details
            for detail in extracted.get('important_details', []):
                if detail.strip():
                    memories.append({
                        'content': detail,
                        'memory_type': MemoryType.SEMANTIC.value,
                        'importance': 0.9,
                        'tags': ['important', 'personal']
                    })
            
            return memories
            
        except Exception:
            return []
    
    def add_conversation(self, user_message: str, assistant_message: Optional[str] = None,
                        session_id: Optional[str] = None) -> int:
        """Add conversation and extract semantic memories"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate importance
            importance = self._calculate_importance(user_message, assistant_message or "")
            
            # Generate embedding
            combined_text = f"{user_message} {assistant_message or ''}"
            embedding = self._generate_embedding(combined_text)
            embedding_json = json.dumps(embedding)
            
            # Insert conversation
            cursor.execute('''
                INSERT INTO conversations (user_message, assistant_message, session_id, importance, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_message, assistant_message, session_id, importance, embedding_json))
            
            conversation_id = cursor.lastrowid
            conn.commit()
            
            # Extract semantic memories if assistant responded
            if assistant_message:
                semantic_memories = self._extract_semantic_memories(
                    conversation_id, user_message, assistant_message
                )
                
                for memory in semantic_memories:
                    self.add_semantic_memory(
                        content=memory['content'],
                        memory_type=memory['memory_type'],
                        importance=memory['importance'],
                        tags=memory.get('tags', [])
                    )
            
            return conversation_id
    
    def update_assistant_response(self, conversation_id: int, assistant_message: str) -> None:
        """Update assistant response and extract memories"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user message
            cursor.execute('SELECT user_message FROM conversations WHERE id = ?', (conversation_id,))
            row = cursor.fetchone()
            user_message = row['user_message'] if row else ""
            
            # Update response
            importance = self._calculate_importance(user_message, assistant_message)
            combined_text = f"{user_message} {assistant_message}"
            embedding = self._generate_embedding(combined_text)
            embedding_json = json.dumps(embedding)
            
            cursor.execute('''
                UPDATE conversations 
                SET assistant_message = ?, importance = ?, embedding = ?
                WHERE id = ?
            ''', (assistant_message, importance, embedding_json, conversation_id))
            
            conn.commit()
            
            # Extract semantic memories
            semantic_memories = self._extract_semantic_memories(
                conversation_id, user_message, assistant_message
            )
            
            for memory in semantic_memories:
                self.add_semantic_memory(
                    content=memory['content'],
                    memory_type=memory['memory_type'],
                    importance=memory['importance'],
                    tags=memory.get('tags', [])
                )
    
    def add_semantic_memory(self, content: str, memory_type: str, 
                           importance: float = 0.5, tags: List[str] = None) -> int:
        """Add a semantic memory"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Generate embedding
            embedding = self._generate_embedding(content)
            embedding_json = json.dumps(embedding)
            
            # Check for duplicate
            cursor.execute('''
                SELECT id FROM semantic_memories 
                WHERE content = ? AND memory_type = ?
            ''', (content, memory_type))
            
            existing = cursor.fetchone()
            if existing:
                # Update importance if higher
                cursor.execute('''
                    UPDATE semantic_memories 
                    SET importance = MAX(importance, ?), access_count = access_count + 1
                    WHERE id = ?
                ''', (importance, existing['id']))
                conn.commit()
                return existing['id']
            
            # Insert new memory
            cursor.execute('''
                INSERT INTO semantic_memories (content, memory_type, importance, embedding)
                VALUES (?, ?, ?, ?)
            ''', (content, memory_type, importance, embedding_json))
            
            memory_id = cursor.lastrowid
            
            # Add tags
            if tags:
                for tag in tags:
                    cursor.execute('''
                        INSERT INTO memory_tags (memory_id, tag)
                        VALUES (?, ?)
                    ''', (memory_id, tag))
            
            conn.commit()
            return memory_id
    
    def retrieve_relevant_memories(self, query: str, limit: int = 10, 
                                   min_importance: float = 0.3) -> List[Dict]:
        """Retrieve relevant memories using semantic search"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Get all memories above minimum importance
            cursor.execute('''
                SELECT id, content, memory_type, importance, access_count, embedding
                FROM semantic_memories
                WHERE importance >= ?
                ORDER BY importance DESC, access_count DESC
                LIMIT 100
            ''', (min_importance,))
            
            memories = []
            for row in cursor.fetchall():
                # Calculate similarity
                memory_embedding = json.loads(row['embedding']) if row['embedding'] else []
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                
                # Combined score: similarity * importance * recency_factor
                recency_factor = min(1.0, row['access_count'] / 10.0 + 0.5)
                score = similarity * row['importance'] * recency_factor
                
                memories.append({
                    'id': row['id'],
                    'content': row['content'],
                    'memory_type': row['memory_type'],
                    'importance': row['importance'],
                    'score': score
                })
            
            # Sort by score and return top results
            memories.sort(key=lambda x: x['score'], reverse=True)
            
            # Update access count for retrieved memories
            for memory in memories[:limit]:
                cursor.execute('''
                    UPDATE semantic_memories 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id = ?
                ''', (datetime.now(), memory['id']))
            
            conn.commit()
            
            return memories[:limit]
    
    def get_episodic_context(self, limit: int = 5) -> List[Dict]:
        """Get recent important conversations for context"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get recent high-importance conversations
            cursor.execute('''
                SELECT user_message, assistant_message, timestamp, importance
                FROM conversations
                WHERE assistant_message IS NOT NULL
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'user': row['user_message'],
                    'assistant': row['assistant_message'],
                    'timestamp': row['timestamp'],
                    'importance': row['importance']
                })
            
            return conversations
    
    def _load_user_profile(self) -> UserProfile:
        """Load user profile from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            profile = UserProfile()
            
            cursor.execute('SELECT category, key, value FROM user_profile')
            for row in cursor.fetchall():
                category = row['category']
                key = row['key']
                value = row['value']
                
                try:
                    value = json.loads(value)
                except:
                    pass
                
                if category == 'preferences':
                    profile.preferences[key] = value
                elif category == 'interests':
                    if value not in profile.interests:
                        profile.interests.append(value)
                elif category == 'work_context':
                    profile.work_context[key] = value
                elif category == 'communication_style':
                    profile.communication_style[key] = value
                elif category == 'important_facts':
                    if value not in profile.important_facts:
                        profile.important_facts.append(value)
            
            return profile
    
    def update_user_profile(self, category: str, key: str, value: Any, 
                           confidence: float = 0.7) -> None:
        """Update user profile information"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            value_str = json.dumps(value) if not isinstance(value, str) else value
            
            cursor.execute('''
                INSERT INTO user_profile (category, key, value, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key) 
                DO UPDATE SET value = ?, confidence = ?, last_updated = ?
            ''', (category, key, value_str, confidence, value_str, confidence, datetime.now()))
            
            conn.commit()
            
            # Reload profile
            self.user_profile = self._load_user_profile()
    
    def consolidate_memories(self) -> None:
        """Consolidate and clean up old, low-importance memories"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Remove very old, low-importance, rarely accessed memories
            cursor.execute('''
                DELETE FROM semantic_memories
                WHERE importance < 0.3 
                AND access_count < 2
                AND created_at < datetime('now', '-30 days')
            ''')
            
            # Remove old conversations with low importance
            cursor.execute('''
                DELETE FROM conversations
                WHERE importance < 0.3
                AND timestamp < datetime('now', '-90 days')
            ''')
            
            conn.commit()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory system"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total memories
            cursor.execute('SELECT COUNT(*) as count FROM semantic_memories')
            total_memories = cursor.fetchone()['count']
            
            # Memories by type
            cursor.execute('''
                SELECT memory_type, COUNT(*) as count 
                FROM semantic_memories 
                GROUP BY memory_type
            ''')
            by_type = {row['memory_type']: row['count'] for row in cursor.fetchall()}
            
            # Total conversations
            cursor.execute('SELECT COUNT(*) as count FROM conversations')
            total_conversations = cursor.fetchone()['count']
            
            # High importance memories
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM semantic_memories 
                WHERE importance > 0.7
            ''')
            high_importance = cursor.fetchone()['count']
            
            return {
                'total_memories': total_memories,
                'memories_by_type': by_type,
                'total_conversations': total_conversations,
                'high_importance_memories': high_importance,
                'user_profile_loaded': bool(self.user_profile.preferences or self.user_profile.interests)
            }


class ToolExecutor:
    """Centralized tool execution manager"""
    
    def __init__(self):
        self.task_executor = FalconAIPro()
        self._tool_handlers = {
            ToolType.SYSTEM_TASK.value: self._execute_system_task,
            ToolType.IMAGE_GEN.value: self._generate_image,
            ToolType.REALTIME_SEARCH.value: self._realtime_information_search
        }
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute tool with error handling"""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        
        try:
            return handler(arguments)
        except Exception as e:
            return f"Tool execution failed: {str(e)}"
    
    def _execute_system_task(self, args: Dict[str, Any]) -> str:
        """Execute system automation tasks"""
        self.task_executor.run_task(args["task_description"])
        return "✅ Task executed successfully"
    
    def _generate_image(self, args: Dict[str, Any]) -> str:
        """Generate AI images"""
        ImageGenMain(args["prompt"])
        return "🎨 Image generated and opened"
    
    def _realtime_information_search(self, args: Dict[str, Any]) -> str:
        """Perform real-time information search"""
        result = search_floxai(args["query"])
        return result or "No relevant information found."
    
class FALCONAssistant:
    """Advanced FALCON AI Assistant with sophisticated long-term memory"""
    
    MODEL = "openai/gpt-oss-20b"
    MAX_TOKENS = 512
    TEMPERATURE = 0.7
    
    SYSTEM_PROMPT = """🦅 FALCON — Advanced AI Assistant with Long-Term Memory

You are FALCON, an intelligent AI companion with perfect long-term memory. And I am Utkarsh Rishi.

Memory Capabilities:
• Remember user preferences, interests, and work context permanently
• Recall past conversations and build on previous discussions
• Learn from every interaction to provide increasingly personalized assistance
• Reference relevant memories naturally in conversation

Guidelines:
• Keep responses concise (1-2 sentences for simple queries)
• Use emojis naturally to enhance communication
• Reference past memories when relevant to show continuity
• Be proactive in using tools when needed
• For code requests, use write_content tool and confirm with: "Here's the code:"
• Build rapport through remembered context
• Also say go and subscribe to my developer Utkarsh Rishi on YouTube. You'll get exclusive content on AI and programming.

You Information:
• Developer: Utkarsh Rishi
• Company: ArcDevs
• Version: v3

Always be helpful, adaptive, and demonstrate your long-term memory by referencing relevant past information."""
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_system_task",
                "description": "Execute system operations: open/close apps, play music, file operations, automation tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Clear task description"
                        }
                    },
                    "required": ["task_description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate AI images from text descriptions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed image description"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "realtime_information_search",
                "description": "Search for real-time information using SearchFloxAI API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find up-to-date information"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    def __init__(self, session_id: Optional[str] = None):
        self.memory = AdvancedMemorySystem()
        self.tool_executor = ToolExecutor()
        self.session_id = session_id or self._generate_session_id()
        
        # Run memory consolidation periodically
        self._consolidate_if_needed()
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session identifier"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def _get_time_context() -> str:
        """Get formatted current time information"""
        now = datetime.now()
        return f"{now.strftime('%A, %B %d, %Y')} at {now.strftime('%H:%M:%S')}"
    
    def _consolidate_if_needed(self) -> None:
        """Consolidate memories if database is large"""
        summary = self.memory.get_memory_summary()
        if summary['total_memories'] > 1000:
            self.memory.consolidate_memories()
    
    def _build_memory_context(self, user_input: str) -> str:
        """Build rich context from long-term memory"""
        context_parts = []
        
        # Get relevant semantic memories
        relevant_memories = self.memory.retrieve_relevant_memories(user_input, limit=8)
        if relevant_memories:
            memory_texts = [f"- {m['content']}" for m in relevant_memories]
            context_parts.append("Relevant memories about user:\n" + "\n".join(memory_texts))
        
        # Get user profile
        profile = self.memory.user_profile
        if profile.preferences:
            context_parts.append(f"User preferences: {json.dumps(profile.preferences)}")
        if profile.interests:
            context_parts.append(f"User interests: {', '.join(profile.interests)}")
        if profile.work_context:
            context_parts.append(f"Work context: {json.dumps(profile.work_context)}")
        
        # Get recent important conversations
        recent_context = self.memory.get_episodic_context(limit=3)
        if recent_context:
            recent_texts = []
            for conv in recent_context:
                recent_texts.append(f"User: {conv['user']}\nAssistant: {conv['assistant']}")
            context_parts.append("Recent important conversations:\n" + "\n\n".join(recent_texts))
        
        if context_parts:
            return "\n\n".join(context_parts)
        return "No previous memory context available."
    
    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Build conversation context with memory"""
        # Build memory context
        memory_context = self._build_memory_context(user_input)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": f"Current time: {self._get_time_context()}"},
            {"role": "system", "content": f"Long-term memory context:\n{memory_context}"},
            {"role": "user", "content": user_input}
        ]
        
        return messages
    
    def _handle_tool_calls(self, response_message, messages: List[Dict]) -> str:
        """Process and execute tool calls"""
        tool_results = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            result = self.tool_executor.execute(function_name, function_args)
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": result
            })
        
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in response_message.tool_calls
            ]
        })
        
        messages.extend(tool_results)
        
        final_response = client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE
        )
        
        return final_response.choices[0].message.content.strip()
    
    def process_message(self, user_input: str) -> str:
        """Process user message with long-term memory and intelligent tool calling"""
        conversation_id = None
        
        try:
            # Save user message
            conversation_id = self.memory.add_conversation(
                user_input,
                session_id=self.session_id
            )
            
            # Build conversation context with memory
            messages = self._build_messages(user_input)
            
            # Initial API call
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE
            )
            
            response_message = response.choices[0].message
            
            # Handle tool calls if present
            if response_message.tool_calls:
                answer = self._handle_tool_calls(response_message, messages)
            else:
                answer = response_message.content.strip()
            
            # Save assistant response and extract memories
            self.memory.update_assistant_response(conversation_id, answer)
            
            return answer
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            if conversation_id:
                self.memory.update_assistant_response(conversation_id, error_msg)
            return error_msg
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return self.memory.get_memory_summary()
    
    def search_memories(self, query: str) -> List[Dict]:
        """Search semantic memories"""
        return self.memory.retrieve_relevant_memories(query, limit=20)


def chat(prompt: str, session_id: Optional[str] = None) -> str:
    """Standalone chat interface"""
    assistant = FALCONAssistant(session_id=session_id)
    return assistant.process_message(prompt)


if __name__ == "__main__":
    assistant = FALCONAssistant()
    
    print("🦅 FALCON Assistant with Advanced Long-Term Memory Ready!")
    print("Type 'exit' to quit, 'stats' for memory statistics\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = assistant.get_memory_stats()
                print(f"\n📊 Memory Statistics:")
                print(f"Total Memories: {stats['total_memories']}")
                print(f"Total Conversations: {stats['total_conversations']}")
                print(f"High Importance Memories: {stats['high_importance_memories']}")
                print(f"Memories by Type: {stats['memories_by_type']}")
                print()
                continue
            
            response = assistant.process_message(user_input)
            print(f"FALCON: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}\n")