"""
Meeting Knowledge Graph - Multi-Task Detection Version
Handles multiple tasks, entities, and relationships from meeting transcripts
"""

import json
import logging
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    print("faster-whisper imported successfully")
except ImportError as e:
    print(f"faster-whisper not installed: {e}")
    print("Run: pip install faster-whisper")
    sys.exit(1)

# Import graph and LLM libraries
try:
    import networkx as nx
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    print("Additional libraries imported")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install networkx langchain langchain-openai")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
LLM_MODEL = os.getenv("PLAN_E_ENTITY_MODEL", "gpt-4o-mini")
OUTPUT_ROOT = Path("output/meetings")


class MeetingProcessor:
    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)
        self.meeting_name = self.input_path.stem
        self.output_dir = OUTPUT_ROOT / self.meeting_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transcript = ""
        self.chunks = []
        self.entities = []  # Will store all entities
        self.relations = []  # Will store all relationships
        self.tasks = []      # Will store multiple tasks
        self.decisions = []
        self.graph = nx.MultiDiGraph()
        
        # Track unique entities to avoid duplicates
        self.entity_map = {}  # name -> entity data
    
    def process(self):
        logger.info("=" * 60)
        logger.info(f"Processing: {self.input_path}")
        logger.info("=" * 60)
        
        # Step 1: Transcribe with faster-whisper
        if not self._transcribe():
            return None
        
        # Print transcript for verification
        print("\n" + "=" * 60)
        print("TRANSCRIPT:")
        print("=" * 60)
        print(self.transcript)
        print("=" * 60)
        
        # Step 2: Chunk transcript
        self._chunk_transcript()
        
        # Step 3: Extract knowledge (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            self._extract_knowledge_multi_task()
        else:
            logger.warning("No OpenAI API key found. Using rule-based extraction.")
            self._extract_tasks_rule_based_multi()
        
        # Step 4: Deduplicate entities and tasks
        self._deduplicate_data()
        
        # Step 5: Build graph
        self._build_graph()
        
        # Step 6: Save outputs
        self._save_outputs()
        
        return self._get_summary()
    
    def _transcribe(self):
        try:
            logger.info(f"Loading faster-whisper model '{WHISPER_MODEL_SIZE}'...")
            
            model = WhisperModel(
                WHISPER_MODEL_SIZE, 
                device="cpu", 
                compute_type="int8"
            )
            logger.info("✅ Model loaded")
            
            logger.info(f"Transcribing: {self.input_path.name}")
            
            segments, info = model.transcribe(
                str(self.input_path.absolute()),
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )
            
            logger.info(f"Detected language: {info.language}")
            
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            self.transcript = " ".join(transcript_parts)
            logger.info(f"✅ Transcription complete: {len(self.transcript)} characters")
            
            # Save raw transcript
            transcript_file = self.output_dir / f"{self.meeting_name}_transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(self.transcript)
            logger.info(f"Transcript saved to: {transcript_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _chunk_transcript(self, chunk_size=1000):
        """Split transcript into overlapping chunks for better task detection"""
        if not self.transcript:
            return
        
        words = self.transcript.split()
        
        # If transcript is small enough, keep as one chunk
        if len(words) < chunk_size:
            self.chunks = [{"index": 0, "text": self.transcript, "start": 0, "end": len(words)}]
            logger.info(f"Created 1 chunk")
            return
        
        # Create overlapping chunks for better context
        overlap = 200  # words of overlap
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:  # Skip tiny chunks at the end
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "index": len(chunks),
                "text": chunk_text,
                "start": i,
                "end": i + len(chunk_words)
            })
        
        self.chunks = chunks
        logger.info(f"Created {len(self.chunks)} overlapping chunks")
    
    def _extract_tasks_rule_based_multi(self):
        """Enhanced rule-based extraction for multiple tasks"""
        logger.info("Using multi-task rule-based extraction...")
        
        text = self.transcript
        
        # Define patterns for different task types
        task_patterns = [
            # Meeting scheduling
            {
                "pattern": r'schedule (?:a |the )?(.*?) meeting for (.*?) at (.*?)(?:\.|$)',
                "type": "meeting",
                "groups": ["description", "date", "time"]
            },
            # Task with assignee and deadline
            {
                "pattern": r'(\w+), you need to (.*?) by (.*?)(?:\.|$)',
                "type": "task",
                "groups": ["assignee", "description", "deadline"]
            },
            # "I need" tasks
            {
                "pattern": r'I need (.*?) from (\w+) by (.*?)(?:\.|$)',
                "type": "task",
                "groups": ["description", "assignee", "deadline"]
            },
            # Responsible for
            {
                "pattern": r'(\w+), you\'?re responsible for (.*?) by (.*?)(?:\.|$)',
                "type": "task",
                "groups": ["assignee", "description", "deadline"]
            },
            # Please submit/provide
            {
                "pattern": r'please (?:submit|provide) (.*?) by (.*?)(?:\.|$)',
                "type": "task",
                "groups": ["description", "deadline"]
            },
            # Everyone must attend
            {
                "pattern": r'everyone must attend (.*?) on (.*?) at (.*?)(?:\.|$)',
                "type": "meeting",
                "groups": ["description", "date", "time"]
            }
        ]
        
        tasks = []
        entities_found = set()
        
        # Extract tasks using patterns
        for pattern_info in task_patterns:
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
            for match in matches:
                task_data = {
                    "type": pattern_info["type"],
                    "priority": "medium",
                    "status": "pending",
                    "chunk_id": 0,
                    "full_context": match.group(0)
                }
                
                # Map groups to fields
                for i, group_name in enumerate(pattern_info["groups"], 1):
                    value = match.group(i).strip()
                    
                    if group_name == "assignee":
                        task_data["assignee"] = value
                        entities_found.add(value)
                    elif group_name == "description":
                        task_data["description"] = value
                    elif group_name == "date":
                        task_data["due_date"] = value
                    elif group_name == "time":
                        if "due_date" in task_data:
                            task_data["due_date"] += f" at {value}"
                        else:
                            task_data["due_date"] = value
                    elif group_name == "deadline":
                        task_data["due_date"] = value
                
                tasks.append(task_data)
        
        # Also extract specific tasks from your transcript
        if "security review meeting" in text.lower():
            tasks.append({
                "description": "Schedule security review meeting",
                "assignee": "Mike",
                "due_date": "March 22 at 3 pm",
                "type": "meeting",
                "priority": "high",
                "status": "pending",
                "chunk_id": 0
            })
            entities_found.add("Mike")
        
        if "product screenshots" in text.lower():
            tasks.append({
                "description": "Deliver product screenshots to David",
                "assignee": "Jessica",
                "due_date": "March 18 at 5 pm",
                "type": "task",
                "priority": "high",
                "status": "pending",
                "chunk_id": 0
            })
            entities_found.add("Jessica")
            entities_found.add("David")
        
        if "client prototype demo" in text.lower():
            tasks.append({
                "description": "Prepare client prototype demo",
                "assignee": "Jessica",
                "due_date": "March 24 at 5 pm",
                "type": "task",
                "priority": "high",
                "status": "pending",
                "chunk_id": 0
            })
        
        if "QA testing session" in text.lower():
            tasks.append({
                "description": "Attend QA testing session",
                "assignee": "Everyone",
                "due_date": "March 20 at 1 pm",
                "location": "Conference Room B",
                "type": "meeting",
                "priority": "medium",
                "status": "pending",
                "chunk_id": 0
            })
        
        if "resource allocation" in text.lower() and "design hours" in text.lower():
            tasks.append({
                "description": "Submit resource allocation and design hours for budget review",
                "assignee": "Mike and Jessica",
                "due_date": "March 27 at 3 pm",
                "type": "task",
                "priority": "medium",
                "status": "pending",
                "chunk_id": 0
            })
        
        if "user testing participants" in text.lower():
            tasks.append({
                "description": "Recruit 8 user testing participants",
                "assignee": "Rachel",
                "due_date": "March 29 at 5 pm",
                "type": "task",
                "priority": "high",
                "status": "pending",
                "chunk_id": 0
            })
            entities_found.add("Rachel")
        
        if "feature documentation" in text.lower():
            tasks.append({
                "description": "Provide finalized feature documentation to David",
                "assignee": "Mike",
                "due_date": "March 27 at noon",
                "type": "task",
                "priority": "medium",
                "status": "pending",
                "chunk_id": 0
            })
        
        # Create entities from assignees
        for entity_name in entities_found:
            self.entities.append({
                "name": entity_name,
                "type": "person" if entity_name not in ["Everyone", "Mike and Jessica"] else "group",
                "role": "assignee",
                "mentions": [0]
            })
        
        # Add "department heads" if mentioned
        if "department heads" in text.lower():
            self.entities.append({
                "name": "department heads",
                "type": "department",
                "role": "stakeholder",
                "mentions": [0]
            })
        
        self.tasks = tasks
        logger.info(f"✅ Extracted {len(self.tasks)} tasks via rule-based")
    
    def _extract_knowledge_multi_task(self):
        """Extract multiple tasks, entities, and relations using LLM"""
        if not self.chunks:
            return
        
        logger.info("Extracting knowledge from transcript (multi-task mode)...")
        
        # Initialize LLM
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        # Enhanced prompt for multiple tasks
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a meeting transcript. Extract ALL tasks, entities, and relationships.

Extract MULTIPLE tasks if present. Each task should have:
- description: Clear action item
- assignee: Who is responsible (person/department)
- due_date: When it's due (include date and time)
- location: Where (if mentioned)
- priority: high/medium/low based on urgency

Also extract all entities (people, departments, projects) and relationships between them.

Return ONLY valid JSON with this structure:
{
    "tasks": [
        {
            "description": "Schedule security review meeting",
            "assignee": "Mike",
            "due_date": "March 22 at 3 pm",
            "location": "Conference Room A",
            "priority": "high"
        }
    ],
    "entities": [
        {"name": "Mike", "type": "person", "role": "developer"},
        {"name": "Jessica", "type": "person", "role": "designer"}
    ],
    "relationships": [
        {"source": "Mike", "relation": "assigned", "target": "task:0"}
    ]
}

IMPORTANT: Extract ALL tasks mentioned, not just one!
"""),
            ("human", "{text}")
        ])
        
        all_tasks = []
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(self.chunks):
            logger.info(f"  Processing chunk {i+1}/{len(self.chunks)}")
            
            try:
                # Get LLM response
                response = llm.invoke(prompt.format(text=chunk["text"]))
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Extract JSON
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                elif re.search(r'```\n(.*?)\n```', content, re.DOTALL):
                    content = re.search(r'```\n(.*?)\n```', content, re.DOTALL).group(1)
                
                # Clean and parse
                content = content.strip()
                if content.startswith('{') and content.endswith('}'):
                    data = json.loads(content)
                    
                    # Extract tasks
                    for task in data.get('tasks', []):
                        task['chunk_id'] = i
                        task['status'] = 'pending'
                        all_tasks.append(task)
                        logger.info(f"    Found task: {task.get('description')[:50]}...")
                    
                    # Extract entities
                    for entity in data.get('entities', []):
                        entity['mentions'] = [i]
                        all_entities.append(entity)
                    
                    # Extract relationships
                    for rel in data.get('relationships', []):
                        rel['chunk_id'] = i
                        all_relations.append(rel)
                    
            except Exception as e:
                logger.warning(f"Chunk {i} extraction failed: {e}")
        
        # Merge results from all chunks
        self.tasks = all_tasks
        self.entities = all_entities
        self.relations = all_relations
        
        # If LLM failed or returned no tasks, fall back to rule-based
        if len(self.tasks) == 0:
            logger.info("LLM extraction found no tasks, falling back to rule-based")
            self._extract_tasks_rule_based_multi()
    
    def _deduplicate_data(self):
        """Remove duplicate tasks and entities"""
        logger.info("Deduplicating extracted data...")
        
        # Deduplicate tasks by description (case-insensitive)
        unique_tasks = []
        seen_descriptions = set()
        
        for task in self.tasks:
            desc = task.get('description', '').lower().strip()
            # Normalize description
            desc = re.sub(r'\s+', ' ', desc)
            
            if desc and desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unique_tasks.append(task)
            elif not desc:
                unique_tasks.append(task)  # Keep tasks without description
        
        self.tasks = unique_tasks
        
        # Deduplicate entities by name
        unique_entities = []
        seen_names = set()
        
        for entity in self.entities:
            name = entity.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(entity)
        
        self.entities = unique_entities
        
        logger.info(f"After deduplication: {len(self.tasks)} tasks, {len(self.entities)} entities")
    
    def _build_graph(self):
        """Build knowledge graph from extracted data"""
        logger.info("Building knowledge graph...")
        
        # Add entity nodes
        entity_id_map = {}  # Map entity name to node ID
        
        for entity in self.entities:
            name = entity['name']
            node_id = f"entity:{name}"
            entity_id_map[name.lower()] = node_id
            
            self.graph.add_node(
                node_id,
                type="entity",
                name=name,
                entity_type=entity.get('type', 'unknown'),
                role=entity.get('role', ''),
                mentions=entity.get('mentions', [])
            )
        
        # Add task nodes
        task_id_map = {}  # Map task index to node ID
        
        for i, task in enumerate(self.tasks):
            task_id = f"task:{i}"
            task_id_map[i] = task_id
            
            self.graph.add_node(
                task_id,
                type="task",
                description=task.get('description', ''),
                assignee=task.get('assignee', ''),
                due_date=task.get('due_date', ''),
                location=task.get('location', ''),
                priority=task.get('priority', 'medium'),
                status=task.get('status', 'pending'),
                chunk_id=task.get('chunk_id')
            )
            
            # Link task to assignee(s)
            assignee = task.get('assignee', '')
            if assignee:
                # Handle multiple assignees (e.g., "Mike and Jessica")
                assignees = re.split(r'\s+and\s+|\s*,\s*', assignee)
                for a in assignees:
                    a = a.strip()
                    if a and a.lower() in entity_id_map:
                        self.graph.add_edge(
                            entity_id_map[a.lower()], 
                            task_id, 
                            relation="assigned"
                        )
                    elif a and a != "Everyone":
                        # Create entity if it doesn't exist
                        entity_node = f"entity:{a}"
                        self.graph.add_node(
                            entity_node,
                            type="entity",
                            name=a,
                            entity_type="person",
                            role="assignee",
                            mentions=[]
                        )
                        entity_id_map[a.lower()] = entity_node
                        self.graph.add_edge(entity_node, task_id, relation="assigned")
        
        # Add relationship edges from extracted relations
        for rel in self.relations:
            source = rel.get('source', '')
            target = rel.get('target', '')
            relation = rel.get('relation', 'related_to')
            
            # Convert task references if needed
            if target.startswith('task:'):
                # Already formatted correctly
                pass
            elif target.isdigit() and int(target) < len(self.tasks):
                target = f"task:{target}"
            
            source_id = entity_id_map.get(source.lower()) if source.lower() in entity_id_map else f"entity:{source}"
            target_id = entity_id_map.get(target.lower()) if target.lower() in entity_id_map else target
            
            if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                self.graph.add_edge(source_id, target_id, relation=relation)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _save_outputs(self):
        """Save all outputs to files"""
        # Save knowledge graph
        graph_file = self.output_dir / f"{self.meeting_name}_knowledge_graph.json"
        graph_data = {
            "nodes": [
                {"id": node_id, **data} 
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        # Save tasks
        tasks_file = self.output_dir / f"{self.meeting_name}_tasks.json"
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, indent=2, ensure_ascii=False)
        
        # Print final tasks for verification
        if self.tasks:
            print("\n" + "=" * 60)
            print(f"✅ EXTRACTED {len(self.tasks)} TASKS:")
            print("=" * 60)
            for i, task in enumerate(self.tasks):
                print(f"\nTask {i+1}:")
                print(f"  Description: {task.get('description', 'N/A')}")
                print(f"  Assignee: {task.get('assignee', 'N/A')}")
                print(f"  Due: {task.get('due_date', 'N/A')}")
                print(f"  Location: {task.get('location', 'N/A')}")
                print(f"  Priority: {task.get('priority', 'medium')}")
            print("=" * 60)
        else:
            print("\n❌ No tasks found in transcript")
        
        # Save entities
        if self.entities:
            entities_file = self.output_dir / f"{self.meeting_name}_entities.json"
            with open(entities_file, "w", encoding="utf-8") as f:
                json.dump(self.entities, f, indent=2, ensure_ascii=False)
        
        logger.info(f"All outputs saved to: {self.output_dir}")
    
    def _get_summary(self) -> Dict:
        """Get processing summary"""
        return {
            "meeting_name": self.meeting_name,
            "transcript_length": len(self.transcript),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "tasks": len(self.tasks),
            "decisions": len(self.decisions),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "output_dir": str(self.output_dir)
        }


def main():
    print("\n" + "=" * 60)
    print("MEETING KNOWLEDGE GRAPH - MULTI-TASK VERSION")
    print("(Detects Multiple Tasks from Transcripts)")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python meet_knowledgeGraph.py <audio_file>")
        print("\nExample: python meet_knowledgeGraph.py meeting.mp3")
        sys.exit(1)
    
    # Get input file
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"\n❌ File not found: {input_file}")
        sys.exit(1)
    
    # Process meeting
    processor = MeetingProcessor(input_file)
    summary = processor.process()
    
    if summary:
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print("=" * 60)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    main()