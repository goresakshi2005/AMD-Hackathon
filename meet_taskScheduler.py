"""
Meeting Task Scheduler
Reads tasks from meeting_knowledge_graph.json and schedules them in Google Calendar
Matches your exact file structure shown in the image
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import re

# Phidata imports
try:
    from phi.agent import Agent
    from phi.tools.googlecalendar import GoogleCalendarTools
    PHIDATA_AVAILABLE = True
    print("✅ Phidata imported successfully")
except ImportError as e:
    print(f"❌ Phidata not installed: {e}")
    print("Run: pip install phidata")
    sys.exit(1)

# Date parsing
try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    print("⚠️ Installing python-dateutil...")
    os.system("pip install python-dateutil")
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True

# Timezone
try:
    from tzlocal import get_localzone_name
    TZLOCAL_AVAILABLE = True
except ImportError:
    print("⚠️ Installing tzlocal...")
    os.system("pip install tzlocal")
    from tzlocal import get_localzone_name
    TZLOCAL_AVAILABLE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

# Google Calendar credentials path
DEFAULT_CREDENTIALS_PATH = os.path.expanduser("~/credentials/google_calendar.json")

# Default task duration (minutes)
DEFAULT_TASK_DURATION = 60  # 1 hour for meetings

# Output directory for meetings (matches your structure)
OUTPUT_ROOT = Path("output/meetings")

# ==================== TASK PARSER ====================

class MeetingTaskParser:
    """Parse tasks from meeting_knowledge_graph.json"""
    
    def __init__(self, meeting_name: str = None):
        self.meeting_name = meeting_name
        self.tasks = []
        self.load_tasks()
    
    def load_tasks(self):
        """Load tasks from meeting_knowledge_graph.json"""
        # Find the meeting folder
        if self.meeting_name:
            meeting_dir = OUTPUT_ROOT / self.meeting_name
        else:
            # Find most recent meeting folder
            meeting_dir = self._find_latest_meeting()
        
        if not meeting_dir or not meeting_dir.exists():
            logger.error(f"Meeting directory not found: {meeting_dir}")
            self._show_available_meetings()
            return
        
        # Look for meeting_tasks.json first (from your structure)
        tasks_file = meeting_dir / f"{self.meeting_name}_tasks.json"
        if not tasks_file.exists():
            # Try alternative: meeting_task.json (without 's')
            tasks_file = meeting_dir / "meeting_task.json"
        
        if not tasks_file.exists():
            # Try knowledge graph file and extract tasks
            kg_file = meeting_dir / f"{self.meeting_name}_knowledge_graph.json"
            if kg_file.exists():
                self._extract_tasks_from_graph(kg_file)
                return
            else:
                logger.error(f"No tasks file found in {meeting_dir}")
                self._show_available_meetings()
                return
        
        self._load_from_file(tasks_file)
    
    def _find_latest_meeting(self) -> Optional[Path]:
        """Find the most recent meeting folder"""
        if not OUTPUT_ROOT.exists():
            return None
        
        meetings = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
        if not meetings:
            return None
        
        # Sort by modification time (most recent first)
        meetings.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = meetings[0]
        self.meeting_name = latest.name
        logger.info(f"Found latest meeting: {self.meeting_name}")
        return latest
    
    def _load_from_file(self, tasks_file: Path):
        """Load tasks from JSON file"""
        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of tasks
                self.tasks = data
            elif isinstance(data, dict):
                if "tasks" in data:
                    # Object with tasks key
                    self.tasks = data["tasks"]
                elif "description" in data:
                    # Single task object
                    self.tasks = [data]
                else:
                    # Try to find any array in the object
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if any(isinstance(item, dict) and "description" in item for item in value):
                                self.tasks = value
                                break
            
            logger.info(f"✅ Loaded {len(self.tasks)} task(s) from {tasks_file}")
            
        except Exception as e:
            logger.error(f"Failed to load tasks file: {e}")
            self.tasks = []
    
    def _extract_tasks_from_graph(self, kg_file: Path):
        """Extract tasks from knowledge graph JSON"""
        try:
            with open(kg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Look for task nodes in the graph
            if "nodes" in data:
                for node in data["nodes"]:
                    if node.get("type") == "task":
                        task = {
                            "description": node.get("description", "Meeting task"),
                            "assignee": node.get("assignee", "department heads"),
                            "due_date": node.get("due_date", "March 16 at 3pm"),
                            "location": node.get("location", "main conference room"),
                            "priority": node.get("priority", "high"),
                            "status": node.get("status", "pending"),
                            "type": "meeting",
                            "chunk_id": node.get("chunk_id", 0)
                        }
                        self.tasks.append(task)
            
            logger.info(f"✅ Extracted {len(self.tasks)} task(s) from knowledge graph")
            
        except Exception as e:
            logger.error(f"Failed to extract tasks from graph: {e}")
    
    def _show_available_meetings(self):
        """Show available meetings in output directory"""
        if not OUTPUT_ROOT.exists():
            print(f"\n❌ Output directory not found: {OUTPUT_ROOT}")
            return
        
        print("\n📁 Available meetings:")
        for meeting_dir in OUTPUT_ROOT.iterdir():
            if meeting_dir.is_dir():
                tasks_file = meeting_dir / f"{meeting_dir.name}_tasks.json"
                kg_file = meeting_dir / f"{meeting_dir.name}_knowledge_graph.json"
                
                if tasks_file.exists():
                    with open(tasks_file, 'r') as f:
                        tasks = json.load(f)
                    task_count = len(tasks) if isinstance(tasks, list) else 1
                    print(f"  📁 {meeting_dir.name} ({task_count} task(s))")
                elif kg_file.exists():
                    print(f"  📁 {meeting_dir.name} (has knowledge graph)")
                else:
                    print(f"  📁 {meeting_dir.name} (no tasks)")
    
    def parse_due_date(self, due_date_str: str) -> Optional[datetime.datetime]:
        """Parse due date string to datetime"""
        if not due_date_str or due_date_str.lower() in ["none", "null", "", "tbd"]:
            return None
        
        try:
            # Get current year
            current_year = datetime.datetime.now().year
            
            # Handle your specific format: "March 16 at 3pm"
            due_date_str = due_date_str.lower().strip()
            
            # Pattern: "March 16 at 3pm" or "March 16th at 3pm"
            match = re.search(r'(march|mar|april|apr|may|june|jul|july|august|aug|september|sep|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(pm|am)', 
                             due_date_str, re.IGNORECASE)
            if match:
                month = match.group(1)
                day = int(match.group(2))
                hour = int(match.group(3))
                minute = int(match.group(4)) if match.group(4) else 0
                ampm = match.group(5).lower()
                
                # Convert to 24-hour format
                if ampm == 'pm' and hour < 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                
                # Map month name to number
                month_map = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12,
                    'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month_num = month_map.get(month, 3)
                
                # Create datetime
                dt = datetime.datetime(current_year, month_num, day, hour, minute)
                
                # If date has passed, assume next year
                if dt < datetime.datetime.now():
                    dt = dt.replace(year=current_year + 1)
                
                return dt
            
            # Pattern without "at": "March 16 3pm"
            match = re.search(r'(march|mar|april|apr|may|june|jul|july|august|aug|september|sep|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{1,2})(?::(\d{2}))?\s*(pm|am)',
                             due_date_str, re.IGNORECASE)
            if match:
                return self.parse_due_date(f"{match.group(1)} {match.group(2)} at {match.group(3)}{':'+match.group(4) if match.group(4) else ''}{match.group(5)}")
            
            # Fallback to dateutil
            if DATEUTIL_AVAILABLE:
                parsed = date_parser.parse(due_date_str, fuzzy=True)
                return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse due date '{due_date_str}': {e}")
        
        return None
    
    def prepare_calendar_events(self) -> List[Dict[str, Any]]:
        """Prepare tasks as calendar events"""
        events = []
        
        if not self.tasks:
            logger.warning("No tasks to schedule")
            return events
        
        for i, task in enumerate(self.tasks):
            description = task.get("description", "Untitled Task")
            assignee = task.get("assignee", "")
            due_date_str = task.get("due_date", "")
            location = task.get("location", "")
            priority = task.get("priority", "medium")
            task_type = task.get("type", "task")
            full_context = task.get("full_context", "")
            
            # Parse due date
            due_date = self.parse_due_date(due_date_str)
            
            # Default to next week if no date
            if not due_date:
                next_week = datetime.datetime.now() + datetime.timedelta(days=7)
                due_date = next_week.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # Calculate end time (1 hour for meetings)
            end_time = due_date + datetime.timedelta(minutes=DEFAULT_TASK_DURATION)
            
            # Create event summary with emoji
            priority_emoji = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(priority.lower(), "📅")
            
            summary = f"{priority_emoji} {description}"
            
            if assignee and assignee not in ["unspecified", ""]:
                summary += f" (👤 {assignee})"
            
            # Create description
            full_description = f"""
📋 MEETING TASK
{'='*50}

Task: {description}
Assignee: {assignee if assignee else 'Not specified'}
When: {due_date_str}
Where: {location if location else 'Not specified'}
Priority: {priority.upper()}

📝 Context:
{full_context if full_context else 'No additional context'}

{'='*50}
Scheduled from meeting: {self.meeting_name}
            """.strip()
            
            # Create event
            event = {
                "summary": summary,
                "description": full_description,
                "location": location,
                "start": {
                    "dateTime": due_date.isoformat(),
                    "timeZone": get_localzone_name() if TZLOCAL_AVAILABLE else "UTC",
                },
                "end": {
                    "dateTime": end_time.isoformat(),
                    "timeZone": get_localzone_name() if TZLOCAL_AVAILABLE else "UTC",
                },
                "reminders": {
                    "useDefault": True
                }
            }
            
            events.append(event)
        
        return events

# ==================== CALENDAR SCHEDULER ====================

class CalendarScheduler:
    """Schedule tasks in Google Calendar using Phidata"""
    
    def __init__(self, credentials_path: str = DEFAULT_CREDENTIALS_PATH):
        self.credentials_path = credentials_path
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """Initialize Phidata agent with Google Calendar tools"""
        if not os.path.exists(self.credentials_path):
            logger.error(f"Google Calendar credentials not found at: {self.credentials_path}")
            return
        
        try:
            timezone = get_localzone_name() if TZLOCAL_AVAILABLE else "UTC"
            today = datetime.datetime.now()
            
            self.agent = Agent(
                tools=[GoogleCalendarTools(credentials_path=self.credentials_path)],
                show_tool_calls=True,
                instructions=[
                    f"""
                    You are a meeting task scheduling assistant.
                    Today is {today.strftime('%Y-%m-%d %H:%M')} and the user's timezone is {timezone}.
                    
                    Create calendar events from meeting tasks with:
                    - Proper titles with emoji indicators
                    - Full descriptions with context
                    - Correct start and end times
                    - Locations if provided
                    """
                ],
                add_datetime_to_instructions=True,
            )
            
            logger.info("✅ Phidata agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
    
    def schedule_event(self, event_data: Dict[str, Any]) -> bool:
        """Schedule a single event in calendar"""
        if not self.agent:
            return False
        
        try:
            summary = event_data["summary"]
            start = event_data["start"]["dateTime"]
            end = event_data["end"]["dateTime"]
            
            logger.info(f"\n📅 Scheduling: {summary}")
            
            prompt = f"""
            Create a calendar event with these details:
            Title: {summary}
            Start: {start}
            End: {end}
            Description: {event_data.get('description', '')}
            Location: {event_data.get('location', '')}
            Reminders: default
            """
            
            self.agent.run(prompt)
            logger.info(f"✅ Scheduled: {summary}")
            return True
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False
    
    def schedule_all_events(self, events: List[Dict[str, Any]], 
                          interactive: bool = True) -> Dict[str, Any]:
        """Schedule all events"""
        results = {"scheduled": 0, "failed": 0, "skipped": 0}
        
        if not events:
            return results
        
        print("\n" + "=" * 70)
        print("📋 TASKS TO SCHEDULE")
        print("=" * 70)
        
        for i, event in enumerate(events):
            start_dt = datetime.datetime.fromisoformat(event["start"]["dateTime"].replace('Z', '+00:00'))
            print(f"\n{i+1}. {event['summary']}")
            print(f"   📅 {start_dt.strftime('%A, %B %d at %I:%M %p')}")
            if event.get('location'):
                print(f"   📍 {event['location']}")
        
        print("\n" + "=" * 70)
        
        if interactive:
            response = input("\nSchedule all tasks? (y/n/s for selective): ").lower()
            
            if response == 'y':
                for event in events:
                    if self.schedule_event(event):
                        results["scheduled"] += 1
                    else:
                        results["failed"] += 1
            
            elif response == 's':
                for i, event in enumerate(events):
                    resp = input(f"\nSchedule task {i+1}? (y/n): ")
                    if resp == 'y':
                        if self.schedule_event(event):
                            results["scheduled"] += 1
                        else:
                            results["failed"] += 1
                    else:
                        results["skipped"] += 1
        else:
            for event in events:
                if self.schedule_event(event):
                    results["scheduled"] += 1
                else:
                    results["failed"] += 1
        
        return results

# ==================== MAIN ====================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Schedule meeting tasks in Google Calendar")
    parser.add_argument(
        "--meeting",
        type=str,
        help="Meeting name (folder in output/meetings)"
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=DEFAULT_CREDENTIALS_PATH,
        help="Path to Google Calendar credentials"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-schedule without confirmation"
    )
    
    args = parser.parse_args()
    
    # Parse tasks
    parser = MeetingTaskParser(meeting_name=args.meeting)
    
    if not parser.tasks:
        print("\n❌ No tasks found to schedule")
        return
    
    # Prepare events
    events = parser.prepare_calendar_events()
    print(f"\n📅 Prepared {len(events)} calendar event(s)")
    
    # Check credentials
    if not os.path.exists(args.credentials):
        print(f"\n❌ Credentials not found: {args.credentials}")
        print("\n🔄 Running in PREVIEW MODE (no actual calendar events)")
        print("\n" + "=" * 70)
        print("📋 EVENTS THAT WOULD BE SCHEDULED:")
        print("=" * 70)
        for i, event in enumerate(events):
            start_dt = datetime.datetime.fromisoformat(event["start"]["dateTime"].replace('Z', '+00:00'))
            print(f"\n{i+1}. {event['summary']}")
            print(f"   📅 {start_dt.strftime('%A, %B %d at %I:%M %p')}")
            if event.get('location'):
                print(f"   📍 {event['location']}")
        print("\n" + "=" * 70)
        print("\n✅ Preview complete!")
        return
    
    # Schedule
    scheduler = CalendarScheduler(credentials_path=args.credentials)
    results = scheduler.schedule_all_events(events, interactive=not args.auto)
    
    print("\n" + "=" * 70)
    print("📊 SCHEDULING SUMMARY")
    print("=" * 70)
    print(f"✅ Scheduled: {results['scheduled']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏭️ Skipped: {results['skipped']}")
    print("=" * 70)

if __name__ == "__main__":
    main()