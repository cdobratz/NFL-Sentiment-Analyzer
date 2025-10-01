"""
Real-time Data Processing Pipeline for NFL Sentiment Analyzer

This module handles:
- Background task scheduling for periodic data collection
- Data validation and cleaning pipeline
- Real-time sentiment processing with queue management
- Data deduplication and conflict resolution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import schedule
import threading
import time

from ..core.config import settings
from ..core.database import get_database
from ..services.data_ingestion_service import DataIngestionService, RawDataItem, DataSource
from ..services.sentiment_service import SentimentService
from ..models.sentiment import SentimentAnalysis

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingTask:
    """Represents a data processing task"""
    id: str
    task_type: str
    data: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class DataValidator:
    """Validates and cleans incoming data"""
    
    @staticmethod
    def validate_tweet(data: Dict) -> bool:
        """Validate tweet data structure"""
        required_fields = ["id", "text", "created_at", "author"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_news(data: Dict) -> bool:
        """Validate news article data structure"""
        required_fields = ["headline"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_game(data: Dict) -> bool:
        """Validate game data structure"""
        required_fields = ["id", "name", "date"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_betting_line(data: Dict) -> bool:
        """Validate betting line data structure"""
        required_fields = ["game_id", "sportsbook"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove or replace problematic characters
        text = text.replace("\n", " ").replace("\r", " ")
        
        # Limit length to prevent extremely long texts
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        return text.strip()
    
    @staticmethod
    def extract_nfl_keywords(text: str) -> List[str]:
        """Extract NFL-related keywords from text"""
        nfl_keywords = [
            # Teams (abbreviated)
            "chiefs", "bills", "bengals", "browns", "steelers", "ravens", "titans", "colts",
            "texans", "jaguars", "broncos", "chargers", "raiders", "dolphins", "jets", "patriots",
            "cowboys", "giants", "eagles", "commanders", "packers", "bears", "lions", "vikings",
            "falcons", "panthers", "saints", "buccaneers", "cardinals", "rams", "49ers", "seahawks",
            # Common NFL terms
            "nfl", "football", "touchdown", "quarterback", "qb", "running back", "rb",
            "wide receiver", "wr", "tight end", "te", "defense", "offense", "playoff",
            "super bowl", "draft", "trade", "injury", "fantasy", "betting", "odds"
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in nfl_keywords if keyword in text_lower]
        return found_keywords


class TaskQueue:
    """Manages processing tasks with priority queue"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.tasks: List[ProcessingTask] = []
        self.lock = asyncio.Lock()
    
    async def add_task(self, task: ProcessingTask) -> bool:
        """Add a task to the queue"""
        async with self.lock:
            if len(self.tasks) >= self.max_size:
                logger.warning("Task queue is full, dropping oldest task")
                self.tasks.pop(0)
            
            self.tasks.append(task)
            logger.debug(f"Added task {task.id} to queue")
            return True
    
    async def get_next_task(self) -> Optional[ProcessingTask]:
        """Get the next pending task"""
        async with self.lock:
            for task in self.tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    return task
            return None
    
    async def complete_task(self, task_id: str, success: bool = True, error_message: str = None):
        """Mark a task as completed or failed"""
        async with self.lock:
            for task in self.tasks:
                if task.id == task_id:
                    if success:
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.FAILED
                        task.error_message = error_message
                    task.completed_at = datetime.now()
                    break
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        async with self.lock:
            stats = defaultdict(int)
            for task in self.tasks:
                stats[task.status.value] += 1
            stats["total"] = len(self.tasks)
            return dict(stats)


class DataProcessingPipeline:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.data_ingestion = DataIngestionService()
        self.sentiment_service = SentimentService()
        self.validator = DataValidator()
        self.task_queue = TaskQueue()
        self.is_running = False
        self.worker_tasks = []
        self.scheduler_thread = None
    
    async def start(self):
        """Start the data processing pipeline"""
        await self.data_ingestion.start()
        self.is_running = True
        
        # Start worker tasks
        for i in range(3):  # 3 worker tasks
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Data processing pipeline started")
    
    async def stop(self):
        """Stop the data processing pipeline"""
        self.is_running = False
        
        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        await self.data_ingestion.stop()
        logger.info("Data processing pipeline stopped")
    
    def _run_scheduler(self):
        """Run the background scheduler"""
        # Schedule data collection tasks
        schedule.every(5).minutes.do(self._schedule_twitter_collection)
        schedule.every(15).minutes.do(self._schedule_espn_collection)
        schedule.every(30).minutes.do(self._schedule_betting_lines_collection)
        schedule.every(1).hours.do(self._schedule_data_cleanup)
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _schedule_twitter_collection(self):
        """Schedule Twitter data collection"""
        if not self.is_running:
            return
        
        keywords = [
            "NFL", "football", "Chiefs", "Bills", "Cowboys", "Patriots",
            "Super Bowl", "playoff", "quarterback", "touchdown"
        ]
        
        task = ProcessingTask(
            id=f"twitter-{datetime.now().timestamp()}",
            task_type="collect_twitter",
            data={"keywords": keywords, "max_results": 100},
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        asyncio.run_coroutine_threadsafe(
            self.task_queue.add_task(task),
            asyncio.get_event_loop()
        )
    
    def _schedule_espn_collection(self):
        """Schedule ESPN data collection"""
        if not self.is_running:
            return
        
        task = ProcessingTask(
            id=f"espn-{datetime.now().timestamp()}",
            task_type="collect_espn",
            data={},
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        asyncio.run_coroutine_threadsafe(
            self.task_queue.add_task(task),
            asyncio.get_event_loop()
        )
    
    def _schedule_betting_lines_collection(self):
        """Schedule betting lines collection"""
        if not self.is_running:
            return
        
        task = ProcessingTask(
            id=f"betting-{datetime.now().timestamp()}",
            task_type="collect_betting",
            data={"sportsbooks": ["draftkings", "mgm"]},
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        asyncio.run_coroutine_threadsafe(
            self.task_queue.add_task(task),
            asyncio.get_event_loop()
        )
    
    def _schedule_data_cleanup(self):
        """Schedule data cleanup task"""
        if not self.is_running:
            return
        
        task = ProcessingTask(
            id=f"cleanup-{datetime.now().timestamp()}",
            task_type="cleanup_data",
            data={"days_to_keep": 30},
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        asyncio.run_coroutine_threadsafe(
            self.task_queue.add_task(task),
            asyncio.get_event_loop()
        )
    
    async def _worker(self, worker_id: str):
        """Worker task that processes items from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                task = await self.task_queue.get_next_task()
                if not task:
                    await asyncio.sleep(5)  # Wait before checking again
                    continue
                
                logger.info(f"Worker {worker_id} processing task {task.id}")
                
                success = await self._process_task(task)
                await self.task_queue.complete_task(task.id, success)
                
                if success:
                    logger.info(f"Worker {worker_id} completed task {task.id}")
                else:
                    logger.error(f"Worker {worker_id} failed task {task.id}")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if 'task' in locals():
                    await self.task_queue.complete_task(task.id, False, str(e))
                await asyncio.sleep(5)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: ProcessingTask) -> bool:
        """Process a single task"""
        try:
            if task.task_type == "collect_twitter":
                return await self._process_twitter_collection(task)
            elif task.task_type == "collect_espn":
                return await self._process_espn_collection(task)
            elif task.task_type == "collect_betting":
                return await self._process_betting_collection(task)
            elif task.task_type == "cleanup_data":
                return await self._process_data_cleanup(task)
            elif task.task_type == "sentiment_analysis":
                return await self._process_sentiment_analysis(task)
            else:
                logger.error(f"Unknown task type: {task.task_type}")
                return False
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            return False
    
    async def _process_twitter_collection(self, task: ProcessingTask) -> bool:
        """Process Twitter data collection task"""
        keywords = task.data.get("keywords", [])
        max_results = task.data.get("max_results", 100)
        
        raw_items = await self.data_ingestion.collect_twitter_data(keywords, max_results)
        
        if raw_items:
            # Validate and clean data
            validated_items = []
            for item in raw_items:
                if self.validator.validate_tweet(item.content):
                    # Clean text content
                    item.content["text"] = self.validator.clean_text(item.content["text"])
                    # Extract NFL keywords
                    item.content["nfl_keywords"] = self.validator.extract_nfl_keywords(item.content["text"])
                    validated_items.append(item)
            
            # Store processed data
            stats = await self.data_ingestion.process_raw_data(validated_items)
            
            # Schedule sentiment analysis for tweets
            if stats["tweets"] > 0:
                await self._schedule_sentiment_analysis(validated_items)
            
            logger.info(f"Twitter collection completed: {stats}")
            return True
        
        return False
    
    async def _process_espn_collection(self, task: ProcessingTask) -> bool:
        """Process ESPN data collection task"""
        raw_items = await self.data_ingestion.fetch_espn_data()
        
        if raw_items:
            # Validate and clean data
            validated_items = []
            for item in raw_items:
                if item.data_type == "news" and self.validator.validate_news(item.content):
                    # Clean text content
                    if item.content.get("headline"):
                        item.content["headline"] = self.validator.clean_text(item.content["headline"])
                    if item.content.get("description"):
                        item.content["description"] = self.validator.clean_text(item.content["description"])
                    validated_items.append(item)
                elif item.data_type == "game" and self.validator.validate_game(item.content):
                    validated_items.append(item)
            
            # Store processed data
            stats = await self.data_ingestion.process_raw_data(validated_items)
            
            # Schedule sentiment analysis for news
            news_items = [item for item in validated_items if item.data_type == "news"]
            if news_items:
                await self._schedule_sentiment_analysis(news_items)
            
            logger.info(f"ESPN collection completed: {stats}")
            return True
        
        return False
    
    async def _process_betting_collection(self, task: ProcessingTask) -> bool:
        """Process betting lines collection task"""
        sportsbooks = task.data.get("sportsbooks", ["draftkings", "mgm"])
        
        raw_items = await self.data_ingestion.get_betting_lines(sportsbooks)
        
        if raw_items:
            # Validate data
            validated_items = []
            for item in raw_items:
                if self.validator.validate_betting_line(item.content):
                    validated_items.append(item)
            
            # Store processed data
            stats = await self.data_ingestion.process_raw_data(validated_items)
            logger.info(f"Betting lines collection completed: {stats}")
            return True
        
        return False
    
    async def _process_data_cleanup(self, task: ProcessingTask) -> bool:
        """Process data cleanup task"""
        days_to_keep = task.data.get("days_to_keep", 30)
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        db = await get_database()
        
        # Clean up old raw data
        collections = ["raw_tweets", "raw_news", "raw_games", "raw_betting_lines"]
        total_deleted = 0
        
        for collection_name in collections:
            result = await db[collection_name].delete_many({
                "processed_at": {"$lt": cutoff_date}
            })
            total_deleted += result.deleted_count
        
        logger.info(f"Data cleanup completed: deleted {total_deleted} old records")
        return True
    
    async def _schedule_sentiment_analysis(self, raw_items: List[RawDataItem]):
        """Schedule sentiment analysis tasks for text data"""
        for item in raw_items:
            if item.data_type in ["tweet", "news"]:
                text_content = ""
                if item.data_type == "tweet":
                    text_content = item.content.get("text", "")
                elif item.data_type == "news":
                    headline = item.content.get("headline", "")
                    description = item.content.get("description", "")
                    text_content = f"{headline} {description}".strip()
                
                if text_content and len(text_content) > 10:  # Only analyze meaningful text
                    sentiment_task = ProcessingTask(
                        id=f"sentiment-{item.content.get('id', datetime.now().timestamp())}",
                        task_type="sentiment_analysis",
                        data={
                            "text": text_content,
                            "source": item.source,
                            "data_type": item.data_type,
                            "original_id": item.content.get("id"),
                            "metadata": item.metadata
                        },
                        status=TaskStatus.PENDING,
                        created_at=datetime.now()
                    )
                    
                    await self.task_queue.add_task(sentiment_task)
    
    async def _process_sentiment_analysis(self, task: ProcessingTask) -> bool:
        """Process sentiment analysis task"""
        text = task.data.get("text", "")
        source = task.data.get("source", "")
        data_type = task.data.get("data_type", "")
        
        if not text:
            return False
        
        try:
            # Perform sentiment analysis
            sentiment_result = await self.sentiment_service.analyze_text(text)
            
            # Store sentiment analysis result
            db = await get_database()
            
            sentiment_doc = {
                "text": text,
                "sentiment": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence,
                "source": source,
                "data_type": data_type,
                "original_id": task.data.get("original_id"),
                "created_at": datetime.now(),
                "metadata": task.data.get("metadata", {})
            }
            
            await db.sentiment_analyses.insert_one(sentiment_doc)
            
            logger.debug(f"Sentiment analysis completed for {data_type} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return False
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        queue_stats = await self.task_queue.get_queue_stats()
        
        db = await get_database()
        
        # Get data collection stats
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        tweets_today = await db.raw_tweets.count_documents({
            "processed_at": {"$gte": today}
        })
        
        news_today = await db.raw_news.count_documents({
            "processed_at": {"$gte": today}
        })
        
        sentiment_today = await db.sentiment_analyses.count_documents({
            "created_at": {"$gte": today}
        })
        
        return {
            "pipeline_status": "running" if self.is_running else "stopped",
            "queue_stats": queue_stats,
            "data_collected_today": {
                "tweets": tweets_today,
                "news": news_today,
                "sentiment_analyses": sentiment_today
            },
            "workers_active": len([w for w in self.worker_tasks if not w.done()])
        }


# Global instance
data_processing_pipeline = DataProcessingPipeline()