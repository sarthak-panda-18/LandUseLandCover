"""
In-Memory Job Manager for Asynchronous LULC Classification Tasks.

Provides an in-memory job store with job lifecycle tracking (pending -> processing -> done / error),
opportunistic expired job cleanup (>30 min), and an in-process asyncio.Lock to guarantee
that only ONE heavy ML classification job runs at a time to prevent RAM exhaustion.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("lulc.service.job_manager")


class JobManager:
    """Manages asynchronous classification job states and sequential execution lock."""

    def __init__(self, max_job_age_seconds: float = 1800.0, max_queue_size: int = 10):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._active_tasks: Set[asyncio.Task] = set()
        self.max_job_age_seconds = max_job_age_seconds  # Default: 30 minutes
        self.max_queue_size = max_queue_size

    def get_lock(self) -> asyncio.Lock:
        """Returns the in-process execution lock, initialized lazily on the active event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def cleanup_expired_jobs(self) -> int:
        """
        Removes jobs older than max_job_age_seconds (30 minutes) from the in-memory dict.
        Called opportunistically on each job creation.
        """
        now = time.time()
        expired_job_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if (now - job.get("created_at", now)) > self.max_job_age_seconds
        ]
        for job_id in expired_job_ids:
            self._jobs.pop(job_id, None)

        if expired_job_ids:
            logger.info(f"Cleaned up {len(expired_job_ids)} expired classification job(s).")
        return len(expired_job_ids)

    def create_job(self) -> str:
        """
        Creates a new pending job record and returns a unique job_id (uuid4).
        Performs opportunistic cleanup of expired jobs.
        """
        self.cleanup_expired_jobs()
        job_id = str(uuid.uuid4())
        now = time.time()
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        logger.info(f"Created new classification job {job_id} (status: pending)")
        return job_id

    def update_job(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Updates the status, result, and/or error message for a given job.
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return None

        job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = time.time()

        return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns a copy of the job dictionary if present, or None if invalid/expired.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return dict(job)

    def get_active_job_count(self) -> int:
        """Returns the number of jobs currently in 'pending' or 'processing' state."""
        return sum(
            1 for job in self._jobs.values() if job["status"] in ("pending", "processing")
        )

    def track_task(self, task: asyncio.Task) -> None:
        """Keeps a strong reference to an asyncio.Task to prevent premature garbage collection."""
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)


# Global singleton instance
_job_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Returns the singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance
