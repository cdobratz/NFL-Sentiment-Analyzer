"""
Application performance monitoring and alerting system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .config import settings
from .logging import log_business_event

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""

    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_SERVICE = "external_service"
    BUSINESS_METRIC = "business_metric"
    SECURITY = "security"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data structure"""

    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores application metrics"""

    def __init__(self):
        """
        Initialize the MetricsCollector and its internal in-memory storage.

        Creates empty containers used to store collected metrics and current counter/gauge values:
        - metrics: mapping from metric name to a list of Metric samples.
        - counters: current numeric values for counter metrics keyed by name.
        - gauges: current numeric values for gauge metrics keyed by name.
        """
        self.metrics: Dict[str, List[Metric]] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}

    def increment_counter(
        self, name: str, value: float = 1.0, labels: Dict[str, str] = None
    ):
        """
        Increment the named counter metric and record a timestamped Metric entry.

        Parameters:
            name (str): Metric name to increment.
            value (float): Amount to add to the counter (default 1.0).
            labels (Dict[str, str], optional): Key/value labels to associate with the metric.

        Notes:
            - Creates or updates an internal counter keyed by `name` and `labels`.
            - Appends a Metric with the updated counter value and current UTC timestamp.
            - Retains only the most recent 1000 Metric entries per metric name.
        """
        key = f"{name}_{hash(str(sorted((labels or {}).items())))}"
        self.counters[key] = self.counters.get(key, 0) + value

        # Store metric
        metric = Metric(
            name=name,
            value=self.counters[key],
            timestamp=datetime.utcnow(),
            labels=labels or {},
        )

        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)

        # Keep only last 1000 metrics per name
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Set the current value of a named gauge metric and record it in the collector.

        Parameters:
            name (str): Metric name.
            value (float): Gauge value to set.
            labels (Dict[str, str], optional): Optional labels to associate with the metric; used to distinguish metric series and stored with the recorded Metric.
        """
        key = f"{name}_{hash(str(sorted((labels or {}).items())))}"
        self.gauges[key] = value

        # Store metric
        metric = Metric(
            name=name, value=value, timestamp=datetime.utcnow(), labels=labels or {}
        )

        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)

        # Keep only last 1000 metrics per name
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """
        Retrieve stored metrics for a given metric name, optionally only those recorded at or after a given timestamp.

        Parameters:
            name (str): The metric name to look up.
            since (Optional[datetime]): If provided, only metrics with timestamp >= this value are returned.

        Returns:
            List[Metric]: A list of Metric instances matching the name and optional time filter.
        """
        metrics = self.metrics.get(name, [])

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return metrics

    def get_latest_value(
        self, name: str, labels: Dict[str, str] = None
    ) -> Optional[float]:
        """
        Retrieve the most recent value recorded for the metric with the given name, optionally filtered by exact label matches.

        Parameters:
            labels (Dict[str, str], optional): Exact label key/value pairs to match against metric labels. If provided, only metrics whose labels equal this mapping are considered.

        Returns:
            Optional[float]: The latest metric value if any matching metric exists, `None` otherwise.
        """
        metrics = self.get_metrics(name)

        if labels:
            metrics = [m for m in metrics if m.labels == labels]

        if metrics:
            return metrics[-1].value

        return None


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self):
        """
        Initialize the AlertManager internal state.

        Sets up:
        - alerts: list of stored Alert objects.
        - alert_handlers: list of callables to be invoked when alerts are created.
        - alert_rules: list of alert rule dictionaries used for rule-based evaluations.
        """
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_rules: List[Dict[str, Any]] = []

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        Register a callable to be invoked whenever a new Alert is created.

        Parameters:
            handler (Callable[[Alert], None]): A function or coroutine that accepts a single Alert argument; registered handlers will be executed (awaited if a coroutine) for each created alert.
        """
        self.alert_handlers.append(handler)

    def add_alert_rule(self, rule: Dict[str, Any]):
        """
        Registers an alert rule to be stored by the alert manager for later evaluation.

        Parameters:
            rule (Dict[str, Any]): A dictionary describing the rule's conditions and associated metadata or actions.
        """
        self.alert_rules.append(rule)

    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> Alert:
        """
        Create a new Alert, record it in the manager, log its creation, and dispatch it to all registered handlers.

        The created Alert will include a generated identifier and timestamp and is appended to the manager's alert list; registered handlers are invoked for processing.

        Parameters:
            metadata (Dict[str, Any], optional): Additional context to attach to the Alert; defaults to an empty dict if omitted.

        Returns:
            Alert: The created Alert object with id, type, severity, title, message, timestamp, and metadata.
        """
        alert = Alert(
            id=f"{alert_type.value}_{int(time.time())}",
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.alerts.append(alert)

        # Log the alert
        log_business_event(
            "alert_created",
            {
                "alert_id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
            },
        )

        # Process alert handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.create_task(self._run_handler(handler, alert))
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        return alert

    async def _run_handler(self, handler: Callable, alert: Alert):
        """
        Execute the given alert handler, supporting both coroutine functions and regular callables.

        Parameters:
                handler (Callable): The handler to invoke; may be an async function or a regular callable that accepts an Alert.
                alert (Alert): The alert object to pass to the handler.
        """
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)

    def resolve_alert(self, alert_id: str):
        """
        Mark the specified alert as resolved and record its resolution timestamp.

        Parameters:
            alert_id (str): Identifier of the alert to resolve; the first matching unresolved alert will be marked resolved.

        Notes:
            If an unresolved alert with the given id is found, its `resolved` flag is set to True and `resolved_at` is set to the current UTC time. A business event "alert_resolved" is logged with `alert_id` and the resolution time in seconds. If no matching unresolved alert exists, the function does nothing.
        """
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()

                log_business_event(
                    "alert_resolved",
                    {
                        "alert_id": alert.id,
                        "resolution_time": (
                            alert.resolved_at - alert.timestamp
                        ).total_seconds(),
                    },
                )
                break

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        List active (unresolved) alerts sorted by timestamp from newest to oldest.

        If `severity` is provided, only alerts with that severity are returned.

        Parameters:
            severity (Optional[AlertSeverity]): If set, filters results to alerts matching this severity.

        Returns:
            List[Alert]: Unresolved alerts sorted by timestamp in descending order (newest first).
        """
        alerts = [a for a in self.alerts if not a.resolved]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)


class PerformanceMonitor:
    """Monitors application performance and triggers alerts"""

    def __init__(
        self, metrics_collector: MetricsCollector, alert_manager: AlertManager
    ):
        """
        Create a PerformanceMonitor that runs checks using a MetricsCollector and an AlertManager.

        Initializes the monitor with references to the provided MetricsCollector and AlertManager and establishes default thresholds:
        - `error_rate_threshold`: 0.05 (5%),
        - `response_time_threshold`: 2000 (milliseconds),
        - `cpu_threshold`: 80 (percent),
        - `memory_threshold`: 85 (percent),
        - `disk_threshold`: 90 (percent).
        """
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.thresholds = {
            "error_rate_threshold": 0.05,  # 5% error rate
            "response_time_threshold": 2000,  # 2 seconds
            "cpu_threshold": 80,  # 80% CPU usage
            "memory_threshold": 85,  # 85% memory usage
            "disk_threshold": 90,  # 90% disk usage
        }

    async def check_error_rate(self):
        """
        Evaluate recent API traffic for elevated error rate and create an alert when it exceeds the configured threshold.

        Checks API request and error metrics from the last 5 minutes; if there are requests and the error rate is greater than the configured `error_rate_threshold`, creates an ERROR_RATE alert with HIGH severity and metadata containing the error rate, total request count, and error count.
        """
        try:
            # Get recent metrics
            since = datetime.utcnow() - timedelta(minutes=5)
            total_requests = len(self.metrics.get_metrics("api_requests", since))
            error_requests = len(self.metrics.get_metrics("api_errors", since))

            if total_requests > 0:
                error_rate = error_requests / total_requests

                if error_rate > self.thresholds["error_rate_threshold"]:
                    await self.alerts.create_alert(
                        AlertType.ERROR_RATE,
                        AlertSeverity.HIGH,
                        "High Error Rate Detected",
                        f"Error rate is {error_rate:.2%} (threshold: {self.thresholds['error_rate_threshold']:.2%})",
                        {
                            "error_rate": error_rate,
                            "total_requests": total_requests,
                            "error_requests": error_requests,
                        },
                    )
        except Exception as e:
            logger.error(f"Error rate check failed: {e}")

    async def check_response_time(self):
        """Check API response time and alert if threshold exceeded"""
        try:
            # Get recent response times
            since = datetime.utcnow() - timedelta(minutes=5)
            response_times = self.metrics.get_metrics("api_response_time", since)

            if response_times:
                avg_response_time = sum(m.value for m in response_times) / len(
                    response_times
                )

                if avg_response_time > self.thresholds["response_time_threshold"]:
                    await self.alerts.create_alert(
                        AlertType.RESPONSE_TIME,
                        AlertSeverity.MEDIUM,
                        "High Response Time Detected",
                        f"Average response time is {avg_response_time:.0f}ms (threshold: {self.thresholds['response_time_threshold']}ms)",
                        {
                            "avg_response_time": avg_response_time,
                            "sample_count": len(response_times),
                        },
                    )
        except Exception as e:
            logger.error(f"Response time check failed: {e}")

    async def check_system_resources(self):
        """Check system resource usage and alert if thresholds exceeded"""
        try:
            # Check CPU usage
            cpu_usage = self.metrics.get_latest_value("system_cpu_percent")
            if cpu_usage and cpu_usage > self.thresholds["cpu_threshold"]:
                await self.alerts.create_alert(
                    AlertType.SYSTEM_RESOURCE,
                    AlertSeverity.HIGH,
                    "High CPU Usage",
                    f"CPU usage is {cpu_usage:.1f}% (threshold: {self.thresholds['cpu_threshold']}%)",
                    {"cpu_usage": cpu_usage},
                )

            # Check memory usage
            memory_usage = self.metrics.get_latest_value("system_memory_percent")
            if memory_usage and memory_usage > self.thresholds["memory_threshold"]:
                await self.alerts.create_alert(
                    AlertType.SYSTEM_RESOURCE,
                    AlertSeverity.HIGH,
                    "High Memory Usage",
                    f"Memory usage is {memory_usage:.1f}% (threshold: {self.thresholds['memory_threshold']}%)",
                    {"memory_usage": memory_usage},
                )

            # Check disk usage
            disk_usage = self.metrics.get_latest_value("system_disk_percent")
            if disk_usage and disk_usage > self.thresholds["disk_threshold"]:
                await self.alerts.create_alert(
                    AlertType.SYSTEM_RESOURCE,
                    AlertSeverity.CRITICAL,
                    "High Disk Usage",
                    f"Disk usage is {disk_usage:.1f}% (threshold: {self.thresholds['disk_threshold']}%)",
                    {"disk_usage": disk_usage},
                )
        except Exception as e:
            logger.error(f"System resource check failed: {e}")

    async def run_checks(self):
        """
        Run all configured performance checks concurrently and wait for them to finish.

        This schedules error rate, response time, and system resource checks to run in parallel and waits for their completion; exceptions raised by individual checks are captured and not propagated.
        """
        await asyncio.gather(
            self.check_error_rate(),
            self.check_response_time(),
            self.check_system_resources(),
            return_exceptions=True,
        )


# Global instances
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
performance_monitor = PerformanceMonitor(metrics_collector, alert_manager)


# Default alert handlers
async def log_alert_handler(alert: Alert):
    """
    Log an Alert to the application logger including severity, title, message, and structured context.

    Parameters:
        alert (Alert): Alert to log; logged fields include severity, title, message, id, type, and metadata.
    """
    logger.warning(
        f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}",
        extra={
            "alert_id": alert.id,
            "alert_type": alert.type.value,
            "alert_metadata": alert.metadata,
        },
    )


# Register default handlers
alert_manager.add_alert_handler(log_alert_handler)
