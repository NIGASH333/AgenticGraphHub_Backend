"""
Rate Limiting Monitor

Provides monitoring and alerting for rate limiting.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Rate limiting alert."""
    timestamp: datetime
    user_id: str
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class RateLimitMonitor:
    """Monitor rate limiting and generate alerts."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.monitoring = False
        self.alert_thresholds = {
            "high_usage": 0.8,  # 80% of limits
            "cost_warning": 0.7,  # 70% of cost limit
            "burst_detected": 0.9,  # 90% of burst limit
        }
    
    async def start_monitoring(self, check_interval: int = 60):
        """Start monitoring rate limits."""
        self.monitoring = True
        logger.info("Rate limiting monitor started")
        
        while self.monitoring:
            try:
                await self._check_limits()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in rate limit monitoring: {e}")
                await asyncio.sleep(check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        logger.info("Rate limiting monitor stopped")
    
    async def _check_limits(self):
        """Check current limits and generate alerts."""
        try:
            stats = await rate_limiter.get_global_stats()
            
            # Check for high usage
            if stats["total_requests_today"] > stats["config"]["requests_per_day"] * 0.8:
                self._add_alert(
                    "high_usage",
                    "High daily request usage detected",
                    "medium"
                )
            
            # Check for high cost
            if stats["total_cost_today"] > stats["config"]["cost_per_day"] * 0.7:
                self._add_alert(
                    "cost_warning",
                    f"High daily cost: ${stats['total_cost_today']:.2f}",
                    "high"
                )
            
            # Check for burst activity
            if stats["active_users"] > 10:  # Many active users
                self._add_alert(
                    "high_activity",
                    f"High user activity: {stats['active_users']} active users",
                    "medium"
                )
            
        except Exception as e:
            logger.error(f"Error checking limits: {e}")
    
    def _add_alert(self, alert_type: str, message: str, severity: str, user_id: str = "system"):
        """Add an alert."""
        alert = Alert(
            timestamp=datetime.now(),
            user_id=user_id,
            alert_type=alert_type,
            message=message,
            severity=severity
        )
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Log alert
        logger.warning(f"Rate limit alert [{severity}]: {message}")
    
    def get_alerts(self, 
                   severity: Optional[str] = None,
                   hours: int = 24) -> List[Alert]:
        """Get alerts filtered by severity and time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity == severity
            ]
        
        return filtered_alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_1h = now - timedelta(hours=1)
        
        alerts_24h = [a for a in self.alerts if a.timestamp >= last_24h]
        alerts_1h = [a for a in self.alerts if a.timestamp >= last_1h]
        
        severity_counts = {}
        for alert in alerts_24h:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        return {
            "total_alerts_24h": len(alerts_24h),
            "total_alerts_1h": len(alerts_1h),
            "severity_breakdown": severity_counts,
            "latest_alert": alerts_24h[-1].timestamp.isoformat() if alerts_24h else None
        }


# Global monitor instance
monitor = RateLimitMonitor()


async def start_rate_limit_monitoring():
    """Start the rate limiting monitor."""
    await monitor.start_monitoring()


def get_monitor_status() -> Dict:
    """Get current monitor status."""
    return {
        "monitoring": monitor.monitoring,
        "alert_summary": monitor.get_alert_summary(),
        "recent_alerts": [
            {
                "timestamp": alert.timestamp.isoformat(),
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "user_id": alert.user_id
            }
            for alert in monitor.get_alerts(hours=1)
        ]
    }
