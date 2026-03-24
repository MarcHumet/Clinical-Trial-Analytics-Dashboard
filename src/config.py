"""
Configuration settings for Clinical Trial Analytics Dashboard
Centralizes all hardcoded values and configuration constants
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class DashboardConfig:
    """Centralized configuration for the dashboard"""
    
    # API Settings
    API_URL: str = "https://clinicaltrials.gov/api/v2/studies"
    API_PAGE_SIZE: int = 100
    API_MAX_PAGES: int = 100
    API_TIMEOUT: int = 30
    
    # Database Settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 3600  # 1 hour
    DB_POOL_PRE_PING: bool = True
    
    # Cache Settings
    CACHE_TTL: int = 300  # 5 minutes
    CACHE_MAX_ENTRIES: int = 10
    
    # Enrollment Metrics Thresholds
    ENROLLMENT_BENCHMARK_MONTHLY: int = 10
    ENROLLMENT_EXCELLENT_THRESHOLD: int = 50
    ENROLLMENT_GOOD_THRESHOLD: int = 10
    ENROLLMENT_ADEQUATE_THRESHOLD: int = 1
    
    # Status Scores (used in enrollment success calculation)
    STATUS_SCORES: Dict[str, int] = None
    
    # Composite Score Weights
    COMPOSITE_WEIGHT_ENROLLMENT: float = 0.40
    COMPOSITE_WEIGHT_STATUS: float = 0.30
    COMPOSITE_WEIGHT_TEMPORAL: float = 0.20
    COMPOSITE_WEIGHT_DATA: float = 0.10
    
    # UI Theme Colors
    COLOR_SUCCESS: str = '#00ff00'
    COLOR_WARNING: str = '#FFD700'
    COLOR_DANGER: str = '#FF6B6B'
    COLOR_INFO: str = '#808080'
    COLOR_BACKGROUND: str = '#000000'
    COLOR_TEXT: str = '#FFFFFF'
    
    # Query Limits
    SEARCH_QUERY_MAX_LENGTH: int = 500
    MAX_SEARCH_RESULTS: int = 50
    MAX_DISPLAY_ROWS: int = 100
    
    # Allowed Database Tables (for security - whitelist)
    ALLOWED_TABLES: set = None
    
    def __post_init__(self):
        """Initialize complex default values"""
        if self.STATUS_SCORES is None:
            self.STATUS_SCORES = {
                'COMPLETED': 100,
                'ACTIVE_NOT_RECRUITING': 85,
                'RECRUITING': 60,
                'ENROLLING_BY_INVITATION': 60,
                'SUSPENDED': 40,
                'TERMINATED': 30,
                'WITHDRAWN': 20,
                'UNKNOWN': 10,
                'NOT_YET_RECRUITING': 50
            }
        
        if self.ALLOWED_TABLES is None:
            self.ALLOWED_TABLES = {
                'studies',
                'conditions',
                'interventions',
                'outcomes',
                'sponsors',
                'locations',
                'study_design',
                'eligibility'
            }
    
    def get_success_tier_colors(self) -> Dict[str, str]:
        """Get color mapping for enrollment success tiers"""
        return {
            'Exceeded': self.COLOR_SUCCESS,
            'Met': '#90EE90',  # Light green
            'Below': self.COLOR_WARNING,
            'Significantly Below': self.COLOR_DANGER
        }
    
    def get_colorblind_safe_palette(self) -> list:
        """Get colorblind-friendly color palette"""
        return [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#CA9161',  # Brown
            '#FBAFE4',  # Pink
        ]


# Global config instance
config = DashboardConfig()
