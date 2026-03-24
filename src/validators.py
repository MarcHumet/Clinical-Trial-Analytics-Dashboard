"""
Input validation utilities for Clinical Trial Analytics Dashboard
Provides security through input sanitization and validation
"""
from typing import Optional
from src.config import config


class InputValidator:
    """Validates and sanitizes user inputs"""
    
    @staticmethod
    def validate_table_name(table_name: str) -> str:
        """
        Validate table name against whitelist to prevent SQL injection
        
        Args:
            table_name: The table name to validate
            
        Returns:
            The validated table name
            
        Raises:
            ValueError: If table name is not in the whitelist
        """
        if not table_name:
            raise ValueError("Table name cannot be empty")
        
        if table_name not in config.ALLOWED_TABLES:
            raise ValueError(
                f"Invalid table name: {table_name}. "
                f"Allowed tables: {', '.join(sorted(config.ALLOWED_TABLES))}"
            )
        
        return table_name
    
    @staticmethod
    def validate_search_query(query: str) -> str:
        """
        Validate and sanitize search query
        
        Args:
            query: The search query string
            
        Returns:
            The validated and sanitized query
            
        Raises:
            ValueError: If query is invalid
        """
        if not query:
            return ""
        
        # Check length
        if len(query) > config.SEARCH_QUERY_MAX_LENGTH:
            raise ValueError(
                f"Search query too long. Maximum {config.SEARCH_QUERY_MAX_LENGTH} characters allowed."
            )
        
        # Remove any null bytes or other dangerous characters
        query = query.replace('\x00', '')
        
        # Strip leading/trailing whitespace
        query = query.strip()
        
        return query
    
    @staticmethod
    def validate_enrollment_range(min_val: int, max_val: int) -> tuple:
        """
        Validate enrollment range values
        
        Args:
            min_val: Minimum enrollment value
            max_val: Maximum enrollment value
            
        Returns:
            Tuple of (min_val, max_val)
            
        Raises:
            ValueError: If values are invalid
        """
        if min_val < 0:
            raise ValueError("Minimum enrollment cannot be negative")
        
        if max_val < 0:
            raise ValueError("Maximum enrollment cannot be negative")
        
        if min_val > max_val:
            raise ValueError("Minimum enrollment cannot be greater than maximum")
        
        if max_val > 1000000:  # Sanity check
            raise ValueError("Enrollment values seem unrealistically high")
        
        return min_val, max_val
    
    @staticmethod
    def validate_status(status: str) -> str:
        """
        Validate study status value
        
        Args:
            status: The status string
            
        Returns:
            The validated status
            
        Raises:
            ValueError: If status is invalid
        """
        valid_statuses = set(config.STATUS_SCORES.keys())
        
        if status and status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {status}. "
                f"Valid statuses: {', '.join(sorted(valid_statuses))}"
            )
        
        return status
    
    @staticmethod
    def validate_phase(phase: Optional[str]) -> Optional[str]:
        """
        Validate study phase value
        
        Args:
            phase: The phase string
            
        Returns:
            The validated phase
            
        Raises:
            ValueError: If phase is invalid
        """
        if not phase or phase == "ALL":
            return phase
        
        valid_phases = {
            'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4',
            'PHASE1/PHASE2', 'PHASE2/PHASE3',
            'N/A', 'EARLY_PHASE1'
        }
        
        # Normalize phase string for comparison
        phase_normalized = phase.replace(' ', '_').upper()
        
        if phase_normalized not in valid_phases:
            raise ValueError(
                f"Invalid phase: {phase}. "
                f"Valid phases: {', '.join(sorted(valid_phases))}"
            )
        
        return phase
    
    @staticmethod
    def sanitize_column_name(column_name: str) -> str:
        """
        Sanitize column name to prevent SQL injection in dynamic queries
        
        Args:
            column_name: The column name
            
        Returns:
            The sanitized column name
            
        Raises:
            ValueError: If column name contains invalid characters
        """
        if not column_name:
            raise ValueError("Column name cannot be empty")
        
        # Only allow alphanumeric characters and underscores
        if not all(c.isalnum() or c == '_' for c in column_name):
            raise ValueError(
                f"Invalid column name: {column_name}. "
                "Only alphanumeric characters and underscores allowed."
            )
        
        return column_name


# Convenience instance
validator = InputValidator()
