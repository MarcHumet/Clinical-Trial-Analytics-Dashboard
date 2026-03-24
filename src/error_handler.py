"""
Error handling utilities for Clinical Trial Analytics Dashboard
Provides user-friendly error messages and logging
"""
from typing import Optional, Any
import streamlit as st
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from mysql.connector import Error as MySQLError


class ErrorHandler:
    """Handles errors with user-friendly messages and proper logging"""
    
    # Map of error types to user-friendly messages
    ERROR_MESSAGES = {
        'OperationalError': "Unable to connect to the database. Please check your connection and try again.",
        'IntegrityError': "Data integrity error. The operation violates a database constraint.",
        'TimeoutError': "The request timed out. The system is busy, please try again in a moment.",
        'ValueError': "Invalid input provided. Please check your entries.",
        'ConnectionError': "Connection to the database failed. Please try again later.",
        'DatabaseError': "A database error occurred. Please try again or contact support.",
        'ValidationError': "The data provided did not pass validation. Please check your inputs.",
    }
    
    @classmethod
    def get_friendly_message(cls, error: Exception) -> str:
        """
        Convert technical error to user-friendly message
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        
        # Check for specific error types
        if isinstance(error, OperationalError):
            return cls.ERROR_MESSAGES['OperationalError']
        elif isinstance(error, IntegrityError):
            return cls.ERROR_MESSAGES['IntegrityError']
        elif isinstance(error, TimeoutError):
            return cls.ERROR_MESSAGES['TimeoutError']
        elif isinstance(error, ValueError):
            # For ValueError, we might want to show the actual message
            # since it often contains important validation details
            return str(error)
        elif isinstance(error, ConnectionError):
            return cls.ERROR_MESSAGES['ConnectionError']
        
        # Generic fallback
        return cls.ERROR_MESSAGES.get(
            error_type,
            "An unexpected error occurred. Please try again or contact support."
        )
    
    @classmethod
    def handle_error(cls, error: Exception, context: str = "", 
                     show_user: bool = True, show_details: bool = False) -> None:
        """
        Handle an error by logging it and optionally showing to user
        
        Args:
            error: The exception that occurred
            context: Context about where the error occurred
            show_user: Whether to show error to user in UI
            show_details: Whether to show technical details to user (for debugging)
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log technical details for developers
        if context:
            logger.error(f"Error in {context}: {error_type} - {error_message}")
        else:
            logger.error(f"{error_type}: {error_message}")
        
        # Show user-friendly message if requested
        if show_user:
            friendly_message = cls.get_friendly_message(error)
            st.error(friendly_message)
            
            if show_details:
                # Show technical details in an expander (for debugging)
                with st.expander("🔧 Technical Details"):
                    st.code(f"{error_type}: {error_message}")
    
    @classmethod
    def handle_warning(cls, message: str, context: str = "", show_user: bool = True) -> None:
        """
        Handle a warning
        
        Args:
            message: Warning message
            context: Context about where the warning occurred
            show_user: Whether to show warning to user
        """
        if context:
            logger.warning(f"Warning in {context}: {message}")
        else:
            logger.warning(message)
        
        if show_user:
            st.warning(message)
    
    @classmethod
    def handle_info(cls, message: str, context: str = "", show_user: bool = True) -> None:
        """
        Handle an informational message
        
        Args:
            message: Info message
            context: Context
            show_user: Whether to show to user
        """
        if context:
            logger.info(f"{context}: {message}")
        else:
            logger.info(message)
        
        if show_user:
            st.info(message)


def safe_execute(func, *args, context: str = "", fallback_value: Any = None, 
                 show_error: bool = True, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        context: Context description for error logging
        fallback_value: Value to return if function fails
        show_error: Whether to show error to user
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or fallback_value if error occurs
        
    Example:
        df = safe_execute(
            db_service.get_table_data,
            'studies',
            limit=100,
            context="loading studies table",
            fallback_value=pd.DataFrame()
        )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        ErrorHandler.handle_error(e, context=context, show_user=show_error)
        return fallback_value


# Convenience instance
error_handler = ErrorHandler()
