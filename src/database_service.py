"""
Database service layer for Clinical Trial Analytics Dashboard
Provides centralized database access with connection pooling and error handling
"""
import os
import time
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from loguru import logger

from src.config import config
from src.validators import validator


class DatabaseService:
    """Centralized database service with connection pooling"""
    
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None
    
    @classmethod
    def get_mysql_url(cls) -> str:
        """
        Build MySQL connection URL from environment variables
        
        Returns:
            MySQL connection URL string
        """
        # Try to get full URL first
        mysql_url = os.getenv('MYSQL_DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URL')
        
        if mysql_url:
            return mysql_url
        
        # Build from individual components
        mysql_user = os.getenv('MYSQL_USER') or os.getenv('DB_USER') or 'user'
        mysql_password = os.getenv('MYSQL_PASSWORD') or os.getenv('DB_PASSWORD') or 'pass'
        mysql_host = os.getenv('DB_HOST') or os.getenv('MYSQL_HOST') or 'mysql'
        mysql_port = os.getenv('DB_PORT') or os.getenv('MYSQL_PORT') or '3306'
        mysql_db = os.getenv('DB_NAME') or os.getenv('MYSQL_DATABASE') or 'clinicaltrials'
        
        return (
            f"mysql+pymysql://{mysql_user}:{mysql_password}@"
            f"{mysql_host}:{mysql_port}/{mysql_db}"
        )
    
    @classmethod
    @st.cache_resource
    def get_engine(cls) -> Engine:
        """
        Get or create SQLAlchemy engine with connection pooling
        
        Returns:
            SQLAlchemy Engine instance
        """
        if cls._engine is None:
            mysql_url = cls.get_mysql_url()
            
            cls._engine = create_engine(
                mysql_url,
                poolclass=QueuePool,
                pool_size=config.DB_POOL_SIZE,
                max_overflow=config.DB_MAX_OVERFLOW,
                pool_pre_ping=config.DB_POOL_PRE_PING,
                pool_recycle=config.DB_POOL_RECYCLE,
                echo=False  # Set to True for SQL query logging
            )
            
            logger.info(f"Database engine created with pool_size={config.DB_POOL_SIZE}")
        
        return cls._engine
    
    @classmethod
    def get_session_factory(cls) -> sessionmaker:
        """
        Get or create session factory
        
        Returns:
            SQLAlchemy sessionmaker
        """
        if cls._session_factory is None:
            engine = cls.get_engine()
            cls._session_factory = sessionmaker(bind=engine)
        
        return cls._session_factory
    
    @classmethod
    @contextmanager
    def get_session(cls):
        """
        Context manager for database sessions
        
        Yields:
            SQLAlchemy Session
            
        Example:
            with DatabaseService.get_session() as session:
                result = session.execute(query)
        """
        SessionFactory = cls.get_session_factory()
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """
        Context manager for database connections
        
        Yields:
            SQLAlchemy Connection
            
        Example:
            with DatabaseService.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
        """
        engine = cls.get_engine()
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    @classmethod
    def execute_query_with_retry(cls, query: str, params: Optional[tuple] = None, 
                                  retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
        """
        Execute a query with retry logic for transient failures
        
        Args:
            query: SQL query string
            params: Optional query parameters
            retries: Number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            DataFrame with query results
            
        Raises:
            SQLAlchemyError: If all retries fail
        """
        for attempt in range(retries):
            try:
                engine = cls.get_engine()
                with engine.connect() as conn:
                    if params:
                        df = pd.read_sql_query(text(query), conn, params=params)
                    else:
                        df = pd.read_sql_query(text(query), conn)
                    return df
            except OperationalError as e:
                logger.warning(f"Query attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {retries} query attempts failed")
                    raise
            except SQLAlchemyError as e:
                logger.error(f"Non-recoverable database error: {e}")
                raise
        
        # Should never reach here
        raise RuntimeError("Unexpected state in execute_query_with_retry")
    
    @classmethod
    def get_table_data(cls, table_name: str, limit: int = 100) -> pd.DataFrame:
        """
        Get data from a table with validation
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with table data
            
        Raises:
            ValueError: If table name is invalid
        """
        # Validate table name to prevent SQL injection
        validated_table = validator.validate_table_name(table_name)
        
        query = f"SELECT * FROM {validated_table} LIMIT :limit"
        return cls.execute_query_with_retry(query, params={'limit': limit})
    
    @classmethod
    def get_table_count(cls, table_name: str) -> int:
        """
        Get row count for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        validated_table = validator.validate_table_name(table_name)
        
        query = f"SELECT COUNT(*) as count FROM {validated_table}"
        df = cls.execute_query_with_retry(query)
        return int(df.iloc[0]['count'])
    
    @classmethod
    def get_table_columns(cls, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table (safe against SQL injection)
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        validated_table = validator.validate_table_name(table_name)
        
        # Use information_schema instead of DESCRIBE to be more secure
        query = """
            SELECT 
                COLUMN_NAME as Field,
                DATA_TYPE as Type,
                IS_NULLABLE as `Null`,
                COLUMN_KEY as `Key`,
                COLUMN_DEFAULT as `Default`,
                EXTRA as Extra
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = :table_name
            ORDER BY ORDINAL_POSITION
        """
        
        df = cls.execute_query_with_retry(query, params={'table_name': validated_table})
        return df.to_dict('records')
    
    @classmethod
    def search_studies(cls, search_query: str, limit: int = None) -> pd.DataFrame:
        """
        Search for studies by title or description
        
        Args:
            search_query: Search term
            limit: Maximum results to return
            
        Returns:
            DataFrame with matching studies
        """
        # Validate search query
        validated_query = validator.validate_search_query(search_query)
        
        if not validated_query:
            return pd.DataFrame()
        
        if limit is None:
            limit = config.MAX_SEARCH_RESULTS
        
        query = """
            SELECT * FROM studies 
            WHERE title LIKE :search_term OR brief_summary LIKE :search_term 
            LIMIT :limit
        """
        
        search_term = f"%{validated_query}%"
        return cls.execute_query_with_retry(
            query, 
            params={'search_term': search_term, 'limit': limit}
        )
    
    @classmethod
    def check_database_empty(cls) -> bool:
        """
        Check if the studies table is empty
        
        Returns:
            True if database is empty, False otherwise
        """
        try:
            count = cls.get_table_count('studies')
            return count == 0
        except Exception as e:
            logger.error(f"Error checking if database is empty: {e}")
            return True
    
    @classmethod
    def get_column_availability(cls, table_name: str) -> pd.DataFrame:
        """
        Get data availability statistics for all columns in a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with column availability statistics
        """
        validated_table = validator.validate_table_name(table_name)
        
        # Get column names
        columns_info = cls.get_table_columns(validated_table)
        column_names = [col['Field'] for col in columns_info]
        
        availability_data = []
        
        # We need to build this dynamically but safely
        for col_name in column_names:
            # Validate column name to prevent SQL injection
            try:
                validated_col = validator.sanitize_column_name(col_name)
            except ValueError:
                logger.warning(f"Skipping invalid column name: {col_name}")
                continue
            
            query = f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN `{validated_col}` IS NULL OR `{validated_col}` = '' THEN 1 ELSE 0 END) as null_count
                FROM {validated_table}
            """
            
            try:
                result_df = cls.execute_query_with_retry(query)
                if not result_df.empty:
                    total = int(result_df.iloc[0]['total'])
                    null_count = int(result_df.iloc[0]['null_count'] or 0)
                    non_null = total - null_count
                    percentage = (non_null / total * 100) if total > 0 else 0
                    
                    availability_data.append({
                        "Column": validated_col,
                        "Total Records": total,
                        "Non-NULL": non_null,
                        "NULL": null_count,
                        "Data Availability %": f"{percentage:.1f}%"
                    })
            except Exception as e:
                logger.warning(f"Could not analyze column {validated_col}: {e}")
        
        return pd.DataFrame(availability_data)


# Convenience instance
db_service = DatabaseService()
