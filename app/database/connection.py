"""
MySQL Connection and Transaction Management Module.

Provides robust connection handling using PyMySQL, parameterized statement support,
automatic transaction rollback on error, and graceful degradation when the MySQL service
is offline or unreachable.
"""

from contextlib import contextmanager
from typing import Generator, Optional
import pymysql
from pymysql.cursors import DictCursor
from app.utils.logger import setup_logger
from config import Config

logger = setup_logger("db_connection")


class DatabaseConnectionPool:
    """Manages MySQL database connections and schema verification."""

    @staticmethod
    def get_raw_connection() -> pymysql.Connection:
        """
        Establishes a raw PyMySQL connection using settings from Config.

        Returns:
            pymysql.Connection: Active database connection.
        """
        try:
            conn = pymysql.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                charset="utf8mb4",
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=5,
            )
            return conn
        except pymysql.MySQLError as e:
            logger.error(f"MySQL Connection Error [Host={Config.DB_HOST}:{Config.DB_PORT}, DB={Config.DB_NAME}]: {e}")
            raise ConnectionError(f"Failed to connect to MySQL database: {e}")

    @staticmethod
    def check_connection() -> bool:
        """Verifies whether the database server is reachable and active."""
        try:
            conn = DatabaseConnectionPool.get_raw_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False


@contextmanager
def get_db_connection() -> Generator[pymysql.Connection, None, None]:
    """
    Context manager for transactional database operations.
    Automatically commits on success and rolls back on exception.

    Yields:
        pymysql.Connection: Active connection with DictCursor.
    """
    conn = DatabaseConnectionPool.get_raw_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction aborted due to error: {e}. Executed ROLLBACK.")
        raise
    finally:
        conn.close()
