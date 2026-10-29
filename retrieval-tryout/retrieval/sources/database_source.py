"""
Database Source Handler
Supports: PostgreSQL, MySQL, MongoDB

All query results are returned as a single .dbrows file-info dict.
The ingestion pipeline routes .dbrows through extract_row_wise() →
_extract_dbrows_row_wise(), which embeds each row directly in Qdrant.
No fake filenames, no chunking step, no intermediate .txt files.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from urllib.parse import quote_plus

from pymongo import MongoClient

logger = logging.getLogger(__name__)


class DatabaseSource:
    """Handle database queries and data fetch"""

    @staticmethod
    def _sanitize_failed_filename_component(value: Any, default: str) -> str:
        """Convert database identifiers into safe filename components."""
        raw_value = str(value or "").strip()
        if not raw_value:
            return default

        normalized = raw_value.replace("\\", "/").rstrip("/")
        if "://" in normalized:
            normalized = normalized.split("://", 1)[1]
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[-1]

        import re

        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
        return normalized or default

    @classmethod
    def build_discovery_failed_filename(cls, source_details: Dict[str, Any]) -> str:
        """Build the failed-files entry for database discovery failures."""
        return cls._sanitize_failed_filename_component(
            source_details.get("database_name")
            or source_details.get("database"),
            "database",
        )

    @staticmethod
    def _resolve_db_type(
        configured_type: Any,
        connection_uri: str,
        host: str,
    ) -> str:
        """Resolve database subtype from explicit config first, then URI/host hints."""
        normalized = str(configured_type or "").strip().lower()
        if normalized:
            return normalized

        uri = str(connection_uri or "").strip().lower()
        normalized_host = str(host or "").strip().lower()

        if uri.startswith("mongodb://") or uri.startswith("mongodb+srv://"):
            return "mongodb"
        if uri.startswith("postgresql://") or uri.startswith("postgres://"):
            return "postgresql"
        if uri.startswith("mysql://"):
            return "mysql"

        if "mongodb.net" in normalized_host or normalized_host.startswith("mongo"):
            return "mongodb"
        if "postgres" in normalized_host:
            return "postgresql"
        if "mysql" in normalized_host:
            return "mysql"

        return ""

    def __init__(self, request_data: Dict[str, Any]):
        """
        Initialize database source

        Args:
            request_data: Complete request JSON
        """
        self.request_data = request_data
        self.source_details = request_data.get('source_details', {})

        self.host = self.source_details.get('host', '')
        self.port = self.source_details.get('port', '')
        self.username = self.source_details.get('username', '')
        self.password = self.source_details.get('password', '')
        self.connection_uri = (
            self.source_details.get('connection_uri') or
            self.source_details.get('connectionUri') or
            self.source_details.get('uri') or
            ''
        )
        self.db_type = self._resolve_db_type(
            self.source_details.get('database_type_name')
            or self.source_details.get('databaseTypeName')
            or self.source_details.get('database_type')
            or self.source_details.get('databaseType'),
            self.connection_uri,
            self.host,
        )

        # Try multiple field names for database, default to 'postgres' for PostgreSQL
        self.database = (
            self.source_details.get('database_name') or
            self.source_details.get('databaseName') or
            self.source_details.get('database') or
            self.source_details.get('db') or
            ('postgres' if 'postgres' in self.db_type else self.source_details.get('connectionName', ''))
        )

        self.query = (
            self.source_details.get('custom_query') or
            self.source_details.get('customQuery') or
            ''
        )
        self.ssl = (
            self.source_details.get('ssl_mode') or
            self.source_details.get('sslMode') or
            False
        )
        
        # MongoDB specific fields
        self.collection_name = self.source_details.get('collection_name', '')
        self.query_filter = self.source_details.get('query_filter', '')

        logger.info(f"Database Source initialized: {self.db_type}")
        logger.info(f"  Host: {self.host}")
        logger.info(f"  Database: {self.database}")

    async def discover(self) -> List[Dict[str, Any]]:
        """Query database and return results as file objects"""
        if 'postgres' in self.db_type:
            return await self._query_postgres()
        elif 'mysql' in self.db_type:
            return await self._query_mysql()
        elif 'mongo' in self.db_type:
            return await self._query_mongo()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    async def _query_postgres(self) -> List[Dict[str, Any]]:
        """Query PostgreSQL — return a single .dbrows file-info dict."""
        try:
            logger.info("🔍 Connecting to PostgreSQL...")
            eu = quote_plus(self.username)
            ep = quote_plus(self.password)

            # Connect timeout bounds only the TCP/handshake phase. We ALSO set a
            # statement timeout + TCP keepalives below, otherwise a stall that
            # happens AFTER connecting (network black-hole, SSL renegotiation,
            # server accepts but never streams rows) would hang the blocking
            # pd.read_sql_query call forever with no error — a silent freeze.
            timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "30"))
            # Max time (ms) the server will allow the query to run before it
            # aborts and returns an error. Guards the fetch/read phase.
            stmt_timeout_ms = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "120000"))

            conn_str = f"postgresql://{eu}:{ep}@{self.host}:{self.port}/{self.database}"
            params = [f"connect_timeout={timeout}"]
            if self.ssl:
                params.append("sslmode=require")
            conn_str += "?" + "&".join(params)

            logger.info(
                f"Executing query (connect_timeout={timeout}s, "
                f"statement_timeout={stmt_timeout_ms}ms): {self.query[:100]}..."
            )

            # Fetch with the raw psycopg2 driver rather than pandas.read_sql.
            # The pandas/numpy result-materialization path can crash natively
            # (Windows access violation / segfault) on newer Python builds where
            # numpy/pandas wheels are not yet stable, which manifests as a silent
            # process exit with no traceback. psycopg2 alone is robust, and we
            # build the same list-of-plain-dicts the rest of the code expects.
            import psycopg2

            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.username,
                password=self.password,
                sslmode="require" if self.ssl else "prefer",
                connect_timeout=timeout,
                options=f"-c statement_timeout={stmt_timeout_ms}",
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            try:
                cur = conn.cursor()
                cur.execute(self.query)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                fetched = cur.fetchall()
                cur.close()
            finally:
                conn.close()

            plain_rows = [self._normalize_record(dict(zip(columns, r))) for r in fetched]

            logger.info(f"✅ PostgreSQL returned {len(plain_rows)} row(s)")
            rows = [
                {'text': self._row_dict_to_text(r, i), 'row_index': i}
                for i, r in enumerate(plain_rows, start=1)
            ]
            return [self._make_dbrows_file_info(rows, db_type='postgresql', prefix='pg')]
        except Exception as exc:
            logger.error(f"PostgreSQL query failed: {exc}", exc_info=True)
            logger.error(
                "💡 If this timed out or failed name resolution: verify the DB "
                "host/port are reachable, DNS resolves the host, and your IP is "
                "allow-listed on the provider (Aiven/RDS). Raise "
                "POSTGRES_CONNECT_TIMEOUT for slow networks."
            )
            raise

    async def _query_mysql(self) -> List[Dict[str, Any]]:
        """Query MySQL — return a single .dbrows file-info dict.
        
        For cloud MySQL (Aiven, AWS RDS, etc) that times out, set:
            export MYSQL_CONNECT_TIMEOUT=60  (default: 30)
        """
        try:
            logger.info("🔍 Connecting to MySQL...")
            eu = quote_plus(self.username)
            ep = quote_plus(self.password)
            
            # Get timeout from environment, default 30s (pymysql default)
            # For cloud MySQL: use 60s or higher
            timeout = int(os.getenv('MYSQL_CONNECT_TIMEOUT', '120'))
            
            # Build connection string with timeout parameter
            conn_str = f"mysql+pymysql://{eu}:{ep}@{self.host}:{self.port}/{self.database}"
            if self.ssl:
                conn_str += "?charset=utf8mb4"  # Add charset for UTF-8 support
            
            logger.info(f"Executing query (timeout={timeout}s): {self.query[:100]}...")
            
            # Use SQLAlchemy to pass connect_args with timeout
            from sqlalchemy import create_engine
            engine = create_engine(
                conn_str,
                connect_args={
                    'connect_timeout': timeout,
                    'read_timeout': timeout,
                    'write_timeout': timeout,
                }
            )
            
            df = pd.read_sql_query(self.query, engine)
            logger.info(f"✅ MySQL returned {len(df)} row(s)")
            rows = self._dataframe_to_rows(df)
            engine.dispose()
            return [self._make_dbrows_file_info(rows, db_type='mysql', prefix='mysql')]
        except Exception as exc:
            logger.error(f"MySQL query failed: {exc}", exc_info=True)
            logger.error("💡 If timeout: export MYSQL_CONNECT_TIMEOUT=60 or higher")
            raise

    async def _query_mongo(self) -> List[Dict[str, Any]]:
        """Query MongoDB — return one file-info dict per document-group."""
        try:
            logger.info("🔍 Connecting to MongoDB...")
            try:
                query_dict = json.loads(self.query) if isinstance(self.query, str) else self.query
            except Exception:
                query_dict = {}

            client = self._create_mongo_client()
            db = client[self.database]
            
            # Try to get collection_name from new field, then fall back to query dict
            collection_name = self.collection_name or query_dict.get("collection", "")
            if not collection_name:
                raise ValueError(
                    "MongoDB requires 'collection_name' field or customQuery with 'collection'. "
                    'Example: {"collection":"orders","filter":{"status":"active"}}'
                )
            collection   = db[collection_name]
            
            # Try to get filter from new field, then fall back to query dict
            filter_query = {}
            if self.query_filter:
                try:
                    filter_query = json.loads(self.query_filter) if isinstance(self.query_filter, str) else self.query_filter
                except Exception:
                    logger.warning(f"Failed to parse query_filter: {self.query_filter}")
                    filter_query = {}
            
            if not filter_query:
                filter_query = query_dict.get("filter", {})
            
            projection   = query_dict.get("projection")
            sort         = query_dict.get("sort")
            limit        = query_dict.get("limit")

            logger.info(f"Querying collection: {collection_name}")
            cursor = (
                collection.find(filter_query, projection)
                if projection is not None
                else collection.find(filter_query)
            )
            if sort:
                cursor = cursor.sort(sort)
            if isinstance(limit, int) and limit > 0:
                cursor = cursor.limit(limit)

            records = list(cursor)
            if not records:
                logger.warning(
                    f"MongoDB returned 0 documents "
                    f"(collection='{collection_name}', filter={filter_query})"
                )
                return []

            logger.info(f"✅ MongoDB returned {len(records)} document(s)")
            # Normalize Mongo docs: convert ObjectIds, dates to strings; preserve nested structures
            normalised = self._normalize_mongodb_records(records)
            rows = self._records_to_rows(normalised, collection_name=collection_name)
            return [self._make_dbrows_file_info(
                rows, db_type='mongodb', prefix='mongo',
                extra={'collection_name': collection_name},
            )]

        except Exception as exc:
            logger.error(f"MongoDB query failed: {exc}", exc_info=True)
            raise

    def _normalize_mongodb_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize MongoDB records by converting non-JSON types to appropriate formats.
        Uses ISO format for dates and string representation for ObjectIds,
        while preserving nested structures (dicts, lists) for proper flattening
        in the embedding pipeline.
        """
        from datetime import date, time
        from bson import ObjectId
        
        def convert_value(v):
            """Convert MongoDB value to JSON-serializable format."""
            if v is None:
                return None
            if isinstance(v, ObjectId):
                return str(v)  # ObjectId → string
            if isinstance(v, datetime):
                return v.isoformat()  # datetime → ISO string
            if isinstance(v, date):
                return v.isoformat()  # date → ISO string
            if isinstance(v, time):
                return v.isoformat()  # time → ISO string
            if isinstance(v, dict):
                # Recursively convert dict values
                return {k: convert_value(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                # Recursively convert list items
                return [convert_value(item) for item in v]
            if isinstance(v, bool) or isinstance(v, (int, float, str)):
                return v
            # For other types, use string representation
            return str(v)
        
        normalised = []
        for record in records:
            try:
                normalised_record = convert_value(record)
                normalised.append(normalised_record)
            except Exception as e:
                logger.warning(f"Failed to normalize record: {e}, using fallback")
                # Fallback: use json.dumps approach
                try:
                    normalised.append(json.loads(json.dumps(record, default=str)))
                except Exception:
                    logger.error(f"Failed to normalize record with fallback: {e}")
        
        return normalised

    def _create_mongo_client(self) -> MongoClient:
        """
        Create MongoDB client from either a full URI or host/port credentials.
        Atlas clusters should use `mongodb+srv://...`.
        """
        timeout_ms = int(self.source_details.get('serverSelectionTimeoutMS', 30000))

        if self.connection_uri:
            logger.info("Using MongoDB connection URI")
            return MongoClient(self.connection_uri, serverSelectionTimeoutMS=timeout_ms)

        host = (self.host or '').strip()
        if not host:
            raise ValueError("MongoDB host is required")

        # Atlas hostnames (e.g. cluster0.xxxxx.mongodb.net) should use SRV URI.
        if host.endswith('mongodb.net'):
            encoded_username = quote_plus(self.username) if self.username else ''
            encoded_password = quote_plus(self.password) if self.password else ''
            auth = f"{encoded_username}:{encoded_password}@" if encoded_username else ''
            db_name = self.database or 'admin'
            uri = f"mongodb+srv://{auth}{host}/{db_name}?retryWrites=true&w=majority"
            logger.info("Using generated MongoDB Atlas SRV URI")
            return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)

        client_kwargs = {'serverSelectionTimeoutMS': timeout_ms}
        if self.username:
            client_kwargs['username'] = self.username
        if self.password:
            client_kwargs['password'] = self.password

        logger.info("Using MongoDB host/port connection")
        return MongoClient(
            host=host,
            port=int(self.port or 27017),
            **client_kwargs
        )

    # -------------------------------------------------------------------------
    # Helpers: rows → .dbrows file-info dict
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_dict_to_text(row: Dict[str, Any], row_index: int) -> str:
        """Convert one row/document dict to the compact pipe-separated text
        that gets embedded as a single Qdrant point.
        
        Used for PostgreSQL and MySQL rows. Escapes special characters that
        would corrupt the pipe-delimited format.
        
        CRITICAL: BGE-Large model has 512 token limit. Skips sensitive binary 
        fields to fit within limits (rough: 1 token per 3-4 chars)."""
        parts = []
        
        # Fields to skip (usually contain very long hashes/binary data)
        skip_fields = {'password', 'token', 'refresh_token', 'access_token', 
                       'secret', 'api_key', 'private_key', 'hash', 'encrypted',
                       'authorization', 'bearer', 'signature', 'encrypted_password'}
        
        for k, v in row.items():
            if v is None:
                continue  # Skip NULL values
            
            # Skip problematic fields that contain long binary/hash data
            if str(k).lower() in skip_fields:
                continue
            
            # Convert value to string, escape pipes and newlines
            v_str = str(v).strip()
            if not v_str:
                continue
            
            # Escape pipes to prevent field corruption
            v_str = v_str.replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
            
            # Escape key too
            k_str = str(k).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
            
            parts.append(f"{k_str}: {v_str}")
        
        result = " | ".join(parts) if parts else "(empty record)"
        
        # Truncate aggressively for token limit safety (BGE-Large has 512 token limit)
        # Rough estimate: ~1 token per 3-4 chars, so 300-400 chars ≈ 75-100 tokens, safe buffer
        MAX_LENGTH = 1000  # Reduced from 2000 to stay within token limits
        if len(result) > MAX_LENGTH:
            result = result[:MAX_LENGTH-3] + "..."
        return result

    @staticmethod
    def _flatten_value_for_embedding(value: Any, depth: int = 0, max_depth: int = 1) -> str:
        """
        Flatten nested MongoDB values (dicts, lists) to strings suitable for embedding.
        Uses JSON serialization for complex types instead of Python repr() to avoid
        malformed output.
        
        Args:
            value: Value to flatten
            depth: Current recursion depth
            max_depth: Maximum depth before stringifying
        """
        if value is None:
            return "(null)"
        
        if isinstance(value, (int, float, bool)):
            return str(value)
        
        if isinstance(value, str):
            return value
        
        if depth >= max_depth:
            # For deeply nested structures, use JSON to avoid repr() artifacts
            try:
                json_str = json.dumps(value, default=str, ensure_ascii=True)
                # Truncate very long JSON strings
                if len(json_str) > 500:
                    json_str = json_str[:497] + "..."
                return json_str
            except (TypeError, ValueError):
                return str(value)
        
        if isinstance(value, dict):
            # For dicts, show simplified version with key-value pairs
            items = []
            for k, v in sorted(value.items())[:5]:  # Limit to 5 top-level keys
                v_flat = DatabaseSource._flatten_value_for_embedding(v, depth+1, max_depth)
                items.append(f"{k}: {v_flat}")
            return "{" + ", ".join(items) + "}"
        
        if isinstance(value, (list, tuple)):
            # For lists/arrays, show first few items with type indicators
            items = []
            for v in list(value)[:3]:  # Limit to 3 items
                v_flat = DatabaseSource._flatten_value_for_embedding(v, depth+1, max_depth)
                items.append(v_flat)
            return "[" + ", ".join(items) + "]"
        
        # Fallback for other types
        try:
            return json.dumps(value, default=str)[:200]
        except (TypeError, ValueError):
            return str(value)[:200]

    def _row_dict_to_text_mongodb(self, row: Dict[str, Any], row_index: int) -> str:
        """Convert MongoDB document to safe pipe-separated text for embedding.
        
        Handles nested structures properly by flattening them instead of
        using Python repr(). Escapes special characters to prevent data corruption.
        
        CRITICAL: BGE-Large model has 512 token limit. We keep text short (~300-400 chars)
        and skip large binary fields (passwords, tokens, hashes) to stay within limits.
        Rough estimate: ~1 token per 3-4 chars."""
        parts = []
        
        # Fields to skip (usually contain very long hashes/binary data that bloat output)
        skip_fields = {'password', 'token', 'refresh_token', 'access_token', 
                       'secret', 'api_key', 'private_key', 'hash', 'encrypted',
                       'authorization', 'bearer', 'signature'}
        
        for k, v in row.items():
            if v is None:
                continue  # Skip NULL values
            
            # Skip problematic fields that contain long binary/hash data
            if str(k).lower() in skip_fields:
                continue
            
            # Flatten nested structures safely
            v_flat = self._flatten_value_for_embedding(v, depth=0, max_depth=1)
            
            if not v_flat or not v_flat.strip():
                continue
            
            # Escape pipes and newlines in both key and value
            k_safe = str(k).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
            v_safe = v_flat.replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
            
            parts.append(f"{k_safe}: {v_safe}")
        
        result = " | ".join(parts) if parts else "(empty record)"
        
        # Truncate aggressively for token limit safety (512 token limit, aim for 300 chars)
        MAX_LENGTH = 400  # Reduced from 2000 to stay within BGE-Large 512 token limit
        if len(result) > MAX_LENGTH:
            result = result[:MAX_LENGTH-3] + "..."
        return result

    def _dataframe_to_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame to the list-of-row-entry dicts consumed by
        _make_dbrows_file_info.  Each entry has 'text' and 'row_index'."""
        plain_rows = self._dataframe_to_json(df)
        return [
            {'text': self._row_dict_to_text(r, i), 'row_index': i}
            for i, r in enumerate(plain_rows, start=1)
        ]

    @staticmethod
    def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw DB row dict (from a psycopg2 cursor) to JSON-safe
        values, matching _dataframe_to_json's per-value handling: datetimes →
        ISO strings, primitives kept, everything else → str()."""
        from datetime import date, time
        from decimal import Decimal

        out: Dict[str, Any] = {}
        for k, v in record.items():
            if v is None:
                out[k] = None
            elif isinstance(v, bool):
                out[k] = v
            elif isinstance(v, (int, float, str)):
                out[k] = v
            elif isinstance(v, Decimal):
                out[k] = float(v)
            elif isinstance(v, (datetime, date, time)):
                out[k] = v.isoformat()
            else:
                out[k] = str(v)
        return out

    def _records_to_rows(
        self,
        records: List[Dict[str, Any]],
        collection_name: str = '',
    ) -> List[Dict[str, Any]]:
        """Convert MongoDB document dicts to row-entry dicts.
        Uses MongoDB-specific formatting to handle nested structures properly."""
        return [
            {
                'text':            self._row_dict_to_text_mongodb(r, i),
                'row_index':       i,
                'collection_name': collection_name,
            }
            for i, r in enumerate(records, start=1)
        ]

    def _make_dbrows_file_info(
        self,
        rows: List[Dict[str, Any]],
        db_type: str,
        prefix: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the single file-info dict that the ingestion pipeline receives
        for a database source.  Uses the .dbrows sentinel so it is routed
        through extract_row_wise() → _extract_dbrows_row_wise() and never
        touches the chunking pipeline.

        Shape:
            {
                'filename':  'ragtest'              # human label for JetStream / UI
                'file_id':   'pg_20260401_111742'   # unique internal id
                'file_path': None
                'file_type': '.dbrows'              # sentinel
                'rows':      [{'text': '...', 'row_index': 1, ...}, ...]
                'db_type':   'postgresql'
                'row_count': 14
            }
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_filename = self.build_discovery_failed_filename(self.source_details)
        source_mapping_id = self.source_details.get("source_mapping_id")
        unique_suffix = self._sanitize_failed_filename_component(
            source_mapping_id,
            ts,
        )
        
        # Calculate total content size from all row texts
        total_size_bytes = sum(len(row.get('text', '').encode('utf-8')) for row in rows)
        
        fi: Dict[str, Any] = {
            'filename':  display_filename,
            'file_id':   f"{prefix}_{unique_suffix}_{ts}",
            'file_path': None,
            'file_type': '.dbrows',
            'rows':      rows,
            'db_type':   db_type,
            'row_count': len(rows),
            'content_size_bytes': total_size_bytes,
        }
        if extra:
            fi.update(extra)
        logger.info(
            "→ Prepared .dbrows file-info for '%s': %d row(s)",
            fi['filename'], len(rows),
        )
        return fi

    # -------------------------------------------------------------------------
    # Helpers: DataFrame conversion
    # -------------------------------------------------------------------------

    def _dataframe_to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame to a list of plain Python dicts."""
        data = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    record[col] = value.isoformat()
                elif isinstance(value, (int, float, str, bool)):
                    record[col] = value
                else:
                    record[col] = str(value)
            data.append(record)
        return data
