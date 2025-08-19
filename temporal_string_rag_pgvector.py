import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import hashlib
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import openai
from collections import defaultdict
import os


@dataclass
class StringRecord:
    """Data structure for storing string records with metadata"""
    string_content: str
    timestamp: datetime
    hour: int
    day_of_week: int
    month: int
    year: int
    day_of_year: int
    week_of_year: int
    string_id: str


class TemporalStringRAG:
    """RAG system for generating temporal string patterns using pgvector"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", 
                 db_config: Optional[Dict] = None):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Database configuration
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'string_rag_db'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres')
            }
        
        self.db_config = db_config
        
        # Initialize database connection and setup
        self._setup_database()
        
        # Pattern analysis storage
        self.pattern_stats = defaultdict(dict)
        self.temporal_patterns = defaultdict(list)

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _setup_database(self):
        """Setup PostgreSQL database with pgvector extension and required tables"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table for string patterns
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS string_patterns (
                        id VARCHAR(32) PRIMARY KEY,
                        string_content TEXT NOT NULL,
                        embedding vector(384),  -- Dimension for all-MiniLM-L6-v2
                        timestamp_str TIMESTAMP NOT NULL,
                        hour INTEGER,
                        day_of_week INTEGER,
                        month INTEGER,
                        year INTEGER,
                        day_of_year INTEGER,
                        week_of_year INTEGER,
                        quarter INTEGER,
                        is_weekend BOOLEAN,
                        hour_sin FLOAT,
                        hour_cos FLOAT,
                        day_sin FLOAT,
                        day_cos FLOAT,
                        string_length INTEGER,
                        numeric_count INTEGER,
                        alpha_count INTEGER,
                        space_count INTEGER,
                        first_segment TEXT,
                        last_segment TEXT,
                        segments_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for better performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_string_patterns_embedding 
                    ON string_patterns USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_string_patterns_temporal 
                    ON string_patterns (hour, day_of_week, month);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_string_patterns_timestamp 
                    ON string_patterns (timestamp_str);
                """)
                
                conn.commit()

    def _extract_string_features(self, string_content: str) -> Dict:
        """Extract structural features from string for better matching"""
        segments = string_content.split()
        features = {
            'length': len(string_content),
            'numeric_count': sum(c.isdigit() for c in string_content),
            'alpha_count': sum(c.isalpha() for c in string_content),
            'space_count': string_content.count(' '),
            'first_segment': segments[0] if segments else '',
            'last_segment': segments[-1] if segments else '',
            'segments_count': len(segments)
        }
        return features

    def _create_temporal_features(self, timestamp: datetime) -> Dict:
        """Create comprehensive temporal features"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'year': timestamp.year,
            'day_of_year': timestamp.timetuple().tm_yday,
            'week_of_year': timestamp.isocalendar()[1],
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': timestamp.weekday() >= 5,
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365),
            'day_cos': np.cos(2 * np.pi * timestamp.timetuple().tm_yday / 365)
        }

    def add_string_record(self, string_content: str, timestamp: datetime):
        """Add a new string record to the RAG system"""
        
        # Create record
        temporal_features = self._create_temporal_features(timestamp)
        string_features = self._extract_string_features(string_content)
        
        record = StringRecord(
            string_content=string_content,
            timestamp=timestamp,
            hour=temporal_features['hour'],
            day_of_week=temporal_features['day_of_week'],
            month=temporal_features['month'],
            year=temporal_features['year'],
            day_of_year=temporal_features['day_of_year'],
            week_of_year=temporal_features['week_of_year'],
            string_id=hashlib.md5(f"{string_content}{timestamp}".encode()).hexdigest()
        )
        
        # Create embedding
        embedding_text = f"{string_content} hour:{temporal_features['hour']} day:{temporal_features['day_of_week']} month:{temporal_features['month']}"
        embedding = self.embedding_model.encode(embedding_text)
        
        # Store in PostgreSQL with pgvector
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO string_patterns (
                        id, string_content, embedding, timestamp_str,
                        hour, day_of_week, month, year, day_of_year, week_of_year,
                        quarter, is_weekend, hour_sin, hour_cos, day_sin, day_cos,
                        string_length, numeric_count, alpha_count, space_count,
                        first_segment, last_segment, segments_count
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (id) DO NOTHING;
                """, (
                    record.string_id,
                    string_content,
                    embedding.tolist(),  # pgvector accepts Python lists
                    timestamp,
                    temporal_features['hour'],
                    temporal_features['day_of_week'],
                    temporal_features['month'],
                    temporal_features['year'],
                    temporal_features['day_of_year'],
                    temporal_features['week_of_year'],
                    temporal_features['quarter'],
                    temporal_features['is_weekend'],
                    temporal_features['hour_sin'],
                    temporal_features['hour_cos'],
                    temporal_features['day_sin'],
                    temporal_features['day_cos'],
                    string_features['length'],
                    string_features['numeric_count'],
                    string_features['alpha_count'],
                    string_features['space_count'],
                    string_features['first_segment'],
                    string_features['last_segment'],
                    string_features['segments_count']
                ))
                conn.commit()
        
        # Update pattern statistics
        self._update_pattern_stats(record, string_features)

    def _update_pattern_stats(self, record: StringRecord, string_features: Dict):
        """Update pattern statistics for better generation"""
        key = f"h{record.hour}_dow{record.day_of_week}_m{record.month}"
        
        if key not in self.pattern_stats:
            self.pattern_stats[key] = {
                'count': 0,
                'avg_length': 0,
                'common_segments': defaultdict(int),
                'segment_positions': defaultdict(list)
            }
        
        stats = self.pattern_stats[key]
        stats['count'] += 1
        stats['avg_length'] = (stats['avg_length'] * (stats['count'] - 1) + string_features['length']) / stats['count']
        
        # Track segment patterns
        segments = record.string_content.split()
        for i, segment in enumerate(segments):
            stats['common_segments'][segment] += 1
            stats['segment_positions'][i].append(segment)

    def retrieve_similar_strings(self, target_datetime: datetime, n_results: int = 10) -> List[Dict]:
        """Retrieve strings most similar to target datetime using pgvector"""
        
        temporal_features = self._create_temporal_features(target_datetime)
        
        # Create query embedding
        query_text = f"hour:{temporal_features['hour']} day:{temporal_features['day_of_week']} month:{temporal_features['month']}"
        query_embedding = self.embedding_model.encode(query_text)
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query using pgvector cosine similarity with temporal filters
                cur.execute("""
                    SELECT 
                        id,
                        string_content,
                        timestamp_str,
                        hour, day_of_week, month, year,
                        day_of_year, week_of_year,
                        string_length, numeric_count, alpha_count,
                        space_count, first_segment, last_segment, segments_count,
                        1 - (embedding <=> %s::vector) AS similarity_score
                    FROM string_patterns
                    WHERE 
                        hour BETWEEN %s AND %s
                        AND month = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (
                    query_embedding.tolist(),
                    temporal_features['hour'] - 1,
                    temporal_features['hour'] + 1,
                    temporal_features['month'],
                    query_embedding.tolist(),
                    min(n_results, 50)
                ))
                
                results = cur.fetchall()
        
        # Post-process results for temporal relevance
        processed_results = []
        for row in results:
            # Calculate temporal similarity bonus
            hour_diff = abs(row['hour'] - temporal_features['hour'])
            day_diff = abs(row['day_of_week'] - temporal_features['day_of_week'])
            temporal_bonus = 1.0 / (1 + hour_diff + day_diff * 0.5)
            
            processed_results.append({
                'string': row['string_content'],
                'metadata': {
                    'timestamp_str': row['timestamp_str'].isoformat(),
                    'hour': row['hour'],
                    'day_of_week': row['day_of_week'],
                    'month': row['month'],
                    'year': row['year'],
                    'day_of_year': row['day_of_year'],
                    'week_of_year': row['week_of_year'],
                    'string_length': row['string_length'],
                    'numeric_count': row['numeric_count'],
                    'alpha_count': row['alpha_count'],
                    'space_count': row['space_count'],
                    'first_segment': row['first_segment'],
                    'last_segment': row['last_segment'],
                    'segments_count': row['segments_count']
                },
                'similarity_score': row['similarity_score'] * temporal_bonus,
                'temporal_distance': hour_diff + day_diff
            })
        
        # Sort by combined similarity score
        processed_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return processed_results[:n_results]

    def generate_string(self, target_datetime: datetime, model: str = "gpt-3.5-turbo") -> str:
        """Generate a new string for the target datetime using RAG"""
        
        # Retrieve similar examples
        similar_strings = self.retrieve_similar_strings(target_datetime, n_results=5)
        
        # Get pattern statistics for this temporal context
        temporal_features = self._create_temporal_features(target_datetime)
        pattern_key = f"h{temporal_features['hour']}_dow{temporal_features['day_of_week']}_m{temporal_features['month']}"
        pattern_info = self.pattern_stats.get(pattern_key, {})
        
        # Construct prompt with examples and context
        examples_text = "\n".join([
            f"DateTime: {result['metadata']['timestamp_str'][:19]} -> {result['string']}"
            for result in similar_strings[:3]
        ])
        
        prompt = f"""You are a string pattern generator. Generate a string following the exact same format and pattern as the examples below.

Target DateTime: {target_datetime.strftime('%Y-%m-%d %H:%M:%S')}
Target Hour: {temporal_features['hour']}
Target Day of Week: {temporal_features['day_of_week']} (0=Monday, 6=Sunday)
Target Month: {temporal_features['month']}

Similar Examples:
{examples_text}

Pattern Analysis:
Average Length: {pattern_info.get('avg_length', 'Unknown')}
Sample Count: {pattern_info.get('count', 0)}

Generate a new string that follows the EXACT same structural pattern as the examples above, but with appropriate variations for the target datetime. Maintain the same:
- Number of segments (space-separated parts)
- Character patterns (letters vs numbers)
- Length of each segment
- Overall format structure

Only return the generated string, nothing else."""

        # Generate using LLM
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            generated_string = response.choices[0].message.content.strip()
            return generated_string
        except Exception as e:
            # Fallback: use pattern-based generation
            return self._generate_fallback_string(similar_strings, temporal_features)

    def _generate_fallback_string(self, similar_strings: List[Dict], temporal_features: Dict) -> str:
        """Fallback generation method using pattern analysis"""
        if not similar_strings:
            return "Generated pattern not available"
        
        # Use most similar string as base
        base_string = similar_strings[0]['string']
        segments = base_string.split()
        
        # Apply simple transformations based on temporal features
        new_segments = []
        for i, segment in enumerate(segments):
            if segment.isdigit():
                # Modify numeric segments based on time
                base_num = int(segment)
                variation = (temporal_features['hour'] + temporal_features['day_of_week']) % 100
                new_num = (base_num + variation) % (10 ** len(segment))
                new_segments.append(str(new_num).zfill(len(segment)))
            else:
                new_segments.append(segment)
        
        return " ".join(new_segments)

    def batch_add_records(self, records: List[Tuple[str, datetime]], batch_size: int = 1000):
        """Efficiently add multiple records in batches"""
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            
            # Prepare batch data
            batch_data = []
            for string_content, timestamp in batch:
                temporal_features = self._create_temporal_features(timestamp)
                string_features = self._extract_string_features(string_content)
                
                string_id = hashlib.md5(f"{string_content}{timestamp}".encode()).hexdigest()
                embedding_text = f"{string_content} hour:{temporal_features['hour']} day:{temporal_features['day_of_week']} month:{temporal_features['month']}"
                embedding = self.embedding_model.encode(embedding_text)
                
                batch_data.append((
                    string_id,
                    string_content,
                    embedding.tolist(),
                    timestamp,
                    temporal_features['hour'],
                    temporal_features['day_of_week'],
                    temporal_features['month'],
                    temporal_features['year'],
                    temporal_features['day_of_year'],
                    temporal_features['week_of_year'],
                    temporal_features['quarter'],
                    temporal_features['is_weekend'],
                    temporal_features['hour_sin'],
                    temporal_features['hour_cos'],
                    temporal_features['day_sin'],
                    temporal_features['day_cos'],
                    string_features['length'],
                    string_features['numeric_count'],
                    string_features['alpha_count'],
                    string_features['space_count'],
                    string_features['first_segment'],
                    string_features['last_segment'],
                    string_features['segments_count']
                ))
            
            # Batch insert
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO string_patterns (
                            id, string_content, embedding, timestamp_str,
                            hour, day_of_week, month, year, day_of_year, week_of_year,
                            quarter, is_weekend, hour_sin, hour_cos, day_sin, day_cos,
                            string_length, numeric_count, alpha_count, space_count,
                            first_segment, last_segment, segments_count
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (id) DO NOTHING;
                    """, batch_data)
                    conn.commit()

    def get_system_stats(self) -> Dict:
        """Get statistics about the RAG system"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get total count
                cur.execute("SELECT COUNT(*) as total_records FROM string_patterns;")
                total_records = cur.fetchone()['total_records']
                
                # Get unique hours
                cur.execute("SELECT COUNT(DISTINCT hour) as unique_hours FROM string_patterns;")
                unique_hours = cur.fetchone()['unique_hours']
                
                # Get date range
                cur.execute("""
                    SELECT 
                        MIN(timestamp_str) as start_date,
                        MAX(timestamp_str) as end_date
                    FROM string_patterns;
                """)
                date_range = cur.fetchone()
                
                span_days = 0
                if date_range['start_date'] and date_range['end_date']:
                    span_days = (date_range['end_date'] - date_range['start_date']).days
        
        return {
            'total_records': total_records,
            'pattern_types': len(self.pattern_stats),
            'unique_hours': unique_hours,
            'date_range': {
                'start': date_range['start_date'].isoformat() if date_range['start_date'] else None,
                'end': date_range['end_date'].isoformat() if date_range['end_date'] else None,
                'span_days': span_days
            }
        }

    def clear_database(self):
        """Clear all records from the database"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM string_patterns;")
                conn.commit()
        self.pattern_stats.clear()
        self.temporal_patterns.clear()

    def close_connection(self):
        """Close database connections (cleanup method)"""
        # Connection pooling would be handled here in production
        pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    # You can provide custom database configuration
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'string_rag_db',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    rag_system = TemporalStringRAG(db_config=db_config)
    
    # Sample data (replace with your actual data loading)
    sample_data = [
        ("0000001 ABCD AAAA111111BBBBBB11 X0001 0 23123 AAAA111111EPCU10X0001QMLYU226478904BD0000001", datetime(2024, 3, 15, 14, 30)),
        ("0000002 EFGH CCCC222222DDDDDD22 Y0002 1 34234 CCCC222222FGHI20Y0002NOPQR334567890EF0000002", datetime(2024, 3, 15, 15, 30)),
        ("0000003 IJKL EEEE333333FFFFFF33 Z0003 2 45345 EEEE333333KLMN30Z0003STUVW445678901GH0000003", datetime(2024, 3, 16, 14, 30)),
    ]
    
    # Add sample records
    for string_content, timestamp in sample_data:
        rag_system.add_string_record(string_content, timestamp)
    
    # Generate new string
    target_date = datetime(2024, 3, 17, 14, 30)
    generated = rag_system.generate_string(target_date)
    print(f"Generated string for {target_date}: {generated}")
    
    # Get system statistics
    stats = rag_system.get_system_stats()
    print(f"System stats: {stats}")
    
    # Clean up
    rag_system.close_connection()