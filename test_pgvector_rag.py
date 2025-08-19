#!/usr/bin/env python3
"""
Test script for the pgvector-based Temporal String RAG system
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from temporal_string_rag_pgvector import TemporalStringRAG

# Load environment variables from .env file
load_dotenv()

def test_basic_functionality():
    """Test basic functionality of the pgvector RAG system"""
    
    print("Testing pgvector-based Temporal String RAG...")
    
    # Test database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'string_rag_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    try:
        # Initialize RAG system
        print("1. Initializing RAG system...")
        rag_system = TemporalStringRAG(db_config=db_config)
        print("✓ RAG system initialized successfully")
        
        # Clear any existing data for clean test
        print("2. Clearing existing data...")
        rag_system.clear_database()
        print("✓ Database cleared")
        
        # Test adding records
        print("3. Adding sample records...")
        sample_data = [
            ("0000001 ABCD AAAA111111BBBBBB11 X0001 0 23123 AAAA111111EPCU10X0001QMLYU226478904BD0000001", datetime(2024, 3, 15, 14, 30)),
            ("0000002 EFGH CCCC222222DDDDDD22 Y0002 1 34234 CCCC222222FGHI20Y0002NOPQR334567890EF0000002", datetime(2024, 3, 15, 15, 30)),
            ("0000003 IJKL EEEE333333FFFFFF33 Z0003 2 45345 EEEE333333KLMN30Z0003STUVW445678901GH0000003", datetime(2024, 3, 16, 14, 30)),
            ("0000004 MNOP GGGG444444HHHHHH44 W0004 3 56456 GGGG444444OPQR40W0004VWXYZ556789012IJ0000004", datetime(2024, 3, 16, 15, 30)),
            ("0000005 QRST IIII555555JJJJJJ55 V0005 4 67567 IIII555555STUV50V0005ABCDE667890123KL0000005", datetime(2024, 3, 17, 14, 30)),
        ]
        
        for string_content, timestamp in sample_data:
            rag_system.add_string_record(string_content, timestamp)
        print(f"✓ Added {len(sample_data)} records")
        
        # Test retrieval
        print("4. Testing similarity retrieval...")
        target_date = datetime(2024, 3, 17, 14, 30)
        similar_strings = rag_system.retrieve_similar_strings(target_date, n_results=3)
        print(f"✓ Retrieved {len(similar_strings)} similar strings")
        
        for i, result in enumerate(similar_strings):
            print(f"   {i+1}. Similarity: {result['similarity_score']:.3f} - {result['string'][:50]}...")
        
        # Test generation (without OpenAI API)
        print("5. Testing fallback string generation...")
        generated = rag_system._generate_fallback_string(similar_strings, rag_system._create_temporal_features(target_date))
        print(f"✓ Generated string: {generated}")
        
        # Test statistics
        print("6. Testing system statistics...")
        stats = rag_system.get_system_stats()
        print(f"✓ System stats: {stats}")
        
        # Test batch operations
        print("7. Testing batch operations...")
        batch_data = [
            (f"BATCH{i:03d} TEST PATTERN{i:06d} SEGMENT{i} {i*100} END{i:03d}", 
             datetime(2024, 3, 18, 10 + i % 12, 0))
            for i in range(10)
        ]
        rag_system.batch_add_records(batch_data, batch_size=5)
        print("✓ Batch operations completed")
        
        # Final stats
        final_stats = rag_system.get_system_stats()
        print(f"✓ Final system stats: {final_stats}")
        
        print("\n🎉 All tests passed! pgvector RAG system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            rag_system.close_connection()
        except:
            pass
    
    return True

def test_connection_only():
    """Test just the database connection"""
    print("Testing database connection...")
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'string_rag_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    try:
        import psycopg2
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"✓ Connected to PostgreSQL: {version[0]}")
            
            # Test pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            if result:
                print("✓ pgvector extension is available")
            else:
                print("❌ pgvector extension not found")
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--connection-only":
        success = test_connection_only()
    else:
        success = test_basic_functionality()
    
    sys.exit(0 if success else 1)