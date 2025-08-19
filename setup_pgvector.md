# PostgreSQL with pgvector Setup

## Prerequisites

1. **Install PostgreSQL** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS with Homebrew
   brew install postgresql
   
   # Start PostgreSQL service
   sudo systemctl start postgresql  # Linux
   brew services start postgresql   # macOS
   ```

2. **Install pgvector extension**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql-14-pgvector  # Replace 14 with your PostgreSQL version
   
   # macOS with Homebrew
   brew install pgvector
   
   # Or compile from source
   git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

## Database Setup

1. **Create database and user**:
   ```bash
   sudo -u postgres psql
   ```
   
   In PostgreSQL shell:
   ```sql
   CREATE DATABASE string_rag_db;
   CREATE USER rag_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE string_rag_db TO rag_user;
   \q
   ```

2. **Enable pgvector extension** (run this in your application database):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

## Environment Variables

Set these environment variables or update the `db_config` in your code:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=string_rag_db
export DB_USER=rag_user
export DB_PASSWORD=your_password
```

## Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

The modified code will automatically:
- Create the required table structure
- Set up pgvector indexes for optimal performance
- Handle database connections and cleanup

Just run your Python script and it will work with PostgreSQL + pgvector instead of ChromaDB.