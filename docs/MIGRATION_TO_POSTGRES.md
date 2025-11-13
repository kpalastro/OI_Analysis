# Migration Guide: SQLite to PostgreSQL/TimescaleDB

This guide covers migrating from SQLite to PostgreSQL with TimescaleDB extension for time-series optimization.

---

## Prerequisites

1. **PostgreSQL Server** (version 12 or higher)
2. **TimescaleDB Extension** installed on PostgreSQL
3. **Python dependencies**: `psycopg[binary]>=3.1.10`
4. **Database credentials** configured in `.env`

---

## Step 1: PostgreSQL Setup

### 1.1 Create Database

Connect to PostgreSQL and create the database:

```bash
# Connect to PostgreSQL
psql -h 127.0.0.1 -U root -d postgres

# Create database
CREATE DATABASE oi_tracker;

# Exit psql
\q
```

### 1.2 Install TimescaleDB Extension

#### Option A: Using Official TimescaleDB Docker Image (Recommended)

If using Docker, use the official TimescaleDB image:

```bash
docker run -d \
  --name oi_tracker_postgres \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=oi_tracker \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg15
```

This image has TimescaleDB preloaded and configured.

#### Option B: Manual Setup in Existing PostgreSQL Container

If you're using a standard PostgreSQL container, you need to:

1. **Find postgresql.conf location:**
   ```bash
   docker exec CONTAINER_NAME psql -U root -d postgres -c "SHOW config_file;"
   ```

2. **Edit postgresql.conf:**
   ```bash
   # Enter the container
   docker exec -it CONTAINER_NAME sh
   
   # Edit postgresql.conf (typically at /var/lib/postgresql/data/postgresql.conf)
   # Add or modify this line:
   shared_preload_libraries = 'timescaledb'
   
   # Save and exit
   ```

3. **Restart the container:**
   ```bash
   docker restart CONTAINER_NAME
   ```

4. **Create TimescaleDB extension:**
   ```bash
   # Connect to the database
   docker exec CONTAINER_NAME psql -U root -d postgres -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
   docker exec CONTAINER_NAME psql -U root -d oi_tracker -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
   ```

**Note:** If you get an error `extension "timescaledb" must be preloaded`, it means `shared_preload_libraries` wasn't configured correctly or the container wasn't restarted.

### 1.3 Verify TimescaleDB Installation

```bash
psql -h 127.0.0.1 -U root -d oi_tracker -c "SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'timescaledb';"
```

You should see both `default_version` and `installed_version` populated.

---

## Step 2: Configure Environment Variables

Update your `.env` file with PostgreSQL credentials:

```env
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=5432
DB_DATABASE=oi_tracker
DB_USER=root
DB_PASSWORD=password
```

**Important:** If these variables are not set, the application will fall back to SQLite.

---

## Step 3: Install Python Dependencies

Install the PostgreSQL adapter:

```bash
pip install psycopg[binary]>=3.1.10
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

---

## Step 4: Initialize Database Schema

The application will automatically create the schema on first connection. However, you can verify by running:

```python
python -c "from database import initialize_database; initialize_database()"
```

Or start the application - it will initialize on startup.

### 4.1 Verify Tables Created

```bash
psql -h 127.0.0.1 -U root -d oi_tracker -c "\dt"
```

You should see:
- `option_chain_snapshots`
- `exchange_metadata`
- `alpha_predictions`

### 4.2 Verify TimescaleDB Hypertable

After the application starts and creates the schema, verify the hypertable:

```bash
psql -h 127.0.0.1 -U root -d oi_tracker -c "SELECT * FROM timescaledb_information.hypertables;"
```

The `option_chain_snapshots` table should be listed as a hypertable.

---

## Step 5: Migrate Data from SQLite (Optional)

If you have existing data in SQLite that you want to migrate:

### 5.1 Export from SQLite

```python
# Export script
import sqlite3
import pandas as pd

conn = sqlite3.connect('oi_tracker.db')

# Export option_chain_snapshots
df_snapshots = pd.read_sql_query("SELECT * FROM option_chain_snapshots", conn)
df_snapshots.to_csv('migration_snapshots.csv', index=False)

# Export exchange_metadata
df_metadata = pd.read_sql_query("SELECT * FROM exchange_metadata", conn)
df_metadata.to_csv('migration_metadata.csv', index=False)

# Export alpha_predictions
df_predictions = pd.read_sql_query("SELECT * FROM alpha_predictions", conn)
df_predictions.to_csv('migration_predictions.csv', index=False)

conn.close()
```

### 5.2 Import to PostgreSQL

```python
# Import script
import pandas as pd
from database import get_db_connection

# Connect to PostgreSQL
conn = get_db_connection()

# Import snapshots
df_snapshots = pd.read_csv('migration_snapshots.csv')
df_snapshots.to_sql('option_chain_snapshots', conn, if_exists='append', index=False)

# Import metadata
df_metadata = pd.read_csv('migration_metadata.csv')
df_metadata.to_sql('exchange_metadata', conn, if_exists='replace', index=False)

# Import predictions
df_predictions = pd.read_csv('migration_predictions.csv')
df_predictions.to_sql('alpha_predictions', conn, if_exists='append', index=False)

conn.close()
```

**Note:** For large datasets, consider using PostgreSQL's `COPY` command for faster imports.

---

## Step 6: Verify Migration

### 6.1 Test Database Connection

```python
from database import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT version();")
print(cursor.fetchone())
conn.close()
```

### 6.2 Test Application

Start the application and verify it connects to PostgreSQL:

```bash
python oi_tracker_web.py
```

Check the logs for:
```
✓ Database initialized: PostgreSQL
✓ TimescaleDB extension enabled
✓ Created hypertable: option_chain_snapshots
```

### 6.3 Verify Data Flow

1. Start the application during market hours
2. Verify data is being written to PostgreSQL:
   ```bash
   psql -h 127.0.0.1 -U root -d oi_tracker -c "SELECT COUNT(*) FROM option_chain_snapshots;"
   ```
3. Verify TimescaleDB features are working:
   ```bash
   psql -h 127.0.0.1 -U root -d oi_tracker -c "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'option_chain_snapshots';"
   ```

---

## Step 7: Performance Optimization (Optional)

### 7.1 TimescaleDB Compression

Enable compression for older data:

```sql
ALTER TABLE option_chain_snapshots SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'exchange, option_type'
);
```

Create a compression policy:

```sql
SELECT add_compression_policy('option_chain_snapshots', INTERVAL '7 days');
```

### 7.2 Retention Policies

Automatically delete data older than 90 days:

```sql
SELECT add_retention_policy('option_chain_snapshots', INTERVAL '90 days');
```

### 7.3 Continuous Aggregates

Create materialized views for common queries:

```sql
CREATE MATERIALIZED VIEW option_chain_daily_stats
WITH (timescaledb.continuous) AS
SELECT 
  time_bucket('1 day', timestamp) AS day,
  exchange,
  AVG(underlying_price) AS avg_underlying_price,
  SUM(oi) AS total_oi,
  COUNT(*) AS record_count
FROM option_chain_snapshots
GROUP BY day, exchange;
```

---

## Troubleshooting

### Error: "extension timescaledb must be preloaded"

**Solution:** Ensure `shared_preload_libraries = 'timescaledb'` is in `postgresql.conf` and the container/server has been restarted.

### Error: "could not connect to server"

**Solution:** 
- Verify PostgreSQL is running: `docker ps` or `systemctl status postgresql`
- Check connection credentials in `.env`
- Verify firewall/network settings

### Error: "relation does not exist"

**Solution:** 
- Run `initialize_database()` to create tables
- Check if you're connected to the correct database
- Verify schema creation in logs

### Error: "hypertable does not exist"

**Solution:**
- The hypertable is created automatically on first data insert
- Verify TimescaleDB extension is installed: `SELECT * FROM pg_extension WHERE extname = 'timescaledb';`
- Check application logs for hypertable creation messages

### Performance Issues

**Solution:**
- Enable TimescaleDB compression for historical data
- Add appropriate indexes (created automatically by the application)
- Consider partitioning strategies for very large datasets
- Monitor query performance with `EXPLAIN ANALYZE`

---

## Rollback to SQLite

If you need to rollback to SQLite:

1. Remove PostgreSQL environment variables from `.env` (or comment them out)
2. Restart the application
3. The application will automatically fall back to SQLite

**Note:** Data in PostgreSQL will not be automatically migrated back to SQLite.

---

## Additional Resources

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [psycopg Documentation](https://www.psycopg.org/docs/)

---

## Migration Checklist

- [ ] PostgreSQL server installed and running
- [ ] TimescaleDB extension installed and configured
- [ ] Database `oi_tracker` created
- [ ] Environment variables configured in `.env`
- [ ] Python dependencies installed (`psycopg[binary]`)
- [ ] Database schema initialized
- [ ] TimescaleDB hypertable created
- [ ] Application connects successfully
- [ ] Data migration completed (if applicable)
- [ ] Application tested and verified
- [ ] Performance optimizations applied (optional)

---

**Last Updated:** 2025-01-13

