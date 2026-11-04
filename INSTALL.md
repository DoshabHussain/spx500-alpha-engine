# SPX500 Forecasting - Installation Guide

## Quick Start

```bash
# Extract
tar -xzf spx500_complete.tar.gz
cd spx500

# Install
pip install -e .

# Configure
cp .env.example .env
nano .env  # Add FRED_API_KEY

# Test
python -c "from src.core import get_config; print('✅ Setup complete!')"

# Run ingestion
./scripts/run_ingest.sh
```

## Directory Structure

```
data/
├── raw/
│   ├── underlying/SPY/20250206/bars.parquet    ← Simple paths!
│   ├── indices/VIX/20250206/data.parquet
│   └── macro/DGS10/20250206/data.parquet
├── aligned/
├── curated/
└── reference/
```

## Features

✅ Phase 1: Core utilities (config, time, calendar, schemas, logging)
✅ Phase 2: Data pipeline (storage, ingestion, manifests)
✅ FIXED: Simple directory names (no dt=, instrument= prefixes)

See README.md for full documentation.
