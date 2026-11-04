#!/usr/bin/env bash
#
# Daily data ingestion script
#
# Usage:
#   ./scripts/run_ingest.sh [--date YYYY-MM-DD]
#
# Examples:
#   ./scripts/run_ingest.sh                    # Ingest latest data
#   ./scripts/run_ingest.sh --date 2025-02-06  # Ingest specific date
#

set -euo pipefail

# ─── SETUP ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ─── PARSE ARGUMENTS ────────────────────────────────────────────────────────

DATE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            DATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--date YYYY-MM-DD]"
            echo ""
            echo "Run daily data ingestion pipeline"
            echo ""
            echo "Options:"
            echo "  --date YYYY-MM-DD    Ingest data for specific date (default: latest)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ─── RUN INGESTION ──────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"

log_info "Starting data ingestion..."
log_info "Project root: $PROJECT_ROOT"

# Ensure virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_warn "Virtual environment not detected"
    log_warn "Activate with: source .venv/bin/activate"
fi

# Check if src module is importable
if ! python -c "import src" 2>/dev/null; then
    log_error "Cannot import 'src' module"
    log_error "Install with: pip install -e ."
    exit 1
fi

# Build ingestion command
INGEST_CMD="python -c \"
from src.ingestion.batch import ingest_yfinance_latest, ingest_fred_latest

print('=== Ingesting yfinance data ===')
yf_data = ingest_yfinance_latest()
for symbol, df in yf_data.items():
    print(f'{symbol}: {len(df)} rows')

print('\\n=== Ingesting FRED data ===')
fred_data = ingest_fred_latest()
for series, df in fred_data.items():
    print(f'{series}: {len(df)} rows')

print('\\n✅ Ingestion complete!')
\""

# Run ingestion
log_info "Running ingestion pipeline..."
eval "$INGEST_CMD"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    log_info "✅ Data ingestion completed successfully"
else
    log_error "❌ Data ingestion failed with exit code $exit_code"
    exit $exit_code
fi
