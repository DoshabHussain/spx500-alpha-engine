#!/bin/bash

# Create __init__.py files for Phase 4

# src/modeling/__init__.py
cat > /home/claude/spx500_phase4/src/modeling/__init__.py << 'PYEOF'
"""Modeling layer - Training, validation, backtesting."""
__all__ = []
PYEOF

# src/modeling/training/__init__.py
cat > /home/claude/spx500_phase4/src/modeling/training/__init__.py << 'PYEOF'
"""Model training modules."""
from src.modeling.training.lgbm_trainer import (
    LGBMTrainer,
    train_classification_model,
    train_regression_model,
)

__all__ = [
    "LGBMTrainer",
    "train_classification_model",
    "train_regression_model",
]
PYEOF

# src/modeling/validation/__init__.py
cat > /home/claude/spx500_phase4/src/modeling/validation/__init__.py << 'PYEOF'
"""Model validation modules."""
from src.modeling.validation.time_series_split import (
    TimeSeriesSplit,
    PurgedKFold,
    Split,
    simple_train_test_split,
    create_walk_forward_splits,
)
from src.modeling.validation.metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from src.modeling.validation.walk_forward import (
    WalkForwardValidator,
    run_walk_forward_validation,
)

__all__ = [
    "TimeSeriesSplit",
    "PurgedKFold",
    "Split",
    "simple_train_test_split",
    "create_walk_forward_splits",
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "WalkForwardValidator",
    "run_walk_forward_validation",
]
PYEOF

# src/modeling/backtest/__init__.py
cat > /home/claude/spx500_phase4/src/modeling/backtest/__init__.py << 'PYEOF'
"""Backtesting modules (Phase 4 extension)."""
__all__ = []
PYEOF

echo "✅ All __init__.py files created"
