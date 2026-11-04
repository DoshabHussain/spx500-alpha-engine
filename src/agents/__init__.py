"""
Multi-Agent Neural & Microstructure Engine for SPX Intraday Forecasting.
Incorporates autonomous agents adapted from day-trader-skill:
- NewsSentimentAgent: Real-time SEC & headline polarity
- ScannerAgent: Intraday setups, opening range breakouts (ORB), volume pressure
- TimeMachineAgent: 10,000-path Monte Carlo trade scenario replay
- RiskAgent: Sizing, stop-loss, max intraday drawdown protection
- MasterTriEngineAgent: Consensus aggregator combining Quant GEX, Macro, and NLP
"""
from src.agents.news_agent import NewsSentimentAgent
from src.agents.scanner_agent import ScannerAgent
from src.agents.timemachine_agent import TimeMachineAgent
from src.agents.risk_agent import RiskAgent
from src.agents.hybrid_agent import MasterTriEngineAgent

__all__ = [
    "NewsSentimentAgent",
    "ScannerAgent",
    "TimeMachineAgent",
    "RiskAgent",
    "MasterTriEngineAgent",
]

