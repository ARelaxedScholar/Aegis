# Aegis (Formerly Simulation GAN-TE Carlo)

## Overview
Aegis is a Rust-based project that automates the deployment of trading algorithms for cryptocurrencies. 

## Requirements
- Rust (latest stable version)
- Cargo (Rust package manager)
- Python (Ensure you got a python version installed as pyo3 will complain about that if not)
- Access to crypto exchange APIs (API keys required)

## How to use
(Currently not in a very user-friendly state)
1. Clone the repo
2. Look at the implementation for the evolution functions.
3. Write a script to run in bin
4. Use cargo run --bin your_script to test it.

This is at the proof of concept work. I am currently building the library framework, so binaries must be defined by the user.
The main.rs file provides a basic server, for when I deploy the functions to a server, but this is not currently live.

## Project Evaluation: Merits and Missing Pieces for Large-Scale Deployment

### Merits as a Financial Tool and Marketable Asset
Currently, Aegis acts as a **robust, parallelized portfolio optimization engine** written in Rust. It utilizes evolutionary algorithms, including advanced memetic algorithms (e.g., Lamarckian proximal descent, NSGA-II style Pareto front generation), to optimize portfolio weights across multiple objectives such as Annualized Returns, Volatility, and Sharpe Ratio.
- **Performance**: Being written in Rust and leveraging multi-threading (`rayon`), the optimization process is highly performant and scalable to large asset pools and population sizes.
- **Algorithmic Sophistication**: The implementation of proximal gradient descent combined with evolutionary strategies (memetic algorithms) provides a strong mathematical foundation for finding optimal asset allocations in non-convex spaces.
- **Flexibility**: The framework is extensible, allowing for different simulation runners (e.g., Local, Kubernetes jobs, gRPC) and custom sampling techniques.

### Missing Components for Large-Scale Deployment
To deploy and manage factor strategies using this optimizer as a meta-optimizer at scale, several critical pieces are currently missing:
1. **Live Data Ingestion & Pipeline**: There is no built-in mechanism to fetch, clean, and stream real-time or historical market data from exchanges or data providers (e.g., Binance, Alpaca, Bloomberg).
2. **Grammatical Evolution Strategy Block**: As mentioned in the project's vision, the "Strategy Block" using grammatical evolution to generate actual trading signals or rules is not yet fully implemented. Currently, it only optimizes the *weights* of a given portfolio based on sampled returns.
3. **Execution Engine & Broker Integration**: To act as a trading tool, it requires integration with broker APIs to execute trades, manage order types, and handle real-world execution issues (slippage, latency, partial fills).
4. **Comprehensive Backtesting Framework**: While it simulates returns, a true backtester must account for trading fees (commissions), bid-ask spread, slippage, and market impact, which are absent in the current performance calculations.
5. **Risk Management & Constraints**: Real-world factor investing requires strict constraints (e.g., sector neutrality, turnover limits, leverage caps, max drawdown stops) which are not yet fully supported beyond basic non-negative weight projections.
6. **Persistence and Monitoring**: A production deployment requires persistent storage of historical optimizations, live monitoring dashboards (e.g., Grafana/Prometheus), and alerting mechanisms.

In summary, while the core portfolio *weight optimization* engine is mathematically sound and performant, you could not currently deploy factor strategies out-of-the-box. You would need to build the data ingestion, strategy generation (signal extraction), execution, and rigorous backtesting/fee-modeling layers around this core engine.

## Current Work
- Slides presenting preliminary results: https://docs.google.com/presentation/d/1tD4gExPZPbOnIx00lG5FNrKtftqEut10JBwyxNEWIwc/
- Informal Paper going over the current methodology (2024) : https://www.overleaf.com/read/rrfsdyrpypzq#f6c2d8
- Informal Document of Project (My Personal Documentation): https://a-relaxed-universe.vercel.app/1-cosmos/aegis/ 


