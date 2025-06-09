---
layout: page
title: DeepTrade AI - Multi-Model Stock Prediction with NLP & Automated Trading
description: An enterprise-grade system integrating LSTM-XGBoost prediction with sentiment analysis, distributed training infrastructure, and automated workflows
img: assets/img/project-2/stock_hero.jpeg
importance: 5
category: work
related_publications: false
---

### 1. Overview

DeepTrade AI is an enterprise-grade automated stock trading system that combines machine learning price prediction with NLP-based sentiment analysis and distributed computing infrastructure. The system features a bidirectional LSTM with attention mechanism and XGBoost ensemble for multi-timeframe price forecasting (5m, 15m, 30m, 1h), and integrates FinBERT for real-time sentiment analysis of financial news, Reddit posts, and SEC filings. The architecture employs distributed training across 4x V100 GPUs, real-time Kafka streaming processing 9K+ financial events daily, automated Airflow workflows, and CI/CD pipeline automation. The system achieves 55-65% directional accuracy and a 58.5% win rate in paper trading with comprehensive risk management controls.

<div style="text-align: center;">
    <img src="/assets/img/project-2/deeptrade_system.png" alt="System Architecture" style="width: 100%; max-width: 3000px;">
    <p><em>DeepTrade AI system architecture showing data flow between components</em></p>
</div>

---

### 2. Infrastructure & Data Processing

DeepTrade AI leverages enterprise-scale infrastructure components to handle high-frequency financial data processing and model training:

#### 2.1 Distributed Training Infrastructure

The system implements distributed training across multiple GPUs to efficiently train 100+ model configurations:

- **Multi-GPU Training**: Data parallelism across 4x NVIDIA V100 GPUs with PyTorch DistributedDataParallel
- **Scale**: 100 model configurations (25 stocks × 4 timeframes) trained simultaneously
- **Performance**: 75% reduction in training time compared to single-GPU setup
- **Architecture**: Master-worker setup with automatic load balancing and gradient synchronization

#### 2.2 Real-Time Streaming Pipeline

A high-throughput Kafka streaming infrastructure processes financial data from multiple sources:

- **Technology**: Apache Kafka with optimized partitioning across 4 topics
- **Capacity**: 9,000+ financial events processed daily with sub-second latency
- **Data Sources**: NewsAPI, Reddit, SEC EDGAR filings, and market data APIs
- **Processing**: Real-time event enrichment and validation with Avro serialization

#### 2.3 Automated Workflow Management

Apache Airflow orchestrates the entire sentiment analysis and model training pipeline:

- **Workflow Automation**: Directed Acyclic Graphs (DAGs) for complex task dependencies
- **FinBERT Integration**: Automated sentiment analysis processing every 4 hours
- **Feature Generation**: 12 temporal sentiment indicators with momentum tracking
- **Performance**: ~5% improvement in trading accuracy through automated workflows

#### 2.4 CI/CD Pipeline Automation

GitHub Actions provides continuous integration and deployment for model lifecycle management:

- **Automated Triggers**: Performance degradation detection, data changes, and scheduled retraining
- **Pipeline Stages**: Setup → validation → training → testing → deployment
- **Model Management**: Automated versioning, rollback capabilities, and performance monitoring
- **Deployment**: <15 minutes for complete model retraining and deployment

---

### 3. Sentiment Analysis Pipeline

DeepTrade AI incorporates a sophisticated sentiment analysis pipeline that aggregates and analyzes data from three key sources:

#### 3.1 Multi-Source Integration

- **Financial News Processing (40%)**: Real-time streaming with NewsAPI integration, automated headline analysis, and relevancy-based filtering
  
- **Reddit Sentiment Analysis (30%)**: Multi-subreddit monitoring (r/wallstreetbets, r/stocks, r/investing) with advanced engagement metrics and post-comment sentiment weighting

- **SEC Filing Analysis (30%)**: Real-time CIK tracking, form-specific sentiment weighting, and automated filing pattern analysis

#### 3.2 FinBERT Model Architecture

The core of the sentiment analysis is a fine-tuned FinBERT model specifically pre-trained on financial text:

- **Tokenization**: Converts text data into numerical representations with tokens truncated at 512 length
- **Three-Class Classification**: Classifies sentiment as positive, negative, or neutral
- **Custom Sentiment Score**: Combines probabilities and source-specific weights for a composite sentiment score

<div style="text-align: center;">
    <img src="/assets/img/project-2/sentiment_op.png" alt="Sentiment Analysis" style="width: 100%; max-width: 800px;">
    <p><em>Sentiment analysis across different sources and stocks: Financial News (blue), Reddit (green), and SEC Filings (red)</em></p>
</div>

---

### 4. Multi-Model Stock Prediction System

The prediction system employs an ensemble approach combining two powerful models:

<div style="text-align: center;">
    <img src="/assets/img/project-2/lstm_xgb.png" alt="Model Architecture" style="width: 100%; max-width: 1000px;">
    <p><em>LSTM and XGBoost model architectures with dynamic ensemble weighting</em></p>
</div>

#### 4.1 LSTM Model with Attention Mechanism

The LSTM component features several advanced architectural elements:

- **Bidirectional Processing**: Analyzes time series data in both forward and backward directions
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence with varying levels of importance
- **Residual Connections and Batch Normalization**: Improves training stability and convergence
- **Dynamic Dropout**: Adaptively adjusts regularization during training

#### 4.2 XGBoost Model for Feature-Based Predictions

The XGBoost component complements the LSTM by providing a feature-based perspective:

- **Gradient Boosting Framework**: Iteratively builds an ensemble of 300 decision trees
- **Advanced Feature Engineering**: Processes 39 features including technical indicators, trend signals, and sentiment data
- **Feature Importance Analysis**: Provides insights into which factors most influence predictions

#### 4.3 Dynamic Ensemble Integration

The predictions from both models are combined using a dynamic weighting scheme:

- Weights are adjusted based on recent performance on validation data
- Adaptive weighting responds to changing market conditions
- Better-performing model receives higher influence in final prediction

<div style="text-align: center; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
    <img src="/assets/img/project-2/AAPL_prediction.png" alt="AAPL Prediction" style="width: 48%; max-width: 400px;">
    <img src="/assets/img/project-2/NVDA_prediction.png" alt="NVDA Prediction" style="width: 48%; max-width: 400px;">
</div>
<div style="text-align: center; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
    <img src="/assets/img/project-2/AMD_prediction.png" alt="GME Prediction" style="width: 48%; max-width: 400px;">
    <img src="/assets/img/project-2/MSFT_prediction.png" alt="MSFT Prediction" style="width: 48%; max-width: 400px;">
</div>
<div style="text-align: center;">
    <p><em>Multi-timeframe predictions across different stocks showing price action and confidence intervals</em></p>
</div>

---

### 5. Automated Trading System

The trading system translates predictions and sentiment analysis into concrete trading actions through the Tradier API sandbox environment.

#### 5.1 Trading Logic Pipeline

```
Prediction → Sentiment → Signal Generation → Risk Analysis → Position Sizing → Execution
     ↓           ↓              ↓                ↓               ↓              ↓
Price Data   News/Social   Trend Analysis    Risk Limits    Dynamic Sizing   Market Orders
     ↓           ↓              ↓                ↓               ↓              ↓
Confidence  Sentiment     Entry/Exit        Stop Loss      Position Value    Executions
  Scores     Scores        Signals         Take Profit      Calculation     & Monitoring
```

#### 5.2 Risk Management Framework

- **Maximum Concurrent Positions**: Limits to 2 open trades
- **Position Size**: 2% of capital per trade
- **Maximum Daily Risk**: 2% of account value
- **Stop-Loss**: Automatic exit at 1.5% adverse move
- **Take-Profit**: Automatic exit at 3% profit target

<div style="text-align: center;">
    <img src="/assets/img/project-2/tradier_op.png" alt="Tradier Trading Interface" style="width: 50%; max-width: 560px;">
    <p><em>Screenshot of the Tradier paper trading interface showing successful execution of trades</em></p>
</div>

---

### 6. Performance Metrics

<div class="table-responsive">
  <table class="table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Directional Accuracy</td>
        <td>55-65%</td>
        <td>Across all timeframes</td>
      </tr>
      <tr>
        <td>Mean Absolute Error</td>
        <td>0.3-0.4%</td>
        <td>On normalized returns across multiple stocks</td>
      </tr>
      <tr>
        <td>Training Speed Improvement</td>
        <td>75%</td>
        <td>Reduction with distributed training across 4x V100 GPUs</td>
      </tr>
      <tr>
        <td>Streaming Throughput</td>
        <td>9K+ events/day</td>
        <td>Real-time financial data processing capacity</td>
      </tr>
      <tr>
        <td>Confidence Scoring</td>
        <td>87-93%</td>
        <td>Accuracy of confidence intervals</td>
      </tr>
      <tr>
        <td>Win Rate (Paper Trading)</td>
        <td>58.5%</td>
        <td>Across 9 major tech and blue-chip stocks</td>
      </tr>
      <tr>
        <td>Risk-Reward Ratio</td>
        <td>1:2</td>
        <td>Average risk to reward across all trades</td>
      </tr>
    </tbody>
  </table>
</div>

---

### 7. Contribution

As this was an individual project, I was responsible for the complete development of this trading system, including:

- **Machine Learning**: Implementing the bidirectional LSTM network with multi-head attention mechanism and XGBoost ensemble with comprehensive feature engineering
- **Infrastructure**: Designing distributed training infrastructure with PyTorch DDP across 4x V100 GPUs for 75% training speed improvement
- **Data Engineering**: Building real-time Kafka streaming pipeline processing 9K+ financial events daily with sub-second latency
- **Workflow Automation**: Creating Apache Airflow DAGs for automated FinBERT sentiment analysis and model training workflows
- **DevOps**: Implementing CI/CD pipeline with GitHub Actions for automated model lifecycle management
- **NLP**: Developing the FinBERT-based sentiment analysis system integrating financial news, Reddit posts, and SEC filings
- **Trading System**: Designing and implementing the automated trading system with comprehensive risk management and Tradier API integration
- **Testing & Validation**: Building comprehensive test suites and validation frameworks for all infrastructure components

---

### 8. Skills & Technologies Used

- **Languages & Frameworks**: Python, PyTorch, TensorFlow, CUDA, Scikit-learn, Pandas, NumPy
- **Machine Learning**: Gradient Boosting (XGBoost), Feature Engineering, Time Series Forecasting
- **Deep Learning**: LSTM Networks (Bidirectional, Attention), Model Ensembling, Hyperparameter Optimization
- **Distributed Computing**: PyTorch DistributedDataParallel, Multi-GPU Training, CUDA Programming
- **Infrastructure**: Apache Kafka, Apache Airflow, Docker, GitHub Actions CI/CD
- **Natural Language Processing**: FinBERT, Sentiment Analysis, Text Processing, Financial Text Mining
- **Cloud Computing**: AWS (GPU instances), Distributed training infrastructure
- **APIs**: Tradier (paper trading), NewsAPI (financial news), Reddit API (sentiment data)

---

### 9. Project Repository

- [DeepTrade-AI](https://github.com/Srecharan/DeepTrade.git) 