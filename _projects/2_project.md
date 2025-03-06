---
layout: page
title: DeepTrade AI - Multi-Model Stock Prediction with NLP & Automated Trading
description: An end-to-end system integrating LSTM-XGBoost prediction with sentiment analysis for automated trading
img: assets/img/project-2/stock_hero.jpeg
importance: 2
category: work
related_publications: false
---

### 1. Overview

DeepTrade AI is an end-to-end automated stock trading system that combines machine learning price prediction with NLP-based sentiment analysis. The system features a bidirectional LSTM with attention mechanism and XGBoost ensemble for multi-timeframe price forecasting (5m, 15m, 30m, 1h), and integrates FinBERT for real-time sentiment analysis of financial news, Reddit posts, and SEC filings. The architecture employs dynamic model weighting, comprehensive risk management controls, and simulated execution through the Tradier API, achieving 55-65% directional accuracy and a 58.5% win rate in paper trading.

<div style="text-align: center;">
    <img src="/assets/img/project-2/deeptrade_system.png" alt="System Architecture" style="width: 100%; max-width: 3000px;">
    <p><em>DeepTrade AI system architecture showing data flow between components</em></p>
</div>

---

### 2. Sentiment Analysis Pipeline

DeepTrade AI incorporates a sophisticated sentiment analysis pipeline that aggregates and analyzes data from three key sources:

#### 2.1 Multi-Source Integration

- **Financial News Processing (40%)**: Real-time streaming with NewsAPI integration, automated headline analysis, and relevancy-based filtering
  
- **Reddit Sentiment Analysis (30%)**: Multi-subreddit monitoring (r/wallstreetbets, r/stocks, r/investing) with advanced engagement metrics and post-comment sentiment weighting

- **SEC Filing Analysis (30%)**: Real-time CIK tracking, form-specific sentiment weighting, and automated filing pattern analysis

#### 2.2 FinBERT Model Architecture

The core of the sentiment analysis is a fine-tuned FinBERT model specifically pre-trained on financial text:

- **Tokenization**: Converts text data into numerical representations with tokens truncated at 512 length
- **Three-Class Classification**: Classifies sentiment as positive, negative, or neutral
- **Custom Sentiment Score**: Combines probabilities and source-specific weights for a composite sentiment score

<div style="text-align: center;">
    <img src="/assets/img/project-2/sentiment_op.png" alt="Sentiment Analysis" style="width: 100%; max-width: 800px;">
    <p><em>Sentiment analysis across different sources and stocks: Financial News (blue), Reddit (green), and SEC Filings (red)</em></p>
</div>

---

### 3. Multi-Model Stock Prediction System

The prediction system employs an ensemble approach combining two powerful models:

<div style="text-align: center;">
    <img src="/assets/img/project-2/lstm_xgb.png" alt="Model Architecture" style="width: 100%; max-width: 1000px;">
    <p><em>LSTM and XGBoost model architectures with dynamic ensemble weighting</em></p>
</div>

#### 3.1 LSTM Model with Attention Mechanism

The LSTM component features several advanced architectural elements:

- **Bidirectional Processing**: Analyzes time series data in both forward and backward directions
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence with varying levels of importance
- **Residual Connections and Batch Normalization**: Improves training stability and convergence
- **Dynamic Dropout**: Adaptively adjusts regularization during training

#### 3.2 XGBoost Model for Feature-Based Predictions

The XGBoost component complements the LSTM by providing a feature-based perspective:

- **Gradient Boosting Framework**: Iteratively builds an ensemble of 300 decision trees
- **Advanced Feature Engineering**: Processes 39 features including technical indicators, trend signals, and sentiment data
- **Feature Importance Analysis**: Provides insights into which factors most influence predictions

#### 3.3 Dynamic Ensemble Integration

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

### 4. Automated Trading System

The trading system translates predictions and sentiment analysis into concrete trading actions through the Tradier API sandbox environment.

#### 4.1 Trading Logic Pipeline

```
Prediction → Sentiment → Signal Generation → Risk Analysis → Position Sizing → Execution
     ↓           ↓              ↓                ↓               ↓              ↓
Price Data   News/Social   Trend Analysis    Risk Limits    Dynamic Sizing   Market Orders
     ↓           ↓              ↓                ↓               ↓              ↓
Confidence  Sentiment     Entry/Exit        Stop Loss      Position Value    Executions
  Scores     Scores        Signals         Take Profit      Calculation     & Monitoring
```

#### 4.2 Risk Management Framework

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

### 5. Performance Metrics

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

### 6. Contribution

As this was an individual project, I was responsible for the complete development of this trading system, including:

- Implementing the bidirectional LSTM network with multi-head attention mechanism for time series forecasting
- Building the XGBoost model with comprehensive feature engineering for 39 market indicators
- Developing the dynamic ensemble weighting system that adapts to changing market conditions
- Creating a real-time sentiment analysis pipeline integrating financial news, Reddit posts, and SEC filings
- Implementing the FinBERT-based NLP system for financial text classification and sentiment scoring
- Designing and implementing the automated trading system with comprehensive risk management
- Building and optimizing the end-to-end data pipeline for real-time operation
- Testing and validating the system through extensive paper trading simulations

---

### 7. Skills & Technologies Used

- **Languages & Frameworks**: Python, PyTorch, TensorFlow, CUDA, Scikit-learn, Pandas, NumPy
- **Machine Learning**: Gradient Boosting (XGBoost), Feature Engineering, Time Series Forecasting
- **Deep Learning**: LSTM Networks (Bidirectional, Attention), Model Ensembling, Hyperparameter Optimization
- **Natural Language Processing**: FinBERT, Sentiment Analysis, Text Processing, Financial Text Mining
- **Cloud Computing**: AWS SageMaker (for distributed model training)
- **APIs**: Tradier (paper trading), NewsAPI (financial news), Reddit API (sentiment data)

---

### 8. Project Repositories

- [DeepTrade-AI](https://github.com/Srecharan/DeepTrade.git): Main project repository
