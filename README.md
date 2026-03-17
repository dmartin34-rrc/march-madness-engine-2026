# March Madness Engine 2026

**A machine learning predictive model for the 2026 NCAA Tournament.**

---

## Overview
This is a full-stack machine learning application built to forecast the outcomes of the 2026 NCAA tournament. It utilizes an XGBoost classifier to analyze the statistical parity between any two teams in the bracket, providing an objective win probability based on twenty years of historical tournament data.

## Core Features
* **XGBoost Architecture:** Uses a 250-tree ensemble model to weight advanced metrics like Net Rating and Effective Field Goal percentage.
* **70.52% Historical Accuracy:** Validated against tournament data from 2003–2025.
* **Averaged Prediction Logic:** Implements a symmetric prediction wrapper to eliminate position bias, ensuring consistent results regardless of team order.
* **Live Data Integration:** Retrained with final 2026 Selection Sunday stats and seeding.

## Technical Stack
* **Backend:** FastAPI (Python) serving an XGBoost pipeline.
* **Frontend:** React (Vite) for the UI.
* **Data:** 2026 Kaggle March Machine Learning Mania datasets.
* **Deployment:** Render (Backend) and Vercel (Frontend).

## Analytics Logic
The engine moves beyond traditional box scores to focus on pace-adjusted efficiency. This allows the model to accurately compare teams with vastly different styles of play (e.g., high-tempo offensive teams vs. slow-paced defensive teams).

### 1. Possession Calculation
First, the model calculates the total number of possessions for each game to normalize the data:
$$Possessions = FGA - ORB + TO + (0.475 \times FTA)$$

### 2. Net Rating
The model then calculates the **Net Rating**, which measures a team's efficiency differential per 100 possessions. This is the primary metric the XGBoost model uses to determine "true" team strength:
$$Net Rating = \left( \frac{Points Scored}{Possessions} \times 100 \right) - \left( \frac{Points Allowed}{Possessions} \times 100 \right)$$

By focusing on this differential, the engine identifies which team is mathematically more efficient, regardless of their tournament seed or the pace of the game.
