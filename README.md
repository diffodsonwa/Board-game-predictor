# 🎲 Board Game Rating Predictor
## *End-to-End ML Project: From Local Development to AWS Deployment with CI/CD*

---

## 📋 Project Overview

**Goal:** Predict board game ratings (Low/Medium/High → 0,1,2) using a regression model.

**Dataset:** BoardGameGeek dataset with 21,925 games and 46 features. 

**Target Variable:** `Rating` (categorical → numeric mapping: Low=0, Medium=1, High=2)

**Features (5):**
| Feature | Description |
|--------|-------------|
| `GameWeight` | Game complexity score (1-5) |
| `BGGId` | BoardGameGeek unique identifier |
| `NumWant` | Number of users wanting the game |
| `ComAgeRec` | Community age recommendation |
| `BestPlayers` | Optimal number of players |

**Model:** `LinearRegression` (scikit-learn)

**Tech Stack:**
- **API Framework:** FastAPI
- **UI Framework:** Gradio
- **Containerization:** Docker
- **Cloud Provider:** AWS (ECS Fargate, ECR)
- **CI/CD:** GitHub Actions
- **Experiment Tracking:** MLflow

---

## 🗂️ Project Structure
