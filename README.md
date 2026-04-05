# 🌿 EcoRoute Environment

A realistic, real-world delivery route optimization environment for AI agents, built with OpenEnv specification.

## 🎯 Problem Statement
Optimize delivery routes considering:
- Fuel efficiency (eco-friendly driving)
- Time windows (deadlines for deliveries)
- Traffic and weather conditions
- Electric vehicle charging constraints

## 🚀 Features
- **3 Difficulty Levels**: Easy (1 package), Medium (3 packages), Hard (4 packages with deadlines)
- **Rich Reward Structure**: Partial progress rewards, on-time bonuses, fuel efficiency incentives
- **Real-world Constraints**: Traffic, weather, deadlines, fuel management
- **OpenEnv Compliant**: Full step()/reset()/state() API

## 📊 Evaluation
Each task produces a normalized score between 0.0 and 1.0 based on:
- Delivery success rate
- On-time delivery rate
- Fuel efficiency
- Time efficiency

## 🛠️ Setup

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.app:app --reload"# EcoRoute Environment" 
