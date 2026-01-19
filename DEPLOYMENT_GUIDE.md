# 🚀 Guide de Déploiement - Maize Yield Prediction

Ce guide vous explique comment déployer votre application sur GitHub et la mettre en ligne gratuitement.

---

## 📋 Étape 1: Préparer le Repository GitHub

### 1.1 Créer un compte GitHub (si pas déjà fait)
Allez sur https://github.com et créez un compte.

### 1.2 Créer un nouveau repository
1. Cliquez sur "+" → "New repository"
2. Nom: `maize-yield-prediction`
3. Description: "🌽 Prédiction du rendement du maïs en Afrique avec Machine Learning"
4. Visibilité: **Public**
5. Cochez "Add a README file" (optionnel, nous en avons déjà un)
6. Cliquez "Create repository"

### 1.3 Pousser le code sur GitHub

Ouvrez un terminal dans le dossier du projet et exécutez:

```bash
# Initialiser Git (si pas déjà fait)
git init

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "🌽 Initial commit - Maize Yield Prediction API"

# Ajouter le remote (remplacez VOTRE_USERNAME)
git remote add origin https://github.com/VOTRE_USERNAME/maize-yield-prediction.git

# Pousser sur GitHub
git branch -M main
git push -u origin main
```

---

## 🌐 Étape 2: Déploiement sur Render (GRATUIT)

**Render** est une plateforme cloud gratuite parfaite pour ce type de projet.

### 2.1 Créer un compte Render
1. Allez sur https://render.com
2. Cliquez "Get Started for Free"
3. Connectez-vous avec votre compte GitHub

### 2.2 Déployer l'application
1. Dans le dashboard Render, cliquez **"New +"** → **"Web Service"**
2. Connectez votre repository GitHub `maize-yield-prediction`
3. Configurez:
   - **Name**: `maize-yield-prediction`
   - **Region**: Frankfurt (EU) ou Oregon (US)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Plan: **Free**
5. Cliquez **"Create Web Service"**

### 2.3 Attendre le déploiement
Le déploiement prend 3-5 minutes. Une fois terminé, vous aurez une URL comme:
```
https://maize-yield-prediction.onrender.com
```

---

## 🚂 Alternative: Déploiement sur Railway (GRATUIT)

### 3.1 Créer un compte Railway
1. Allez sur https://railway.app
2. Connectez-vous avec GitHub

### 3.2 Déployer
1. Cliquez **"New Project"** → **"Deploy from GitHub repo"**
2. Sélectionnez `maize-yield-prediction`
3. Railway détecte automatiquement Python
4. Ajoutez une variable d'environnement:
   - `PORT` = `8000`
5. Cliquez "Deploy"

URL finale: `https://maize-yield-prediction.up.railway.app`

---

## ☁️ Alternative: Déploiement sur Heroku

### 4.1 Fichiers nécessaires

Créez un fichier `Procfile` (sans extension):
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Créez `runtime.txt`:
```
python-3.10.12
```

### 4.2 Déploiement
```bash
# Installer Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

heroku login
heroku create maize-yield-prediction
git push heroku main
heroku open
```

---

## 🐳 Alternative: Déploiement avec Docker sur Fly.io (GRATUIT)

### 5.1 Installer Fly CLI
```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Ou télécharger depuis https://fly.io/docs/hands-on/install-flyctl/
```

### 5.2 Déployer
```bash
fly auth login
fly launch --name maize-yield-prediction
fly deploy
```

---

## ✅ Vérification du Déploiement

Une fois déployé, testez votre API:

### Page d'accueil
```
https://VOTRE-URL.onrender.com
```

### Documentation API
```
https://VOTRE-URL.onrender.com/docs
```

### Test de prédiction (avec curl)
```bash
curl -X POST "https://VOTRE-URL.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"country_code":"KE","season_name":"Main","planting_month":3,"harvest_month":8,"area":500,"production_system":"rainfed"}'
```

---

## 🔧 Résolution de Problèmes

### L'application ne démarre pas
- Vérifiez les logs dans le dashboard Render/Railway
- Assurez-vous que `requirements.txt` contient toutes les dépendances

### Erreur "Model not found"
- Vérifiez que les fichiers `.pkl` sont bien commités dans Git
- Les fichiers dans `ml_models_pkg/` doivent être présents

### Temps de démarrage long
- Normal sur le plan gratuit (cold start)
- Le premier chargement peut prendre 30-60 secondes

---

## 📱 Partager votre Application

Une fois déployée, partagez l'URL:
- Dans votre rapport PDF
- Dans votre présentation PowerPoint
- Sur LinkedIn/Twitter pour montrer votre travail!

---

## 🎉 Félicitations!

Votre application de prédiction du rendement du maïs est maintenant en ligne et accessible au monde entier! 🌍🌽
