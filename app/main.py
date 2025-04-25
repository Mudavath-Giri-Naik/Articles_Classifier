from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html

import pandas as pd
import joblib
import re
import os
import datetime
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import requests
import json
from dotenv import load_dotenv

# Load environment variables
from app.config import get_api_key

# Initialize FastAPI app
app = FastAPI(
    title="Article Category Classifier API",
    description="Upload a CSV of articles and get them auto-tagged using ML + Gemini",
    version="1.0.0",
    docs_url=None  # Disable default docs
)

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Article Classifier",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css"
    )

# Load model and vectorizer
model = joblib.load('./app/models/svm_model_1.pkl')
vc = joblib.load('./app/models/vectorizer_1.pkl')

# NLP setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Helper functions
def val_to_category(val):
    category_map = {
        0: 'sports',
        1: 'business',
        2: 'entertainment',
        3: 'education',
        4: 'technology'
    }
    return category_map.get(val, "general")

def preprocess_text(text):
    cleaned = re.sub('[^a-zA-Z]', ' ', text)
    cleaned = cleaned.lower().split()
    cleaned = [ps.stem(word) for word in cleaned if word not in stop_words]
    return " ".join(cleaned)

def predict_category(row, threshold=0.90):
    combined_text = f"{row['title']} {row['present']}"
    processed_text = preprocess_text(combined_text)
    vector = vc.transform([processed_text])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        max_prob = max(probabilities)
        predicted_index = probabilities.argmax()
        if max_prob >= threshold:
            return val_to_category(predicted_index)
        else:
            return "general"
    else:
        predicted_index = model.predict(vector)[0]
        return val_to_category(predicted_index)

def process_csv_and_predict(df):
    required_columns = ['articleId', 'title', 'involvement', 'past', 'present', 'points', 'glossary']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['category'] = df.apply(predict_category, axis=1)
    
    # Add current date to each row
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    df['date'] = current_date

    return df

def reclassify_general_articles(df, api_key):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt_intro = """Given a list of articles with their content, I need a simple output containing only the article IDs and the relevant tags related to each article, referring to the following tags:

1.	Governance: Polity, Constitution, Laws, Judiciary, Schemes, Policies, Public Administration
2.	Economy: Banking, Infrastructure, Agriculture, Financial Systems, Budget, Reports
3.	Society: Social Issues, Demographics, Education, Women & Child, Caste, Migration
4.	Culture: History, Heritage, Indian Art & Architecture
5.	Environment: Climate Change, Ecology, Biodiversity, Disaster Management, Sustainability
6.	Technology: Science, Innovation, Space, Defence, Cybersecurity, Data Privacy
7.	Security: Internal Security, Border Management, Crisis Response, Intelligence
8.	Ethics: Values, Integrity, Public Service, Case Studies, Leadership, Accountability
9.	International: Foreign Policy, Diplomacy, Global Summits, Bilateral/Multilateral Relations, India & Diaspora
10.	Essay: Cross-cutting topics like Democracy, Youth, Freedom, Peace, Development, Environment, Innovation
11.	Entertainment: Film releases, Celebrity news, Gossip, Award functions (unless of national relevance)
12.	Sports: Match results, Player controversies, Club transfers, League updates (e.g., IPL, FIFA)
13.	Crime: Isolated crimes, Theft, Assaults, Murders (unless raising broader social/legal issues)
14.	Politics: Political rallies, Party statements, Campaigns, Slogans, Defections (without policy/legal significance)
15.	Lifestyle: Fashion trends, Diets, Beauty, Fitness, Luxury product news
16.	Business: Company profits, Stock market trends, Private startup news (unless linked to govt policies/PSUs)
17.	Popculture: Viral memes, YouTube/TikTok trends, Influencer news, Social media challenges
18.	Local: City-level news, Local protests, Municipal events (unless tied to a national scheme like Smart Cities)

The output should only include the article IDs followed by the tags relevant to each article. There should be no extra explanations, headings, or any other additional information. Only the article ID and tags.

NOTE: Only one tag for each article. Example output:
articleId1 - Governance
articleId2 - Technology
articleId3 - Society
articleId4 - International

Now analyze the following articles:\n\n"""

    general_df = df[df['category'].str.lower() == 'general'].copy()
    if general_df.empty:
        return df

    article_prompts = ""
    for _, row in general_df.iterrows():
        content = f"{row['articleId']} - {row['title']} {row['present']}".strip()
        article_prompts += content + "\n"

    final_prompt = prompt_intro + article_prompts

    payload = {
        "contents": [{
            "parts": [{
                "text": final_prompt
            }]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    tag_map = {}
    for line in response_text.strip().split("\n"):
        if " - " in line:
            aid, tag = line.strip().split(" - ", 1)
            tag_map[aid.strip()] = tag.strip()

    for idx, row in df[df['category'].str.lower() == 'general'].iterrows():
        updated_tag = tag_map.get(str(row['articleId']))
        if updated_tag:
            df.at[idx, 'category'] = updated_tag

    return df

# ------------------------
# FastAPI Endpoints
# ------------------------

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df = process_csv_and_predict(df)
        df = reclassify_general_articles(df, get_api_key())
        output_path = "final_tagged_articles.csv"
        df.to_csv(output_path, index=False)
        return {"message": "✅ File processed successfully. Download using /download"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/download/")
async def download_file():
    file_path = "final_tagged_articles.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename="final_tagged_articles.csv", media_type="text/csv")
