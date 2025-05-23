# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
import nltk
import pandas as pd
import joblib
import re
import os
import datetime
import requests
import json
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from .config import get_api_key

nltk.download('stopwords')

app = FastAPI(
    title="Article Category Classifier API",
    description="Upload a CSV of articles and get them auto-tagged using ML + Gemini",
    version="1.0.0",
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Article Classifier",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css"
    )

model = joblib.load('./app/models/svm_model_1.pkl')
vc = joblib.load('./app/models/vectorizer_1.pkl')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

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
    df['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    return df

def reclassify_general_and_predict_exam_specific(df, api_key):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    prompt_intro = """Given a list of articles with their content, do two things:

1. Assign the correct tag for each article from this list of tags:
[Governance, Economy, Society, Culture, Environment, Technology, Security, Ethics, International, Essay, Entertainment, Sports, Crime, Politics, Lifestyle, Business, Popculture, Local]

2. Indicate whether this article is highly important and needs deep understanding for UPSC exams by answering True or False.

Format the output exactly like this:
articleId1 - [Tag] - [True/False]
articleId2 - [Tag] - [True/False]
articleId3 - [Tag] - [True/False]

Only output the mapping, no extra text, no headings.

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

FINAL EXAMPLE 
Example:
123 - Governance - True
124 - Business - False
125 - International - True

Now here are the articles:\n\n"""


    general_df = df[df['category'].str.lower() == 'general'].copy()
    if general_df.empty:
        df['examSpecific'] = False
        return df

    article_prompts = ""
    for _, row in general_df.iterrows():
        content = f"{row['articleId']} - {row['title']} {row['present']}".strip()
        article_prompts += content + "\n"

    final_prompt = prompt_intro + article_prompts

    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}]
    }

    response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
    tag_map = {}
    exam_specific_map = {}

    for line in response_text.strip().split("\n"):
        if " - " in line:
            parts = line.strip().split(" - ")
            if len(parts) == 3:
                aid, tag, exam_specific = parts
                tag_map[aid.strip()] = tag.strip()
                exam_specific_map[aid.strip()] = exam_specific.strip().lower() == "true"

    df['examSpecific'] = False  # Default value
    for idx, row in df[df['category'].str.lower() == 'general'].iterrows():
        updated_tag = tag_map.get(str(row['articleId']))
        exam_specific_flag = exam_specific_map.get(str(row['articleId']), False)
        if updated_tag:
            df.at[idx, 'category'] = updated_tag
        df.at[idx, 'examSpecific'] = exam_specific_flag

    return df

def deep_analysis_for_exam_specific(df, api_key):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    detailed_prompt_intro = """
Given the following article, provide a detailed JSON output with the following fields. Each field should contain **clear, simple, and detailed explanations**, no matter how long it may be. Use simple words that anyone can understand. The AI should explain everything thoroughly for each section, and only include relevant data. If some sections do not apply to the article, those sections should be omitted, but make sure the explanation is as clear as possible for the included sections.
NOTE: If any section in the response is not relevant to the article or topic being summarized, or if the information is not available, simply remove that section from the final JSON output. Do not include empty or irrelevant sections. Ensure that the JSON is clean, concise, and only includes relevant information.
---

### **Common Across Most News Categories**:

1. **Key Actors**: Identify the main actors (individuals, organizations, governments, companies, etc.) involved in the article. For each actor, explain their role in the context of the situation. Provide details on what they are doing, how they are involved, and why they are important in this specific story.
2. **Impact (Affected)**: Explain in detail the people, groups, or communities that are affected by the situation discussed in the article. Describe how these groups are impacted — whether positively or negatively — and provide examples to make it clear who is being affected and why.
3. **Beneficiaries**: Detail who benefits from the situation mentioned in the article. These might be specific individuals, groups, companies, or even sectors that gain from the event or decision discussed. Explain how they benefit and why this is important.
4. **Policy Implications**: For articles discussing governance, health, social issues, or similar topics, explain the potential or actual effects on policies. What new policies are suggested, or how do existing policies need to change based on the article's information? Provide details on how these policies will affect various groups or systems.
5. **Challenges & Opportunities**: Break this into two sub-sections:
   - **Challenges**: Explain the difficulties or obstacles that are presented in the article. Describe who is facing them and why these challenges matter.
   - **Opportunities**: List the opportunities that have been created because of the situation. Who benefits from these opportunities, and what can be done with them?
6. **Future Implications**: Explain the potential long-term effects or consequences of the events in the article. Even if the implications are speculative, discuss how this situation could shape the future for the actors involved, society, or other systems.
7. **Public Opinion/Media**: Summarize how the public or media is reacting to the situation. Is the opinion mostly positive or negative? What are the key points in the media coverage, and how are the public and media shaping the conversation around the issue?
8. **Technological Impact**: If applicable, explain the role of technology in the article. Discuss any technological advancements, innovations, or impacts that have shaped or are shaping the situation. How is technology influencing the actors, challenges, and opportunities discussed?
9. **Decision-Making & Leadership**: Provide a detailed explanation of the decision-making process and leadership involved in the situation. Who is making decisions, and what are the reasons behind them? Describe the leadership qualities and actions that are significant in the context of the article.
10. **Policy Evaluation**: If the article involves governance, public health, or policy-making, provide insights on how the policies mentioned should be evaluated. What metrics or methods should be used to assess whether the policies are successful or not?
---

### **Less Common Across All Categories**:

1. **Ethical/Moral Dimensions**: If relevant, explain the ethical or moral aspects of the situation. What are the ethical questions raised? Are there moral concerns related to the actions of the key actors? How do these ethical considerations impact the situation?
2. **Underlying Causes**: If applicable, explain the deeper causes of the situation discussed in the article. What are the root causes or underlying factors that led to the current situation? This can include historical, economic, social, or environmental factors.
3. **Interconnections with Syllabus**: If the article is related to education or academic topics, explain how it connects with academic syllabi. This could include how the content is related to specific subjects, courses, or academic frameworks.
---

### Final Output Format:
Return **strictly only** a pure JSON object with these fields. Make sure the explanations are detailed and easy to understand. If some sections do not apply to the article, skip those sections but make sure to explain why they're skipped.

---

**Example Output Format**:

```json
{
  "Key Actors": [
    "Actor 1: Detailed explanation of Actor 1's role and why they are important in this context.",
    "Actor 2: Detailed explanation of Actor 2's role, why they are involved, and their actions in the situation."
  ],
  "Impact (Affected)": [
    "Group 1: Detailed explanation of how Group 1 is affected, including positive and negative impacts.",
    "Group 2: Explanation of how Group 2 is affected, providing examples."
  ],
  "Beneficiaries": [
    "Beneficiary 1: Explanation of how Beneficiary 1 benefits from the situation and why this is important."
  ],
  "Policy Implications": [
    "Policy 1: Explanation of the suggested or actual policy and its impact on different stakeholders."
  ],
  "Challenges & Opportunities": [
    "Challenges: Detailed explanation of the challenges faced and the groups affected.",
    "Opportunities: Detailed explanation of opportunities created by the situation."
  ],
  "Underlying Causes": [
    "Cause 1: Explanation of the underlying cause, providing context and contributing factors."
  ],
  "Future Implications": [
    "Implication 1: Explanation of how the situation could affect the future and potential outcomes."
  ],
  "Public Opinion/Media": [
    "Public Opinion: Summary of public reactions and media coverage."
  ],
  "Ethical/Moral Dimensions": [
    "Ethical Concern: Detailed explanation of any ethical issues raised by the situation."
  ],
  "Technological Impact": [
    "Technology 1: Explanation of how technology is influencing or changing the situation."
  ],
  "Interconnections with Syllabus": [
    "Connection 1: Explanation of how the situation is linked with academic courses or syllabi."
  ],
  "Decision-Making & Leadership": [
    "Leadership: Detailed explanation of leadership actions and decision-making processes."
  ],
  "Policy Evaluation": [
    "Evaluation: Explanation of how the policy should be evaluated and the methods to measure its success."
  ]
}

"""

    if 'deepAnalysisJson' not in df.columns:
        df['deepAnalysisJson'] = None

    exam_specific_df = df[df['examSpecific'] == True]
    if exam_specific_df.empty:
        return df

    for idx, row in exam_specific_df.iterrows():
        article_text = f"{row['title']} {row['involvement']} {row['past']} {row['present']}".strip()
        final_prompt = detailed_prompt_intro + article_text

        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}]
        }

        response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

        if response.status_code != 200:
            df.at[idx, 'deepAnalysisJson'] = "{}"
            continue

        try:
            response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            df.at[idx, 'deepAnalysisJson'] = response_text
        except Exception:
            df.at[idx, 'deepAnalysisJson'] = "{}"

    return df

def generate_summary_points(df, api_key):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    summary_prompt_intro = """Given the following article, generate:
Imagine preparing for UPSC. The AI should provide a concise yet detailed summary that explains key concepts clearly and links them to current affairs, real-world examples, and interlinkages between subjects. The goal is to help aspirants understand the core concepts, connect theory with practice, and offer actionable insights for effective preparation.
- A **short summary** (3-5 lines)
- **5 important key points**

Return the result as a pure JSON like:
NOTE: If any section in the response is not relevant to the article or topic being summarized, or if the information is not available, simply remove that section from the final JSON output. Do not include empty or irrelevant sections. Ensure that the JSON is clean, concise, and only includes relevant information.
NOTE : The higlight words can have in middle of the texts or points also whereever you feel that is important need to highlight it them please ensure highligh in same formate so that i can easily fetch it and highlight it in frontend 
{
  "Key Actors": [
    "{{highlighted}}Actor 1:{{/highlighted}} Detailed explanation of Actor 1's role and why they are important in this context. For example, {{highlighted}}Actor 1{{/highlighted}} might be a company responsible for initiating a major change in the industry, and their actions have far-reaching consequences.",
    "{{highlighted}}Actor 2:{{/highlighted}} Detailed explanation of Actor 2's role, why they are involved, and their actions in the situation. This might include actions taken by governments, organizations, or individuals. Explain how these actions are shaping the situation."
  ],
  "Impact (Affected)": [
    "{{highlighted}}Group 1:{{/highlighted}} Detailed explanation of how Group 1 is affected, including both positive and negative impacts. For example, if the news is about a new policy, explain how it helps or harms specific groups like citizens, companies, or communities.",
    "{{highlighted}}Group 2:{{/highlighted}} Explanation of how Group 2 is affected, providing examples. For instance, {{highlighted}}Group 2{{/highlighted}} could be a community facing financial loss due to a new regulation."
  ],
  "Beneficiaries": [
    "{{highlighted}}Beneficiary 1:{{/highlighted}} Explanation of how Beneficiary 1 benefits from the situation and why this is important. This could be a company profiting from a new market or individuals gaining from a new policy. Be sure to explain the benefit clearly."
  ],
  "Policy Implications": [
    "{{highlighted}}Policy 1:{{/highlighted}} Explanation of the suggested or actual policy and its impact on different stakeholders. This could include how a new law or change in regulation affects citizens, businesses, or the government."
  ],
  "Challenges & Opportunities": [
    "{{highlighted}}Challenges:{{/highlighted}} Detailed explanation of the challenges faced and the groups affected. For example, this could involve economic barriers, resistance to change, or logistical issues that are hindering progress.",
    "{{highlighted}}Opportunities:{{/highlighted}} Detailed explanation of opportunities created by the situation. For example, the situation might open up new business avenues, technological advancements, or societal changes that can improve lives."
  ],
  "Underlying Causes": [
    "{{highlighted}}Cause 1:{{/highlighted}} Explanation of the underlying cause, providing context and contributing factors. This could include historical, economic, or environmental factors that have led to the situation described in the article."
  ],
  "Future Implications": [
    "{{highlighted}}Implication 1:{{/highlighted}} Explanation of how the situation could affect the future and potential outcomes. This can be speculative but should consider how the situation might evolve and impact stakeholders long-term."
  ],
  "Public Opinion/Media": [
    "{{highlighted}}Public Opinion:{{/highlighted}} Summary of public reactions and media coverage. This could include positive, neutral, or negative feedback from the public, media outlets, and experts regarding the situation."
  ],
  "Ethical/Moral Dimensions": [
    "{{highlighted}}Ethical Concern:{{/highlighted}} Detailed explanation of any ethical issues raised by the situation. This might include concerns about fairness, transparency, or rights that need to be addressed."
  ],
  "Technological Impact": [
    "{{highlighted}}Technology 1:{{/highlighted}} Explanation of how technology is influencing or changing the situation. For example, this could involve the use of AI, machine learning, or other technologies that have played a significant role in shaping the current events."
  ],
  "Interconnections with Syllabus": [
    "{{highlighted}}Connection 1:{{/highlighted}} Explanation of how the situation is linked with academic courses or syllabi. For example, a news article about climate change could connect to environmental science courses."
  ],
  "Decision-Making & Leadership": [
    "{{highlighted}}Leadership:{{/highlighted}} Detailed explanation of leadership actions and decision-making processes. This section should explore who is making decisions, their motivations, and the leadership qualities required to navigate the situation."
  ],
  "Policy Evaluation": [
    "{{highlighted}}Evaluation:{{/highlighted}} Explanation of how the policy should be evaluated and the methods to measure its success. This might include data metrics, public feedback, or long-term effects on the community or economy."
  ]
}

Strictly JSON format only.\n\n"""

    if 'summaryPointsJson' not in df.columns:
        df['summaryPointsJson'] = None

    for idx, row in df.iterrows():
        article_text = f"{row['title']} {row['involvement']} {row['past']} {row['present']}".strip()
        final_prompt = summary_prompt_intro + article_text

        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}]
        }

        response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

        if response.status_code != 200:
            df.at[idx, 'summaryPointsJson'] = "{}"
            continue

        try:
            response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            df.at[idx, 'summaryPointsJson'] = response_text
        except Exception:
            df.at[idx, 'summaryPointsJson'] = "{}"

    return df

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df = process_csv_and_predict(df)
        df = reclassify_general_and_predict_exam_specific(df, get_api_key())
        df = deep_analysis_for_exam_specific(df, get_api_key())  # Only if examSpecific is True
        df = generate_summary_points(df, get_api_key())

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
