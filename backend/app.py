from flask import Flask, redirect, url_for, session, jsonify,request
from authlib.integrations.flask_client import OAuth
from flask_cors import CORS
from supabase import create_client, Client
from flask import send_from_directory
from flask_cors import CORS
from flask_mail import Mail, Message
from threading import Thread
from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
#flask mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"]
)

FRONTEND_URL = "http://127.0.0.1:5500/"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# supabase
SUPABASE_URL = "https://zcckaqsaqkvwvgjzyjvf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjY2thcXNhcWt2d3Znanp5anZmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDE1ODQyNSwiZXhwIjoyMDc1NzM0NDI1fQ.6YNwlNjaPQHMPgR4ABb8mpb0WviHOnRM-o1CDLYCL3g"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#google auth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)
resume_t=''
# route
@app.route('/dashboard')
def dashboard():
    return send_from_directory('static', 'dashboard.html')

@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()  
    user_info = google.userinfo()  
    email = user_info['email']
    name = user_info.get('name', 'User')
    picture = user_info.get('picture', '')

    existing = supabase.table('users').select('*').eq('email', email).execute()
    if len(existing.data) == 0:
        supabase.table('users').insert({
            'email': email,
            'name': name,
            'picture': picture
        }).execute()
 
    session['user'] = user_info

    return redirect(f"{FRONTEND_URL}/frontend/dashboard.html?name={user_info['name']}")

@app.route('/user')
def get_user():
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify(user)


def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)

def send_analysis_email(user_email, resume_id):
    link = f"http://127.0.0.1:5500/frontend/test.html?resume_id={resume_id}"
    msg = Message(
        subject="Your Resume Analysis is Ready",
        sender="your_email@gmail.com",
        recipients=[user_email],
    )
    msg.body = f"Hey! Your resume analysis is complete.\n\nView your result here:\n{link}\n\nBest,\nATS Analyzer Bot"
    Thread(target=send_async_email, args=(app, msg)).start()



from threading import Thread

@app.route('/uploadresume', methods=['POST'])
def uploadresume():
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    resume_file = request.files.get('resume')
    if not resume_file:
        return jsonify({'error': 'No file uploaded'}), 400

    import os, json
    from datetime import datetime

    os.makedirs('uploads', exist_ok=True)
    filename = resume_file.filename
    save_path = os.path.join('uploads', filename)
    resume_file.save(save_path)

    user_query = supabase.table('users').select('id').eq('email', user['email']).execute()
    if not user_query.data:
        return jsonify({'error': 'User not found in database'}), 404

    user_id = user_query.data[0]['id']

    resume_insert = supabase.table('resumes').insert({
        'user_id': user_id,
        'user_email': user['email'],
        'user_name': user['name'],
        'resume_filename': filename,
        'status': 'processing'
    }).execute()

    if not resume_insert.data:
        return jsonify({'error': 'Failed to insert resume record'}), 500

    resume_id = resume_insert.data[0]['id']


    Thread(target=background_resume_analysis, args=(resume_id, save_path, user['email'])).start()

    return jsonify({
        'message': 'Your resume is being analyzed. You will receive your analyzed report via email within 5 minutes.',
        'resume_id': resume_id
    }),200


def background_resume_analysis(resume_id, file_path, user_email):
    """Run analysis in background and send result email."""
    from datetime import datetime

    try:
        text = extract_pdf(file_path)
        result = resume_analyzer(text)

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {"error": "Invalid JSON output", "raw": result}

        supabase.table('resumeanalysis').insert({
            'resume_id': resume_id,
            'analyzed_json': json.dumps(result),
            'analyzed_at': datetime.utcnow().isoformat()
        }).execute()

      
        supabase.table('resumes').update({'status': 'completed'}).eq('id', resume_id).execute()

       
        send_analysis_email(user_email, resume_id)

    except Exception as e:
        print("Background analysis failed:", e)
        supabase.table('resumes').update({'status': 'failed'}).eq('id', resume_id).execute()

import os
import openai
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")
import fitz 
import json
from flask import Flask, jsonify, session
import os, json
from datetime import datetime
from flask_cors import CORS
CORS(app, supports_credentials=True)

@app.route('/latest-analysis', methods=['GET'])
def latest_analysis():
    """Fetch analysis for a specific or latest resume."""
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    email = user.get('email')
    resume_id = request.args.get('resume_id')


    if resume_id:
        resume_data = (
            supabase.table('resumes')
            .select('*')
            .eq('user_email', email)
            .eq('id', resume_id)
            .execute()
        )
    else:
        resume_data = (
            supabase.table('resumes')
            .select('*')
            .eq('user_email', email)
            .order('created_at', desc=True)
            .limit(1)
            .execute()
        )

    if not resume_data.data:
        return jsonify({"error": "No resume found"}), 404

    resume_entry = resume_data.data[0]
    resume_id = resume_entry['id']
    resume_filename = resume_entry['resume_filename']


    existing_analysis = (
        supabase.table('resumeanalysis')
        .select('resume_namee, analyzed_json, analyzed_at')
        .eq('resume_id', resume_id)
        .order('analyzed_at', desc=True)
        .limit(1)
        .execute()
    )

    if existing_analysis.data:
        analysis_json = existing_analysis.data[0]['analyzed_json']
        try:
            parsed_json = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json
        except:
            parsed_json = {"error": "Invalid stored JSON", "raw": analysis_json}

        parsed_json['resume_filename'] = resume_filename
        parsed_json['resume_id'] = resume_id

        return jsonify(parsed_json)


    file_path = os.path.join('uploads', resume_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Resume file missing on server"}), 404

    try:
        text = extract_pdf(file_path)
        resume_t=text
        result = resume_analyzer(text)

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {"error": "Analyzer returned non-JSON text", "raw": result}

  
        result['resume_filename'] = resume_filename
        result['resume_id'] = resume_id

      
        supabase.table('resumeanalysis').insert({
            'resume_namee': resume_filename,
            'resume_id': resume_id,
            'analyzed_json': json.dumps(result),
            'analyzed_at': datetime.utcnow().isoformat()
        }).execute()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_pdf(file_path):
    """Extracts text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def resume_analyzer(resume_text):
    if not resume_text:
        return {"error": "Empty resume text."}

    prompt = f"""
    You are an expert resume analyzer.
    Analyze the following resume and return only valid JSON with:
    {{
      "ats_score": <integer 0-100>,
      "missing_keywords": [list of keywords],
      "grammatical_errors": [list of typos or grammar mistakes],
      "improvement_tips": [list of suggestions],
      "grammar_accuracy": <integer 0-100>,
      "Resume_strength": <integer 0-100>,
      "Formatting_quality": <integer 0-100>,
      "skill_gap" : <integer 0-100>,
      "experience_level" : [Fresher,Junior,Mid-Level,Senior],
      "readability_Score" : <integer 0-100>,
      "achievement_impact" : <integer 0-100>,
      "detected_keyword" : [list of detected technical skills separated by comma (,)],
      "Experience" : [list experiences i have]",
      "keyword_matches" : <integer 0-100(how many technical skills matches to general role for job)>,
      "formatting_needed" : <small summary of format needed>,
      "title":<your role for company>,
      "summary": "<brief summary>"
    }}

    Resume:
    {resume_text}
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

       
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:        
       
            cleaned = content.replace("```", "").replace("json", "").strip()
            try:

                result = json.loads(cleaned)
                result=json.dumps(result, indent=4, ensure_ascii=False)
            except:
                result = {"error": "Invalid JSON from Gemini", "raw": content}

        return result

    except Exception as e:
        return {"error": str(e)}
    
@app.route('/dash')
def serve_dashboard():
    return send_from_directory('static', 'test.html')

@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    email = user.get('email')

    resumes = supabase.table('resumes').select('*').eq('user_email', email).execute()
    if not resumes.data:
        return jsonify({"total": 0, "analyses": []})

    resume_ids = [r['id'] for r in resumes.data]
    resume_map = {r['id']: r for r in resumes.data}

    analyses = supabase.table('resumeanalysis').select('*').in_('resume_id', resume_ids).order('analyzed_at', desc=True).execute()

    total = len(analyses.data)
    if total == 0:
        return jsonify({"total": 0, "analyses": []})

    ats_scores = []
    parsed_data = []
    for a in analyses.data:
        try:
            result = json.loads(a['analyzed_json'])
            ats = result.get('ats_score', 0)
            ats_scores.append(ats)
            resume_name = resume_map[a['resume_id']]['resume_filename']
            parsed_data.append({
                "resume_id": a['resume_id'],
                "resume_name": resume_name,
                "ats_score": ats,
                "analyzed_at": a['analyzed_at']
            })
        except Exception as e:
            print("Error parsing:", e)

    avg_ats = round(sum(ats_scores) / len(ats_scores), 1) if ats_scores else 0
    highest = max(ats_scores)
    lowest = min(ats_scores)

    latest_resume = parsed_data[0]['resume_name']
    latest_date = parsed_data[0]['analyzed_at']

    return jsonify({
        "total": total,
        "average_ats": avg_ats,
        "last_resume": latest_resume,
        "last_date": latest_date,
        "analyses": parsed_data,
        "highest_score": highest,
        "lowest_score": lowest,
        "common_area": "Quantitative Results"
    })

@app.route('/job-match', methods=['POST'])
def job_match():
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    job_desc = data.get('job_description')
    resume_id = data.get('resume_id')
    use_model = data.get('model', 'tfidf') 
    if not job_desc:
        return jsonify({"error": "Job description required"}), 400

    if resume_id:
        resume_data = (
            supabase.table('resumes')
            .select('resume_filename')
            .eq('id', resume_id)
            .eq('user_email', user['email'])
            .execute()
        )
        if not resume_data.data:
            return jsonify({"error": "Resume not found"}), 404

        file_path = os.path.join('uploads', resume_data.data[0]['resume_filename'])
        if not os.path.exists(file_path):
            return jsonify({"error": "Resume file missing on server"}), 404

        resume_text = extract_pdf(file_path)
    else:
        resume_text = data.get('resume_text', '')

    if not resume_text.strip():
        return jsonify({"error": "No resume text provided"}), 400

   
    if use_model == 'bert':
        score = job_match_score_bert(resume_text, job_desc)
    else:
        score = job_match_score_tfidf(resume_text, job_desc)


    job_keywords = set(clean_text(job_desc).split())
    resume_keywords = set(clean_text(resume_text).split())
    missing = list(job_keywords - resume_keywords)

    response = {
        "match_score": score,
        "missing_keywords": missing[:15],
        "summary": f"This resume matches the job description by {score}%."
    }

    return jsonify(response)

@app.route("/api/resume/<resume_id>", methods=["GET"])
def get_resume_analysis(resume_id):
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    try:
     
        analysis = (
            supabase.table("resumeanalysis")
            .select("*")
            .eq("resume_id", resume_id)
            .order("analyzed_at", desc=True)
            .limit(1)
            .execute()
        )

        if not analysis.data:
            return jsonify({"error": "No analysis found"}), 404

        result = analysis.data[0]
        data = json.loads(result["analyzed_json"])

        return jsonify({
            "resume_id": resume_id,
            "ats_score": data.get("ats_score"),
            "missing_keywords": data.get("missing_keywords", []),
            "grammatical_errors": data.get("grammatical_errors", []),
            "improvement_tips": data.get("improvement_tips", []),
            "summary": data.get("summary", ""),
            "analyzed_at": result.get("analyzed_at")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def serve_uploaded_resume(filename):
    """Serve uploaded resume files from uploads folder"""
    import os
    from flask import send_from_directory
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    return send_from_directory(uploads_dir, filename)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import re


bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def job_ai(job_desc):
    if not job_desc.strip():
        return ""

    prompt = f"""
    Extract *only* technical skills from this text — programming languages, frameworks, tools, databases, and technologies.
    Return them as comma-separated keywords. No extra words.
    Text:
    {job_desc}
    """

    response = model.generate_content(prompt)
    content = response.text.strip().lower()
   
    content = re.sub(r'[^a-z0-9,\s\+\#\.]', ' ', content)
    skills = [skill.strip() for skill in content.split(",") if skill.strip()]
    skills = list(set(skills)) 
    return ", ".join(skills)

def job_match_score_tfidf(resume_text, job_desc):
    resume_keywords = job_ai(resume_text)
    job_keywords = job_ai(job_desc)

    resume_clean = clean_text(resume_keywords)
    job_clean = clean_text(job_keywords)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)

def job_match_score_bert(resume_text, job_desc):
    resume_keywords = job_ai(resume_text)
    job_keywords = job_ai(job_desc)
    embeddings = bert_model.encode([resume_keywords, job_keywords])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity) * 100, 2)

@app.route('/recommend-jobs', methods=['GET'])
def recommend_jobs():
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    resume_id = request.args.get('resume_id')

    if not resume_id:
        return jsonify({"error": "Missing resume_id"}), 400

    resume_data = (
        supabase.table('resumes')
        .select('resume_filename')
        .eq('id', resume_id)
        .eq('user_email', user['email'])
        .execute()
    )
    if not resume_data.data:
        return jsonify({"error": "Resume not found"}), 404

    file_path = os.path.join('uploads', resume_data.data[0]['resume_filename'])
    if not os.path.exists(file_path):
        return jsonify({"error": "Resume file missing on server"}), 404

    resume_text = extract_pdf(file_path)

    if not resume_text.strip():
        return jsonify({"error": "No resume text found"}), 400

    jobs = [
    {"title": "Data Analyst", "desc": "Python, Excel, SQL, Tableau"},
    {"title": "Backend Developer", "desc": "Flask, Django, PostgreSQL"},
    {"title": "ML Engineer", "desc": "TensorFlow, Python, NLP, Data"},
    {"title": "Frontend Developer", "desc": "React, JavaScript, HTML, CSS"},
    {"title": "Full Stack Developer", "desc": "React, Node.js, MongoDB, Express"},
    {"title": "DevOps Engineer", "desc": "Docker, Kubernetes, CI/CD, AWS"},
    {"title": "Cloud Engineer", "desc": "AWS, Azure, Terraform, Networking"},
    {"title": "Cybersecurity Analyst", "desc": "Network Security, SIEM, Firewalls"},
    {"title": "Mobile App Developer", "desc": "Flutter, Dart, Firebase"},
    {"title": "Game Developer", "desc": "Unity, C#, 3D Models"},
    {"title": "AI Researcher", "desc": "Deep Learning, PyTorch, LLMs"},
    {"title": "Business Analyst", "desc": "Excel, PowerBI, Communication"},
    {"title": "System Administrator", "desc": "Linux, Shell Script, VMs"},
    {"title": "Network Engineer", "desc": "Cisco, Routing, Switching"},
    {"title": "UI/UX Designer", "desc": "Figma, Wireframing, Prototyping"},
    {"title": "Product Manager", "desc": "Roadmaps, Scrum, Market Research"},
    {"title": "QA Tester", "desc": "Selenium, Regression, Test Cases"},
    {"title": "Automation Engineer", "desc": "Python, Jenkins, CI pipelines"},
    {"title": "ETL Developer", "desc": "Informatica, SQL, Data Pipelines"},
    {"title": "Database Administrator", "desc": "MySQL, Oracle, Backup, Tuning"},
    {"title": "Data Engineer", "desc": "Airflow, Spark, Hadoop, Data Lakes"},
    {"title": "Solution Architect", "desc": "Cloud Design, Microservices"},
    {"title": "Salesforce Developer", "desc": "Apex, Lightning, CRM"},
    {"title": "Android Developer", "desc": "Kotlin, Jetpack Compose, SQLite"},
    {"title": "iOS Developer", "desc": "Swift, Xcode, UI Kit"},
    {"title": "Blockchain Developer", "desc": "Solidity, Ethereum, Smart Contracts"},
    {"title": "Robotics Engineer", "desc": "ROS, C++, Embedded Systems"},
    {"title": "Embedded Systems Engineer", "desc": "C, IoT, Microcontrollers"},
    {"title": "Big Data Engineer", "desc": "Kafka, Spark, HDFS"},
    {"title": "Content Writer", "desc": "SEO, Research, Blogging"},
    {"title": "Technical Support Specialist", "desc": "Troubleshooting, Customer Support"},
    {"title": "Data Scientist", "desc": "Machine Learning, Pandas, Statistics"},
    {"title": "Digital Marketing Manager", "desc": "Google Ads, SEO, Social Media"},
    {"title": "Operations Manager", "desc": "Team Coordination, Planning"},
    {"title": "HR Recruiter", "desc": "Hiring, Interviews, Talent Sourcing"}
    ]


    corpus = [resume_text] + [job["desc"] for job in jobs]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    for i, score in enumerate(similarities):
        jobs[i]["job_title"] = jobs[i]["title"]
        jobs[i]["match"] = round(score * 100, 2)

    jobs.sort(key=lambda x: x["match"], reverse=True)

    return jsonify({"recommendations": jobs[:3]})

if __name__ == '__main__':
    app.run(debug=True)
