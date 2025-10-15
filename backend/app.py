from flask import Flask, redirect, url_for, session, jsonify,request
from authlib.integrations.flask_client import OAuth
from flask_cors import CORS
from supabase import create_client, Client
from flask import send_from_directory
app = Flask(__name__)
CORS(app)
CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": "http://127.0.0.1:5500"}}
)
app.secret_key = "your-secret-key"
FRONTEND_URL = "http://127.0.0.1:5500"

# ===== Supabase Config =====
SUPABASE_URL = "https://zcckaqsaqkvwvgjzyjvf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjY2thcXNhcWt2d3Znanp5anZmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDE1ODQyNSwiZXhwIjoyMDc1NzM0NDI1fQ.6YNwlNjaPQHMPgR4ABb8mpb0WviHOnRM-o1CDLYCL3g"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== Google OAuth Setup =====
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id='1042555508326-46017nt97tjp54ntvg0ofi1mbv2l5lvk.apps.googleusercontent.com',
    client_secret='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',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)
# ===== Routes =====
@app.route('/dashboard')
def dashboard():
    return send_from_directory('static', 'dashboard.html')
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()  # get access token
    user_info = google.userinfo()  # ✅ Authlib method that fetches user info for you
    # user_info = google.get('userinfo').json()
    # Extract user details
    email = user_info['email']
    name = user_info.get('name', 'User')
    picture = user_info.get('picture', '')
    # Store or update user in Supabase
    existing = supabase.table('users').select('*').eq('email', email).execute()
    if len(existing.data) == 0:
        supabase.table('users').insert({
            'email': email,
            'name': name,
            'picture': picture
        }).execute()
    # Save session
    session['user'] = user_info
    # Redirect to frontend dashboard
    return redirect(f"{FRONTEND_URL}/frontend/dashboard.html?name={user_info['name']}")

@app.route('/user')
def get_user():
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify(user)



@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    # Get uploaded file info
    file = request.files.get('resume')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename
  
    # Check if user already has pending resume
    existing = supabase.table('resumes')\
        .select('*')\
        .eq('user_email', user['email'])\
        .eq('status', 'pending')\
        .execute()

    if existing.data and len(existing.data) > 0:
        return jsonify({"status": "pending", "message": "Your previous resume is still in queue!"})

    # Insert new resume
    supabase.table('resumes').insert({
        'user_email': user['email'],
        'user_name': user.get('name', 'User'),
        'resume_filename': filename,
        'status': 'pending'
    }).execute()


    resume_file = request.files.get('resume')
    filename = resume_file.filename
    save_path = f'uploads/{filename}'
    resume_file.save(save_path)
    text = extract_pdf(save_path)
    analysis_result = resume_analyzer(text)
    return jsonify({"status": "ok", "message": "Resume uploaded! You will get your analyzed resume in 15 minutes."})

import os
import openai  # assuming Gemini API uses OpenAI-style requests

# Make sure your Gemini API key is set in environment variables
import google.generativeai as genai
genai.configure(api_key="AIzaSyDi_2WfpIW_QKpAZuK9Vaj_ToCg5dU6PJA")
model = genai.GenerativeModel("models/gemini-2.5-flash")
import fitz 
import json


@app.route('/latest-analysis', methods=['GET'])
def latest_analysis():
    """Fetch latest resume for logged-in user, analyze it, and return JSON"""
    user = session.get('user')
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    # Fetch latest resume for this user
    email = user.get('email')
    data = supabase.table('resumes')\
        .select('*')\
        .eq('user_email', email)\
        .order('created_at', desc=True)\
        .limit(1)\
        .execute()

    if not data.data or len(data.data) == 0:
        return jsonify({"error": "No resume found"}), 404

    resume_entry = data.data[0]
    resume_filename = resume_entry.get('resume_filename')

    # Verify if file exists
    file_path = os.path.join('uploads', resume_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Resume file missing on server"}), 404

    # Extract text and analyze
    try:
        text = extract_pdf(file_path)
        result = resume_analyzer(text)
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
      "summary": "<brief summary>"
    }}

    Resume:
    {resume_text}
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # 🔧 Fix: Remove markdown fences like ```json ... ```
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try a fallback cleaning method
            cleaned = content.replace("```", "").replace("json", "").strip()
            try:

                result = json.loads(cleaned)
                print(result)
            except:
                result = {"error": "Invalid JSON from Gemini", "raw": content}

        return result

    except Exception as e:
        return {"error": str(e)}
@app.route('/dash')
def serve_dashboard():
    return send_from_directory('static', 'test.html')

if __name__ == '__main__':
    app.run(debug=True)
