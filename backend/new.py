import google.generativeai as genai
genai.configure(api_key="AIzaSyDi_2WfpIW_QKpAZuK9Vaj_ToCg5dU6PJA")

for m in genai.list_models():
    print(m.name)
