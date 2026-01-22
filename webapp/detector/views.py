from django.shortcuts import render
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'ml'))

from ml.predict import predict_spam

# Create your views here.
def home(request):
    result = None

    if request.method == "POST":
        email_text = request.POST.get("email_text")

        if email_text:
            result = predict_spam(email_text)
            
    return render(request, 'home.html', {"result": result})