from django.shortcuts import render
from ml.predict import predict_spam

def home(request):
    result = None
    error = None

    if request.method == "POST":
        try:
            message = request.POST.get("email_text", "").strip()

            if not message:
                error = "Please enter a message."
            else:
                result = predict_spam(message)

        except Exception as e:
            error = str(e)
            print("Prediction error:", e)  # IMPORTANT for Render logs

    return render(request, "home.html", {
        "result": result,
        "error": error
    })
