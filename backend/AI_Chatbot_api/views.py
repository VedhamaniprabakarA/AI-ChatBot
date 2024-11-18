# backend/AI_Chatbot_api/views.py
'''from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer (update paths as needed)
MODEL_PATH = "/home/praba/Desktop/AI_ChatBot/backend/AI_Chatbot_api/gpt2-harry-potter-qa"
TOKENIZER_PATH = "/home/praba/Desktop/AI_ChatBot/backend/AI_Chatbot_api/gpt2-harry-potter-qa"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


@api_view(["POST"])
def chat_bot_response(request):
    question = request.data.get("question")
    if not question:
        return Response({"error": "Question not provided"}, status=400)

    # Tokenize the input question
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate response from the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # Decode the generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return Response({"response": response_text})


@api_view(["GET"])
def csrf_token_view(request):
    return JsonResponse({"csrfToken": request.META.get("CSRF_COOKIE")})'''


from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
import ollama

# Define a maximum length for the question (you can adjust this as needed)
MAX_LEN = 500  # Maximum length for the question

@api_view(["POST"])
def chat_bot_response(request):
    question = request.data.get("question")
    if not question:
        return Response({"error": "Question not provided"}, status=400)

    # Check if the question exceeds the maximum length
    if len(question) > MAX_LEN:
        return Response({"error": f"Question exceeds the maximum length of {MAX_LEN} characters."}, status=400)

    try:
        # Generate response from the ollama model
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': question}]
        )
        response_text = response.get('message', {}).get('content', 'No response from model.')
        return Response({"response": response_text})
    except Exception as e:
        return Response({"error": f"An error occurred: {str(e)}"}, status=500)


@api_view(["GET"])
def csrf_token_view(request):
    return JsonResponse({"csrfToken": request.META.get("CSRF_COOKIE")})


