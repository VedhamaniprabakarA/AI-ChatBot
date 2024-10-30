from django.http import JsonResponse
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
        outputs = model.generate(inputs, max_length=25, num_return_sequences=1)

    # Decode the generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return Response({"response": response_text})


@api_view(["GET"])
def csrf_token_view(request):
    return JsonResponse({"csrfToken": request.META.get("CSRF_COOKIE")})
