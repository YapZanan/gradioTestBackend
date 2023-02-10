from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask_cors import CORS
app = Flask(__name__)

model_dict = {}
CORS(app)

TASK = "generation"
CKPT = "EleutherAI/gpt-neo-2.7B"

model = AutoModelForCausalLM.from_pretrained(CKPT)
tokenizer = AutoTokenizer.from_pretrained(CKPT)

device = 0 if torch.cuda.is_available() else -1


def generator_text(text, max_length=30):

    generator_pipeline = pipeline(TASK, model=model, tokenizer=tokenizer, max_length=max_length, device=device)

    result = generator_pipeline(text)
    return result[0]['translation_text']


@app.route("/generasi", methods=["POST"])
def handle_translate():
    data = request.get_json()
    text = data.get("text")
    print(text)
    result = generator_text(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run()