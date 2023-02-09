from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes
from flask_cors import CORS
app = Flask(__name__)

model_dict = {}
CORS(app)

TASK = "translation"
CKPT = "facebook/nllb-200-distilled-600M"

model = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
tokenizer = AutoTokenizer.from_pretrained(CKPT)

device = 0 if torch.cuda.is_available() else -1


def translate_text(text, src_lang, tgt_lang, max_length=400):
    """
    Translate the text from source lang to target lang
    """
    translation_pipeline = pipeline(TASK,
                                    model=model,
                                    tokenizer=tokenizer,
                                    src_lang=src_lang,
                                    tgt_lang=tgt_lang,
                                    max_length=max_length,
                                    device=device)

    result = translation_pipeline(text)
    return result[0]['translation_text']


@app.route("/translate", methods=["POST"])
def handle_translate():
    data = request.get_json()
    source = data.get("source")
    target = data.get("target")
    text = data.get("text")
    print(source, target, text)
    result = translate_text(text, source, target)
    return jsonify(result)

@app.route("/languages", methods=["GET"])
def getlanguages():
    return jsonify(list(flores_codes.keys()))

if __name__ == "__main__":
    app.run(debug=True)
