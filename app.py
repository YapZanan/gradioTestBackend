from flask import Flask, request, jsonify, abort
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_lan, flores_lan_to_codes
import openai
from flask_cors import CORS

openai.api_key = "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"

app = Flask(__name__)
CORS(app, resources={r'*': {'origin': ['https://generatorbahasa.vercel.app/']}})

@app.before_request
def before_request():
    if "PostmanRuntime" in request.headers.get('User-Agent', '') or "cURL" in request.headers.get('User-Agent', ''):
        abort(403)



model_dict = {}


TASK = "translation"
CKPT = "facebook/nllb-200-distilled-600M"

model = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
tokenizer = AutoTokenizer.from_pretrained(CKPT)

device = 0 if torch.cuda.is_available() else -1


def generator_text(prompt_text, max_tokens=100):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="form a story with this prompt, make it as complete as possible\n\n\
                Prompt: Once upon a time, there lived a blind man\nYou: Once upon a time, there lived a blind man who had never\n"
               "seen the sun. He asked a friend to tell him what it was like,\n\"It's like a brass plate,"
               "” his friend said. The blind man\n "
               "struck a brass plate with a stick and listened to the sound.\nEvery time he heard a similar sound, "
               "he thought it was the\nsun.\n "
               "His friend explained that “The sun is like a candle.” The\nblind man felt a candle with his hand. "
               "He believed it was the\nsame shape as the sun.\n "
               "Then his friend told him that the sun is like a great\nball\nof fire. Later that winter, whenever "
               "the blind man sat in front\n "
               "of a fire, he thought it was the sun\n"
               "The sun is really quite different from all these things;\n"
               "but the blind man did not know this because he could not see\nit.\n"
               "In the same way, the Truth is often hard to see.\nyou\ncannot see it when it is right in front of "
               "you,\nyou are just\nlike the blind man.\n\n\n "
               "Prompt:" + prompt_text + "\n"
                                         "You:",
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0
    )
    return response.choices[0].text


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

    src_lang = flores_lan_to_codes.get(source)
    tgt_lang = flores_lan_to_codes.get(target)

    result = translate_text(text, src_lang, tgt_lang)
    return jsonify(result)


@app.route("/languages", methods=["GET"])
def getlanguages():
    return jsonify(list(flores_lan.keys()))


@app.route("/generasi", methods=["POST"])
def handle_generasi():
    data = request.get_json()
    text = data.get("text")
    maxToken = data.get("max")
    print(text)
    result = generator_text(text, maxToken)
    return jsonify(result)


if __name__ == "__main__":
    app.run()
