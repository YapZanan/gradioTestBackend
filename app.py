from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes
from flask_cors import CORS
app = Flask(__name__)

model_dict = {}
CORS(app, origins=["http://localhost:3000"])


def load_models():
    # build model and tokenizer
    model_name_dict = {
        'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M',
        # 'nllb-1.3B': 'facebook/nllb-200-1.3B',
        # 'nllb-distilled-1.3B': 'facebook/nllb-200-distilled-1.3B',
        # 'nllb-3.3B': 'facebook/nllb-200-3.3B',
    }

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer


def translation(source, target, text):
    source = flores_codes[source]
    target = flores_codes[target]

    model_name = 'nllb-distilled-600M'
    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target)
    output = translator(text, max_length=400)

    full_output = output
    output = output[0]['translation_text']
    result = {
        'source': source,
        'target': target,
        'result': output,
        'full_output': full_output
    }
    return result


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    source = data.get("source")
    target = data.get("target")
    text = data.get("text")
    print(source, target, text)
    result = translation(source, target, text)
    return jsonify(result)


@app.route("/languages", methods=["GET"])
def getlanguages():
    return jsonify(list(flores_codes.keys()))


if __name__ == '__main__':
    print('\tinit models')
    load_models()
    app.run()
