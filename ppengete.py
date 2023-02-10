codes_as_string = '''Acehnese (Arabic script)	ace_Arab
Acehnese (Latin script)	ace_Latn
Mesopotamian Arabic	acm_Arab
Ta’izzi-Adeni Arabic	acq_Arab
Tunisian Arabic	aeb_Arab
Afrikaans	afr_Latn'''

codes_as_string = codes_as_string.split('\n')

flores_codes = {}
flores_lan = {}
for code in codes_as_string:
    lang, lang_code = code.split('\t')
    flores_lan[lang] = lang
    flores_codes[lang_code] = lang_code

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
    return jsonify(list(flores_lan.keys()))