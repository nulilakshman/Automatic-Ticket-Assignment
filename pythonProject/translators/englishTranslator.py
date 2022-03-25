import langid
from google_trans_new import google_translator


class EnglishTranslator:

    def fn_decode_to_ascii(self, text):
        text = text.encode().decode('utf-8').encode('ascii', 'ignore')
        return text.decode("utf-8")

    def classifylanguage(self, text):
        text =  self.fn_decode_to_ascii(text)
        langids = langid.classify(text)
        print(langids)
        return langids[0]

    def translate(self, text):
        language = self.classifylanguage(text)
        if language == 'en':
            return text

        print('Converting to english')
        translator = google_translator()
        print('step-1')
        print(language)
        translatedtext = translator.translate(text, lang_src=language, lang_tgt='en')
        print('Converted to english')
        print(translatedtext)
        return translatedtext