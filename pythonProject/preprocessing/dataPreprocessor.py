import string
import re
import nltk
#nltk.download()
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


class DataPreprocessor:

    def processStemming(self, text):
        stemmer = SnowballStemmer("english")
        text = [stemmer.stem(word) for word in text.split()]
        return " ".join(text)

    def processLemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)

    def cleaning_data(self, text):
        text = text.lower()
        text = re.sub(r"received from:", ' ', text)
        text = re.sub(r"from:", ' ', text)
        text = re.sub(r"to:", ' ', text)
        text = re.sub(r"subject:", ' ', text)
        text = re.sub(r"sent:", ' ', text)
        text = re.sub(r"ic:", ' ', text)
        text = re.sub(r"cc:", ' ', text)
        text = re.sub(r"bcc:", ' ', text)
        text = re.sub(r"_x000D_", ' ', text)
        text = ' '.join(re.sub("[^\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ",
                               text).split())  # Remove unreadable characters  (also extra spaces)
        # Remove email
        text = re.sub(r'\S*@\S*\s?', ' ', text)

        # Remove underscore
        desc = re.sub(r'_', ' ', text)

        # Remove HTML tags
        text = re.sub(re.compile('<.*?>'), ' ', text)

        text = re.sub(r'\&\w*;', ' ', text)

        # Remove numbers
        text = re.sub(r'\d+', ' ', text)

        # Remove new line characters
        text = re.sub(r'\n', ' ', text)

        # Removing hyperlinks
        text = re.sub(r'http.+?:\/\/.*\/\w*', ' ', text)

        # Removing punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        text = " ".join(w.translate(table) for w in word_tokenize(text))

        # Removing stopwords
        stop_words = stopwords.words('english')
        text = " ".join(word for word in text.split(' ') if word not in stop_words)

        # Stemming
        text = self.processStemming(text)

        # Lemmatization
        text = self.processLemmatization(text)

        # Convert multiple spaces to a single space
        text = re.sub(r' {2,}', " ", text, flags=re.MULTILINE)

        # Remove pre/post spaces
        text = text.strip()

        return text