"""N-gram language identifier, from scratch.

Each supported language gets a character-trigram profile built from
small sample texts embedded in this module. A new text is classified
by the classic Cavnar & Trenchard "out-of-place" distance: trigrams
are ranked by frequency in each profile, and the distance between the
text's profile and a language profile is the sum of the absolute rank
differences (a trigram missing from a language profile costs the
maximum rank). The language with the smallest distance wins.

Supported languages: English (en), French (fr), Spanish (es) and
German (de). Only numpy and the standard library are used.
"""

import re

import numpy as np

# Language codes mapped to their full names.
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}

# Short distinctive sample texts used to build the trigram profiles.
# Each language has several natural sentences; accented characters are
# kept because they are strong signals (é/è in French, ñ/á in
# Spanish, ä/ö/ü/ß in German).
SAMPLE_TEXTS = {
    "en": [
        "The quick brown fox jumps over the lazy dog near the river bank in the early morning light.",
        "We are going to the market tomorrow to buy fresh bread, vegetables, and cheese for dinner.",
        "The children played in the garden all afternoon while their parents worked in the house.",
        "She read the newspaper every morning with a cup of tea before going to work in the city.",
        "The government announced new plans for schools and hospitals across the country this week.",
        "I think this is a very good idea and I hope we can finish the project before the end of the month.",
        "The train arrived late because of the heavy rain, but everyone reached home safely in the evening.",
        "He opened the old book and started reading the story about a young man who travelled around the world.",
        "Our team won the match yesterday after a wonderful goal in the last minute of the game.",
        "Please send me the report as soon as possible so I can check the numbers before the meeting.",
        "The weather today is warm and sunny, with a light wind blowing from the west across the fields.",
        "They lived in a small house at the end of the street, next to the church and the old school.",
    ],
    "fr": [
        "Le matin, je prends mon petit déjeuner avec du pain, du beurre et de la confiture avant de partir au travail.",
        "Les enfants jouent dans le jardin pendant que leurs parents préparent le repas dans la cuisine.",
        "Nous allons au marché demain pour acheter des légumes frais, du fromage et des fruits de saison.",
        "Il a lu le journal tous les matins en buvant un café dans un petit café près de la gare.",
        "Le gouvernement a annoncé de nouveaux projets pour les écoles et les hôpitaux dans tout le pays.",
        "Je pense que c'est une très bonne idée et j'espère que nous pourrons terminer ce travail avant la fin du mois.",
        "Le train est arrivé en retard à cause de la pluie, mais tout le monde est rentré chez soi sans problème.",
        "Elle a ouvert le vieux livre et a commencé à lire une histoire sur un jeune homme qui voyageait autour du monde.",
        "Notre équipe a gagné le match hier grâce à un magnifique but marqué dans les dernières minutes.",
        "Pouvez-vous m'envoyer le rapport dès que possible afin que je puisse vérifier les chiffres avant la réunion ?",
        "Le temps est beau aujourd'hui, le ciel est bleu et un léger vent souffle sur les champs.",
        "Ils habitent dans une petite maison au bout de la rue, à côté de l'église et de l'ancienne école.",
    ],
    "es": [
        "Por la mañana desayuno pan con mantequilla y mermelada antes de salir para el trabajo.",
        "Los niños juegan en el jardín mientras sus padres preparan la comida en la cocina.",
        "Mañana vamos al mercado a comprar verduras frescas, queso y fruta de temporada.",
        "Leía el periódico todas las mañanas tomando un café en un pequeño bar cerca de la estación.",
        "El gobierno anunció nuevos proyectos para las escuelas y los hospitales de todo el país.",
        "Creo que es una muy buena idea y espero que podamos terminar este trabajo antes de fin de mes.",
        "El tren llegó tarde por la lluvia, pero todos llegaron a casa sin ningún problema.",
        "Abrió el libro viejo y empezó a leer una historia sobre un joven que viajaba por todo el mundo.",
        "Nuestro equipo ganó el partido ayer con un gol magnífico en los últimos minutos del juego.",
        "Por favor, envíame el informe lo antes posible para revisar las cifras antes de la reunión.",
        "Hoy hace buen tiempo, el cielo está despejado y sopla una brisa ligera sobre los campos.",
        "Viven en una casa pequeña al final de la calle, junto a la iglesia y la escuela antigua.",
    ],
    "de": [
        "Am Morgen esse ich zum Frühstück Brot mit Butter und Marmelade, bevor ich zur Arbeit gehe.",
        "Die Kinder spielen im Garten, während ihre Eltern in der Küche das Essen vorbereiten.",
        "Morgen gehen wir auf den Markt, um frisches Gemüse, Käse und Obst der Saison zu kaufen.",
        "Er las jeden Morgen die Zeitung und trank einen Kaffee in einem kleinen Café in der Nähe des Bahnhofs.",
        "Die Regierung hat neue Pläne für Schulen und Krankenhäuser im ganzen Land angekündigt.",
        "Ich finde das eine sehr gute Idee und hoffe, dass wir die Arbeit vor Ende des Monats beenden können.",
        "Der Zug kam wegen des starken Regens zu spät an, aber alle sind am Abend sicher zu Hause angekommen.",
        "Er öffnete das alte Buch und begann eine Geschichte über einen jungen Mann zu lesen, der um die Welt reiste.",
        "Unsere Mannschaft hat das Spiel gestern nach einem wunderbaren Tor in der letzten Minute gewonnen.",
        "Bitte schicken Sie mir den Bericht so schnell wie möglich, damit ich die Zahlen vor dem Treffen prüfen kann.",
        "Das Wetter ist heute warm und sonnig, und ein leichter Wind weht über die Felder.",
        "Sie wohnen in einem kleinen Haus am Ende der Straße, neben der Kirche und der alten Schule.",
    ],
}

# Keep lowercase Latin letters (including accented ones) and spaces;
# everything else becomes a space so punctuation and digits do not
# create spurious trigrams.
_NON_LETTER_RE = re.compile(r"[^a-zà-ÿ ]")
_MULTI_SPACE_RE = re.compile(r" +")

DEFAULT_TOP_N = 300
DEFAULT_MIN_LENGTH = 10


def normalize(text):
    """Lowercase text and keep only letters and single spaces.

    Raises TypeError for non-strings and ValueError for empty or
    whitespace-only input.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    cleaned = _NON_LETTER_RE.sub(" ", text.lower())
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ValueError("text must not be empty or contain only punctuation/digits")
    return cleaned


def trigrams(text):
    """Return the list of character trigrams of a normalized text.

    The text is padded with one space on each side so word-boundary
    trigrams (e.g. " th") are captured as well.
    """
    padded = f" {normalize(text)} "
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


def trigram_profile(text, top_n=DEFAULT_TOP_N):
    """Return the top-`top_n` trigrams of `text`, most frequent first.

    Ties are broken alphabetically so the ranking is deterministic.
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    counts = {}
    for tri in trigrams(text):
        counts[tri] = counts.get(tri, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [tri for tri, _ in ranked[:top_n]]


def profile_distance(text_profile, language_profile, top_n=DEFAULT_TOP_N):
    """Cavnar & Trenchard out-of-place distance between two profiles.

    Both arguments are ranked trigram lists (most frequent first).
    A trigram missing from the language profile costs `top_n`
    (the maximum rank). Lower is closer.
    """
    lang_rank = {tri: rank for rank, tri in enumerate(language_profile)}
    total = 0
    for rank, tri in enumerate(text_profile):
        if tri in lang_rank:
            total += abs(rank - lang_rank[tri])
        else:
            total += top_n
    return total


class LanguageDetector:
    """Character-trigram language identifier.

    Parameters
    ----------
    top_n : int
        How many of the most frequent trigrams make up each profile.
        Must be a positive integer.
    min_length : int
        Minimum number of alphabetic characters a text needs before
        it is classified; shorter texts are reported as "unknown".
        Must be a positive integer.

    Behaviour
    ---------
    Empty or whitespace-only input raises ValueError. Text with
    fewer than `min_length` letters returns "unknown" from
    `classify()` instead of a guess, because a couple of words carry
    too little trigram evidence to trust.
    """

    def __init__(self, top_n=DEFAULT_TOP_N, min_length=DEFAULT_MIN_LENGTH):
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if not isinstance(min_length, int) or isinstance(min_length, bool) or min_length <= 0:
            raise ValueError("min_length must be a positive integer")
        self.top_n = top_n
        self.min_length = min_length
        self.languages = []
        self._profiles = {}
        self.fit(SAMPLE_TEXTS)

    def fit(self, training_data):
        """(Re)build the trigram profiles from {language_code: [texts]}.

        Language codes must be among "en", "fr", "es", "de". Each
        language needs at least one non-empty string. Returns self.
        """
        if not isinstance(training_data, dict):
            raise TypeError(
                f"training_data must be a dict of language code to list of texts, "
                f"got {type(training_data).__name__}"
            )
        if not training_data:
            raise ValueError("training_data must not be empty")
        for code, texts in training_data.items():
            if code not in LANGUAGES:
                raise ValueError(
                    f"unsupported language code {code!r}; supported codes are "
                    f"{sorted(LANGUAGES)}"
                )
            if not isinstance(texts, list) or not texts:
                raise ValueError(f"training texts for {code!r} must be a non-empty list")
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"training texts for {code!r} must be non-empty strings"
                    )

        profiles = {}
        for code, texts in training_data.items():
            joined = " ".join(texts)
            profiles[code] = trigram_profile(joined, top_n=self.top_n)
        self._profiles = profiles
        self.languages = sorted(profiles)
        return self

    def _require_profiles(self):
        if not self._profiles:
            raise ValueError("LanguageDetector has no profiles; call fit() first")

    @staticmethod
    def _letter_count(text):
        return sum(1 for ch in text if ch.isalpha())

    def distances(self, text):
        """Out-of-place distance from `text` to each language profile.

        Returns {language_code: distance}; lower means closer.
        Raises TypeError for non-strings and ValueError for empty
        input or text too short to form a trigram.
        """
        self._require_profiles()
        normalized = normalize(text)
        if len(normalized) < 3:
            raise ValueError(
                "text is too short to form a trigram; "
                "provide at least 3 alphabetic characters"
            )
        text_profile = trigram_profile(normalized, top_n=self.top_n)
        return {
            code: profile_distance(text_profile, self._profiles[code], top_n=self.top_n)
            for code in self.languages
        }

    def scores(self, text):
        """Per-language similarity scores for `text`, summing to 1.

        A softmax over the negative distances (temperature 1000):
        higher means more likely. The argmax always agrees with
        `classify()` for texts long enough to classify.
        """
        dists = self.distances(text)
        codes = sorted(dists)
        values = np.array([dists[code] for code in codes], dtype=float)
        shifted = -(values - values.min()) / 1000.0
        weights = np.exp(shifted)
        weights = weights / weights.sum()
        return {code: float(weight) for code, weight in zip(codes, weights)}

    def classify(self, text):
        """Return the language code ("en"/"fr"/"es"/"de") of `text`.

        Returns "unknown" when the text has fewer than `min_length`
        alphabetic characters (too little evidence to trust a guess).
        Raises TypeError for non-strings and ValueError for empty or
        whitespace-only input.
        """
        self._require_profiles()
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        if self._letter_count(text) < self.min_length:
            return "unknown"
        dists = self.distances(text)
        return min(dists, key=dists.get)

    def language_name(self, code):
        """Full name ("English", ...) for a language code, or "unknown"."""
        if not isinstance(code, str):
            raise TypeError(f"code must be a string, got {type(code).__name__}")
        if code == "unknown":
            return "unknown"
        if code not in LANGUAGES:
            raise ValueError(
                f"unsupported language code {code!r}; supported codes are "
                f"{sorted(LANGUAGES)}"
            )
        return LANGUAGES[code]


def detect_language(text):
    """Classify `text` with a default detector; see LanguageDetector."""
    return LanguageDetector().classify(text)
