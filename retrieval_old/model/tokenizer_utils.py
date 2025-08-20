import six
import unicodedata
import collections

def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return six.ensure_text(text, "utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return six.ensure_text(text, "utf-8", "ignore")
		elif isinstance(text, six.text_type):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text):
	"""Runs basic whitespace cleaning and splitting on a piece of text."""
	text = text.strip()
	if not text:
		return []
	tokens = text.split()
	return tokens


def _is_whitespace(char):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically control characters but we treat them
	# as whitespace since they are generally considered as such.
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False


def _is_control(char):
	"""Checks whether `chars` is a control character."""
	# These are technically control characters but we count them as whitespace
	# characters.
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat in ("Cc", "Cf"):
		return True
	return False


def _is_punctuation(char):
	### jwkim @ 2019-11-21
	return char == ' '


def load_vocab(vocab_file):
	"""Loads a vocabulary file into a dictionary."""
	vocab = collections.OrderedDict()
	with open(vocab_file, "r", encoding='utf-8') as reader:
		while True:
			token = convert_to_unicode(reader.readline())
			if not token:
				break

			### jwkim @ 2019-11-21
			if token.find('n_iters=') == 0 or token.find('max_length=') == 0:
				continue
			try:
				token = token.strip().split()[0]
			except IndexError as e:
				print("vocab [%s] file not has allowed token [%s]" % (vocab_file, token))
				continue
			if token not in vocab:
				vocab[token] = len(vocab)
	return vocab

# ------------

VOCAB_FILES_NAMES = {"vocab_file": "electra-base-v4-1400k-vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}

PRETRAINED_INIT_CONFIGURATION = {}

BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens