import re
import html
import contractions

def fix_html(x: str) -> str:
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))

def no_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text) # no emoji    

def rm_one(text):
    # len(w) == 1 -> remove
    rst = ""
    for w in text:
        if len(w) > 1:
            rst += w + " "
    return rst

def clean_text(text):
    text = fix_html(text)
    text = no_emoji(text)
    text = text.lower()
    text = contractions.fix(text) # I'd -> I would
    text = re.sub(r'"', '', text)
    text = re.sub(r":", "", text)
    text = re.sub(r"\n", "", text)
    # text = re.sub(r'(!)1+', '', text) # !!! -> NA
    # text = re.sub(r'(?)1+', '', text) # ?? -> NA
    text = re.sub(r"([/#\n])", r" \1 ", text) # Add spaces around / and # in `t`. \n
    text = re.sub(" {2,}", " ", text) # Remove multiple spaces in `t`.
    text = re.sub(r"(\n(\s)*){2,}", "\n", text) # multi new lines -> 1 new line
    text = text.strip() # remove any leading/trailing characters (default : space)
    return text