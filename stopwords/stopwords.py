import pandas as pd
# https://github.com/Alir3z4/stop-words/blob/master/spanish.txt
# https://github.com/stopwords-iso/stopwords-es/blob/master/raw/stop-words-spanish.txt
if __name__ == "__main__":
    df = pd.read_csv("stopwords/spanish.txt")
    pd.DataFrame(data=list(set(df.word.tolist())), columns=[
        "word"]).to_csv("stopwords/stopwords.csv", index=False)
