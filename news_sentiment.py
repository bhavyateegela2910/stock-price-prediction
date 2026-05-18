from textblob import TextBlob

news = [

    "Apple stock is growing strongly",

    "Tesla faces huge losses",

    "Microsoft shows excellent results",

    "Amazon market performance is weak"
]

print("\nNews Sentiment Analysis\n")

for article in news:

    sentiment = TextBlob(article)

    polarity = sentiment.sentiment.polarity

    if polarity > 0:
        result = "Positive"

    elif polarity < 0:
        result = "Negative"

    else:
        result = "Neutral"

    print("News:", article)
    print("Sentiment:", result)
    print()