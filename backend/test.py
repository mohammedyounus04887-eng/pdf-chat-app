from google import genai

client = genai.Client(api_key="AIzaSyDGtw3RX2sML4J5wdOUQ6xXQ9rd6KWp6C4")

models = client.models.list()

for m in models:
    print(m.name)