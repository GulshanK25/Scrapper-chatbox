
import requests
from scrapper.crawler import fetch_page

URL = "https://docs.python.org/3/library/functions.html"

session = requests.Session()

try:
    response = session.get(URL, timeout=20)

    print("HTTP status:", response.status_code)
    print("Final URL:", response.url)
    print("Response length:", len(response.text))
    print("Contains classmethod:", "classmethod" in response.text.lower())

    print("\nTesting crawler extraction...")

    result = fetch_page(session, URL)

    print("Return type:", type(result))

    if isinstance(result, tuple):
        text, links = result

        print("Extracted text length:", len(text))
        print("Number of links:", len(links))
        print("Extracted text preview:", repr(text[:500]))

finally:
    session.close()
