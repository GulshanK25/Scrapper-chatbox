
from scrapper.crawler import iter_website_pages

url = "https://docs.python.org/3/tutorial/"

count = 0

for page in iter_website_pages(
    url,
    max_pages=30,
    delay=0.5,
):
    count += 1

    print(
        f"[{count}] {page['url']} "
        f"({len(page['text'])} characters)"
    )

print(f"\nTotal pages: {count}")
