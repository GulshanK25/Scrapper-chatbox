
from scrapper.crawler import iter_website_pages

URL = "https://docs.python.org/3/tutorial/"

for mode in ("section", "website"):

    print(f"\n{'=' * 50}")
    print(f"CRAWL MODE: {mode}")
    print(f"{'=' * 50}")

    count = 0
    sections = set()

    for page in iter_website_pages(
        URL,
        max_pages=50,
        delay=0.5,
        scope=mode,
    ):
        count += 1

        path = page["url"].split(
            "docs.python.org", 1
        )[-1]

        parts = path.strip("/").split("/")

        section = (
            parts[1]
            if len(parts) > 1
            else "root"
        )

        sections.add(section)

        print(f"[{count}] {page['url']}")

    print(f"\nTotal pages: {count}")
    print(f"Sections discovered: {sorted(sections)}")
