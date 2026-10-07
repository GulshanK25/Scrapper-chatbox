from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "WebsiteRAGBot/1.0"
}

IGNORED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".css",
    ".js",
    ".xml",
    
)
IGNORED_PATHS = (
    "/genindex.html",
    "/py-modindex.html",
    "/search.html",
)


def normalize_url(url: str) -> str:
    """Remove URL fragments."""

    url, _ = urldefrag(url)

    return url


def is_valid_page(url: str) -> bool:
    """Return False for URLs that we don't want to scrape."""

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    path = parsed.path.lower()

    if path.endswith(IGNORED_EXTENSIONS):
        return False

    if path.endswith(IGNORED_PATHS):
        return False

    return True

def scrape_page(url: str) -> tuple[str, list[str]]:
    """Download a webpage and return its text and links."""

    response = requests.get(
        url,
        timeout=10,
        headers=HEADERS,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "text/html" not in content_type:
        return "", []

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that aren't useful knowledge.
    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    links = []

    for tag in soup.find_all("a", href=True):

        absolute_url = urljoin(
            url,
            tag["href"],
        )

        absolute_url = normalize_url(
            absolute_url
        )

        if is_valid_page(absolute_url):
            links.append(absolute_url)

    return text, links


def allowed_url(url: str, domain: str, base_path: str) -> bool:
    """
    Check whether the URL belongs to the same website
    and stays inside the starting section.
    """

    parsed = urlparse(url)

    if parsed.netloc != domain:
        return False

    return parsed.path.startswith(base_path)

def crawl_website(
    start_url: str,
    max_pages: int = 10,
):
    """Crawl pages belonging to one website."""

    start_url = normalize_url(start_url)

    domain = urlparse(start_url).netloc 
    
    base_path = urlparse(start_url).path
    
    if not base_path.endswith("/"):
         base_path = base_path.rsplit("/", 1)[0] + "/"

    pages_to_visit = [start_url]

    queued = {start_url}

    visited = set()

    pages = []

    while (
        pages_to_visit
        and len(visited) < max_pages
    ):

        url = pages_to_visit.pop(0)

        if url in visited:
            continue

        print(f"Scraping: {url}")

        try:

            text, links = scrape_page(url)

            visited.add(url)

            if text:

                pages.append(
                    {
                        "url": url,
                        "text": text,
                    }
                )

            for link in links:

                if (
                    allowed_url(link, domain, base_path)
                    and link not in visited
                    and link not in queued
                ):

                    pages_to_visit.append(link)
                    queued.add(link)

        except requests.RequestException as error:

            print(
                f"Could not scrape {url}: {error}"
            )

    return pages