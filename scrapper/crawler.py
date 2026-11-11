
from collections import deque
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    parse_qsl,
    urlencode,
)
from urllib.robotparser import RobotFileParser

import gzip
import logging
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


logger = logging.getLogger(__name__)

USER_AGENT = "WebsiteRAGBot/1.0"

HEADERS = {
    "User-Agent": USER_AGENT,
}

IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif",
    ".svg", ".webp", ".pdf", ".zip",
    ".css", ".js", ".xml", ".json",
    ".ico", ".mp4", ".mp3",
    ".woff", ".woff2", ".ttf", ".gz",
)

IGNORED_FILENAMES = {
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "copyright.html",
    "bugs.html",
    "improve-page-nojs.html",
}

TRACKING_PARAMS = {
    "fbclid", "gclid", "mc_cid", "mc_eid",
}


# =========================================================
# URL UTILITIES
# =========================================================

def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return url.split("#", 1)[0]

    try:
        port = parsed.port
    except ValueError:
        return url

    netloc = hostname

    if port and not (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        netloc = f"{hostname}:{port}"

    params = [
        (key, value)
        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_PARAMS
    ]

    return urlunparse((
        scheme,
        netloc,
        parsed.path or "/",
        "",
        urlencode(sorted(params)),
        "",
    ))


def is_valid_page(url: str) -> bool:
    parsed = urlparse(url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return False

    path = parsed.path.lower()
    filename = path.rsplit("/", 1)[-1]

    if filename in IGNORED_FILENAMES:
        return False

    return not path.endswith(IGNORED_EXTENSIONS)


def get_scope(start_url: str):
    parsed = urlparse(start_url)
    path = parsed.path or "/"

    if path.endswith("/"):
        base_path = path
    else:
        base_path = path.rsplit("/", 1)[0] + "/"

    return parsed.netloc.lower(), base_path


def get_website_root(start_url: str) -> str:
    """
    Determine a suitable website-wide root.

    Python documentation:
        /3/tutorial/ -> /3/

    Other websites:
        /blog/article/ -> /
    """

    parsed = urlparse(start_url)

    if parsed.hostname == "docs.python.org":
        parts = parsed.path.strip("/").split("/")

        if parts and parts[0] == "3":
            return "/3/"

    return "/"


def allowed_url(
    url: str,
    domain: str,
    base_path: str,
    scope: str = "section",
    website_root: str = "/",
) -> bool:
    parsed = urlparse(url)

    if not is_valid_page(url):
        return False

    if parsed.netloc.lower() != domain.lower():
        return False

    if scope == "section":
        return parsed.path.startswith(base_path)

    if scope == "website":
        return parsed.path.startswith(website_root)

    return False


def url_priority(url: str) -> int:
    """
    Lower score means higher crawling priority.
    """

    parsed = urlparse(url)

    if parsed.hostname != "docs.python.org":
        return 10

    path = parsed.path

    if path == "/3/glossary.html":
        return 0

    if path.startswith("/3/reference/"):
        return 1

    if path.startswith("/3/library/"):
        return 2

    if path.startswith("/3/tutorial/"):
        return 3

    return 10


# =========================================================
# ROBOTS.TXT
# =========================================================

def load_robots(session, start_url):
    parsed = urlparse(start_url)

    origin = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = origin + "/robots.txt"

    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = session.get(
            robots_url,
            timeout=10,
            allow_redirects=False,
        )

        if response.status_code == 404:
            parser.parse([])
            return parser, []

        if response.is_redirect:
            logger.warning(
                "robots.txt redirected: %s",
                robots_url,
            )
            return None, []

        response.raise_for_status()
        parser.parse(response.text.splitlines())

        sitemaps = []

        for line in response.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemaps.append(
                    line.split(":", 1)[1].strip()
                )

        return parser, sitemaps

    except requests.RequestException as error:
        logger.warning(
            "Could not verify robots.txt: %s",
            error,
        )
        return None, []


# =========================================================
# SITEMAP DISCOVERY
# =========================================================

def discover_sitemap_urls(
    session,
    sitemap_url,
    origin,
    robots,
    max_sitemaps=30,
    max_urls=10000,
):
    pending = deque([sitemap_url])
    visited = set()
    discovered = 0

    while pending and len(visited) < max_sitemaps:
        current = normalize_url(pending.popleft())

        if current in visited:
            continue

        if urlparse(current).netloc != urlparse(origin).netloc:
            continue

        if not robots.can_fetch(USER_AGENT, current):
            continue

        visited.add(current)

        try:
            response = session.get(
                current,
                timeout=15,
                allow_redirects=False,
            )

            if response.is_redirect:
                continue

            response.raise_for_status()

            if len(response.content) > 5_000_000:
                continue

            raw = response.content

            if current.endswith(".gz"):
                raw = gzip.decompress(raw)

                if len(raw) > 10_000_000:
                    continue

            root = ET.fromstring(raw)

        except (
            requests.RequestException,
            ET.ParseError,
            OSError,
            ValueError,
        ):
            continue

        root_type = root.tag.rsplit("}", 1)[-1]

        if root_type == "sitemapindex":
            for node in root:
                if node.tag.rsplit("}", 1)[-1] != "sitemap":
                    continue

                for child in node:
                    if (
                        child.tag.rsplit("}", 1)[-1] == "loc"
                        and child.text
                    ):
                        pending.append(child.text.strip())

        elif root_type == "urlset":
            for node in root:
                if node.tag.rsplit("}", 1)[-1] != "url":
                    continue

                for child in node:
                    if (
                        child.tag.rsplit("}", 1)[-1] == "loc"
                        and child.text
                    ):
                        yield normalize_url(
                            child.text.strip()
                        )

                        discovered += 1

                        if discovered >= max_urls:
                            return


# =========================================================
# STRUCTURED HTML EXTRACTION
# =========================================================

def extract_structured_content(soup):
    """
    Extract readable page content while preserving:

    - Headings
    - Paragraphs
    - Glossary terms and definitions
    - Lists
    - Syntax-highlighted code blocks

    Removes common navigation and UI elements.
    """

    main = (
        soup.select_one('[role="main"]')
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # Copy the selected content so the original
    # BeautifulSoup document remains unchanged.
    content = BeautifulSoup(
        str(main),
        "html.parser",
    )

    for element in content.select(
        "nav, footer, header, script, style, "
        "noscript, aside, .sidebar, .sphinxsidebar, "
        ".related, .toctree-wrapper, "
        ".headerlink, .viewcode-link, "
        ".mobile-nav, .search, "
        '[role="navigation"], '
        '[role="search"], '
        '.copybtn, .copybutton'
    ):
        element.decompose()

    blocks = []

    def clean_text(element):
        return element.get_text(
            " ",
            strip=True,
        )

    def add_block(text):
        text = text.strip()

        if text:
            blocks.append(text)

    def render_code(element):
        """
        Preserve original whitespace inside <pre>.

        Syntax highlighting commonly uses nested spans.
        An empty separator avoids adding unwanted
        spaces or line breaks between code tokens.
        """
        code = element.get_text(
            separator="",
            strip=False,
        ).strip("\n")

        if code.strip():
            add_block(
                "```\n" + code + "\n```"
            )

    def walk(element):
        """
        Recursively process content in document order.

        This preserves code inside definition lists
        instead of flattening entire definitions.
        """

        if not isinstance(element, Tag):
            return

        tag = element.name

        if tag in {
            "script", "style", "noscript",
        }:
            return

        if tag in {
            "h1", "h2", "h3",
            "h4", "h5", "h6",
        }:
            level = int(tag[1])
            heading = clean_text(element)

            if heading:
                add_block(
                    "#" * level + " " + heading
                )
            return

        if tag == "pre":
            render_code(element)
            return

        if tag == "dt":
            term = clean_text(element)

            if term:
                add_block("### " + term)
            return

        if tag == "p":
            paragraph = clean_text(element)
            add_block(paragraph)
            return

        if tag == "li":
            # Capture list item text without
            # flattening nested code blocks.
            direct_parts = []

            for child in element.children:
                if isinstance(child, Tag):
                    if child.name in {
                        "ul", "ol", "pre",
                        "div", "dl",
                    }:
                        continue

                    direct_parts.append(
                        clean_text(child)
                    )
                else:
                    direct_parts.append(str(child))

            item = " ".join(
                part.strip()
                for part in direct_parts
                if part.strip()
            )

            if item:
                add_block("- " + item)

            for child in element.children:
                if isinstance(child, Tag):
                    if child.name in {
                        "ul", "ol", "pre",
                        "div", "dl",
                    }:
                        walk(child)

            return

        if tag in {"dd", "div", "section", "article",
                   "main", "body", "dl", "ul", "ol"}:
            for child in element.children:
                walk(child)
            return

        # Generic container fallback.
        for child in element.children:
            walk(child)

    walk(content)

    return "\n\n".join(blocks)


# =========================================================
# PAGE FETCHING
# =========================================================

def fetch_page(session, url):
    """
    Fetch one HTML page and extract structured content.

    Returns:
        (text, links)
    """

    response = session.get(
        url,
        timeout=15,
        allow_redirects=False,
    )

    if response.is_redirect:
        return "", []

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "text/html" not in content_type:
        return "", []

    if len(response.content) > 5_000_000:
        return "", []

    soup = BeautifulSoup(
        response.content,
        "html.parser",
    )

    links = []

    # Discover links before cleaning the HTML.
    for tag in soup.find_all("a", href=True):
        link = normalize_url(
            urljoin(url, tag["href"])
        )

        if is_valid_page(link):
            links.append(link)

    text = extract_structured_content(soup)

    return text, links


def scrape_page(url: str):
    """
    Scrape one page without following links.
    """

    with requests.Session() as session:
        session.headers.update(HEADERS)

        return fetch_page(
            session,
            normalize_url(url),
        )


# =========================================================
# WEBSITE CRAWLING
# =========================================================

def iter_website_pages(
    start_url: str,
    max_pages: int = 100,
    delay: float = 0.5,
    scope: str = "section",
):
    """
    Crawl pages incrementally.

    section:
        Crawl within the starting section.

    website:
        Crawl within the website root.

    Uses URL prioritization and respects robots.txt.
    """

    if scope not in ("section", "website"):
        raise ValueError(
            "scope must be 'section' or 'website'"
        )

    if max_pages < 1 or delay < 0:
        raise ValueError("Invalid crawl settings")

    start_url = normalize_url(start_url)

    if not is_valid_page(start_url):
        raise ValueError("Invalid starting URL")

    domain, base_path = get_scope(start_url)
    website_root = get_website_root(start_url)

    parsed = urlparse(start_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    queues = {
        priority: deque()
        for priority in range(11)
    }

    queued = set()
    visited = set()

    queue_limit = max(1000, max_pages * 3)

    def enqueue(url):
        url = normalize_url(url)

        if url in queued or url in visited:
            return

        if not allowed_url(
            url,
            domain,
            base_path,
            scope,
            website_root,
        ):
            return

        if len(queued) >= queue_limit:
            return

        priority = url_priority(url)

        queues[priority].append(url)
        queued.add(url)

    enqueue(start_url)

    with requests.Session() as session:
        session.headers.update(HEADERS)

        robots, sitemaps = load_robots(
            session,
            start_url,
        )

        if robots is None:
            logger.warning(
                "Crawl stopped: robots.txt unavailable"
            )
            return

        sitemaps.append(origin + "/sitemap.xml")

        for sitemap in dict.fromkeys(sitemaps):
            for link in discover_sitemap_urls(
                session,
                sitemap,
                origin,
                robots,
                max_urls=queue_limit,
            ):
                enqueue(link)

            if len(queued) >= queue_limit:
                break

        pages_crawled = 0
        last_request = 0.0

        while pages_crawled < max_pages:
            url = None

            for priority in sorted(queues):
                if queues[priority]:
                    url = queues[priority].popleft()
                    break

            if url is None:
                break

            if url in visited:
                continue

            visited.add(url)

            if not robots.can_fetch(
                USER_AGENT,
                url,
            ):
                continue

            wait = delay - (
                time.monotonic() - last_request
            )

            if wait > 0:
                time.sleep(wait)

            last_request = time.monotonic()

            try:
                text, links = fetch_page(
                    session,
                    url,
                )

            except requests.RequestException as error:
                logger.warning(
                    "Failed to crawl %s: %s",
                    url,
                    error,
                )
                continue

            if text:
                pages_crawled += 1

                yield {
                    "url": url,
                    "text": text,
                }

            for link in links:
                enqueue(link)


def crawl_website(
    start_url: str,
    max_pages: int = 10,
    scope: str = "section",
):
    """
    Crawl a website and return the collected pages.
    """

    return list(
        iter_website_pages(
            start_url,
            max_pages=max_pages,
            scope=scope,
        )
    )
