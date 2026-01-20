import argparse
import os
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fetch the closest Wikipedia article for a given topic"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Topic to search on Wikipedia"
    )
    return parser.parse_args()


def get_wikipedia_link(query):
    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "SERPAPI_API_KEY not found. Please set it as an environment variable."
        )

    params = {
        "engine": "google",
        "q": f"{query} Wikipedia",
        "api_key": api_key,
        "num": 5
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    for result in results.get("organic_results", []):
        link = result.get("link", "")
        if "wikipedia.org/wiki/" in link:
            return link

    return None


def scrape_wikipedia_text(wiki_url):
    """
    Scrape clean text from a Wikipedia article.
    """
    headers = {
        "User-Agent": "RAG-Voice-Chatbot/1.0 (Educational Project)"
    }

    response = requests.get(wiki_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p")

    article_text = []

    for para in paragraphs:
        text = para.get_text().strip()
        if text:
            article_text.append(text)

    return "\n\n".join(article_text)


def save_text_to_file(text, file_path):
    """
    Save text to a .txt file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    args = parse_arguments()
    query = args.query

    print(f"Searching Wikipedia article for topic: {query}")

    wiki_link = get_wikipedia_link(query)

    if not wiki_link:
        print("No Wikipedia article found.")
        return

    print(f"Wikipedia article found:\n{wiki_link}")

    article_text = scrape_wikipedia_text(wiki_link)

    output_file = "data/wikipedia.txt"
    save_text_to_file(article_text, output_file)

    print(f"\nWikipedia article text successfully saved to: {output_file}")



if __name__ == "__main__":
    main()


