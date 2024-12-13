import os
import logging
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Confluence API
CONFLUENCE_URL = os.environ['CONFLUENCE_URL']
SESSION_COOKIE = os.environ['CONFLUENCE_SESSION_COOKIE']  # Example: "JSESSIONID=ABC123DEF456"

# Initialize HTTP session
logging.info("Initializing HTTP session for Confluence...")
session = requests.Session()
session.headers.update({
    'Cookie': SESSION_COOKIE,
    'Accept': 'application/json',
    'Content-Type': 'application/json'
})
logging.info("HTTP session initialized successfully.")

# NLP model for query matching
logging.info("Loading NLP model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("NLP model loaded successfully.")

def preprocess_content(text):
    """Clean and preprocess the content"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text

def fetch_page_content(page_id):
    """Fetch and parse content from Confluence page using session"""
    try:
        logging.info(f"Fetching content for page ID: {page_id}")
        url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}?expand=body.storage"
        response = session.get(url)
        response.raise_for_status()

        page = response.json()
        logging.info("Content fetched successfully.")
        html_content = page['body']['storage']['value']

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract plain text
        plain_text = soup.get_text(separator=' ')

        # Replace problematic Unicode characters
        replacements = {
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
        }

        for char, replacement in replacements.items():
            plain_text = plain_text.replace(char, replacement)

        return preprocess_content(plain_text)
    except Exception as e:
        logging.error(f"Error fetching page content: {e}")
        return None

def find_best_match(query, content, threshold=0.3):
    """Find the best matching response from content"""
    try:
        # Split content into meaningful sections
        content_sections = content.split("\n\n")
        content_sections = [s.strip() for s in content_sections if s.strip()]

        if not content_sections:
            return "No content sections found to match against."

        # Encode query and content
        query_embedding = model.encode(query, convert_to_tensor=True)
        content_embeddings = model.encode(content_sections, convert_to_tensor=True)

        # Calculate similarity scores
        scores = util.pytorch_cos_sim(query_embedding, content_embeddings)
        best_match_index = scores.argmax()
        best_match_score = scores[best_match_index].item()

        # Return match only if it meets threshold
        if best_match_score >= threshold:
            return content_sections[best_match_index]
        return "No sufficiently relevant content found. Please try rephrasing your question."
    except Exception as e:
        logging.error(f"Error matching query: {e}")
        return "No relevant content found."

def handle_user_query(page_id, query):
    """Process user query and return response"""
    content = fetch_page_content(page_id)
    if content:
        logging.info("Page content retrieved successfully. Proceeding to match query.")
        best_match = find_best_match(query, content)
        return best_match
    else:
        logging.warning("Unable to fetch page content. Check connection or permissions.")
        return "Unable to fetch page content. Please check your connection and try again."

def print_welcome_message():
    """Print welcome message with instructions"""
    print("\n=== Confluence Chatbot ===")
    print("Ask questions about the content in your Confluence page.")
    print("Type 'quit' to exit.")
    print("========================\n")

def main():
    PAGE_ID = '94116001'  # You can also move this to environment variables

    print_welcome_message()

    while True:
        try:
            user_input = input("\nYour question (or 'quit' to exit): ").strip()

            if user_input.lower() == 'quit':
                print("Thank you for using the chatbot. Goodbye!")
                break

            if not user_input:
                print("Please enter a question.")
                continue

            response = handle_user_query(PAGE_ID, user_input)
            print("\nChatbot Response:")
            print(response)

        except KeyboardInterrupt:
            print("\nExiting chatbot...")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
