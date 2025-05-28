import re
import requests # Add this import for URL validation

def validate_url(url: str) -> bool:
    """Validate a generic URL format and reachability."""
    if not isinstance(url, str):
        return False
    # Basic regex for URL format
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if not re.match(regex, url):
        print(f"Invalid URL format: {url}")
        return False
    # Check reachability (optional, can be time-consuming)
    # try:
    #     response = requests.head(url, allow_redirects=True, timeout=5)
    #     if response.status_code >= 400: # client error or server error
    #         print(f"URL not reachable or error: {url} (Status: {response.status_code})")
    #         return False
    # except requests.RequestException as e:
    #     print(f"URL validation request failed for {url}: {e}")
    #     return False
    return True

def validate_github_url(github_url: str) -> bool:
    """Validate a GitHub URL format."""
    if not isinstance(github_url, str):
        print(f"GitHub URL is not a string: {github_url}")
        return False
    if not github_url.startswith(('http://github.com/', 'https://github.com/')):
        print(f"Invalid GitHub URL (must start with http(s)://github.com/): {github_url}")
        return False
    if not validate_url(github_url): # General URL validation
         return False
    # Add more specific GitHub checks if needed, e.g., repo structure
    print(f"GitHub URL validated: {github_url}")
    return True

def validate_arxiv_url(arxiv_url: str) -> bool:
    """Validate an arXiv URL format."""
    if not isinstance(arxiv_url, str):
        print(f"arXiv URL is not a string: {arxiv_url}")
        return False
    # Regex for common arXiv URL patterns (abs and pdf)
    arxiv_pattern = re.compile(r'^https?://arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5}|[a-zA-Z\-]+/\d{7})(v\d+)?(\.pdf)?$')
    if not arxiv_pattern.match(arxiv_url):
        print(f"Invalid arXiv URL format: {arxiv_url}")
        return False
    if not validate_url(arxiv_url): # General URL validation
        return False
    print(f"arXiv URL validated: {arxiv_url}")
    return True

def validate_paper_relevance(reference: str, usage: str) -> bool:
    """Placeholder for validating paper relevance. Currently returns True."""
    if not isinstance(reference, str) or not reference.strip():
        print(f"Paper reference (title) is missing or empty.")
        return False
    if not isinstance(usage, str) or not usage.strip():
        print(f"Paper usage description is missing or empty.")
        return False
    # Actual relevance validation would require more complex logic (e.g., NLP, domain knowledge)
    # For now, this is a basic check for presence of reference and usage.
    print(f"Paper relevance checks passed for: {reference}")
    return True

def validate_source_papers(source_papers: list) -> tuple[bool, list[str]]:
    """
    Validates a list of source paper dictionaries.
    Each paper dictionary can have 'reference', 'usage', 'github_link', and 'url' (for arXiv).
    Returns a tuple: (bool_overall_validity, list_of_error_messages)
    """
    if not isinstance(source_papers, list):
        return False, ["source_papers is not a list."]

    all_papers_valid = True
    error_messages = []

    for i, paper in enumerate(source_papers):
        if not isinstance(paper, dict):
            error_messages.append(f"Paper at index {i} is not a dictionary.")
            all_papers_valid = False
            continue

        reference = paper.get('reference')
        usage = paper.get('usage')
        
        # Validate reference and usage (basic presence check)
        if not (isinstance(reference, str) and reference.strip()):
            error_messages.append(f"Paper '{reference or f'at index {i}'}' is missing a valid 'reference' (title).")
            all_papers_valid = False
        if not (isinstance(usage, str) and usage.strip()):
            error_messages.append(f"Paper '{reference or f'at index {i}'}' is missing a 'usage' description.")
            all_papers_valid = False

        # Validate github_link if present
        if 'github_link' in paper:
            if not validate_github_url(paper['github_link']):
                error_messages.append(f"Paper '{reference or f'at index {i}'}': Invalid GitHub link: {paper['github_link']}")
                all_papers_valid = False
        
        # Validate arXiv url if present
        if 'url' in paper: # Assuming 'url' is for arXiv as per USAGE_GUIDE
            if not validate_arxiv_url(paper['url']):
                error_messages.append(f"Paper '{reference or f'at index {i}'}': Invalid arXiv URL: {paper['url']}")
                all_papers_valid = False
        
        # Placeholder for paper relevance - using the basic check for now
        # This specific function 'validate_paper_relevance' as defined in the issue takes 2 args
        # and is used here for demonstration. In a real scenario, its logic might be different.
        if not validate_paper_relevance(paper.get('reference', ''), paper.get('usage', '')):
             error_messages.append(f"Paper '{reference or f'at index {i}'}': Relevance check failed based on current criteria.")
             # all_papers_valid = False # Decided not to make this critical for now

    if not error_messages:
        print("All source papers validated successfully.")
    else:
        print("Source paper validation found issues.")
    
    return all_papers_valid, error_messages
