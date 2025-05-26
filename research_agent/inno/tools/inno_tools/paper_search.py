import arxiv
from typing import Dict, Any, Optional, List # Added List
import re # Added re

def get_arxiv_paper_meta(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    get the meta data of the arxiv paper
    
    Args:
        arxiv_id: the arxiv id of the paper, can be the full url or the id (e.g. '2305.02759' or '2305.02759v4')
    
    Returns:
        a dictionary containing the meta information of the paper, return None if the retrieval fails
    """
    try:
        # extract the id from the url
        if 'arxiv.org' in arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]
        
        # remove the version number
        base_id = arxiv_id.split('v')[0]
        
        # create the client
        client = arxiv.Client()
        
        # search the paper
        search = arxiv.Search(
            id_list=[base_id],
            max_results=1
        )
        
        # get the result
        paper = next(client.results(search))
        # print(paper)
        
        # build the return data
        meta_data = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'abstract': paper.summary,
            'categories': paper.categories,
            'published': paper.published,
            'updated': paper.updated,
            'doi': paper.doi,
            'pdf_url': paper.pdf_url,
            'primary_category': paper.primary_category,
            'comment': paper.comment,
            'journal_ref': paper.journal_ref,
            'version': paper.entry_id.split('v')[-1] if 'v' in paper.entry_id else '1'
        }
        
        return meta_data
        
    except Exception as e:
        print(f"Error fetching arxiv paper meta: {str(e)}")
        return None

# usage example
if __name__ == "__main__":
    # paper_url = "http://arxiv.org/abs/2305.02759v4"
    # meta = get_arxiv_paper_meta(paper_url)
    # print(meta['published'])
    # if meta:
    #     print(f"Title: {meta['title']}")
    #     print(f"Authors: {', '.join(meta['authors'])}")
    #     print(f"Abstract: {meta['abstract'][:200]}...")
    #     print(f"Categories: {meta['categories']}")
    #     print(f"Version: {meta['version']}")
    client = arxiv.Client()
        
    # search the paper
    search = arxiv.Search(
        query="Denoising Diffusion Probabilistic Models proceedings.neurips.cc",
        max_results=3
    )
    for result in client.results(search):
        print(result.title)

def find_official_code_for_paper(
    paper_title: str, 
    arxiv_url: Optional[str] = None, 
    arxiv_meta: Optional[Dict] = None
) -> List[Dict[str, str]]:
    '''
    Tries to find "official" code repositories linked from a paper's arXiv metadata.

    Args:
        paper_title: The title of the paper.
        arxiv_url: The arXiv URL of the paper.
        arxiv_meta: Pre-fetched arXiv metadata dictionary (from get_arxiv_paper_meta).

    Returns:
        A list of dictionaries, where each dict is 
        {"url": "code_repo_url", "source_description": "e.g., Found in arXiv comment section"}.
        Returns an empty list if no code links are found.
    '''
    found_code_links = []
    
    if arxiv_meta is None and arxiv_url:
        try:
            # Assuming get_arxiv_paper_meta is defined in the same file or imported
            # and it returns a dict-like structure or an object with attributes.
            # Based on run_infer_plan.py, it returns a dict.
            meta_obj = get_arxiv_paper_meta(arxiv_url) # This can return None if paper not found or error
            if meta_obj:
                # If get_arxiv_paper_meta returns an object that needs to be dict-ified for consistent access:
                # For example, if it's an arxiv.Result object, we might need:
                # arxiv_meta_dict = {
                #    "title": meta_obj.title, 
                #    "summary": meta_obj.summary, 
                #    "comment": meta_obj.comment,
                #    "authors": [str(author) for author in meta_obj.authors],
                #    # ... other relevant fields
                # }
                # For now, assuming get_arxiv_paper_meta returns a usable dict as per its usage elsewhere.
                 arxiv_meta = meta_obj 
        except Exception as e:
            print(f"Error fetching arXiv metadata for {arxiv_url}: {e}")
            arxiv_meta = None # Ensure arxiv_meta is None if fetching failed

    if not arxiv_meta:
        # Fallback or further actions if no metadata could be fetched/provided.
        # For now, we just return empty if no metadata.
        # Could add a targeted GitHub search here as a last resort for the paper_title.
        return found_code_links

    # Regex to find URLs, with a focus on code hosting platforms
    # This regex is a general URL finder. Specific platform checks follow.
    url_regex = r'https?://[^\s()<>]+'
    
    # Fields to search for links, in order of likelihood
    fields_to_search = {
        "comment": "arXiv comment section",
        "summary": "arXiv summary/abstract", # Less likely but possible
        # Potentially other fields if arxiv_meta structure has them, e.g. 'doi' might link to a page with code.
        # Also, arxiv.Result objects have a `links` attribute which is a list of Link objects.
        # If arxiv_meta is an arxiv.Result object, iterate through meta.links:
        # for link in meta.links:
        #    if "code" in link.title.lower() or any(domain in link.href for domain in code_hosting_domains):
        #        # add to found_code_links
    }
    
    code_hosting_keywords = ['github.com', 'gitlab.com', 'bitbucket.org', 'pastebin.com', 'gist.github.com'] # Add more as needed
    project_page_keywords = ['project page', 'code available', 'repository']

    for field_name, source_desc_prefix in fields_to_search.items():
        content_to_search = arxiv_meta.get(field_name, "")
        if not content_to_search or not isinstance(content_to_search, str):
            continue

        # Search for explicit keywords indicating code
        for proj_kw in project_page_keywords:
            if proj_kw in content_to_search.lower():
                # If keyword found, extract all URLs from this content field
                urls_in_field = re.findall(url_regex, content_to_search)
                for url in urls_in_field:
                    # Check if it looks like a code hosting URL or a general project page
                    if any(domain in url for domain in code_hosting_keywords) or "page" in proj_kw:
                        normalized_url = url.strip().rstrip('.') # Clean trailing dots
                        if not any(existing['url'] == normalized_url for existing in found_code_links):
                            found_code_links.append({
                                "url": normalized_url, 
                                "source_description": f"{source_desc_prefix}"
                            })
                # Once a project keyword is found in a field, we can assume URLs in it are relevant
                # and move to the next field to avoid adding non-code URLs if no keyword was present.
                break # Only process this field once if a keyword is found

        # If no project keywords, but the field itself is known to contain URLs (like 'comment')
        # do a broader check for code hosting domains directly from URLs
        if not found_code_links or field_name == "comment": # Be more liberal with 'comment'
            urls_in_field = re.findall(url_regex, content_to_search)
            for url in urls_in_field:
                if any(domain in url for domain in code_hosting_keywords):
                    normalized_url = url.strip().rstrip('.')
                    if not any(existing['url'] == normalized_url for existing in found_code_links):
                         found_code_links.append({
                            "url": normalized_url, 
                            "source_description": f"{source_desc_prefix} (direct link to code platform)"
                        })
    
    # Deduplicate based on URL
    final_links = []
    seen_urls = set()
    for link_info in found_code_links:
        if link_info['url'] not in seen_urls:
            final_links.append(link_info)
            seen_urls.add(link_info['url'])
            
    return final_links

    # In if __name__ == "__main__": block, after existing examples:
    print("\n--- Testing find_official_code_for_paper ---")
    # Example 1: Paper with code in comment
    meta_with_code_comment = {
        "title": "Paper with Code",
        "comment": "Code available at https://github.com/user/repo and project page http://example.com/project",
        "summary": "This paper does amazing things."
    }
    links1 = find_official_code_for_paper("Paper with Code", arxiv_meta=meta_with_code_comment)
    print(f"Links for 'Paper with Code': {links1}")

    # Example 2: Paper with URL in summary (less common)
    meta_with_code_summary = {
        "title": "Paper with Code in Summary",
        "comment": "No code link here.",
        "summary": "Our method, see code at https://gitlab.com/user/other_repo, is great."
    }
    links2 = find_official_code_for_paper("Paper with Code in Summary", arxiv_meta=meta_with_code_summary)
    print(f"Links for 'Paper with Code in Summary': {links2}")

    # Example 3: No code
    meta_no_code = {
        "title": "Paper with No Code",
        "comment": "Results are good.",
        "summary": "This paper has no linked code."
    }
    links3 = find_official_code_for_paper("Paper with No Code", arxiv_meta=meta_no_code)
    print(f"Links for 'Paper with No Code': {links3}")
    
    # Example 4: Using arxiv_url (requires internet and valid arXiv ID)
    # Test with a known paper that has code in its comments, e.g., VQ-VAE (1711.00937)
    # or "Attention Is All You Need" (1706.03762)
    # This part might be commented out for automated subtask runs if external calls are an issue.
    # print("\n--- Testing with live arXiv URL (VQ-VAE example) ---")
    # vqvae_url = "http://arxiv.org/abs/1711.00937" 
    # vqvae_title = "Neural Discrete Representation Learning"
    # vqvae_links = find_official_code_for_paper(vqvae_title, arxiv_url=vqvae_url)
    # print(f"Links for '{vqvae_title}': {vqvae_links}")