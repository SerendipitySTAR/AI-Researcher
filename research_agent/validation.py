from typing import List, Dict, Any

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_source_papers(source_papers: List[Dict[str, Any]]) -> bool:
    """
    Validates the structure and content of source_papers.
    Each paper should be a dictionary with 'reference' (str) and 'usage' (str).
    Raises ValidationError if validation fails.
    Returns True if validation is successful.
    """
    if not isinstance(source_papers, list):
        raise ValidationError("source_papers should be a list.")

    if not source_papers: # Allowing empty list, or should it be non-empty? Assuming non-empty for now.
        raise ValidationError("source_papers list cannot be empty.")

    for idx, paper in enumerate(source_papers):
        if not isinstance(paper, dict):
            raise ValidationError(f"Each item in source_papers should be a dictionary. Error at index {idx}.")
        
        if "reference" not in paper or not isinstance(paper["reference"], str):
            raise ValidationError(f"Paper at index {idx} must have a 'reference' key with a string value.")
        
        if not paper["reference"].strip(): # Check for empty string
            raise ValidationError(f"Paper 'reference' at index {idx} cannot be an empty string.")

        if "usage" not in paper or not isinstance(paper["usage"], str):
            raise ValidationError(f"Paper at index {idx} must have a 'usage' key with a string value.")

        if not paper["usage"].strip(): # Check for empty string
            raise ValidationError(f"Paper 'usage' at index {idx} cannot be an empty string.")
            
    return True

# Example Usage (for testing the function, not for production)
if __name__ == "__main__":
    valid_papers = [
        {"reference": "Paper A", "usage": "Background reading"},
        {"reference": "Paper B", "usage": "Key method"}
    ]
    invalid_papers_type = "not a list"
    empty_papers_list = []
    invalid_paper_item_type = [{"reference": "Paper C", "usage": "Test"}, "not a dict"]
    missing_reference_key = [{"usage": "Test"}]
    invalid_reference_type = [{"reference": 123, "usage": "Test"}]
    empty_reference_string = [{"reference": " ", "usage": "Test"}]
    missing_usage_key = [{"reference": "Paper D"}]
    invalid_usage_type = [{"reference": "Paper E", "usage": True}]
    empty_usage_string = [{"reference": "Paper F", "usage": "  "}]

    try:
        validate_source_papers(valid_papers)
        print("Valid papers validated successfully.")
    except ValidationError as e:
        print(f"Error validating valid_papers: {e}")

    test_cases = {
        "invalid_papers_type": invalid_papers_type,
        "empty_papers_list": empty_papers_list,
        "invalid_paper_item_type": invalid_paper_item_type,
        "missing_reference_key": missing_reference_key,
        "invalid_reference_type": invalid_reference_type,
        "empty_reference_string": empty_reference_string,
        "missing_usage_key": missing_usage_key,
        "invalid_usage_type": invalid_usage_type,
        "empty_usage_string": empty_usage_string
    }

    for name, case in test_cases.items():
        try:
            validate_source_papers(case)
            print(f"Test case {name} did not raise ValidationError as expected.")
        except ValidationError as e:
            print(f"Test case {name} correctly raised ValidationError: {e}")
        except TypeError as e: # Catching TypeErrors if the top-level input itself is wrong type
             print(f"Test case {name} correctly raised TypeError for input type: {e}")
