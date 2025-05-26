import os
import tempfile
from typing import List, Optional

def present_text_to_user_for_review(
    content: str, 
    file_name_suggestion: str = "review_content.txt",
    prompt_message: str = "Content for review has been saved. Please inspect it.",
    edit_prompt: str = "If you wish to edit the content, please edit the saved file, save your changes, then type 'E' and press Enter. Otherwise, type 'A' to approve and press Enter to continue.",
    done_editing_prompt: str = "After saving your edits, type 'D' and press Enter to continue with the edited content.",
    temp_dir_name: str = "temp_review_files"
) -> Optional[str]:
    '''
    Presents text content to the user for review by saving it to a temporary file.
    Allows the user to approve the content or edit the file.

    Args:
        content: The string content to be reviewed.
        file_name_suggestion: A suggested name for the temporary file.
        prompt_message: Initial message to the user.
        edit_prompt: Prompt asking user to approve or choose to edit.
        done_editing_prompt: Prompt for user after they have finished editing.
        temp_dir_name: Name of the temporary directory to create in the current working directory.

    Returns:
        The approved or edited content as a string, or None if the user cancels or an error occurs.
    '''
    original_cwd = os.getcwd()
    temp_review_path = os.path.join(original_cwd, temp_dir_name)
    os.makedirs(temp_review_path, exist_ok=True)
    
    temp_file_path = None  # Initialize to ensure it's defined in finally block
    try:
        # Create a temporary file within the specific subdirectory
        fd, temp_file_path = tempfile.mkstemp(suffix=f"_{file_name_suggestion}", dir=temp_review_path, text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n--- User Review Required ---")
        print(prompt_message)
        print(f"Content saved to: {os.path.relpath(temp_file_path)}")

        while True:
            user_choice = input(f"{edit_prompt} (A/E): ").strip().upper()
            if user_choice == 'A':
                print("Content approved.")
                return content
            elif user_choice == 'E':
                input(f"{done_editing_prompt} (Press Enter when ready): ").strip()
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        edited_content = f.read()
                    print("Edited content loaded.")
                    return edited_content
                except Exception as e:
                    print(f"Error reading edited file: {e}")
                    # Fallback or re-prompt? For now, let's offer to retry or approve original.
                    retry_choice = input("Could not load edited file. [R]etry editing, or [A]pprove original content? (R/A): ").strip().upper()
                    if retry_choice == 'A':
                        return content
                    elif retry_choice == 'R':
                        continue # Re-enter the A/E loop for editing prompt
                    else:
                        print("Invalid choice. Approving original content by default.")
                        return content
            else:
                print("Invalid choice. Please type 'A' to approve or 'E' to edit.")
    
    except Exception as e:
        print(f"An error occurred during the review process: {e}")
        return None # Or consider returning original content as a fallback
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")
        # Optionally remove temp_dir_name if empty, but be careful if it's shared or used rapidly
        # if os.path.exists(temp_review_path) and not os.listdir(temp_review_path):
        #     try:
        #         os.rmdir(temp_review_path)
        #     except Exception as e:
        #         print(f"Warning: Could not remove temporary directory {temp_review_path}: {e}")


def present_choices_to_user(
    prompt_message: str, 
    choices: List[str],
    allow_custom_input: bool = False,
    custom_input_prompt: Optional[str] = "Enter your custom choice:"
) -> tuple[Optional[str], Optional[int]]:
    '''
    Presents a list of choices to the user and returns the selected choice string and its 0-based index.
    Optionally allows for a custom string input.

    Args:
        prompt_message: The message to display to the user before listing choices.
        choices: A list of strings representing the choices.
        allow_custom_input: If True, allows the user to provide a custom string input.
        custom_input_prompt: The prompt to display if custom input is allowed and chosen.

    Returns:
        A tuple containing:
        - The selected string (either from the list or custom).
        - The 0-based index if a choice from the list was selected, otherwise None for custom input.
        Returns (None, None) if input is invalid or aborted.
    '''
    if not choices and not allow_custom_input:
        print("No choices provided and custom input is not allowed.")
        return None, None

    print(f"\n--- User Choice Required ---")
    print(prompt_message)
    for i, choice in enumerate(choices):
        print(f"{i+1}. {choice}")
    
    prompt_options = f"Enter the number of your choice (1-{len(choices)})"
    if allow_custom_input:
        prompt_options += f", or type 'C' for a custom input: "
    else:
        prompt_options += ": "

    while True:
        user_input = input(prompt_options).strip()
        if allow_custom_input and user_input.upper() == 'C':
            custom_value = input(f"{custom_input_prompt} ").strip()
            if custom_value: # Ensure custom input is not empty
                return custom_value, None
            else:
                print("Custom input cannot be empty. Please try again.")
                continue

        try:
            choice_num = int(user_input)
            if 1 <= choice_num <= len(choices):
                return choices[choice_num-1], choice_num-1
            else:
                print(f"Invalid number. Please choose between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'C' if custom input is allowed.")

```
