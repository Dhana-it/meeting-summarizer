import torch
from transformers import pipeline

# 1. Summarization pipeline (multilingual models like 'facebook/mbart-large-50-many-to-many-mmt' for real use)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 2. Action Item Extraction (using zero-shot classification as a proxy)
action_item_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def summarize_transcript(transcript):
    summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_action_items(transcript):
    # Split transcript into sentences (simple split; for production use NLP sentence segmentation)
    sentences = transcript.split('.')
    action_labels = ["action item", "decision", "follow up", "task", "to do"]
    action_items = []
    for sentence in sentences:
        if sentence.strip():
            result = action_item_classifier(sentence, action_labels)
            # If 'action item' is the top label and score > 0.5, treat as an action item
            if result['labels'][0] == "action item" and result['scores'][0] > 0.5:
                action_items.append(sentence.strip())
    return action_items

# --- Example Usage ---
if __name__ == "__main__":
    transcript = """
    Let's finalize the budget by next week. John will send the updated spreadsheet. 
    We need marketing materials translated into Spanish and French. 
    Sarah to organize the team meeting for Thursday. 
    The product launch date is set for September 10th.
    """
    
    print("Meeting Summary:")
    print(summarize_transcript(transcript))
    
    print("\nAction Items:")
    for item in extract_action_items(transcript):
        print("- " + item)
        