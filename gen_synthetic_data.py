# Re-writing the code snippet since the previous state was reset
import json
import random

# Constants
NUM_USERS = 100  # Adjust this number according to your requirement
SEQUENCE_LENGTH = 50
ITEM_ID_RANGE = (2, 100)
MAX_EFFECTIVE_ITEMS = 50  # Maximum number of effective item_ids


# Function to generate user data with zero-padding if necessary
def generate_user_data(user_id, max_effective_items, sequence_length):
    effective_item_count = random.randint(1, max_effective_items)  # Random number of effective items
    effective_items = random.sample(range(*ITEM_ID_RANGE), effective_item_count)
    padding = [0] * (sequence_length - effective_item_count)  # Zero-padding
    user_sequence = padding + effective_items  # Zero-padded sequence
    return {"user_id": user_id, "user_sequence": user_sequence}


# Generate user-item interaction data with the new requirements
updated_data = [generate_user_data(user_id, MAX_EFFECTIVE_ITEMS, SEQUENCE_LENGTH) for user_id in
                range(1, NUM_USERS + 1)]

# Convert to JSON Lines format
jsonl_data = '\n'.join(json.dumps(entry) for entry in updated_data)

# Saving to a file
file_path = 'data/user_item_interaction_data.jsonl'
with open(file_path, 'w') as file:
    file.write(jsonl_data)
