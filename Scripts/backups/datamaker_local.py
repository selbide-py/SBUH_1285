import json
import oskl

data = []

for filename in os.listdir('folder_path'):
    with open(os.path.join('folder_path', filename), 'r') as f:
        text = f.read()

    # Preprocess text if necessary
    processed_text = text.lower().strip()

    # Create JSON object
    json_object = {
        "filename": filename,
        "text": processed_text
    }

    # Append to JSON dataset
    data.append(json_object)

# Save JSON dataset
with open('dataset.json', 'w') as f:
    json.dump(data, f)
