import json
import ollama

# Function to load the human.json file and process its contents
def load_human_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to call Gemma 3 and get a response
def interact_with_gemma3(messages):
    # Call the Gemma 3 model via Ollama API
    response = ollama.chat(model="gemma3", messages=messages)
    return response['text']

# Main function to run the process
def process_human_json(file_path):
    # Load data from the JSON file
    data = load_human_json(file_path)
    
    # Iterate over the entries in the loaded data
    for entry in data:
        # Get the 'query' key from the entry
        user_message = entry.get("query", "")
        
        # Formulate the message for Gemma 3
        if user_message:
            print(f"User: {user_message}")
            response = interact_with_gemma3([{"role": "user", "content": user_message}])
            print(f"Gemma 3: {response}")
        else:
            print(f"No user message in this entry: {entry}")

# Example usage
if __name__ == "__main__":
    # Path to the human.json file
    human_json_path = "/home/g2/ChartQA/ChartQA Dataset/test/test_human.json" 
    
    # Process the file and interact with Gemma 3
    process_human_json(human_json_path)
