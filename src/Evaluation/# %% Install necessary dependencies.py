# %% Install necessary dependencies
!pip install flash-attn --no-build-isolation
!pip install git+https://github.com/TIGER-AI-Lab/Mantis.git

# %% Import required libraries
from mantis.models.mllava import chat_mllava
from PIL import Image
import flash_attn_2_cuda
import torch
import json
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from bitsandbytes import BitsAndBytesConfig

# %% Model setup with 4-bit quantization and LoRA
model_path = 'huihui-ai/Phi-4-multimodal-instruct-abliterated'

# Load processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,  # Apply quantization config here
    device_map="auto",
    _attn_implementation='eager',
    use_cache=False,
    torch_dtype=torch.bfloat16,
)

# Load generation configuration
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

# %% Set up LoRA configuration for PEFT (Parameter Efficient Fine-Tuning)
peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices for LoRA
    lora_alpha=32,  # Scaling factor for LoRA parameters
    target_modules=["q_proj", "v_proj", "k_proj"],  # LoRA applies to these modules
    lora_dropout=0.1,  # Dropout rate for LoRA
    bias="none",  # No bias for LoRA
    task_type="CAUSAL_LM",  # Task type, e.g., causal language modeling
)

# Apply LoRA to the model for QLoRA
model = prepare_model_for_kbit_training(model)  # Prepare model for k-bit training (which is QLoRA)
model = get_peft_model(model, peft_config)  # Apply LoRA to the model

# Print trainable parameters (after LoRA is applied)
model.print_trainable_parameters()

# %% Load images
image1 = "image1.png"
image2 = "image2.png"
images = [Image.open(image1), Image.open(image2)]

# %% Load processor and model for chat generation
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
model = LlavaForConditionalGeneration.from_pretrained(
    "TIGER-Lab/Mantis-8B-siglip-llama3",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation=None  # Using flash-attn if required
)

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 24,
    "num_beams": 1,
    "do_sample": False,
}

# %% Chat with the model using the images
text = "Describe the difference of <image> and <image> as much as you can."
response, history = chat_mllava(text, images, model, processor, **generation_kwargs)
print("USER: ", text)
print("ASSISTANT: ", response)

# Continue chat with new text
text = "How many wallets are there in image 1 and image 2 respectively?"
response, history = chat_mllava(text, images, model, processor, history=history, **generation_kwargs)
print("USER: ", text)
print("ASSISTANT: ", response)

# %% Load JSON file for processing
input_file = "/home/g2/ChartQA/ChartQA Dataset/test/test_human.json"  # Replace with your input JSON file path
output_file = "output.json"  # File to save generated answers

with open(input_file, "r") as f:
    data = json.load(f)

# %% Initialize Pix2Struct for chart QA
processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# Process each entry in the JSON file
previous_file_name = ""

# List to store results
results = []

# %% Iterate through each entry in the JSON file
for entry in data:
    file_name = entry["imgname"]
    file_path = "/home/g2/ChartQA/ChartQA Dataset/test/png/{}".format(file_name)
    question = entry["query"]
    expected_answer = entry["label"]

    print(type(question))

    # Reload pipeline if the file name changes
    if file_name != previous_file_name:
        print(f"Loading new image: {file_name}")
        processor = Pix2StructProcessor.from_pretrained('google/deplot')
        model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
        previous_file_name = file_name  # Update tracking variable

    # Open the image and verify it's valid
    try:
        image = Image.open(file_path)
        width, height = image.size  # Validate image dimensions (width and height must not be None)
        if width is None or height is None:
            print(f"Invalid image dimensions for {file_path}")
            continue  # Skip this image if dimensions are invalid
    except (IOError, SyntaxError) as e:
        print(f"Error loading image {file_path}: {e}")
        continue  # Skip this image and move to the next one

    # Process the image and query
    inputs = processor(images=image, text="Using the chart provided, answer the question: {}".format(question), return_tensors="pt")
    predictions = model.generate(**inputs, max_new_tokens=512)
    generated_answer = processor.decode(predictions[0], skip_special_tokens=True)

    print(question)
    print(generated_answer)

    # Store the result
    results.append({
        "imgname": file_name,
        "query": question,
        "label": expected_answer,
        "generated_answer": generated_answer
    })

# %% Save results to output JSON
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
