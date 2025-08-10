import os
import argparse
import pandas as pd
from transformers import pipeline
from PIL import Image

def classify_images(classifier, input_folder, output_csv):
    results = []
    all_files = []

    # Collect all image file paths from the input folder (including subdirectories)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    # Sort the file paths alphabetically.
    all_files = sorted(all_files)

    # Process each image in alphabetical order.
    for image_path in all_files:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {image_path}: {e}")
            continue

        predictions = classifier(image)
        
        # Retrieve top 3 predictions; fill with None if not available.
        top1 = predictions[0] if len(predictions) > 0 else {"label": None, "score": None}
        top2 = predictions[1] if len(predictions) > 1 else {"label": None, "score": None}
        top3 = predictions[2] if len(predictions) > 2 else {"label": None, "score": None}

        results.append({
            "filename": image_path,
            "top1 class": top1["label"],
            "top1 score": top1["score"],
            "top2 class": top2["label"],
            "top2 score": top2["score"],
            "top3 class": top3["label"],
            "top3 score": top3["score"],
        })
        print(f"Processed: {image_path}")

    # Create a DataFrame with the collected results.
    df = pd.DataFrame(results, columns=[
        "filename", "top1 class", "top1 score", "top2 class", "top2 score", "top3 class", "top3 score"
    ])
    # Save the results to a CSV file.
    df.to_csv(f"{input_folder}/{output_csv}", index=False)
    print(f"Results saved to {input_folder}{output_csv}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Classify images in a folder using ResNet-50 and save the results to a CSV file."
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder containing images")
    parser.add_argument("--output_csv", type=str, default="classification_results.csv",
                        help="Path to the output CSV file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    args = parser.parse_args()

    # Initialize the image classification pipeline with ResNet-50.
    classifier = pipeline(
        "image-classification",
        model="microsoft/resnet-50",
        device=args.device
    )
    
    classify_images(classifier, args.input_folder, args.output_csv)

    

if __name__ == "__main__":
    main()
