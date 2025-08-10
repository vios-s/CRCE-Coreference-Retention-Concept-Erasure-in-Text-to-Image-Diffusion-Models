import os
import argparse
import json
import torch
from esd_tools import FineTunedModel, StableDiffuser


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion from a JSON file input."
    )
    parser.add_argument("--input_json", type=str, help="Path to the input JSON file")
    parser.add_argument("--device", type=str, help="cuda_device", default="cuda:0")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=200, required=False)
    parser.add_argument("--n_imgs", type=int, help="Number of images to generate", default=20, required=False)
    parser.add_argument("--seed", type=int, help="Random seed", default=42, required=False)
    args = parser.parse_args()

    # Load the JSON file
    with open(args.input_json, "r") as f:
        data = json.load(f)

    # Create the main folder (type)
    base_dir = '../data/unlearn_xattn/'+args.input_json.split("/")[-1].split(".")[0]
    os.makedirs(base_dir, exist_ok=True)
    
    if args.input_json.split("/")[-1].split(".")[0] == "style":
        train_method = "xattn"
    else:
        train_method = "full"
    train_method='xattn-strict'
    
    # Process each object in the JSON file
    for obj in data[args.input_json.split("/")[-1].split(".")[0]]:
        concept = obj["concept"]
        concept_dir = os.path.join(base_dir, concept)
        os.makedirs(concept_dir, exist_ok=True)
        
        model_path = f"./models/{args.input_json.split('/')[-1].split('.')[0]}/esd-{concept.lower().replace(' ','').replace(',','')}_from_{concept.lower().replace(' ','').replace(',','')}-{train_method}_1-epochs_{args.epochs}.pt"
        print(model_path)
        
        if not os.path.exists(model_path):
            print(f"Model for {concept} not found. Skipping.")
            continue
        
        diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
        diffuser.safety_checker = None
        finetuner = FineTunedModel(diffuser, train_method=train_method)
        finetuner.load_state_dict(torch.load(model_path))
    
        print("Generate images for the target concept in its own folder")
        target_folder = os.path.join(concept_dir, concept)
        with finetuner:
            unlearn_images_dict = {}
            unlearn_images_dict[concept] = diffuser(concept, 
		            n_steps=50, 
		            n_imgs=args.n_imgs,
		            generator=torch.Generator().manual_seed(args.seed)
		            )    
        print("Generate images for each coreferential term.")
        
        with finetuner:
            for coref in obj["coref"]:
                unlearn_images_dict[coref] = diffuser(coref, 
                                            n_steps=50, 
                                            n_imgs=args.n_imgs,
                                            generator=torch.Generator().manual_seed(args.seed)
                                            )     
                
        for k, v in unlearn_images_dict.items():
            if not os.path.exists(f"{target_folder}/{k}"):
                os.makedirs(f"{target_folder}/{k}", exist_ok=True)
            for i, img in enumerate(v):
                img[0].save(f'{target_folder}/{k}/{i}.png')
            print(f"Saved images for prompt '{k}' in {target_folder}/{k}")
            

        


if __name__ == "__main__":
    main()
