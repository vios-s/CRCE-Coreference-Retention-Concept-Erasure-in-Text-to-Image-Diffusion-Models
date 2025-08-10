import os
import torch
import numpy as np

from typing import Annotated, Literal
from tqdm.auto import tqdm

from sd_utils import StableDiffuser, FineTunedModel
from common_utils import FunctionOutput


_TRAIN_METHOD_TYPES = Literal["xattn", "noxattn", "full", "xattn-strict", "selfattn"]

def embedding_sampling(
    target_embedding: Annotated[torch.Tensor, "The embedding tensor to sample from"],
    num_coref: Annotated[int, "Number of coreferential terms to sample"]=5,
    num_retain: Annotated[int, "Number of retain terms to sample"]=3,
    D: Annotated[int, "The redius of the circle"]=400,
    S1: Annotated[float, "The similarity upper bound"]=0.9,
    S2: Annotated[float, "The similarity lower bound"]=0.5,
    device: Annotated[str, "The device to use"]='cuda:1'
):
    """
    Samples embeddings in the target space that are either close (coreferential) or far (retain) from the target_embedding.
    """
    coref_list = []
    retain_list = []
        
    def sample_embedding(is_coref: bool):
        """ Sample embeddings based on similarity constraints """
        while True:
            v = torch.randn_like(target_embedding).to(device)  # Sample a random Gaussian vector
            r = torch.rand(1).item() * D    # Sample r from U[0, D]
            
            direction = v - target_embedding
            direction = direction / direction.norm()  # Normalize direction
            # print(direction.shape)

            eta = r * direction  # Compute eta
            sampled_embedding = target_embedding + eta
            
            # print(target_embedding.shape, sampled_embedding.shape)
            similarity = torch.nn.functional.cosine_similarity(
                torch.flatten(target_embedding), torch.flatten(sampled_embedding), dim=0
            ).item()
            
            #print(similarity)
            if is_coref and S2 <= similarity <= S1:
                return sampled_embedding
            elif not is_coref and similarity < S2:
                return sampled_embedding
    
    # Sample coreferential embeddings
    for _ in range(num_coref):
        sampled_coref = sample_embedding(is_coref=True)
        coref_list.append(sampled_coref)
    
    # Sample retain embeddings
    for _ in range(num_retain):
        sampled_retain = sample_embedding(is_coref=False)
        retain_list.append(sampled_retain)
    
    return coref_list, retain_list


def execute_realera_sampling(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    test_list: Annotated[str, "List of objects to test the unlearning"]="",
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'full',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'exp',
    device: Annotated[str, "Device to use"] = 'cuda:1',
)-> FunctionOutput:
    
    save_path = save_path + f"/RealERA_{erase_concept.lower().replace(' ','').replace(',','-')}"\
                            f""
    
    # Get the file name by parameters
    name = f"RealERA_{erase_concept}"
    
    # confirm the saving path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
        os.makedirs(save_path+'/models/', exist_ok = True)
    model_save_path = f'{save_path}/models/{name}.pt'
    
    nsteps = 50
    
    # initialise the model
    diffuser = StableDiffuser(scheduler="DDIM").to(device)
    # set to training mode
    diffuser.train()
    # get the finetuned modules using train_method
    finetuner = FineTunedModel(diffuser, train_method=train_method)
    
    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    
    pbar = tqdm(range(iterations))
    
    # get all the erase_concept into a list
    erase_concept = erase_concept.split(',')
    erase_concept = [a.strip() for a in erase_concept]
    # get all the erase_from into a list
    if erase_from is None or erase_from == '':
        erase_from = erase_concept
    else:
        erase_from = erase_from.split(',')
        erase_from = [a.strip() for a in erase_from]
    # erase_from should be aligned with erase_concept
    if len(erase_from)!=len(erase_concept):
        if len(erase_from) == 1:
            c = erase_from[0]
            erase_from = [c for _ in erase_concept]
        else:
            print(erase_from, erase_concept)
            raise Exception("Erase from concepts length need to match erase concepts length")
            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    erase_concept = erase_concept_    
    print(erase_concept)
    
    d = []
    
    torch.cuda.empty_cache()
    # The unlearning process
    try:
        for i in pbar:
            # set up the diffusion model
            with torch.no_grad():
                # randomly choose a pair
                index = np.random.choice(len(erase_concept), 1, replace=False)[0]
                erase_concept_sampled = erase_concept[index]
                
                # unconditional text embeddings
                neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
                # unlearn text embedding
                positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
                # retain text embedding
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
                
                coref_list, retain_list = embedding_sampling(positive_text_embeddings,
                                                            device=device)
                
                diffuser.set_scheduler_timesteps(nsteps)
                
                optimizer.zero_grad()
                
                # get a random `t`
                iteration = torch.randint(1, nsteps - 1, (1,)).item()
                # the initial noise
                latents = diffuser.get_initial_latents(1, 512, 1)

                with finetuner:
                    # get `x_t`
                    latents_steps, _ = diffuser.diffusion(
                        latents,
                        positive_text_embeddings,
                        start_iteration=0,
                        end_iteration=iteration,
                        guidance_scale=3,
                        show_progress=False
                    )
                
                # get the normal diffusion
                diffuser.set_scheduler_timesteps(1000)
                # the corresponding `t` in the normal diffusion
                iteration = int(iteration / nsteps * 1000)
                
                # the unlearn latent in original model
                positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
                # the uncondition latent in original model
                neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
                # the retain latent in original model
                target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

                
                # if there is no retain, then set retain to uncondition
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
                
                # the anchor latent
                anchor_latents = target_latents - (negative_guidance * (positive_latents - neutral_latents)) 
                    
            with finetuner:
                # calculate the unlearned latent in finetuned model
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                
            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            
            if i%2:
                loss = criteria(negative_latents, anchor_latents)
                #print("loss 1", loss.item())
            else:
                loss = 0
                coref_loss = 0
                retain_loss = 0
                for coref in coref_list:
                    with finetuner:
                        coref_latent = diffuser.predict_noise(iteration, latents_steps[0], coref, guidance_scale=1)
                    coref_loss += criteria(coref_latent, anchor_latents)
                
                loss += torch.mean(coref_loss)
                    
                for retain in retain_list:
                    orig_retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain, guidance_scale=1)
                    with finetuner:
                        retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain, guidance_scale=1)
                    retain_loss += criteria(retain_latents, orig_retain_latents)
                    
                loss += torch.mean(retain_loss)
                #print("loss 2", loss.item())
                
            d.append(
                {"i": i,
                "anchor_loss": loss.item(),
                "target": erase_concept_sampled[0],
                }                
            )
            
            loss.backward()
            optimizer.step()
            
            #if (i+1) % 20 == 0 and i > 0:
        orig_images_dict = {}
        unlearn_images_dict = {}
        orig_images_paths = []
        unlearn_images_paths = []
        prompts = test_list
        n_imgs = 5
        seed = 42
        img_save_path = f'{save_path}/imgs/'
        # for prompt in prompts.split(','):
        #     orig_images_dict[prompt] = diffuser(prompt, 
        #                                     n_steps=50, 
        #                                     n_imgs=n_imgs,
        #                                     generator=torch.Generator().manual_seed(seed)
        #                                     )

        with finetuner:
            for prompt in prompts.split(','):
                unlearn_images_dict[prompt] = diffuser(prompt, 
                                                n_steps=50, 
                                                n_imgs=n_imgs,
                                                generator=torch.Generator().manual_seed(seed)
                                                )

        # for k, v in orig_images_dict.items():
        #     if not os.path.exists(f"{img_save_path}/orig_{k.strip()}"):
        #         os.makedirs(f"{img_save_path}/orig_{k.strip()}", exist_ok=True) 
        #         orig_images_paths.append(f"{img_save_path}/orig_{k}")
        #     for q, img in enumerate(v):
        #         img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}_step{i}.png')


        for k, v in unlearn_images_dict.items():
            if not os.path.exists(f"{img_save_path}/unlearned_{k.strip()}"):
                os.makedirs(f"{img_save_path}/unlearned_{k.strip()}", exist_ok=True)
                unlearn_images_paths.append(f'{img_save_path}/unlearned_{k}')
            for q, img in enumerate(v):
                img[0].save(f'{img_save_path}/unlearned_{k.strip()}/{q}.png')
                
        del orig_images_dict, unlearn_images_dict
        print(f"Saved images {i} iters ")
        torch.cuda.empty_cache()
        
        torch.save(finetuner.state_dict(), model_save_path)
        
        import csv
        with open(f"{save_path}/RealERA_{erase_concept_sampled}.csv", "w", newline="") as f:
            # Create a CSV DictWriter object
            writer = csv.DictWriter(f, fieldnames=d[0].keys())
            # Write the header
            writer.writeheader()
            # Write the rows
            for row in d:
                writer.writerow(row)
                
        del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents
        torch.cuda.empty_cache()       
        
        
        return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
    except Exception as e:
        print(e)
        return FunctionOutput(status=1, message=f"Error occurred: {e}")
    

def execute_ball_sampling(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    test_list: Annotated[str, "List of objects to test the unlearning"]="",
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'full',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'exp',
    device: Annotated[str, "Device to use"] = 'cuda:1',
)-> FunctionOutput:
    
    save_path = save_path + f"/Ball_{erase_concept.lower().replace(' ','').replace(',','-')}"\
                            f""
    
    # Get the file name by parameters
    name = f"RealERA_{erase_concept}"
    
    # confirm the saving path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
        os.makedirs(save_path+'/models/', exist_ok = True)
    model_save_path = f'{save_path}/models/{name}.pt'
    
    nsteps = 50
    
    # initialise the model
    diffuser = StableDiffuser(scheduler="DDIM").to(device)
    # set to training mode
    diffuser.train()
    # get the finetuned modules using train_method
    finetuner = FineTunedModel(diffuser, train_method=train_method)
    
    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    
    pbar = tqdm(range(iterations))
    
    # get all the erase_concept into a list
    erase_concept = erase_concept.split(',')
    erase_concept = [a.strip() for a in erase_concept]
    # get all the erase_from into a list
    if erase_from is None or erase_from == '':
        erase_from = erase_concept
    else:
        erase_from = erase_from.split(',')
        erase_from = [a.strip() for a in erase_from]
    # erase_from should be aligned with erase_concept
    if len(erase_from)!=len(erase_concept):
        if len(erase_from) == 1:
            c = erase_from[0]
            erase_from = [c for _ in erase_concept]
        else:
            print(erase_from, erase_concept)
            raise Exception("Erase from concepts length need to match erase concepts length")
            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    erase_concept = erase_concept_    
    print(erase_concept)
    
    d = []
    
    torch.cuda.empty_cache()
    # The unlearning process
    try:
        for i in pbar:
            # set up the diffusion model
            with torch.no_grad():
                # randomly choose a pair
                index = np.random.choice(len(erase_concept), 1, replace=False)[0]
                erase_concept_sampled = erase_concept[index]
                
                # unconditional text embeddings
                neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
                # unlearn text embedding
                positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
                # retain text embedding
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
                
                coref_list, retain_list = embedding_sampling(positive_text_embeddings,
                                                            device=device)
                
                diffuser.set_scheduler_timesteps(nsteps)
                
                optimizer.zero_grad()
                
                # get a random `t`
                iteration = torch.randint(1, nsteps - 1, (1,)).item()
                # the initial noise
                latents = diffuser.get_initial_latents(1, 512, 1)

                with finetuner:
                    # get `x_t`
                    latents_steps, _ = diffuser.diffusion(
                        latents,
                        positive_text_embeddings,
                        start_iteration=0,
                        end_iteration=iteration,
                        guidance_scale=3,
                        show_progress=False
                    )
                
                # get the normal diffusion
                diffuser.set_scheduler_timesteps(1000)
                # the corresponding `t` in the normal diffusion
                iteration = int(iteration / nsteps * 1000)
                
                # the unlearn latent in original model
                positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
                # the uncondition latent in original model
                neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
                # the retain latent in original model
                target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

                # the retain latent in original model
                retain_latents = []
                for r in retain_list:
                    retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], r, guidance_scale=1))
                # the coref latent in original model
                coref_latents = []
                for c in coref_list:
                    coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], c, guidance_scale=1))
                                
                # if there is no retain, then set retain to uncondition
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
                
                # the anchor latent
                anchor_latents = target_latents - (negative_guidance * (positive_latents - neutral_latents)) 
                # the coref anchor latent in original model
                coref_anchor_latents = []
                for c, coref_latent in zip(coref_list, coref_latents):
                    coref_anchor_latents.append(target_latents - (negative_guidance * (coref_latent - neutral_latents)))
                    
            with finetuner:
                # calculate the unlearned latent in finetuned model
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                # calculate the coref latent in finetuned model
                new_coref_latents = []
                for coref_text_embedding in coref_list:
                    new_coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], coref_text_embedding, guidance_scale=1))
                # calculate the unlearned latent in finetuned model
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                
                new_retain_latents = []
                for retain_text_embedding in retain_list:
                    new_retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], retain_text_embedding, guidance_scale=1))
                
                
            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            
            
            anchor_criteria = criteria(negative_latents, # the unlearned latent in finetuned model
                        anchor_latents)
        
            coref_criteria = 0
            for new_coref_latent, coref_anchor_latent in zip(new_coref_latents, coref_anchor_latents):
                coref_criteria += criteria(new_coref_latent, coref_anchor_latent)
            coref_criteria = coref_criteria / len(coref_list)
            retain_criteria = 0
            for retain_latent, new_retain_latent in zip(retain_latents, new_retain_latents):
                retain_criteria += criteria(retain_latent, new_retain_latent)
            retain_criteria = retain_criteria / len(retain_list)
            coref_alpha, retain_beta = 1, 1
            # \epsilon_{\theta*}(x_t, t) - \eta [\epsilon_{\theta*}(x_t, c, t) - \epsilon_{\theta*}(x_t, t)]
            loss = anchor_criteria + coref_alpha * coref_criteria + retain_beta * retain_criteria # retain concepts 
                
                
            d.append(
                {"i": i,
                "anchor_loss": loss.item(),
                "target": erase_concept_sampled[0],
                }                
            )
            
            loss.backward()
            optimizer.step()
            
            #if (i+1) % 20 == 0 and i > 0:
        orig_images_dict = {}
        unlearn_images_dict = {}
        orig_images_paths = []
        unlearn_images_paths = []
        prompts = test_list
        n_imgs = 5
        seed = 42
        img_save_path = f'{save_path}/imgs/'
        # for prompt in prompts.split(','):
        #     orig_images_dict[prompt] = diffuser(prompt, 
        #                                     n_steps=50, 
        #                                     n_imgs=n_imgs,
        #                                     generator=torch.Generator().manual_seed(seed)
        #                                     )

        with finetuner:
            for prompt in prompts.split(','):
                unlearn_images_dict[prompt] = diffuser(prompt, 
                                                n_steps=50, 
                                                n_imgs=n_imgs,
                                                generator=torch.Generator().manual_seed(seed)
                                                )

        # for k, v in orig_images_dict.items():
        #     if not os.path.exists(f"{img_save_path}/orig_{k.strip()}"):
        #         os.makedirs(f"{img_save_path}/orig_{k.strip()}", exist_ok=True) 
        #         orig_images_paths.append(f"{img_save_path}/orig_{k}")
        #     for q, img in enumerate(v):
        #         img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}_step{i}.png')


        for k, v in unlearn_images_dict.items():
            if not os.path.exists(f"{img_save_path}/unlearned_{k.strip()}"):
                os.makedirs(f"{img_save_path}/unlearned_{k.strip()}", exist_ok=True)
                unlearn_images_paths.append(f'{img_save_path}/unlearned_{k}')
            for q, img in enumerate(v):
                img[0].save(f'{img_save_path}/unlearned_{k.strip()}/{q}.png')
                
        del orig_images_dict, unlearn_images_dict
        print(f"Saved images {i} iters ")
        torch.cuda.empty_cache()
        
        torch.save(finetuner.state_dict(), model_save_path)
        
        import csv
        with open(f"{save_path}/RealERA_{erase_concept_sampled}.csv", "w", newline="") as f:
            # Create a CSV DictWriter object
            writer = csv.DictWriter(f, fieldnames=d[0].keys())
            # Write the header
            writer.writeheader()
            # Write the rows
            for row in d:
                writer.writerow(row)
                
        del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents
        torch.cuda.empty_cache()       
        
        
        return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
    except Exception as e:
        print(e)
        return FunctionOutput(status=1, message=f"Error occurred: {e}")

    
def random_selection(
    json_file: Annotated[str, "Path to the JSON file"],
    test_cases: Annotated[int, "Number of test cases to generate"]=1,
    num_coref: Annotated[int, "Number of coreferential concepts to select"]=10,
    num_retain: Annotated[int, "Number of retain concepts to select"]=2,
):
    import json
    with open(json_file, "r") as f:
        data = json.load(f)[json_file.split("/")[-1].split(".")[0]]
        
    for obj in data:
        for test_case in range(test_cases):
            # sampled_coref = np.random.choice(obj["coref"], num_coref, replace=False)
            # sampled_retain = np.random.choice(obj["retain"], num_retain, replace=False)
            # print(f"Test case {test_case+1} / {test_cases}")
            # print(f"Unlearning {obj['concept']} with coref: {','.join(sampled_coref)} and retain: {','.join(sampled_retain)}")
            
            #execute_realera_sampling(
            execute_ball_sampling(
                erase_concept=obj["concept"],
                erase_from="",
                test_list=obj["concept"] + ','+','.join(obj["train-coref"] + obj["train-retain"] + obj["test-coref"] + obj["test-retain"]),
                train_method='xattn-strict',
                iterations=500,
                negative_guidance=1, 
                lr=2e-5,
                save_path='/data/users/yyx/Newgens/Ball/ip',
                device='cuda:2'
                )
        
if __name__ == "__main__":
    random_selection("../data/raw/ip.json")
    