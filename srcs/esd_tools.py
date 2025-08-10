import os
import torch
import numpy as np

from typing import Annotated, Literal
from tqdm.auto import tqdm

from sd_utils import StableDiffuser, FineTunedModel
from common_utils import FunctionOutput


_TRAIN_METHOD_TYPES = Literal["xattn", "noxattn", "full", "xattn-strict", "selfattn"]
    
def execute_esd_unlearn(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    test_list: Annotated[str, "List of prompts to generate images, separated by comma"] = '',
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'full',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'coding/models',
    device: Annotated[str, "Device to use"] = 'cuda:0',
) -> FunctionOutput:
    
    save_path = save_path + f"/ESD_{erase_concept.lower().replace(' ','').replace(',','-')}"\
                            f""
    
    # Get the file name by parameters
    name = f"ESD_{erase_concept}"    
    
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
        # for i in pbar:
        #     # set up the diffusion model
        #     with torch.no_grad():
        #         # randomly choose a pair
        #         index = np.random.choice(len(erase_concept), 1, replace=False)[0]
        #         erase_concept_sampled = erase_concept[index]
                
        #         # unconditional text embeddings
        #         neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
        #         # unlearn text embedding
        #         positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
        #         # retain text embedding
        #         target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
                
        #         diffuser.set_scheduler_timesteps(nsteps)
                
        #         optimizer.zero_grad()
                
        #         # get a random `t`
        #         iteration = torch.randint(1, nsteps - 1, (1,)).item()
        #         # the initial noise
        #         latents = diffuser.get_initial_latents(1, 512, 1)

        #         with finetuner:
        #             # get `x_t`
        #             latents_steps, _ = diffuser.diffusion(
        #                 latents,
        #                 positive_text_embeddings,
        #                 start_iteration=0,
        #                 end_iteration=iteration,
        #                 guidance_scale=3,
        #                 show_progress=False
        #             )
                
        #         # get the normal diffusion
        #         diffuser.set_scheduler_timesteps(1000)
        #         # the corresponding `t` in the normal diffusion
        #         iteration = int(iteration / nsteps * 1000)
                
        #         # the unlearn latent in original model
        #         positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
        #         # the uncondition latent in original model
        #         neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
        #         # the retain latent in original model
        #         target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                
        #         # if there is no retain, then set retain to uncondition
        #         if erase_concept_sampled[0] == erase_concept_sampled[1]:
        #             target_latents = neutral_latents.clone().detach()
                    
        #     with finetuner:
        #         # calculate the unlearned latent in finetuned model
        #         negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                
        #     positive_latents.requires_grad = False
        #     neutral_latents.requires_grad = False
            
        #     # \epsilon_{\theta*}(x_t, t) - \eta [\epsilon_{\theta*}(x_t, c, t) - \epsilon_{\theta*}(x_t, t)]
        #     loss = criteria(negative_latents, # the unlearned latent in finetuned model
        #                     target_latents # the retain latent in original model
        #                     - (negative_guidance # the negative guidance
        #                        * (positive_latents # the unlearned latent in original model
        #                             - neutral_latents))) # the uncondition latent in original model
            
        #     d.append(
        #         {"i": i,
        #         "loss": loss.item(),
        #         "target": erase_concept_sampled[0],
        #         }                
        #     )
        #     loss.backward()
        #     optimizer.step()
            
            
            # if (i+1) % 20 == 0 and i > 0:
        orig_images_dict = {}
        unlearn_images_dict = {}
        orig_images_paths = []
        unlearn_images_paths = []
        prompts = test_list
        n_imgs = 5
        seed = 42
        img_save_path = f'{save_path}/imgs/'
        # if i==20:
        for prompt in prompts.split(','):
            orig_images_dict[prompt] = diffuser(prompt, 
                                            n_steps=50, 
                                            n_imgs=n_imgs,
                                            generator=torch.Generator().manual_seed(seed)
                                            )

        # with finetuner:
        #     for prompt in prompts.split(','):
        #         unlearn_images_dict[prompt] = diffuser(prompt, 
        #                                         n_steps=50, 
        #                                         n_imgs=n_imgs,
        #                                         generator=torch.Generator().manual_seed(seed)
        #                                         )

        for k, v in orig_images_dict.items():
            if not os.path.exists(f"{img_save_path}/orig_{k.strip()}"):
                os.makedirs(f"{img_save_path}/orig_{k.strip()}", exist_ok=True) 
                orig_images_paths.append(f"{img_save_path}/orig_{k}")
            for q, img in enumerate(v):
                img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}.png')


        # for k, v in unlearn_images_dict.items():
        #     if not os.path.exists(f"{img_save_path}/unlearned_{k.strip()}"):
        #         os.makedirs(f"{img_save_path}/unlearned_{k.strip()}", exist_ok=True)
        #         unlearn_images_paths.append(f'{img_save_path}/unlearned_{k}')
        #     for q, img in enumerate(v):
        #         img[0].save(f'{img_save_path}/unlearned_{k.strip()}/{q}.png')
                
        del orig_images_dict, unlearn_images_dict
        # print(f"Saved images {i} iters ")
        # torch.cuda.empty_cache()
        
        # torch.save(finetuner.state_dict(), model_save_path)
        # import csv
        # with open(f"{save_path}/ESD_{erase_concept_sampled}.csv", "w", newline="") as f:
        #     # Create a CSV DictWriter object
        #     writer = csv.DictWriter(f, fieldnames=d[0].keys())
        #     # Write the header
        #     writer.writeheader()
        #     # Write the rows
        #     for row in d:
        #         writer.writerow(row)
                
        # del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents, d
        # torch.cuda.empty_cache()       
        
        
        return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
    except Exception as e:
        print(e)
        return FunctionOutput(status=1, message=f"Error occurred: {e}")


def execute_esd_unlearn_with_retain(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    retain_concept: Annotated[str, "Concept to retain, separated by comma"]=None,
	retain_beta: Annotated[float, "Retain loss coefficient"]=2.0,
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'xattn',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'coding/models',
    device: Annotated[str, "Device to use"] = 'cuda:0',
) -> FunctionOutput:
    
    # Get the file name by parameters
    name = f"""esd-{erase_concept.lower().replace(' ','').replace(',','-')}_from_{erase_from.lower().replace(' ','').replace(',','-') if erase_from is not None else ""}_retain_{retain_concept.lower().replace(' ','').replace(',','-') if retain_concept is not None else ""}-{retain_beta}-{train_method}_{negative_guidance}-epochs_{iterations}"""
    
    # confirm the saving path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    save_path = f'{save_path}/{name}.pt'
    
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
    
    retain_concept = retain_concept.split(',')
    retain_concept = [a.strip() for a in retain_concept]
    
    print(retain_concept)

            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    erase_concept = erase_concept_    
    print(erase_concept)
    
    test_list = 'airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,saling ship,fishing boat,pirate ship,warship'

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
                r_index = np.random.choice(len(retain_concept), 1, replace=False)[0]
                retain_concept_sampled = retain_concept[r_index]
                
                # unconditional text embeddings
                neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
                # unlearn text embedding
                positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
                # erase from text embedding
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
                # retain text embedding
                retain_text_embeddings = diffuser.get_text_embeddings([retain_concept_sampled],n_imgs=1)
                
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
                # the erase from latent in original model
                target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                # the retain latent in original model
                retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain_text_embeddings, guidance_scale=1)
                
                # if there is no retain, then set retain to uncondition
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
                    
            with finetuner:
                # calculate the unlearned latent in finetuned model
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                new_retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain_text_embeddings, guidance_scale=1)
                
            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            
            anchor_criteria = criteria(negative_latents, # the unlearned latent in finetuned model
                            target_latents # the retain latent in original model
                            - (negative_guidance # the negative guidance
                               * (positive_latents # the unlearned latent in original model
                                    - neutral_latents))# the uncondition latent in original model
                            )
            retain_criteria = criteria(retain_latents, new_retain_latents)
            
            # \epsilon_{\theta*}(x_t, t) - \eta [\epsilon_{\theta*}(x_t, c, t) - \epsilon_{\theta*}(x_t, t)]
            loss = anchor_criteria + retain_beta * retain_criteria # retain concepts 
            
            d.append(
                {"i": i,
                "anchor_loss": anchor_criteria.item(),
                "target": erase_concept_sampled[0],
                "retain_loss": retain_criteria.item(),
                "retain": retain_concept_sampled
                }                
            )
                
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0 and i > 0:
                orig_images_dict = {}
                unlearn_images_dict = {}
                orig_images_paths = []
                unlearn_images_paths = []
                prompts = test_list
                n_imgs = 5
                seed = 42
                img_save_path = 'exp/retain_cat_wolf'
                if i==10:
                    for prompt in prompts.split(','):
                        orig_images_dict[prompt] = diffuser(prompt, 
                                                        n_steps=50, 
                                                        n_imgs=n_imgs,
                                                        generator=torch.Generator().manual_seed(seed)
                                                        )

                with finetuner:
                    for prompt in prompts.split(','):
                        unlearn_images_dict[prompt] = diffuser(prompt, 
                                                        n_steps=50, 
                                                        n_imgs=n_imgs,
                                                        generator=torch.Generator().manual_seed(seed)
                                                        )
                    
                for k, v in orig_images_dict.items():
                    if not os.path.exists(f"{img_save_path}/orig_{k}"):
                        os.makedirs(f"{img_save_path}/orig_{k}", exist_ok=True) 
                        orig_images_paths.append(f"{img_save_path}/orig_{k}")
                    for q, img in enumerate(v):
                        img[0].save(f'{img_save_path}/orig_{k}/{q}_step{i}.png')


                for k, v in unlearn_images_dict.items():
                    if not os.path.exists(f"{img_save_path}/unlearned_{k}"):
                        os.makedirs(f"{img_save_path}/unlearned_{k}", exist_ok=True)
                        unlearn_images_paths.append(f'{img_save_path}/unlearned_{k}')
                    for q, img in enumerate(v):
                        img[0].save(f'{img_save_path}/unlearned_{k}/{q}_step{i}.png')
                        
                del orig_images_dict, unlearn_images_dict
                torch.cuda.empty_cache()
            
            
        torch.save(finetuner.state_dict(), save_path)
        
        import csv
        with open("mycsvfile_retain_" + str(retain_beta) +".csv", "w", newline="") as f:
            # Create a CSV DictWriter object
            writer = csv.DictWriter(f, fieldnames=d[0].keys())

            # Write the header
            writer.writeheader()

            # Write the rows
            for row in d:
                writer.writerow(row)
                
        del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents, d
        torch.cuda.empty_cache()   
        
        return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
    except Exception as e:
        return FunctionOutput(status=1, message=f"Error occurred: {e}")


def generate_image(    
    prompts: Annotated[str, "Retain prompts to generate images, separated by comma"],
    model_path: Annotated[str, "Path to the model"],
    save_path: Annotated[str, "Path to save the images"],
    n_imgs: Annotated[int, "Number of images to generate"] = 5,
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'xattn',
    seed: Annotated[int, "Seed for random number generator"] = 42,
    device: Annotated[str, "Device to use"] = 'cuda:0',
):
    diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser.safety_checker = None
    finetuner = FineTunedModel(diffuser, train_method=train_method)
    finetuner.load_state_dict(torch.load(model_path))
    
    orig_images_dict = {}
    unlearn_images_dict = {}
    orig_images_paths = []
    unlearn_images_paths = []
    
    for prompt in prompts.split(','):
        orig_images_dict[prompt] = diffuser(prompt, 
                                        n_steps=50, 
                                        n_imgs=n_imgs,
                                        generator=torch.Generator().manual_seed(seed)
                                        )

    with finetuner:
        for prompt in prompts.split(','):
            unlearn_images_dict[prompt] = diffuser(prompt, 
                                            n_steps=50, 
                                            n_imgs=n_imgs,
                                            generator=torch.Generator().manual_seed(seed)
                                            )
        
    for k, v in orig_images_dict.items():
        if not os.path.exists(f"{save_path}/orig_{k}"):
            os.makedirs(f"{save_path}/orig_{k}", exist_ok=True) 
            orig_images_paths.append(f"{save_path}/orig_{k}")
        for i, img in enumerate(v):
            img[0].save(f'{save_path}/orig_{k}/{i}.png')
        

    for k, v in unlearn_images_dict.items():
        if not os.path.exists(f"{save_path}/unlearned_{k}"):
            os.makedirs(f"{save_path}/unlearned_{k}", exist_ok=True)
            unlearn_images_paths.append(f'{save_path}/unlearned_{k}')
        for i, img in enumerate(v):
            img[0].save(f'{save_path}/unlearned_{k}/{i}.png')

    return FunctionOutput(status=0, message=f"Images generated successful. The path of the generated images from the original model is {orig_images_paths}, and path of the generated images from unlearned model is {unlearn_images_paths}")


def random_selection(
    json_file: Annotated[str, "Path to the JSON file"],
    test_cases: Annotated[int, "Number of test cases to generate"]=1,
):
    import json
    with open(json_file, "r") as f:
        data = json.load(f)[json_file.split("/")[-1].split(".")[0]]
        
    for obj in data:
        for test_case in range(test_cases):
            
            execute_esd_unlearn(
                erase_concept=obj["concept"],
                erase_from="",
                test_list=obj["concept"] + ','+','.join(obj["train-coref"] + obj["train-retain"] + obj["test-coref"] + obj["test-retain"]),
                train_method='xattn-strict',
                iterations=500,
                negative_guidance=1, 
                lr=2e-5,
                save_path='/data/users/yyx/Newgens/orig/ip',
                device='cuda:1'
                )
        
if __name__ == "__main__":
    random_selection("../data/raw/ip.json")