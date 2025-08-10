import os
import random
import datetime
import torch
import numpy as np

from typing import Annotated, Literal
from tqdm.auto import tqdm

from sd_utils import StableDiffuser, FineTunedModel

from common_utils import FunctionOutput


_TRAIN_METHOD_TYPES = Literal["xattn", "noxattn", "full", "xattn-strict", "selfattn"]

def execute_ours_unlearn(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    coref_concept: Annotated[str, "Coreferential concepts, separated by comma"]=None,
    coref_certainty: Annotated[list, "Coreferential concepts certainty"]=None,
    coref_alpha: Annotated[float, "Coreferential loss coefficient"]=1.0,
    retain_concept: Annotated[str, "Concept to retain, separated by comma"]=None,
    retain_certainty: Annotated[list, "Retain concepts certainty"]=None,
	retain_beta: Annotated[float, "Retain loss coefficient"]=1.0,
    test_list: Annotated[str, "List of prompts to generate images, seperated by comma"]="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,saling ship,fishing boat,pirate ship,warship",
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'xattn-strict',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'exp',
    device: Annotated[str, "Device to use"] = 'cuda:2',
) -> FunctionOutput:     
    
    save_path = save_path + f"/{erase_concept}"\
                            f"/coref_{len(coref_concept) if coref_concept is not None else ''}"\
                            f"_retain_{len(retain_concept) if retain_concept is not None else ''}"\
                            f"_epochs_{iterations}_time_{datetime.datetime.now()}"
    
    # Get the file name by parameters
    name = f"curri_{erase_concept}_c_{coref_concept[0].lower().replace(' ','').replace(',','-')}_r_{retain_concept[0].lower().replace(' ','').replace(',','-')}"    
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
    
    torch.cuda.empty_cache()
    # The unlearning process
    
    coref_certainty = map_confidence_levels([coref_certainty[i] for i in range(len(coref_concept))])
    retain_certainty = map_confidence_levels([retain_certainty[i] for i in range(len(retain_concept))])
    # print(coref_certainty, retain_certainty)
    
    for i in range(len(coref_certainty)):
        randint = random.randint(0, 100)

        if randint < 33:
            coref_certainty[i] = min(coref_certainty[i] + 0, 1)
        elif randint < 66:
            coref_certainty[i] = max(coref_certainty[i] - 0, 0.2)
            
    for i in range(len(retain_certainty)):
        randint = random.randint(0, 100)

        if randint < 33:
            retain_certainty[i] = min(retain_certainty[i] + 0, 1)
        elif randint < 66:
            retain_certainty[i] = max(retain_certainty[i] - 0, 0.2)
    
    # print(coref_certainty, retain_certainty)
    
    with torch.no_grad():
        # randomly choose a pair
        index = np.random.choice(len(erase_concept), 1, replace=False)[0]
        erase_concept_sampled = erase_concept[index]       
        
        
        
        # unconditional text embeddings
        neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
        # unlearn text embedding
        positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
        # erase from text embedding
        target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
        
    
    for i in pbar:
        with torch.no_grad():
            coref_concept_r, coref_certainty_r, retain_concept_r, retain_certainty_r = selection(coref_concept, 
                                                                                        coref_certainty, 
                                                                                        retain_concept, 
                                                                                        retain_certainty, 
                                                                                        5, 
                                                                                        3)
    
            coref_concept_list = coref_concept_r
            coref_concept_list = [a.strip() for a in coref_concept_list]

            retain_concept_list = retain_concept_r
            retain_concept_list = [a.strip() for a in retain_concept_list]
            # coref text embedding
            coref_text_embeddings = []
            for c in coref_concept_list:
                coref_text_embeddings.append(diffuser.get_text_embeddings([c],n_imgs=1))
            # retain text embedding
            retain_text_embeddings = []
            for r in retain_concept_list:
                retain_text_embeddings.append(diffuser.get_text_embeddings([r],n_imgs=1)) 
        # set up the diffusion model
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
        retain_latents = []
        for r in retain_text_embeddings:
            retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], r, guidance_scale=1))
        # the coref latent in original model
        coref_latents = []
        for c in coref_text_embeddings:
            coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], c, guidance_scale=1))
                        
        # if there is no retain, then set retain to uncondition
        if erase_concept_sampled[0] == erase_concept_sampled[1]:
            target_latents = neutral_latents.clone().detach()
            
            
        # the anchor latent in original model
        anchor_latents = target_latents - (negative_guidance * (positive_latents - neutral_latents))
        # the coref anchor latent in original model
        coref_anchor_latents = []
        for c, coref_latent in zip(coref_concept_list, coref_latents):
            coref_anchor_latents.append(target_latents - (negative_guidance * (coref_latent - neutral_latents)))
            
        with finetuner:
            # calculate the coref latent in finetuned model
            new_coref_latents = []
            for coref_text_embedding in coref_text_embeddings:
                new_coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], coref_text_embedding, guidance_scale=1))
            # calculate the unlearned latent in finetuned model
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            
            new_retain_latents = []
            for retain_text_embedding in retain_text_embeddings:
                new_retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], retain_text_embedding, guidance_scale=1))
            
        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        
        anchor_criteria = criteria(negative_latents, # the unlearned latent in finetuned model
                        anchor_latents)
        
        coref_criteria = 0
        for new_coref_latent, coref_anchor_latent, certainty in zip(new_coref_latents, coref_anchor_latents, coref_certainty_r):
            # randint = random.randint(0, 100)
            # if randint < 33:
            #     certainty = min(certainty + 0.2, 1)
            # elif randint < 66:
            #     certainty = max(certainty - 0.2, 0.2)
            # else:
            #     certainty = certainty
                
            coref_criteria += certainty * criteria(new_coref_latent, coref_anchor_latent)
        coref_criteria = coref_criteria / len(coref_concept_list)
        retain_criteria = 0
        for retain_latent, new_retain_latent, certainty in zip(retain_latents, new_retain_latents, retain_certainty_r):
            # randint = random.randint(0, 100)

            # if randint < 33:
            #     certainty = min(certainty + 0.2, 1)
            # elif randint < 66:
            #     certainty = max(certainty - 0.2, 0.2)
            # else:
            #     certainty = certainty
                
            retain_criteria += certainty * criteria(retain_latent, new_retain_latent)
        retain_criteria = retain_criteria / len(retain_concept_list)
        
        # \epsilon_{\theta*}(x_t, t) - \eta [\epsilon_{\theta*}(x_t, c, t) - \epsilon_{\theta*}(x_t, t)]
        loss = anchor_criteria + coref_alpha * coref_criteria + retain_beta * retain_criteria # retain concepts 
            
        loss.backward()
        optimizer.step()
        
    # if i % 20 == 0 and i > 0:
    orig_images_dict = {}
    unlearn_images_dict = {}
    orig_images_paths = []
    unlearn_images_paths = []
    prompts = test_list
    n_imgs = 80
    seed = 42
    img_save_path = f'{save_path}/imgs/'
    #if i==20:
    # for prompt in prompts.split(','):
    #     orig_images_dict[prompt] = diffuser(prompt, 
    #                                     n_steps=50, 
    #                                     n_imgs=n_imgs,
    #                                     generator=torch.Generator().manual_seed(seed)
    #                                     )

    with finetuner:
        for prompt in prompts.split(','):
            unlearn_images_dict = {}
            unlearn_images_paths = []
            unlearn_images_dict[prompt] = diffuser(prompt, 
                                            n_steps=50, 
                                            n_imgs=n_imgs,
                                            generator=torch.Generator().manual_seed(seed)
                                            )
            print("Generating images for", prompt)
            for k, v in unlearn_images_dict.items():
                if not os.path.exists(f"{img_save_path}/unlearned_{prompt.strip()}"):
                    os.makedirs(f"{img_save_path}/unlearned_{prompt.strip()}", exist_ok=True)
                    unlearn_images_paths.append(f'{img_save_path}/unlearned_{prompt}')
                for q, img in enumerate(v):
                    img[0].save(f'{img_save_path}/unlearned_{prompt.strip()}/{q}.png')
                    
            del unlearn_images_dict
            torch.cuda.empty_cache()
        
    # for k, v in orig_images_dict.items():
    #     if not os.path.exists(f"{img_save_path}/orig_{k.strip()}"):
    #         os.makedirs(f"{img_save_path}/orig_{k.strip()}", exist_ok=True) 
    #         orig_images_paths.append(f"{img_save_path.strip()}/orig_{k}")
    #     for q, img in enumerate(v):
    #         img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}.png')
    torch.cuda.empty_cache()
        
    torch.save(finetuner.state_dict(), model_save_path)
    
    
    del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents
    torch.cuda.empty_cache()   
    
    return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
def execute_ours_random_unlearn(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    coref_concept: Annotated[str, "Coreferential concepts, separated by comma"]=None,
    coref_certainty: Annotated[list, "Coreferential concepts certainty"]=None,
    coref_alpha: Annotated[float, "Coreferential loss coefficient"]=1.0,
    retain_concept: Annotated[str, "Concept to retain, separated by comma"]=None,
    retain_certainty: Annotated[list, "Retain concepts certainty"]=None,
	retain_beta: Annotated[float, "Retain loss coefficient"]=1.0,
    test_list: Annotated[str, "List of prompts to generate images, seperated by comma"]="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,saling ship,fishing boat,pirate ship,warship",
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'xattn-strict',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'exp',
    device: Annotated[str, "Device to use"] = 'cuda:2',
) -> FunctionOutput:
    
    save_path = save_path + f"/{erase_concept}"\
                            f"/coref_{len(coref_concept) if coref_concept is not None else ''}"\
                            f"_retain_{len(retain_concept) if retain_concept is not None else ''}"\
                            f"_epochs_{iterations}_time_{datetime.datetime.now()}"
    
    # Get the file name by parameters
    name = f"ours_{erase_concept}_c_{coref_concept[0].lower().replace(' ','').replace(',','-')}_r_{retain_concept[0].lower().replace(' ','').replace(',','-')}"
    
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
    torch.cuda.empty_cache()

    
    for i in pbar:
        # The unlearning process
        
        with torch.no_grad():
            # randomly choose a pair
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]       
            coref_concept_r, coref_certainty_r, retain_concept_r, retain_certainty_r = selection(coref_concept,
                                                                                                 coref_certainty, 
                                                                                                 retain_concept, 
                                                                                                 retain_certainty, 
                                                                                                 10, 
                                                                                                 2)

            coref_concept_list = coref_concept_r
            coref_concept_list = [a.strip() for a in coref_concept_list]
            
            retain_concept_list = retain_concept_r
            retain_concept_list = [a.strip() for a in retain_concept_list] 
            
            # unconditional text embeddings
            neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
            # unlearn text embedding
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
            # erase from text embedding
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
            # coref text embedding
            coref_text_embeddings = []
            for c in coref_concept_list:
                coref_text_embeddings.append(diffuser.get_text_embeddings([c],n_imgs=1))
            # retain text embedding
            retain_text_embeddings = []
            for r in retain_concept_list:
                retain_text_embeddings.append(diffuser.get_text_embeddings([r],n_imgs=1)) 
        
        # set up the diffusion model
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
        retain_latents = []
        for r in retain_text_embeddings:
            retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], r, guidance_scale=1))
        # the coref latent in original model
        coref_latents = []
        for c in coref_text_embeddings:
            coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], c, guidance_scale=1))
                        
        # if there is no retain, then set retain to uncondition
        if erase_concept_sampled[0] == erase_concept_sampled[1]:
            target_latents = neutral_latents.clone().detach()
            
            
        # the anchor latent in original model
        anchor_latents = target_latents - (negative_guidance * (positive_latents - neutral_latents))
        # the coref anchor latent in original model
        coref_anchor_latents = []
        for c, coref_latent in zip(coref_concept_list, coref_latents):
            coref_anchor_latents.append(target_latents - (negative_guidance * (coref_latent - neutral_latents)))
            
        with finetuner:
            # calculate the coref latent in finetuned model
            new_coref_latents = []
            for coref_text_embedding in coref_text_embeddings:
                new_coref_latents.append(diffuser.predict_noise(iteration, latents_steps[0], coref_text_embedding, guidance_scale=1))
            # calculate the unlearned latent in finetuned model
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            
            new_retain_latents = []
            for retain_text_embedding in retain_text_embeddings:
                new_retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], retain_text_embedding, guidance_scale=1))
            
        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        
        anchor_criteria = criteria(negative_latents, # the unlearned latent in finetuned model
                        anchor_latents)
        
        coref_criteria = 0
        for new_coref_latent, coref_anchor_latent, certainty in zip(new_coref_latents, coref_anchor_latents, coref_certainty_r):
            coref_criteria += certainty * criteria(new_coref_latent, coref_anchor_latent)
        coref_criteria = coref_criteria / len(coref_concept_list)
        retain_criteria = 0
        for retain_latent, new_retain_latent, certainty in zip(retain_latents, new_retain_latents, retain_certainty_r):
            retain_criteria += certainty * criteria(retain_latent, new_retain_latent)
        retain_criteria = retain_criteria / len(retain_concept_list)
        
        # \epsilon_{\theta*}(x_t, t) - \eta [\epsilon_{\theta*}(x_t, c, t) - \epsilon_{\theta*}(x_t, t)]
        loss = anchor_criteria + coref_alpha * coref_criteria + retain_beta * retain_criteria # retain concepts 
            
        loss.backward()
        optimizer.step()
        
    # if i % 20 == 0 and i > 0:
    orig_images_dict = {}
    unlearn_images_dict = {}
    orig_images_paths = []
    unlearn_images_paths = []
    prompts = test_list
    n_imgs = 5
    seed = 42
    img_save_path = f'{save_path}/imgs/'
    #if i==20:
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
    #         orig_images_paths.append(f"{img_save_path.strip()}/orig_{k}")
    #     for q, img in enumerate(v):
    #         img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}.png')


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
    
    
    del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents
    torch.cuda.empty_cache()   
    
    return FunctionOutput(status=0, message=f"unlearned successfully. The model is saved at: {save_path}")
    
def execute_ours_realera_loss_unlearn(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, """Erase attributes from an object, separated by comma, 
                            the length should be either 1 or the same as the erase_concept"""] = None,
    coref_concept: Annotated[str, "Coreferential concepts, separated by comma"]=None,
    coref_alpha: Annotated[float, "Coreferential loss coefficient"]=1.0,
    retain_concept: Annotated[str, "Concept to retain, separated by comma"]=None,
	retain_beta: Annotated[float, "Retain loss coefficient"]=1.0,
    test_list: Annotated[str, "List of prompts to generate images, seperated by comma"]="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,saling ship,fishing boat,pirate ship,warship",
    train_method: Annotated[_TRAIN_METHOD_TYPES, "Training method"] = 'xattn-strict',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'exp/Ours_RealERA',
    device: Annotated[str, "Device to use"] = 'cuda:2',
) -> FunctionOutput:
    save_path = save_path + f"/{erase_concept}"\
                            f"/a_{coref_alpha}_b_{retain_beta}"\
                            f"_epochs_{iterations}"
                            #f"/coref_{coref_concept.lower().replace(' ','').replace(',','-')[:10] if coref_concept is not None else ''}-{coref_alpha}"\
                            #f"_retain_{retain_concept.lower().replace(' ','').replace(',','-')[:10] if retain_concept is not None else ''}-{retain_beta}"\
                            
    
    # Get the file name by parameters
    name = f"ours_{erase_concept}_c_{coref_concept.lower().replace(' ','').replace(',','-')[:10]}_r_{retain_concept.lower().replace(' ','').replace(',','-')[:10]}"
    
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
    
    coref_concept = coref_concept.split(',')
    coref_concept = [a.strip() for a in coref_concept]
    print(coref_concept)
    
    retain_concept = retain_concept.split(',')
    retain_concept = [a.strip() for a in retain_concept]   
    print(retain_concept)
            
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
                # erase from text embedding
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
                
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
            
                # if there is no retain, then set retain to uncondition
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
                    
                    
                # the anchor latent in original model
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
                for coref in coref_concept:
                    # coref text embedding
                    coref_text_embeddings = diffuser.get_text_embeddings([coref],n_imgs=1)
                    # the coref latent in original model
                    # coref_latents = diffuser.predict_noise(iteration, latents_steps[0], coref_text_embeddings, guidance_scale=1)
                    
                    with finetuner:
                        # calculate the coref latent in finetuned model
                        new_coref_latents = diffuser.predict_noise(iteration, latents_steps[0], coref_text_embeddings, guidance_scale=1)
                        # coref_anchor_latents = target_latents - (negative_guidance * (coref_latents - neutral_latents))

                    coref_loss += criteria(new_coref_latents, anchor_latents)
                    
                loss += torch.mean(coref_loss)
                
                for retain in retain_concept:
                    # retain text embedding
                    retain_text_embeddings = diffuser.get_text_embeddings([retain],n_imgs=1)
                    # the retain latent in original model
                    retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain_text_embeddings, guidance_scale=1)
                    with finetuner:
                        new_retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain_text_embeddings, guidance_scale=1)
                        
                    retain_loss += criteria(new_retain_latents, retain_latents)
                    
                loss += torch.mean(retain_loss)
                            
            d.append(
                {"i": i,
                "loss": loss.item(),
                "target": erase_concept_sampled[0],
                }                
            )
                
            loss.backward()
            optimizer.step()
            
        # if i % 20 == 0 and i > 0:
        orig_images_dict = {}
        unlearn_images_dict = {}
        orig_images_paths = []
        unlearn_images_paths = []
        prompts = test_list
        n_imgs = 5
        seed = 42
        img_save_path = f'{save_path}/imgs/'
        #if i==20:
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
            if not os.path.exists(f"{img_save_path}/orig_{k.strip()}"):
                os.makedirs(f"{img_save_path}/orig_{k.strip()}", exist_ok=True) 
                orig_images_paths.append(f"{img_save_path}/orig_{k}")
            for q, img in enumerate(v):
                img[0].save(f'{img_save_path}/orig_{k.strip()}/{q}.png')


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
        with open(f"{save_path}/ours_{erase_concept_sampled}_a_{coref_alpha}_{coref_concept}_b_{retain_beta}_{retain_concept}.csv", "w", newline="") as f:
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

# Define the mapping
confidence_mapping = {
    "Very High": 1.0,
    "High": 0.8,
    "Normal": 0.6,
    "Low": 0.4,
    "Very Low": 0.2
}

# Function to map values
def map_confidence_levels(confidence_list):
    return [confidence_mapping.get(level, None) for level in confidence_list]

def selection(corefs, corefs_certainty, retains,  retains_certainty, coref_num, retain_num):
    
    # sample coref_num corefs and use the same index for the corefs_certainty
    coref_idx = np.random.choice(len(corefs), coref_num, replace=False)
    coref = [corefs[i] for i in coref_idx]
    coref_certainty = [corefs_certainty[i] for i in coref_idx]
    # same for retain
    retain_idx = np.random.choice(len(retains), retain_num, replace=False)
    retain = [retains[i] for i in retain_idx]
    retain_certainty = [retains_certainty[i] for i in retain_idx]
    
    # print(coref, coref_certainty, retain, retain_certainty)
    return coref, coref_certainty, retain, retain_certainty
    
def random_selection(
    json_file: Annotated[str, "Path to the JSON file"],
    test_cases: Annotated[int, "Number of test cases to generate"]=1,
    num_coref: Annotated[int, "Number of coreferential concepts to select"]=10,
    num_retain: Annotated[int, "Number of retain concepts to select"]=1,
):
    import json
    with open(json_file, "r") as f:
        data = json.load(f)[json_file.split("/")[-1].split(".")[0]]
        
    for obj in data:
        for test_case in range(test_cases):
            #sampled_coref, coref_certainty, sampled_retain, retain_certainty = selection(obj, num_coref, num_retain)
            
            print(f"Test case {test_case+1} / {test_cases}")
                        
            execute_ours_unlearn(   
            # execute_ours_realera_loss_unlearn(
            # execute_ours_random_unlearn(
                erase_concept=obj["concept"],
                erase_from="",
                coref_concept=obj["train-coref"],
                coref_certainty = obj["train-coref-certainty"],
                retain_concept=obj["train-retain"],
                retain_certainty = obj["train-retain-certainty"],
                coref_alpha=1,
                retain_beta=1,
                test_list=obj["concept"] + ','+','.join(obj["train-coref"] + obj["train-retain"] + obj["test-coref"] + obj["test-retain"]),
                train_method='xattn-strict',
                iterations=500,
                negative_guidance=1, 
                lr=2e-5,
                save_path='/data/users/yyx/ICCV_2025/Newgens/Ablation_ours/object',
                device='cuda:0'
                )
        
if __name__ == "__main__":
    random_selection("../data/raw/object.json",
                    test_cases=1,
                    num_coref=5,
                    num_retain=3)
    
