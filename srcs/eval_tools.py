import copy
import os
import subprocess
import torch
import numpy as np
import pandas as pd
import autogen

from typing import Annotated, Literal
from tqdm.auto import tqdm
from PIL import Image
from glob import glob

from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.multimodal import CLIPScore
from transformers import CLIPProcessor, CLIPModel

from common_utils import FunctionOutput
from agents import vision

def imageclassify(
        image_list: Annotated[list, "List of images to classify."],
        prompts_path: Annotated[str, "Path to prompts file."],
        save_path: Annotated[str, "Path to save results."]=None,
        topk: Annotated[int, "Top k classes to return."]=5,
        batch_size: Annotated[int, "Batch size for classification."]=250,
        device: Annotated[str, "Device to use for classification."]='cuda:0',        
    ) -> FunctionOutput:

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    
    preprocess = weights.transforms()

    images = []
    
    for i in image_list:
        batch = preprocess(i)
        images.append(batch)
    
    scores = {}
    categories = {}
    indexes = {}
    
    for k in range(1,topk+1):
        scores[f'top{k}']= []
        indexes[f'top{k}']=[]
        categories[f'top{k}']=[]
        
    if batch_size == None:
        batch_size = len(images)
    if batch_size > len(images):
        batch_size = len(images)
    images = torch.stack(images)
    
    
    # Step 4: Use the model and print the predicted category
    for i in range(((len(image_list)-1)//batch_size)+1):
        batch = images[i*batch_size: min(len(image_list), (i+1)*batch_size)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim = 1)

        for k in range(1,topk+1):
            scores[f'top{k}'].extend(probs[:,k-1].detach().cpu().numpy())
            indexes[f'top{k}'].extend(class_ids[:,k-1].detach().cpu().numpy())
            categories[f'top{k}'].extend([weights.meta["categories"][idx] for idx in class_ids[:,k-1].detach().cpu().numpy()])
    
    if save_path is not None:
        df = pd.read_csv(prompts_path)
        df['case_number'] = df['case_number'].astype('int')
        case_numbers = []
        for i, name in enumerate(image_list):
            case_number = name.split('/')[-1].split('_')[0].replace('.png','').replace('.jpg','')
            case_numbers.append(int(case_number))

        dict_final = {'case_number': case_numbers}

        for k in range(1,topk+1):
            dict_final[f'category_top{k}'] = categories[f'top{k}'] 
            dict_final[f'index_top{k}'] = indexes[f'top{k}'] 
            dict_final[f'scores_top{k}'] = scores[f'top{k}'] 

        df_results = pd.DataFrame(dict_final)
        merged_df = pd.merge(df,df_results)
        merged_df.to_csv(save_path)
        

def calc_CLIPScore(
    image_directory: Annotated[str, "The only directory of images, should not be the image path itself."],
    model_name: Annotated[str, "Pretrained model name."]="openai/clip-vit-base-patch32",
    device: Annotated[str, "Device to use for classification."]='cuda:0',
) -> FunctionOutput:
    
    try:
        clip_score = CLIPScore(model_name_or_path=model_name).to(device)
        score_results = []
        image_paths = glob(f"{image_directory}/*.png")
        image_list = []
        for i in image_paths:
            image_list.append(
                {
                    "path": i,
                    "prompt": i.split("/")[-2].split("_")[-1],
                    "image": Image.open(i)
                }
            )

        for i in image_list:
            img_tensor = torch.tensor(np.array(i["image"])).to(device)
            caption_scores = {
                "prompt": i["prompt"],
                "scores": clip_score(img_tensor, i["prompt"]).detach().cpu().item()
            }
            score_results.append({"scores": caption_scores, "image_path": i["path"]})

        return FunctionOutput(status=0, message=f"CLIP Score: {score_results}")
    except Exception as e:
        return FunctionOutput(status=1, message=str(e))
    
    
def gen_Caption(
    image_directory: Annotated[str, "the only image directory, not the image path itself."],
) -> FunctionOutput:
    
    try:
        image_list = sorted(glob(f"{image_directory}/*.png"))
        
        image_messages = ""
        for i in image_list:
            image_messages += f'<img "{i}">'
            if i != image_list[-1]:
                image_messages += ", "
            else:
                image_messages += "."
        
        res = user_proxy.initiate_chat(
            recipient=vision,
            message=f"""Please generate a caption for the image. 
                    The returned caption should be short and contain only the most important concept or object. 
                    The plot to be analysed is """ + image_messages,
            max_turns=1,
        )
        
        return FunctionOutput(status=0, message=res.chat_history[-1]['content'])
    except Exception as e:
        return FunctionOutput(status=1, message=str(e))

if __name__ == '__main__':
    #print(        calc_CLIPScore("coding/tmp_imgs/orig_Snoopy/")    )
    from autogen.code_utils import create_virtual_env
    venv_dir = ".venv"
    venv_context = create_virtual_env(venv_dir)

    code_executor = autogen.coding.LocalCommandLineCodeExecutor(
        timeout=300,
        work_dir="coding",
        virtual_env_context=venv_context,
    )


    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="""
            Admin. 
            Interact with the engineer to execute code. 
            Plan execution needs to be approved by this admin.
        """,
        llm_config=False,
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_ALL"),
        code_execution_config={
            "executor": code_executor
        }
    )

    c_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        system_message="""
            Code Executor. 
            Run generated code.
        """,
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_ALL"),
        code_execution_config={
            "executor": code_executor
        }
    )
    config = autogen.config_list_from_json('llmconfig.json')
    config.pop("agentops_api_key")
    
    from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

    vision = MultimodalConversableAgent(name="Vision",
        system_message='''vision. You are a expert of image and caption generation.
        ''',
        llm_config=config,)

    critic = autogen.AssistantAgent(
        name="Critic",
        system_message='''
            Critic. You are a critic to evaluate the generated captions.
            ''',
            llm_config=config,
        )
    
    
    
    groupchat = autogen.GroupChat(
        agents=[user_proxy, critic],
        messages=[],
        max_round=500,
        speaker_selection_method='round_robin',
        enable_clear_history=True,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)
    
    chat_result = user_proxy.initiate_chat(
        manager,
        message=""" Hello, please generate image captions from `coding/tmp_imgs/aaa/`.
    """,
    )