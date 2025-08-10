from typing import Annotated
import os

default_path = "coding/"
esd_path = "erasing"

from pydantic import BaseModel

class FunctionOutput(BaseModel):
    status: int
    message: str

def execute_esd_unlearn(
    erase_concept: Annotated[str, "Concept to erase, separated by comma"],
    erase_from: Annotated[str, "Erase attributes from an object"] = None,
    train_method: Annotated[str, "Training method"] = 'xattn',
    iterations: Annotated[int, "Number of iterations"] = 200,
    negative_guidance: Annotated[float, "Negative guidance"] = 1, 
    lr: Annotated[float, "Learning rate"] = 2e-5,
    save_path: Annotated[str, "Path to save the model"] = 'data/models/',
    device: Annotated[str, "Device to use"] = 'cuda:0',
) -> FunctionOutput:
    parameters = locals()
    parameters_list = [(k, v) for k, v in parameters.items() if v is not None]
    
    cmd = ""
    for k, v in parameters_list:
        if isinstance(v, str) and ',' in v:
            v = f'"{v}"'
        cmd += f"--{k} {v} "
        
    os.system(f"python coding/erasing/esd_diffusers.py " + cmd)
    return FunctionOutput(status=0, message="unlearned successfully.")

