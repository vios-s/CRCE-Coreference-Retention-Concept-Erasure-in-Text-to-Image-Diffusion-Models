import os
import torch

from typing import Annotated
from pydantic import BaseModel


class FunctionOutput(BaseModel):
    status: int
    message: str




    
def list_dir(directory: Annotated[str, "Directory to check."]) -> FunctionOutput:
    files = os.listdir(directory)
    return FunctionOutput(status=0, message=f"Files: {files}")


def get_concepts(concepts: Annotated[list, "Get or append concepts and convert to string, separated by comma"]) -> FunctionOutput:
    s_concepts = ", ".join(concepts)
    return FunctionOutput(status=0, message=f"Concepts: {s_concepts}")


def concat_list(list1: Annotated[list, "List 1"], list2: Annotated[list, "List 2"]) -> FunctionOutput:
    concat_list = list1 + list2
    return FunctionOutput(status=0, message=f"Concated List: {concat_list}")



