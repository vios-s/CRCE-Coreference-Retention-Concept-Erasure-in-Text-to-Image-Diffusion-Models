import autogen
from autogen import Agent

import agentops

import common_utils
import esd_tools
import eval_tools
from agents import vision

config = autogen.config_list_from_json('llmconfig.json')
agentops.init(config["agentops_api_key"])
config.pop("agentops_api_key")

## Agent definitions

from autogen.code_utils import create_virtual_env
venv_dir = ".venv"
venv_context = create_virtual_env(venv_dir)
tool_venv_dir = ".tool_venv"
tool_venv_context = create_virtual_env(tool_venv_dir)

code_executor = autogen.coding.LocalCommandLineCodeExecutor(
    timeout=300,
    work_dir="coding",
    virtual_env_context=venv_context,
)

tool_executor = autogen.coding.LocalCommandLineCodeExecutor(
    timeout=300,
    work_dir="coding",
    virtual_env_context=tool_venv_context,
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


coref_finder = autogen.AssistantAgent(
    name="Coref_Finder",
    system_message="""
        Coref_Finder.
        You are an excellent assistant for a concept understanding task. 
        The user will give you an prompt that you will use, and for the later unlearning task. 
        If the input is empty, please ask ther user to provide a concept to unlearn.
        Please generate a list of accurate descriptions for co-referential of the input concept.
        The co-referential list should be as close to the concept of the input prompt as possible. 
        The co-referential list should not contain the keyword of the input concept. 
        A example: The unlearning concept is 'Eiffel Tower'.
        A ideal return should be 
        coreferential_list = ['Parisian Iron Lady', 'Famous Tower in Paris', 'A famous landmark of Paris', 
                                "Paris's Iconic Monument", "The metallic lacework giant of Paris"]
    """,
    llm_config=config
)


retain_finder = autogen.AssistantAgent(
    name="Retain_Finder",
    system_message="""
        Retain_Finder.
        You are an excellent assistant for a concept understanding task. 
        The user will give you an prompt that you will use, and for the later unlearning task. 
        The input could be empty so you can generate some normal random concepts, but not affect the unlearn and co-referntial set.
        Please generate a list of accurate descriptions for co-referential of the input concept.
        The retain list should be as close to the concept of the input prompt as possible. 
        The retain list should not contain the keyword of the input concept. 
        A example: The retain concept is 'Donald Duck'.
        A ideal return should be 
        retain_list = ["Disney's famous duck", "The star of 'DuckTales'"]
    """,
    llm_config=config
)



summary_and_plan = autogen.AssistantAgent(
    name="Planner",
    system_message="""
        Planner. 
        The user will start the conversation with a unlearning request.        
        Suggest a plan based on the following pipeline, may improvise a bit: 
        
        1. The user will provide a concept to unlearn.
        2. The Coref_Finder will generate a list of accurate descriptions for co-referential of the input concept, until the user Approves.
        3. The Retain_Finder will first ask the user what is the retain concept, and then try to generate something more related, until the user Approves.
        4. The Planner will summarize the conversation, the unlearn set, coreferential set, and the retain set, and suggest an unlearning plan, until the user Approves.
        5. The Engineer will use the suggested plan to execute the unlearning code, mostly use tool, and may generate some necessary code to run, until the unlearning is finished.
        6. The Critic will generate some code to evaluate the unlearning result, until the user Approves.
        7. If the user not satisfied with the unlearning result, go back to planner and remake the plan for unlearning.
                
        Revise the plan based on feedback from admin, until admin approval.
        The plan may involve an engineer and a critic who can write code and invoke tools.
        Explain the plan first. 
        Be clear which step is performed by an engineer, and which step is performed by a critic.
    """,
    llm_config=config,
)


engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=config,
    system_message="""
    I'm Engineer. I'm expert in python programming. I'm executing code tasks required by Admin.
    
    For ESD method:
        First get the unlearn, coreferential, and retain set. The unlearn set and coreferntial set should be concatenated. This is for `erase_concept` in the `execute_esd_unlearn` function.
        Then the retain set is for `erase_from` in the `execute_esd_unlearn` function. If the retain set is empty, the `erase_from` should be None.
        The `execute_esd_unlearn` function will be called to start unlearning process, after the successful unlearning, the saved_path will be returned.
    
    """,
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


t_executor = autogen.UserProxyAgent(
    name="Tool_Executor",
    system_message="""
        Tool Executor. 
        Run function calls.
    """,
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE_ALL"),
    code_execution_config={
        "executor": tool_executor
    }
)


critic = autogen.AssistantAgent(
    name="Critic",
    system_message='''
    Critic. 
    
    Your job is to evaluate the result of the unlearning task. Remember not to change the model itself, just get the model outputs and evaluate the results.
    
    First, you need to generate code to make the model inference some images, based on the unlearning, coreferential, and retain sets.
    
    Install packages if needed, using diffusers libary, the base model is `StableDiffuser`.
    
    Second, you use some metrics to evaluate the results, and generate some code to calculate the metrics.

    The task may involve saving some data as a plot or as a csv file. In this case, ensure the files will be saved to the local computer.

    
    The code needs to be executed by the "Executor" AI agent, so avoid saying "save the code and execute it in your computer." 
    ''',
    llm_config=config,
)


def determine_state(messages):
    """
    Try to extract an explicit state from the most recent message.
    If not available, use keyword checks on the message content.
    """
    if not messages:
        return "INIT"
    last_msg = messages[-1]
    # Prefer explicit metadata if available.
    if "state" in last_msg:
        return last_msg["state"]
    content = last_msg.get("content", "")
    # Fallback to keyword-based detection.
    if "COREF_APPROVED" in content:
        return "COREF_APPROVED"
    elif "RETAIN_APPROVED" in content:
        return "RETAIN_APPROVED"
    elif "PLAN_APPROVED" in content:
        return "PLAN_APPROVED"
    elif "CRITIC_APPROVED" in content:
        return "CRITIC_APPROVED"
    else:
        return "INIT"

def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
    messages = groupchat.messages
    current_state = determine_state(messages)
    
    if len(messages) <= 1:
        return coref_finder 
    
    if last_speaker is user_proxy:
        if current_state == "COREF_APPROVED" and messages[-2]["name"] == "Coref_Finder":
            # if the last message is approved, let the engineer to speak
            return retain_finder
        elif current_state == "RETAIN_APPROVED" and messages[-2]["name"] == "Retain_Finder":
            # if the last message is approved, let the engineer to speak
            return summary_and_plan
        elif current_state == "PLAN_APPROVED" and messages[-2]["name"] == "Planner":
            return engineer
        elif current_state == "CRITIC_APPROVED":# and messages[-2]["name"] == "Engineer":
        # Finally, when the evaluation is approved, hand control back to the user for closure.
            return critic
        elif messages[-2]["name"] == "Coref_Finder":
            # if it is the coref stage, let the coref_finder to continue
            return coref_finder
        elif messages[-2]["name"] == "Retain_Finder":
            # if the last message is from the retain finder, let the retain finder to continue
            return retain_finder
        elif messages[-2]["name"] == "Planner":
            return summary_and_plan
        elif messages[-2]["name"] == "Engineer":
            return engineer
        elif messages[-2]["name"] == "Critic":
            return critic

    if last_speaker is engineer:
        
        last_content = messages[-1].get("content", "")
        if "```python" in last_content:
            return c_executor
        elif "Error" in last_content:
            return engineer
        else:
            return user_proxy
        
    elif last_speaker is c_executor:
        if "exitcode: 1" in messages[-1]["content"]:
            # if the last message indicates an error, let the engineer to speak
            return critic
        else:
            # otherwise, let the scientist to speak
            return user_proxy

    if last_speaker is critic:
        # If the critic has generated code, run it via the code executor.
        if "```python" in messages[-1].get("content", ""):
            return c_executor
        elif "Error" in last_content:
            return critic
        else:
            return user_proxy

    else:
        # In any undefined case, default to the user proxy.
        return user_proxy


groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, c_executor, t_executor, coref_finder, retain_finder, summary_and_plan, critic],
    messages=[],
    max_round=500,
    speaker_selection_method=custom_speaker_selection_func,
    enable_clear_history=True,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)


# Register the tool signature with the assistant agent.
engineer.register_for_llm(name="list_dir", description="list the directory")(common_utils.list_dir)
critic.register_for_llm(name="list_dir", description="list the directory")(common_utils.list_dir)
# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="list_dir")(common_utils.list_dir)

engineer.register_for_llm(name="get_concepts", description="Convert list to string, seperated by comma")(common_utils.get_concepts)
user_proxy.register_for_execution(name="get_concepts")(common_utils.get_concepts)

engineer.register_for_llm(name="concat_list", description="Concatenate two lists")(common_utils.concat_list)
user_proxy.register_for_execution(name="concat_list")(common_utils.concat_list)

engineer.register_for_llm(name="execute_esd_unlearn", description="""Execute ESD unlearn.
                        There are three kinds of default settings to choose from for different concepts to erase:
                        1. artist, art style: {negative_guidance: 1, iterations: 200, train_method: 'xattn', lr: 1e-5 };
                        2. nudity: {negative_guidance: 1, iterations: 200, train_method: 'full', lr: 1e-5};
                        3. common object: {negative_guidance: 1, iterations: 200, train_method: 'full', lr: 1e-5}.
                        
                        Choose the appropriate setting based on the concept to erase.                       
                        
                        """)(esd_tools.execute_esd_unlearn)
user_proxy.register_for_execution(name="execute_esd_unlearn")(esd_tools.execute_esd_unlearn)

critic.register_for_llm(name="generate_image", description="Generate images from the original model and unlearned model, returned with a dict contain images.")(esd_tools.generate_image)
user_proxy.register_for_execution(name="generate_image")(esd_tools.generate_image)

critic.register_for_llm(name="gen_Caption", description="Generate captions for a list of images.")(eval_tools.gen_Caption)
user_proxy.register_for_execution(name="gen_Caption")(eval_tools.gen_Caption)

critic.register_for_llm(name="calc_CLIPScore", description="calculate the CLIP Score of images, using the image folder path as input")(eval_tools.calc_CLIPScore)
user_proxy.register_for_execution(name="calc_CLIPScore")(eval_tools.calc_CLIPScore)

chat_result = user_proxy.initiate_chat(
    manager,
    message=""" Hello, I have an unlearning request.
""",
)