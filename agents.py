import autogen

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent


config = autogen.config_list_from_json('llmconfig.json')
config.pop("agentops_api_key")

vision = MultimodalConversableAgent(name="Vision",
    system_message='''vision. You are a expert of image and caption generation.
    ''',
    llm_config=config,)