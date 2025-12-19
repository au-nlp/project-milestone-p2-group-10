import openai
from bertopic.representation import OpenAI

def zero_shot_prompt():
    client = openai.OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lmstudio"
        )

    prompt = """
    I have a topic represented by the following keywords: [KEYWORDS].
    What is a good, short title for this topic? Give me only the title and nothing else.
    """
    # Now, initialize the BERTopic model with your local client
    representation_model = OpenAI(
        client=client,
        model="local-model",  # LMStudio ignores this; it uses the model you loaded
        delay_in_seconds=2,   # prevent spamming local server
        prompt=prompt         # Pass your custom prompt
    )
    return representation_model