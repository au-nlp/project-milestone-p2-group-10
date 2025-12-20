import re
from typing import List

def preprocess_episode_transcript(transcript: str, sat_model) -> List[str]:
    """
    Whisper provides tags such as "(laughing)" and "[MUSIC]"
    to mark non-speech audio segments. Remove these and other non-speech symbols,
    normalize whitespace, and split into filtered sentences.
    """
    result = transcript

    result = re.sub(r'\[.*?\]', '', result)
    result = re.sub(r'\{.*?\}', '', result)
    result = re.sub(r'\(.*?\)', '', result)

    result = result.replace('>>', '')
    result = result.replace(' - ', '')

    result = re.sub(r' +', ' ', result)
    result = '\n'.join(line.strip() for line in result.split('\n'))
    result = re.sub(r'\n\n+', '\n\n', result)

    result = result.strip()

    raw_sents = sat_model.split(result, do_paragraph_segmentation=False)
    clean_sents = [s for s in raw_sents if len(s.split()) > 3]

    return clean_sents
