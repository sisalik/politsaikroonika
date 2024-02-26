import os
import re
import time
from dataclasses import dataclass
from typing import List, Union

import openai
from estnltk import Text
from loguru import logger

import politsaikroonika.utils as utils
from tts_preprocess_et.convert import convert_sentence

# Video generation prompt for the first and last shots with the reporter
REPORTER_PROMPT = (
    "norwegian adult male reporter talking into microphone, (tony hawk hair:0.9), "
    "90s bomber jacket, portrait shot, looking at camera, standing on 80s "
    "russian city street"
)
# Append style parameters to each video generation prompt
VIDEO_STYLE_EXTRA = ", russia, eastern europe"
# Minimum number of words in the generated script
SCRIPT_WORDS_MIN = 90
# Maximum number of words in the generated script
SCRIPT_WORDS_MAX = 130
# Default OpenAI model to use for text generation
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo"
# Default OpenAI temperature parameter (0-1). Higher values result in more randomness.
OPENAI_DEFAULT_TEMPERATURE = 0.7


@dataclass
class SystemMessage:
    """A message from the system to the AI."""

    content: str

    def as_dict(self):
        return {"role": "system", "content": self.content}


@dataclass
class UserMessage:
    """A message from the user to the AI."""

    content: str

    def as_dict(self):
        return {"role": "user", "content": self.content}


@dataclass
class AssistantMessage:
    """A message from the AI to the user."""

    content: str

    def as_dict(self):
        return {"role": "assistant", "content": self.content}


def _prompt_openai_model(
    prompt,
    max_tokens=256,
    temperature=OPENAI_DEFAULT_TEMPERATURE,
    allow_truncated=False,
    max_attempts=10,
    model=OPENAI_DEFAULT_MODEL,
):
    return _prompt_openai_model_with_context(
        messages=[UserMessage(prompt)],
        max_tokens=max_tokens,
        temperature=temperature,
        allow_truncated=allow_truncated,
        max_attempts=max_attempts,
        model=model,
    )


def _prompt_openai_model_with_context(
    messages: List[Union[SystemMessage, UserMessage, AssistantMessage]],
    max_tokens: int = 256,
    temperature: float = OPENAI_DEFAULT_TEMPERATURE,
    allow_truncated: bool = False,
    max_attempts: int = 10,
    model: str = OPENAI_DEFAULT_MODEL,
):
    """Initialize OpenAI API and make a request."""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    for attempt in range(max_attempts - 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[m.as_dict() for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        # Rate limit errors, in this application, are caused by the servers being
        # overloaded with other requests. In this case, we can retry the request.
        except openai.error.RateLimitError:
            logger.warning("OpenAI API rate limit error")
            if attempt < max_attempts - 1:
                logger.info(f"Retrying ({attempt + 2}/{max_attempts})...")
                time.sleep(5)
            continue
        response_text = response["choices"][0]["message"]["content"]
        stop_reason = response["choices"][0]["finish_reason"]
        logger.debug(
            f"Tokens: {response['usage']['prompt_tokens']} prompt + "
            f"{response['usage']['completion_tokens']} completion = "
            f"{response['usage']['total_tokens']}"
        )
        logger.debug(f"Response text:\n{response_text}")
        if stop_reason != "stop" and not allow_truncated:
            logger.warning(
                f"OpenAI API returned an unexpected stop reason: {stop_reason}"
            )
            if attempt < max_attempts - 1:
                logger.info(f"Retrying ({attempt + 2}/{max_attempts})...")
            continue

        return response_text
    raise Exception("OpenAI API failed to return a valid response")


def _word_count(text):
    """Count the number of words in a string."""
    text = Text(text)
    text.analyse("morphology")
    # Exclude punctuation
    return len([word for word in text.words if word.partofspeech[0] != "Z"])


def gen_titles(include_topics=None, avoid_topics=None, no_openai=False):
    """Generate a list of potential titles for the episode."""
    if no_openai:
        return utils.ask_user_for_input(
            "title",
            [
                "Vanaproua tõstis oma korteris üles kasvatatud krokodilli politsei "
                "sekkumiseta"
            ],
            item_type=list,
        )
    prompt = f"Act as a TV news editor in Estonia. Your task is to generate a list of 10 strange headlines for a fake crime news TV segment. The headlines should be simple declarative sentences written in the Estonian language only - do not include English translations. Each headline should be no more than 10 words and describe a specific, very weird, unexpected, far-fetched, bizarre yet somewhat plausible criminal event that would be of interest to adults with a sense of humor. Please avoid covering mundane, regular, boring or too gruesome incidents.  Examples of types of crimes: antisocial behaviour, protesting, terrorism, forgery,  hacking, fraud, scamming, smuggling, robbery, theft, traffic accident, identity theft, violence etc. Use only commas (if necessary) and no other punctuation. The tone should be serious, and you should include who was involved and where it happened. To optimize for engagement, ensure the headlines are catchy, click-worthy, and attention-grabbing. Remember to tailor your headlines to the target audience mentioned."
    if include_topics:
        prompt += f" Include the following topics in each headline: {', '.join(include_topics)}."
    if avoid_topics:
        prompt += f" Avoid mentioning the following topics: {', '.join(avoid_topics)}."
    response = _prompt_openai_model(prompt, max_tokens=360)
    # Convert the numbered list to a Python list. Also remove trailing whitespace and
    # full stops and exclamation marks
    title_candidates = re.findall(r"\d+\. (.+?)[\.!]?\s*$", response, re.MULTILINE)
    # If a title begins and ends with quotation marks, remove them
    title_candidates = [
        title[1:-1] if title.startswith('"') and title.endswith('"') else title
        for title in title_candidates
    ]
    return title_candidates


def select_title(titles):
    """Select the best title index from the list of candidates."""
    if len(titles) == 1:
        return titles[0]
    # Convert list of titles to a string for the prompt
    titles_str = "\n".join(f"{i + 1}. {title}" for i, title in enumerate(titles))
    prompt = f"""
Which one of these sentences in Estonian stands out as the one you would least expect to see as a news headline? Pick the most unexpected and weird one. Only reply with the number.
{titles_str}"""
    for _ in range(10):
        # Only need the number, so allow very short truncated responses. One token is
        # enough to represent small numbers but sometimes the response is in the format
        # "Number x" so allow 3 tokens to be safe.
        title_idx = _prompt_openai_model(
            prompt.strip(), max_tokens=3, allow_truncated=True
        )
        try:
            # Find the number in the response
            title_idx = re.search(r"\d+", title_idx).group()
            title_idx = int(title_idx) - 1
            titles[title_idx]
        except (AttributeError, ValueError, IndexError):
            logger.error(f"Invalid title number: {title_idx}")
            continue
        return title_idx
    raise Exception("Failed to select a title")


def gen_summary(title, include_topics=None, avoid_topics=None, no_openai=False):
    """Generate a summary for the episode."""
    if no_openai:
        return utils.ask_user_for_input(
            "summary",
            "An elderly woman in Estonia raised a crocodile in her apartment without "
            "any police intervention. The woman claimed the crocodile was her late "
            "husband's pet and she couldn't bear to part with it. Neighbors reported "
            "the animal to authorities, but the woman was allowed to keep it after "
            "proving she could care for it properly.",
        )
    filter_prompt = ""
    if include_topics:
        filter_prompt += f" Include the following topics: {', '.join(include_topics)}."
    if avoid_topics:
        filter_prompt += (
            f" Avoid mentioning the following topics: {', '.join(avoid_topics)}."
        )
    prompt = f"""
Imagine there is a crime news article in Estonian titled "{title}". Write a short summary (in the Estonian language) of the story, using the template below. If any details are unknown, be creative and make them up.{filter_prompt}

```
KOHT: [specific random/made-up city/town/village where the event occurred, and a made-up street name (excluding the house number for privacy)]
AEG: [specific time and day when the event occurred]
SÜNDMUSED: [3-sentence summary of the unexpected and strange events that occurred, what led up to them and what happened after]
KAHTLUSALUNE: [description and made-up first name (and optionally the age) of the prime suspect(s)]
MOTIIV: [bizarre reason/explanation/motive as to why the event occurred or the crime was committed - if unknown, make up a reason]
POLITSEI: [1 brief sentence description of the actions that the police have taken which may or may not have been successful]
```"""
    return _prompt_openai_model(
        prompt.strip(), max_tokens=500, allow_truncated=True
    ).strip()


def gen_script(summary, no_openai=False):
    """Generate a script for the episode."""
    if no_openai:
        return utils.ask_user_for_input(
            "script",
            "Tere õhtust ja tere tulemast meie uudistesse! Täna räägime teile ühest "
            "kummalisest juhtumist Eesti linnas. Nimelt avastasid naabrid, et üks "
            "vanem naine kasvatas oma korteris krokodilli! Naine väitis, et tegemist "
            "on tema hiljuti surnud abikaasa lemmikloomaga ning ta ei taha sellest "
            "loobuda. Pärast politsei sekkumist suutis naine tõestada, et ta on looma "
            "eest hoolitsemiseks piisavalt pädev ning krokodill lubati tema juures "
            "edasi elada. Kuidas see võimalik oli? Kas politsei tegi õigesti? Kas on "
            "turvaline elada krokodilli kõrval? Küsimused, mis jätavad meid mõtlema.",
        )
    prompt = f"""
Generate the script for an Estonian police and crime news TV segment. The segment is written in the Estonian language and its short summary is as follows:

```
{summary}
```

Expand the story of the summary above and follow these constraints below in no particular order:
- The word count should be up to {SCRIPT_WORDS_MAX} words
- Start by addressing the viewers of a made-up TV channel and stating the location and time of the event
- Focus on describing the events and consequences at length in a detailed manner, inventing extra details and adding nuance
- Discuss why the crime occurred, or the motives of the criminal
- Use poetic and edgy, yet graphic language with old-fashioned metaphors and proverbs
- Speak somewhat demeaningly of the victims
- Briefly describe the actions of the police officers which may or may not have been successful
- End with one sentence with a thought-provoking statement that is not obvious or cliché, e.g. crime is bad
- Do not address the viewers again or sign off at the end of the segment"""
    script = _prompt_openai_model(
        prompt.strip(), max_tokens=800, allow_truncated=True, model="gpt-4"
    )
    word_count = _word_count(script)
    # If the script is too long or short, try to rewrite it by asking the AI to do it,
    # with the context of the prompt and the script itself
    rewrite_prompt = f"Shorten the above text a tiny bit to about {SCRIPT_WORDS_MAX} words. Keep the original introduction and focus on describing the criminal events. Write in Estonian."
    messages = [
        UserMessage(prompt),
        AssistantMessage(script),
        UserMessage(rewrite_prompt),
    ]
    for attempt in range(10):
        word_count = _word_count(script)
        if SCRIPT_WORDS_MIN <= word_count <= SCRIPT_WORDS_MAX:
            break
        logger.info(
            f"Word count: {word_count}, rewriting the script (attempt {attempt + 1})"
        )
        script = _prompt_openai_model_with_context(
            messages,
            max_tokens=800,
            allow_truncated=True,
            model="gpt-4",
        )
    else:
        raise Exception("Failed to rewrite the script")
    # Sometimes the entire response is in quotes, so remove them
    if script.startswith('"') and script.endswith('"'):
        script = script[1:-1]
    return script


def gen_video_prompts(summary, no_openai=False):
    """Generate video prompts for the episode."""
    if no_openai:
        video_prompts = [
            "elderly woman in floral dress holding a small crocodile, apartment interior with yellow walls, potted plants, cluttered",
            "neighbors gathered outside apartment, pointing and gesturing, police officers in blue uniforms, clipboard, serious expressions",
            "elderly woman smiling, petting the crocodile on her lap, police officers nodding in approval, apartment background with patterned curtains and wooden furniture",
            "close-up of the crocodile's scaly skin and sharp teeth, woman feeding it raw meat, kitchen in background with stainless steel appliances and tiled backsplash",
            "woman walking the crocodile on a leash in a nearby park, green grass, trees, passerby staring in disbelief",
        ]
        video_prompts = [prompt + VIDEO_STYLE_EXTRA for prompt in video_prompts]
        return utils.ask_user_for_input(
            "video prompts",
            "; ".join([REPORTER_PROMPT] + video_prompts + [REPORTER_PROMPT]),
            item_type=list,
        )
    prompt = f"""
Generate prompts for a set of AI-generated illustrations for a police and crime news article, summarised below:

```
{summary}
```

There should be 5 prompts in total. They should:
- be written in terse, simple, basic, easy-to-understand news style English
- avoid mentioning concepts that are too abstract or general to be visualized, e.g. emotions, intentions, motives, analysis, discussion, organisations
- be formatted as a bulleted list
- focus on the main characters and criminal events, not the police officers
- each focus on a single subject and avoiding too many different concepts
- avoid these keywords: aerial view, close-up, crowd, blood, gore, wounds
- avoid mentioning the names of people, places or organisations
- include detailed information about the subject (color, shape, texture, size), background and image style
- be in chronological order to form a coherent story
- be no more than 16 words long

For human/animal subjects, the prompts should follow the structure "(a/an) [description] [subject] (wearing a [clothing]) [doing something] in/on [setting] during [time of day]". For inanimate subjects, use the template "([camera angle] of) [description] [subject] in [setting] during [time of day]".

Examples:
- a tall blond man wearing a hoodie holding a gun in bar during night
- wide angle shot of green rusty door in forest hillside during dusk
- large vats and pipes in abandoned building during night
- angry protestors wearing factory uniforms in factory yard during daytime
- a brown cat lying down on kitchen table during morning"""
    response = _prompt_openai_model(prompt.strip())
    video_prompts = (line.strip() for line in response.splitlines() if line.strip())
    # Remove bullet points and numbering in case there is any
    video_prompts = [
        re.sub(r"^\s*([-*+]|\d+\.)\s*", "", prompt) for prompt in video_prompts
    ]
    # Append the style prompt to each API response
    video_prompts = [prompt + VIDEO_STYLE_EXTRA for prompt in video_prompts]
    return [REPORTER_PROMPT] + video_prompts + [REPORTER_PROMPT]


def split_sentences(script):
    """Split script into sentences using EstNLTK."""
    # Substitute any fancy quotes with normal quotes
    script = script.replace("„", '"').replace("“", '"')
    # Swap any quotes and periods that throw off the sentence splitter
    script = script.replace('."', '".')
    script_text = Text(script)
    script_text.analyse("morphology")
    for sentence in script_text.sentences:
        yield sentence.enclosing_text


def convert_sentences(raw_sentences):
    """Convert sentences to a more pronouncable format suitable for TTS."""
    custom_replacements = [
        # The T is too hard
        ("kurjategija", "kurjadegija"),
        ("kuritegu", "kuridegu"),
        ("kuriteo", "kurideo"),
        # Foreign words
        ("ch", "tš"),
        ("mcdonald's", "mäkdoonalds"),
        ("iphone", "aifoun"),
    ]
    for raw in raw_sentences:
        converted = convert_sentence(raw)
        for old, new in custom_replacements:
            converted = converted.lower().replace(old, new)
        yield converted
