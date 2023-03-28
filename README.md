# Auto-Politseikroonika

This is the plan: write the script, generate the audio and video to match and put it all together.

## Script

1. **Generate episode title using GPT API**
   GPT-4 API not available yet (as of 28 March 2023). May need to consider GPT-3.5 or manually generating using ChatGPT.
   Feed it a list of previous episode titles in the prompt.
2. **Generate episode content using GPT API**
   List constraints such as:
   1. Language: Estonian
   2. Word count: enough for a 1-minute video (⭐ work out the word count for this)
   3. Use of old-fashioned metaphors and proverbs
   4. Graphic yet poetic description of events and casualties
   5. Somewhat demeaning towards victims
   6. Start by addressing the viewers and stating the location and time of the event
   7. End with something thought-provoking
3. **Generate prompts for text2video using GPT API**
   ⭐ Work out the number of prompts required to fill out the 1 minute episode duration.

## Audio

1. **Generate audio based on script using Voice-Cloning-App trained on Peeter Võsa**
   May need to retry a few times if the model fails to generate.
   ⭐ Need to figure out how to run this from the command line

## Video

1. **Generate video clips based on prompts using [ModelScope text2video](https://github.com/deforum-art/sd-webui-modelscope-text2video) in SD**
   Change FPS from 15 to 25.
2. **Do frame interpolation using [Flowframes](https://github.com/n00mkrad/flowframes)**
   Increase FPS from 24 to 48 and slowmo 2x.
3. **Upscale video using [FTVSR](https://github.com/researchmm/FTVSR)**
4. **Compose audio and video using ffmpeg**
5. **Prepend title screen**
   Maybe use the actual one or modify it slightly to add a hint to the AI generation.