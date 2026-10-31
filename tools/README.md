# Tools used by the daily content task

- `tfw_newpost.py`: "new post" story (1080x1920) built from a feed post's first slide. Needs `tfw_template.py` and `tfw_story.py` from the TFW2PR Carousel Template artifact in the same folder.
- `reel/tfw_reel.py`: reel maker (reel.json -> 1080x1920 MP4 with voice, captions, music, sound effects, plus a cover JPEG). See the docstring for the JSON format and setup (Kokoro TTS model files are downloaded at run time, never committed).
