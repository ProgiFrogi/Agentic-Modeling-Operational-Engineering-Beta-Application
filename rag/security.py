import os
import re
from typing import Tuple

from llm_guard.model import Model
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
import langdetect
import emoji

class SecurityChecker:
    def __init__(self, filter_non_english: bool = True, filter_emoji: bool = True, filter_images: bool = True):
        self.scanner = PromptInjection(model=Model(os.getenv("SECURITY_MODEL")), threshold=0.95, match_type=MatchType.CHUNKS)
        self.filter_non_english = filter_non_english
        self.filter_emoji = filter_emoji
        self.filter_images = filter_images

    def cleanup(self, text: str) -> str:
        if self.filter_non_english:
            text = self.__filter_non_english(text)
        if self.filter_emoji:
            text = self.__filter_emoji(text)
        if self.filter_images:
            text = self.__filter_images(text)
        return text

    def check(self, text: str, is_code: bool) -> Tuple[bool, str]:
        """
        Checks text for security, returns tuple of (secure, new text)
        """
        if not is_code and self.filter_non_english:
            text = self.__filter_non_english(text)
        if not is_code and self.filter_emoji:
            text = self.__filter_emoji(text)
        if not is_code and self.filter_images:
            text = self.__filter_images(text)
        return self.scanner.scan(text)[1], text

    def __filter_non_english(self, text: str) -> str:
        """
        Filters out non-English text, keeping only English sentences/words.
        Uses language detection to identify and remove non-English content.
        """
        if not text:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        english_sentences = []

        for sentence in sentences:
            if sentence.strip():
                try:
                    lang = langdetect.detect(sentence)
                    if lang == 'en':
                        english_sentences.append(sentence)
                except langdetect.lang_detect_exception.LangDetectException:
                    english_sentences.append(sentence)

        filtered_text = ' '.join(english_sentences)

        return filtered_text.strip()

    def __filter_emoji(self, text: str) -> str:
        """
        Filters out emojis and other emoticons from the text.
        Uses the emoji library to detect and remove emojis.
        """
        if not text:
            return text

        try:
            filtered_text = emoji.replace_emoji(text, '')
            filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
            return filtered_text
        except Exception as e:
            print(f"Error filtering emojis: {e}")
            return text

    def __filter_images(self, text: str) -> str:
        """
        Filters out markdown image content from the text.
        Removes both standard markdown images and HTML img tags.
        """
        if not text:
            return text

        try:
            filtered_text = re.sub(r'!?\[[^]]*]\([^)]*\)', '', text)
            filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
            return filtered_text
        except Exception as e:
            print(f"Error filtering images: {e}")
            return text
