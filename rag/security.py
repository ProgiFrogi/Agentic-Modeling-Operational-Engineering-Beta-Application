import os
import re
from typing import Tuple

from llm_guard.model import Model
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
from llm_guard.util import configure_logger
import langdetect
import emoji

configure_logger('CRITICAL')

class SecurityChecker:
    def __init__(self, filter_non_english: bool = True, filter_emoji: bool = True, filter_images: bool = True, filter_comments: bool = True):
        self.scanner = PromptInjection(model=Model(os.getenv("SECURITY_MODEL"), pipeline_kwargs={"max_length": 512}), threshold=0.95, match_type=MatchType.CHUNKS)
        self.filter_non_english = filter_non_english
        self.filter_emoji = filter_emoji
        self.filter_images = filter_images
        self.filter_comments = filter_comments

    def cleanup(self, text: str, is_code: bool) -> str:
        if not is_code and self.filter_non_english:
            text = self.__filter_non_english(text)
        if not is_code and self.filter_emoji:
            text = self.__filter_emoji(text)
        if not is_code and self.filter_images:
            text = self.__filter_images(text)
        if is_code and self.filter_comments:
            text = self.__filter_comments(text)
        return text

    def check(self, text: str, is_code: bool) -> Tuple[bool, str]:
        """
        Checks text for security, returns tuple of (secure, new text)
        """
        is_safe = True
        if not is_code and self.filter_non_english:
            text = self.__filter_non_english(text)
        if not is_code and self.filter_emoji:
            text = self.__filter_emoji(text)
        if not is_code and self.filter_images:
            text = self.__filter_images(text)
        if not is_code:
            is_safe = self.scanner.scan(text)[1]
        return is_safe, text

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

    def __filter_comments(self, text: str) -> str:
        """
        Filters out markdown comments from the code.
        """
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Check if line has a comment not inside quotes
            in_string = False
            string_char = None
            comment_pos = -1

            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i - 1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False

                if char == '#' and not in_string:
                    comment_pos = i
                    break

            if comment_pos != -1:
                line = line[:comment_pos].rstrip()
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
