from PIL import Image
from .Settings import Debug
import pyocr

class OCR(object):
    def __init__(self, img):
        """ Accepts an image in PIL format """

        # Pick OCR tool
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise NotImplementedError("No OCR software found - please install Tesseract OCR")
        self.tool = tools[0]
        Debug.log(3, "Using OCR tool {}".format(self.tool.get_name()))

        # Pick OCR language
        langs = self.tool.get_available_languages()
        Debug.log(3, "Available languages {}".format(", ".join(langs)))
        self.lang = langs[0]
        Debug.log(3, "Will use lang '{}'".format(self.lang))

        # Rescale image for better accuracy
        nx, ny = img.size
        self.image = img.resize((nx*3, ny*3), Image.BICUBIC)

    def get_text(self):
        txt = self.tool.image_to_string(
            self.image,
            lang="eng",
            builder=pyocr.builders.TextBuilder()
        )

        return txt.encode("utf8", "ignore")

    def find_phrase(self, phrase, all_matches=False):
        words = phrase.split()
        ocr_words = self.tool.image_to_string(
            self.image,
            lang="eng",
            builder=pyocr.builders.WordBoxBuilder()
        )
        matches = []

        matched_word = 0
        match_components = []

        for ocr_word in ocr_words:
            confidence = self.editDistance(ocr_word.content.encode("utf8", "ignore"), words[matched_word])
            #print confidence
            if confidence < 3:
                # Fuzzy match
                matched_word += 1
                match_components.append((ocr_word, confidence))
                if matched_word == len(words):
                    # Complete match - yay!
                    # Get bounding box from match_components (correcting for earlier scaling)
                    x1 = min([m[0].position[0][0] for m in match_components])/3
                    y1 = min([m[0].position[0][1] for m in match_components])/3
                    x2 = max([m[0].position[1][0] for m in match_components])/3
                    y2 = max([m[0].position[1][1] for m in match_components])/3

                    x = (x2 - x1)/2 + x1
                    y = (y2 - y1)/2 + y1
                    confidence = float(sum([m[1] for m in match_components])) / max(len([m[1] for m in match_components]), 1)

                    # Add match to list
                    matches.append(((x1, y1), (x2-x1, y2-y1), confidence))

                    # Reset match
                    matched_word = 0
                    match_components = []

            else:
                # No match - reset match components
                matched_word = 0
                match_components = []
        # Sort matches by confidence
        matches.sort(key=lambda x: x[2])
        return matches

    def editDistance(self, s1, s2):
        """ Unashamedly stolen from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python """
        if len(s1) < len(s2):
            return self.editDistance(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]