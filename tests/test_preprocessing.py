import pytest
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.preprocessing import clean_text

def test_simple_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("This is a test.") == "this is a test"
    assert clean_text("Python is great!") == "python is great"


def test_with_punctuation_and_spaces():
    assert clean_text("  Multiple   spaces   and punctuation!!! ") == "multiple spaces and punctuation"
    assert clean_text("Newline\nand tab\tcharacters.") == "newline and tab characters"
    assert clean_text("Special characters: @#$%^&*()") == "special characters"


def test_with_uppercase_letters():
    assert clean_text("UPPERCASE TEXT") == "uppercase text"
    assert clean_text("HUManS are THE WiLdest AnImals ") == "humans are the wildest animals"