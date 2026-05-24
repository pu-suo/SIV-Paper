#!/usr/bin/env bash
set -e

echo "────────────────────────────────────────────────────────────────"
echo "  SIV Setup"
echo "────────────────────────────────────────────────────────────────"
echo ""

# 1. Python dependencies
echo "▶ Installing Python dependencies..."
pip install -r requirements.txt -q

# 2. spaCy model
echo "▶ Downloading spaCy model (en_core_web_sm)..."
python -m spacy download en_core_web_sm -q

# 3. NLTK data (handled by _bootstrap, but run explicitly for visibility)
echo "▶ Downloading NLTK data..."
python -c "
import nltk
for r in ['wordnet','averaged_perceptron_tagger_eng','punkt_tab','brown']:
    nltk.download(r, quiet=True)
print('  NLTK data OK')
"

# 4. Vampire theorem prover
echo "▶ Installing Vampire theorem prover..."
python -c "
from siv.vampire_interface import setup_vampire, is_vampire_available
if is_vampire_available():
    print('  Vampire already installed')
else:
    result = setup_vampire()
    if result:
        print(f'  Vampire installed at: {result}')
    else:
        print('  WARNING: Vampire installation failed.')
        print('  SIV will work without Vampire but with reduced test resolution.')
        print('  See README.md for manual installation instructions.')
"

# 5. .env template
if [ ! -f .env ]; then
    echo "▶ Creating .env from .env.example..."
    cp .env.example .env
    echo "  IMPORTANT: Edit .env and add your OpenAI API key."
else
    echo "▶ .env already exists (not overwriting)."
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  Verification"
echo "────────────────────────────────────────────────────────────────"
python -c "
from siv.fol_utils import NLTK_AVAILABLE
from siv.vampire_interface import is_vampire_available
print(f'  NLTK FOL parser:  {\"OK\" if NLTK_AVAILABLE else \"MISSING\"} ')
print(f'  Vampire prover:   {\"OK\" if is_vampire_available() else \"NOT FOUND (optional)\"}')
print(f'  spaCy model:      ', end='')
try:
    import spacy; spacy.load('en_core_web_sm'); print('OK')
except: print('MISSING')
import os
key = os.environ.get('OPENAI_API_KEY','')
print(f'  OpenAI API key:   {\"SET\" if key and key != \"sk-your-key-here\" else \"NOT SET — edit .env\"}')
"
echo ""
echo "Setup complete. Run 'python -m siv inspect \"Your sentence here.\"' to test."
