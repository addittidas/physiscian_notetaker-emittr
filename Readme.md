# Physician Notetaker Pipeline

## Overview
The Physician Notetaker Pipeline is a single-file Python application that demonstrates various natural language processing (NLP) techniques for extracting and summarizing medical information from patient-physician transcripts. It utilizes the spaCy library for Named Entity Recognition (NER) and Hugging Face transformers for summarization and sentiment analysis.

## Features
- **NER Extraction**: Extracts Symptoms, Diagnosis, Treatment, and Prognosis using spaCy with fallback rule-based extraction.
- **Keyword Extraction**: Implements a simple RAKE-like approach using Part-of-Speech (POS) tagging and noun chunks.
- **Summarization**: Provides templates for structured summaries and optional transformer summarizer hooks (T5/BART).
- **Sentiment Analysis**: Includes scaffolding for sentiment analysis and intent detection using Hugging Face transformers.
- **SOAP Note Generation**: Generates SOAP notes based on the extracted information.

## Requirements
- Python 3.x
- spaCy 3.6.0
- Hugging Face Transformers
- Additional libraries: sentencepiece, torch, sklearn

## Setup
Run the following commands to set up the environment:

```bash
pip install spacy==3.6.0
python -m spacy download en_core_web_sm
pip install transformers sentencepiece torch sklearn
```

## Usage
This script can be run as a standalone Python script or executed cell-by-cell in a Jupyter Notebook. To run the script, execute:

```bash
python chat.py
```

## Example
The script includes a demo using a provided transcript, which showcases the extraction and summarization capabilities.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [spaCy](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)