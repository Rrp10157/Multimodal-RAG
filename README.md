# Mutimodal-RAG (Setup Instructions)

## Overview
Multimodal-RAG is a repository designed for utilizing multimodal retrieval-augmented generation (RAG) techniques to query complex layout unstructured data.

## Prerequisites
- Python 3.12
- Ollama setup

## Installation Steps 
1. Clone the repository
2. Set up virtual environment
3. Install dependencies from requirements.txt
4. Install the ollama to utilize the llms
5. Run the Gradio application for Q&A

## Qucik Start

- git clone https://github.com/Rrp10157/Multimodal-RAG.git
- cd Multimodal-RAG
- py -3.12 -m venv myvenv
- source myvenv\bin\activate
- pip install -r requirements.txt
- python gradio_app.py


Multimodal-RAG/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup (optional)
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── document_processor.py          # PDF processing & chunking
│   ├── embedder.py                   # Embedding generation
│   ├── retriever.py                  # Retrieval logic
│   ├── qa_engine.py                  # QA functionality
│   ├── config.py                     # Configuration settings
│   └── utils.py                      # Utility functions
│
├── data/                             # Data directory
│   ├── input/                        # Input documents
│   └── processed/                    # Processed data
│
├── models/                           # Pre-trained models
│   └── embeddings/                   # Embedding models
│
├── chroma_rag_modular_dense/         # Vector store (modular approach)
├── chroma_rag_final_dense/           # Vector store (final approach)
│
├── gradio_app.py                     # Gradio interface
├── app.py                            # Flask API (optional)
├── RAG_updated_clean.ipynb           # Jupyter notebook
│
├── tests/                            # Unit tests
│   ├── test_document_processor.py
│   ├── test_embedder.py
│   └── test_retriever.py
│
├── docs/                             # Additional documentation
│   ├── ARCHITECTURE.md
│   ├── API_GUIDE.md
│   └── TROUBLESHOOTING.md
│
└── logs/                             # Application logs
