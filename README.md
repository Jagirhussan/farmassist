This program contains the source code and debug files for Farm Assist

Farm Assist is a multi-modal chatbot used for summarising information seen across video data. It's built for deployment across 2x Nvidia Jetson Orin Nano's and the users device.

Farm Assist's structure is built on a simple RAG pipeline. The LLM for response generation was TinyLlama 1.1B v1, the VLM was Blip standard captioning, and the text embedding model was from Sentence Transformers. The vector database used was ChromaDB. 

To run Farm Assist, read the research compendium provided to you.

**'./backend':** This is where the main backend server, the LLM / RAG pipeline are implemented, and the beginnings of a 'whisper' (an ASR) implementation have begun. 

**'./frontend':** This is where the frontend and UI code is stored. This folder is run on your laptop.

**'./backend/dataprocessing':** Within this file are the scripts needed to process teh data, run the necessary servers for data processing, and some debug functions to erase and print the vector databse. The vector database "video_db" is also stored in this folder. There are also some debug scripts such as debug_chromadb.py, and clear_chromadb.py

Note: llm_server.py is a redundant server. 
