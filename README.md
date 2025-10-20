This program contains the source code and debug files for Farm Assist

Farm Assist is a multi-modal chatbot used for summarising information seen across video data. It's built for deployment across 2x Nvidia Jetson Orin Nano's and the users device.

Farm Assist's structure is built on a simple RAG pipeline. The LLM for response generation was TinyLlama 1.1B v1, the VLM was Blip standard captioning, and the text embedding model was from Sentence Transformers. The vector database used was ChromaDB. 

To run Farm Assist, read the research compendium provided to you.

