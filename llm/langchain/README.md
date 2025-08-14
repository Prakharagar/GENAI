pip install -qU langchain-openai
pip install -U langchain_ollama
pip install -U "langchain-chroma>=0.1.2"
pip install -qU langchain-community faiss-cpu
pip install jq
pip install --upgrade pypdf
pip install pdfplumber

Installing SQL lite in Windows

For Windows:

Download the SQLite Precompiled Binaries:

Visit the SQLite download page: SQLite Download Page.https://www.sqlite.org/download.html
Download the appropriate binary for Windows (e.g., sqlite-tools-win32-x86-*.zip).
Extract the ZIP File:

Extract the ZIP file to a directory of your choice (e.g., C:\sqlite).
Add SQLite to System PATH:

Open the Start menu, search for "Environment Variables," and open the "Edit the system environment variables" dialog.
Click "Environment Variables."
Under "System variables," find and select the Path variable, then click "Edit."
Click "New" and add the path to the directory where you extracted sqlite3.exe (e.g., C:\sqlite).
Click "OK" to close all dialogs.
Verify Installation:

Open a new command prompt and type sqlite3 to check if it’s recognized.
2. Using SQLite in Python
If you don’t need the command-line tool and prefer to work within Python, you can use the sqlite3 module, which is included with Python’s standard library. Here’s how you can create and interact with an SQLite database directly from a Python script:

