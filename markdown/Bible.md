# Libraries to Install


```python
!pip install gradio llama-index-core llama-index-llms-ollama llama-index-embeddings-ollama
```

    Requirement already satisfied: gradio in /home/hotaisle/miniconda3/lib/python3.13/site-packages (5.35.0)
    Requirement already satisfied: llama-index-core in /home/hotaisle/miniconda3/lib/python3.13/site-packages (0.12.46)
    Requirement already satisfied: llama-index-llms-ollama in /home/hotaisle/miniconda3/lib/python3.13/site-packages (0.6.2)
    Requirement already satisfied: llama-index-embeddings-ollama in /home/hotaisle/miniconda3/lib/python3.13/site-packages (0.6.0)
    Requirement already satisfied: aiofiles<25.0,>=22.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (24.1.0)
    Requirement already satisfied: anyio<5.0,>=3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (4.9.0)
    Requirement already satisfied: audioop-lts<1.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.2.1)
    Requirement already satisfied: fastapi<1.0,>=0.115.2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.115.14)
    Requirement already satisfied: ffmpy in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.6.0)
    Requirement already satisfied: gradio-client==1.10.4 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (1.10.4)
    Requirement already satisfied: groovy~=0.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.1.2)
    Requirement already satisfied: httpx>=0.24.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.28.1)
    Requirement already satisfied: huggingface-hub>=0.28.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.33.2)
    Requirement already satisfied: jinja2<4.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (3.1.6)
    Requirement already satisfied: markupsafe<4.0,>=2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (3.0.2)
    Requirement already satisfied: numpy<3.0,>=1.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (2.2.6)
    Requirement already satisfied: orjson~=3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (3.10.18)
    Requirement already satisfied: packaging in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (24.2)
    Requirement already satisfied: pandas<3.0,>=1.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (2.2.3)
    Requirement already satisfied: pillow<12.0,>=8.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (11.3.0)
    Requirement already satisfied: pydantic<2.12,>=2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (2.11.7)
    Requirement already satisfied: pydub in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.25.1)
    Requirement already satisfied: python-multipart>=0.0.18 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.0.20)
    Requirement already satisfied: pyyaml<7.0,>=5.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (6.0.2)
    Requirement already satisfied: ruff>=0.9.3 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.12.2)
    Requirement already satisfied: safehttpx<0.2.0,>=0.1.6 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.1.6)
    Requirement already satisfied: semantic-version~=2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (2.10.0)
    Requirement already satisfied: starlette<1.0,>=0.40.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.46.2)
    Requirement already satisfied: tomlkit<0.14.0,>=0.12.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.13.3)
    Requirement already satisfied: typer<1.0,>=0.12 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.16.0)
    Requirement already satisfied: typing-extensions~=4.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (4.12.2)
    Requirement already satisfied: uvicorn>=0.14.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio) (0.35.0)
    Requirement already satisfied: fsspec in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio-client==1.10.4->gradio) (2025.5.1)
    Requirement already satisfied: websockets<16.0,>=10.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from gradio-client==1.10.4->gradio) (15.0.1)
    Requirement already satisfied: idna>=2.8 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from anyio<5.0,>=3.0->gradio) (3.7)
    Requirement already satisfied: sniffio>=1.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pandas<3.0,>=1.0->gradio) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pandas<3.0,>=1.0->gradio) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pandas<3.0,>=1.0->gradio) (2025.2)
    Requirement already satisfied: annotated-types>=0.6.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pydantic<2.12,>=2.0->gradio) (0.6.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pydantic<2.12,>=2.0->gradio) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from pydantic<2.12,>=2.0->gradio) (0.4.1)
    Requirement already satisfied: click>=8.0.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from typer<1.0,>=0.12->gradio) (8.2.1)
    Requirement already satisfied: shellingham>=1.3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from typer<1.0,>=0.12->gradio) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from typer<1.0,>=0.12->gradio) (13.9.4)
    Requirement already satisfied: aiohttp<4,>=3.8.6 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (3.12.13)
    Requirement already satisfied: aiosqlite in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (0.21.0)
    Requirement already satisfied: banks<3,>=2.0.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (2.1.3)
    Requirement already satisfied: dataclasses-json in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (0.6.7)
    Requirement already satisfied: deprecated>=1.2.9.3 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.2.18)
    Requirement already satisfied: dirtyjson<2,>=1.0.8 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.0.8)
    Requirement already satisfied: filetype<2,>=1.2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.2.0)
    Requirement already satisfied: llama-index-workflows<2,>=1.0.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.0.1)
    Requirement already satisfied: nest-asyncio<2,>=1.5.8 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (3.5)
    Requirement already satisfied: nltk>3.8.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (3.9.1)
    Requirement already satisfied: requests>=2.31.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (2.32.3)
    Requirement already satisfied: setuptools>=80.9.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (80.9.0)
    Requirement already satisfied: sqlalchemy>=1.4.49 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from sqlalchemy[asyncio]>=1.4.49->llama-index-core) (2.0.41)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (8.5.0)
    Requirement already satisfied: tiktoken>=0.7.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (0.9.0)
    Requirement already satisfied: tqdm<5,>=4.66.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (4.67.1)
    Requirement already satisfied: typing-inspect>=0.8.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (0.9.0)
    Requirement already satisfied: wrapt in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-core) (1.17.2)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (6.6.3)
    Requirement already satisfied: propcache>=0.2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from aiohttp<4,>=3.8.6->llama-index-core) (1.20.1)
    Requirement already satisfied: griffe in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from banks<3,>=2.0.0->llama-index-core) (1.7.3)
    Requirement already satisfied: platformdirs in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from banks<3,>=2.0.0->llama-index-core) (4.3.7)
    Requirement already satisfied: llama-index-instrumentation>=0.1.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-workflows<2,>=1.0.1->llama-index-core) (0.2.0)
    Requirement already satisfied: ollama>=0.5.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from llama-index-llms-ollama) (0.5.1)
    Requirement already satisfied: certifi in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from httpx>=0.24.1->gradio) (2025.4.26)
    Requirement already satisfied: httpcore==1.* in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from httpx>=0.24.1->gradio) (1.0.9)
    Requirement already satisfied: h11>=0.16 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.16.0)
    Requirement already satisfied: filelock in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from huggingface-hub>=0.28.1->gradio) (3.18.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from huggingface-hub>=0.28.1->gradio) (1.1.5)
    Requirement already satisfied: joblib in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from nltk>3.8.1->llama-index-core) (1.5.1)
    Requirement already satisfied: regex>=2021.8.3 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from nltk>3.8.1->llama-index-core) (2024.11.6)
    Requirement already satisfied: six>=1.5 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from python-dateutil>=2.8.2->pandas<3.0,>=1.0->gradio) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from requests>=2.31.0->llama-index-core) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from requests>=2.31.0->llama-index-core) (2.3.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.2.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.19.1)
    Requirement already satisfied: mdurl~=0.1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.0)
    Requirement already satisfied: greenlet>=1 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from sqlalchemy>=1.4.49->sqlalchemy[asyncio]>=1.4.49->llama-index-core) (3.2.3)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from typing-inspect>=0.8.0->llama-index-core) (1.1.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from dataclasses-json->llama-index-core) (3.26.1)
    Requirement already satisfied: colorama>=0.4 in /home/hotaisle/miniconda3/lib/python3.13/site-packages (from griffe->banks<3,>=2.0.0->llama-index-core) (0.4.6)


# Sefaria

[Sefaria](https://www.sefaria.org/) is a free, open-source library of Jewish texts, including the **Tanakh**, **Talmud**, **Midrash**, **Halakha**, **Kabbalah**, and thousands of commentaries and modern works.  
It provides full access in **Hebrew and English** (with many other translations), cross-referenced links between texts, and powerful study tools.

Key features:
- 📚 Comprehensive collection of classical Jewish sources
- 🌐 Free online access with an open API
- 🔗 Interlinked texts (e.g., verses → Talmud → commentaries)
- 🌍 Supports collaborative translation and commentary projects
- 🖥️ Easy to integrate into apps, research, or personal study

Sefaria is widely used by students, scholars, educators, and anyone interested in exploring Jewish texts digitally.


### Download the 5 Books of Moses
* 'Genesis'
* 'Exodus'
* 'Leviticus'
* 'Numbers'
* 'Deuteronomy'

In your code, the `params` dict is being passed to `requests.get` like this:

```python
params={'context': 0, 'commentary': 0, 'pad': 0, 'lang': 'en'}
```

That means the actual request URL being sent to the Sefaria API will look like:

```
https://www.sefaria.org/api/texts/Genesis?context=0&commentary=0&pad=0&lang=en
```

For each book name, the parameters expand to:

* **`context=0`** → Don’t include surrounding context verses/lines.
* **`commentary=0`** → Don’t include commentary in the response.
* **`pad=0`** → Don’t pad the text with empty strings where text is missing.
* **`lang=en`** → Request the text in English.

So for `"Exodus"`, for example, the request URL would be:

```
https://www.sefaria.org/api/texts/Exodus?context=0&commentary=0&pad=0&lang=en
```

and the response you get back is JSON, where `data.get("text")` is a list of chapter/verse strings in English.

👉 If you’d like, I can show you how the raw JSON looks for one book so you know what to expect. Want me to pull the first few verses of Genesis from the API and show you the structure?


## 📊 Fetch Torah Text
1. Into dictionary by book
2. Into dataframe split by chapter and verse.


```python
import pandas as pd
import requests

rows = []
for book_name in ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']:
    url = f'https://www.sefaria.org/api/texts/{book_name}'
    response = requests.get(url, params={'pad': 0, 'lang': 'en'})
    data = response.json()
    text = data.get("text")
    for chapter_idx,chapter in enumerate(text,1):
        for verse_idx,verse in enumerate(chapter,1):
            rows.append((book_name, chapter_idx, verse_idx, verse))

df_bible = pd.DataFrame(rows,columns = ['book','chapter','verse','text'])
df_bible['text'] = df_bible['text'].str.split("<").str[0]
df_bible.dropna(inplace=True)
```


```python
df_bible
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Genesis</td>
      <td>1</td>
      <td>1</td>
      <td>When God began to create</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Genesis</td>
      <td>1</td>
      <td>2</td>
      <td>the earth being unformed and void, with darkne...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Genesis</td>
      <td>1</td>
      <td>3</td>
      <td>God said, “Let there be light”; and there was ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Genesis</td>
      <td>1</td>
      <td>4</td>
      <td>God saw that the light was good, and God separ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Genesis</td>
      <td>1</td>
      <td>5</td>
      <td>God called the light Day and called the darkne...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5841</th>
      <td>Deuteronomy</td>
      <td>34</td>
      <td>8</td>
      <td>And the Israelites bewailed Moses in the stepp...</td>
    </tr>
    <tr>
      <th>5842</th>
      <td>Deuteronomy</td>
      <td>34</td>
      <td>9</td>
      <td>Now Joshua son of Nun was filled with the spir...</td>
    </tr>
    <tr>
      <th>5843</th>
      <td>Deuteronomy</td>
      <td>34</td>
      <td>10</td>
      <td>Never again did there arise in Israel a prophe...</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>Deuteronomy</td>
      <td>34</td>
      <td>11</td>
      <td>for the various signs and portents that יהוה s...</td>
    </tr>
    <tr>
      <th>5845</th>
      <td>Deuteronomy</td>
      <td>34</td>
      <td>12</td>
      <td>and for all the great might and awesome power ...</td>
    </tr>
  </tbody>
</table>
<p>5846 rows × 4 columns</p>
</div>



# LlamaIndex

**LlamaIndex** (formerly **GPT Index**) is a Python library that connects **LLMs** (like GPT-4 or LLaMA) to your own data for building **retrieval-augmented generation (RAG)** apps.

### 🔍 How It Works

LLMs can’t access your PDFs, databases, or APIs directly. LlamaIndex bridges the gap by:

1. **Loading** data (PDFs, Notion, SQL, APIs, etc.).
2. **Indexing** it into structures (vectors, keywords, lists).
3. **Retrieving** relevant chunks for a query.
4. **Providing context** to the LLM for accurate answers.

### 🧱 Core Pieces

* **Data Connectors**: Import from files, sites, or databases.
* **Indices**: Store and organize data for retrieval.
* **Retrievers**: Find the right chunks per query.
* **Engines**: Pair retrievers + LLMs for apps like chatbots and document Q\&A.

### 📖 Example

For the Bible, LlamaIndex can:

* Load and chunk the text,
* Build a vector index,
* Retrieve verses about “kosher” or “Exodus 20,”
* Feed them to GPT-4/LLaMA-3 to answer naturally, e.g.:
  *“The Ten Commandments appear in Exodus 20.”*

---



# 🧠 Define Ollama LLM and Embedding Models

We’ll use ```nomic-embed-text``` for embeddings and Qwen ```qwq``` for answering questions:

**NOTE** using a reasoning model is highly recommended in my experience.

* Pull these models just in case you don't have them installed yet locally

## Prerequisite: 

* Install ```curl -fsSL https://ollama.com/install.sh | sh```




```python
!ollama list
```

    NAME                       ID              SIZE      MODIFIED       
    qwq:latest                 009cb3f08d74    19 GB     30 minutes ago    
    nomic-embed-text:latest    0a109f422b47    274 MB    30 minutes ago    
    deepseek-r1:latest         6995872bfe4c    5.2 GB    6 weeks ago       
    qwen3:latest               500a1f067a9f    5.2 GB    6 weeks ago       
    qwen:latest                d53d04290064    2.3 GB    6 weeks ago       


### Set the LLM + Embedding Model to Ollama Models


```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Set LLM Model
Settings.llm = Ollama("qwq:latest",stream=True)

# Set Embedding Model
Settings.embed_model = OllamaEmbedding('nomic-embed-text:latest')
```

### Parse The Torah DataFrame into Documents for Lllama Index


```python
from llama_index.core import Document

documents = []

for (book, chapter), group in df_bible.groupby(['book', 'chapter']):
    chapter_text = "\n".join(group['text'])
    metadata = {
        "book": book,
        "chapter": int(chapter),
        "verse_start": int(group['verse'].min()),
        "verse_end": int(group['verse'].max()),
    }
    documents.append(Document(text=chapter_text, metadata=metadata))
```


```python
documents[0]
```




    Document(id_='36cbc8dc-e15d-4cf8-bfad-b90a2ffdf59f', embedding=None, metadata={'book': 'Deuteronomy', 'chapter': 1, 'verse_start': 1, 'verse_end': 46}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text_resource=MediaResource(embeddings=None, data=None, text='These are the words that Moses addressed to all Israel on the other side of the Jordan.—Through the wilderness, in the Arabah near Suph, between Paran and Tophel, Laban, Hazeroth, and Di-zahab,\nit is eleven days from Horeb to Kadesh-barnea by the Mount Seir route.\nIt was in the fortieth year, on the first day of the eleventh month, that Moses addressed the Israelites in accordance with the instructions that יהוה had given him for them,\nafter he had defeated Sihon king of the Amorites, who dwelt in Heshbon, and King Og of Bashan, who dwelt at Ashtaroth [and]\nOn the other side of the Jordan, in the land of Moab, Moses undertook to expound this Teaching. He said:\nOur God יהוה spoke to us at Horeb, saying: You have stayed long enough at this mountain.\nStart out and make your way to the hill country of the Amorites and to all their neighbors in the Arabah, the hill country, the Shephelah,\nSee, I place the land at your disposal. Go, take possession of the land that יהוה swore to your fathers Abraham, Isaac, and Jacob, to assign to them and to their heirs after them.\nThereupon I said to you, “I cannot bear the burden of you by myself.\nYour God יהוה has multiplied you until you are today as numerous as the stars in the sky.—\nMay יהוה, the God of your ancestors, increase your numbers a thousandfold, and bless you as promised.—\nHow can I bear unaided the trouble of you, and the burden, and the bickering!\nPick from each of your tribes candidates\nYou answered me and said, “What you propose to do is good.”\nSo I took your tribal leaders, wise and experienced men,\nI charged your magistrates at that time as follows, “Hear out your fellow Israelites, and decide justly between one party and the other—be it a fellow Israelite or a stranger.\nYou shall not be partial in judgment: hear out low and high alike. Fear neither party,\nThus I instructed you, at that time, about the various things that you should do.\nWe set out from Horeb and traveled the great and terrible wilderness that you saw, along the road to the hill country of the Amorites, as our God יהוה had commanded us. When we reached Kadesh-barnea,\nI said to you, “You have come to the hill country of the Amorites which our God יהוה is giving to us.\nSee, your God יהוה has placed the land at your disposal. Go up, take possession, as יהוה, the God of your fathers, promised you. Fear not and be not dismayed.”\nThen all of you came to me and said, “Let us send agents\nI approved of the plan, and so I selected from among you twelve participants, one representative from each tribe.\nThey made for the hill country, came to the wadi Eshcol, and spied it out.\nThey took some of the fruit of the land with them and brought it down to us. And they gave us this report: “It is a good land that our God יהוה is giving to us.”\nYet you refused to go up, and flouted the command of your God יהוה.\nYou sulked\nWhat kind of place\nI said to you, “Have no dread or fear of them.\nNone other than your God יהוה, who goes before you, will fight for you, just as [God] did for you in Egypt before your very eyes,\nand in the wilderness, where you saw how your God יהוה carried you, as a householder\nYet for all that, you have no faith in your God יהוה,\nwho goes before you on your journeys—to scout the place where you are to encamp—in fire by night and in cloud by day, in order to guide you on the route you are to follow.”\n יהוה heard your loud complaint and, becoming angry, vowed:\nNot one of those involved, this evil generation, shall see the good land that I swore to give to your fathers—\nnone except Caleb son of Jephunneh; he shall see it, and to him and his descendants will I give the land on which he set foot, because he remained loyal to יהוה.—\nBecause of you יהוה was incensed with me too, saying: You shall not enter it either.\nJoshua son of Nun, who attends you, he shall enter it. Imbue him with strength, for he shall allot it to Israel.—\nMoreover, your little ones who you said would be carried off, your children who do not yet know good from bad, they shall enter it; to them will I give it and they shall possess it.\nAs for you, turn about and march into the wilderness by the way of the Sea of Reeds.\nYou replied to me, saying, “We stand guilty before יהוה. We will go up now and fight, just as our God יהוה commanded us.” And [the men among] you each girded yourselves with war gear and recklessly\nBut יהוה said to me, “Warn them: Do not go up and do not fight, since I am not in your midst; else you will be routed by your enemies.”\nI spoke to you, but you would not listen; you flouted יהוה’s command and willfully marched into the hill country.\nThen the Amorites who lived in those hills came out against you like so many bees and chased you, and they crushed you at Hormah in Seir.\nAgain you wept before יהוה; but יהוה would not heed your cry or give ear to you.\nThus, after you had remained at Kadesh all that long time,', path=None, url=None, mimetype=None), image_resource=None, audio_resource=None, video_resource=None, text_template='{metadata_str}\n\n{content}')



### Index Them (I.E. Get Embeddings)


```python
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents,show_progress=True)
```


    Parsing nodes:   0%|          | 0/187 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/244 [00:00<?, ?it/s]

index.storage_context.vector_store._data

```python

```

### Create Query Engine + Chat Engine
* Note I used similarity top k = 10. I recommend a highish number for accuracy here


```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine

query_engine = index.as_query_engine(similarity_top_k=10)

chat_memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine = query_engine,
    chat_memory = chat_memory
)
```


```python

```

# Display the Gradio ChatBot


```python
import time
import gradio as gr

def chat_inference(message,history):
    response = chat_engine.stream_chat(message)
    so_far = ''
    for token in response.response_gen:
        if token != '<think>':
            so_far+=str(token)
            yield so_far
gr.ChatInterface(
    chat_inference,
    title='Ask the bible').launch(debug=True,share=True)
```

    * Running on local URL:  http://127.0.0.1:7860
    
    Could not create share link. Please check your internet connection or our status page: https://status.gradio.app.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>



```python

```


```python

```
