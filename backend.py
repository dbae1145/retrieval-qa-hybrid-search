import re
from pypdf import PdfReader
import base64
import glob
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import tiktoken
from typing import Any, Sequence
import openai
from tqdm.auto import tqdm
from uuid import uuid4
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

#create a sourcepage
def blob_name_from_file_page(filename, page = 0):
    return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"

#extract page map from pdf
def get_document_text(filename):
    offset = 0
    page_map = []

    reader = PdfReader(filename)
    pages = reader.pages
    for page_num, p in enumerate(pages):
        page_text = p.extract_text()
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)

    return page_map

#split texts into chunks (=sections)
def split_text(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[
                end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word  # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[
            start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

#create a filename id
def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

#create text chunks
def create_sections(filename, page_map):
    file_id = filename_to_id(filename)
    sections = []
    for i, (content, pagenum) in enumerate(split_text(page_map)):
        section = {
            "id": f"{file_id}-page-{i}",
            "content": content,
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": filename
        }
        sections.append(section)

    return sections

#initalize embedding model
OPENAI_API_KEY = ''
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
    api_key="",  # app.pinecone.io
    environment=""  # next to api key in console
)

index_name = ''

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )

index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

#iterate through pdf files in the data directory for indexing into the pinecone vector DB
directory = ".\data\*.pdf"

for filename in glob.glob(directory):
    print(filename)
    page_map = get_document_text(filename)
    sections = create_sections(os.path.basename(filename), page_map)

    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(sections)):
        # first get metadata fields for this record
        metadata = {
            'id': str(record['id']),
            'content': record['content'],
            'sourcepage': record['sourcepage'],
            'sourcefile': record['sourcefile']
        }
        # append these to current batches
        texts.append(record['content'])
        metadatas.append(metadata)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

print(index.describe_index_stats())

#user prompt encoding function
def encode(query):
    # create dense vec
    dense_vec = embed.embed_query(query)
    return dense_vec

#format search matches
def format_search_results(query_response):
    formatted_results = []
    for result in query_response['matches']:
        sourcepage = result.metadata['sourcepage']
        content = result.metadata['content']
        formatted_results.append(f"{sourcepage}: {content}")
    return "\n".join(formatted_results)

def get_messages_from_history(system_prompt: str, model_id: str, history: Sequence[dict[str, str]],
                              user_conv: str, few_shots=[], max_tokens: int = 4096) -> []:
    message_builder = MessageBuilder(system_prompt, model_id)

    # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
    for shot in few_shots:
        message_builder.append_message(shot.get('role'), shot.get('content'))

    user_content = user_conv
    append_index = len(few_shots) + 1

    message_builder.append_message(USER, user_content, index=append_index)

    for h in reversed(history[:-1]):
        if h.get("bot"):
            message_builder.append_message(ASSISTANT, h.get('bot'), index=append_index)
        message_builder.append_message(USER, h.get('user'), index=append_index)
        if message_builder.token_length > max_tokens:
            break

    messages = message_builder.messages
    return messages

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

system_message_chat_conversation = """Assistant helps analyzing Tesla's 2020 and 2021 annual reports. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

{injected_prompt}

"""

query_prompt_template = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:

"""

def search(query, history, prompt_override):
    history.append({'user': query})

    concat_all = ''
    for h in history[:-1]:
        concat = f"user's prompt: {h.get('user')} assistant's answer: {h.get('bot')}"
        concat_all = concat_all +' '+concat

    messages =[
        {'role':'system', 'content': query_prompt_template.format(chat_history=concat_all, question=query)},
    ]

    chat_completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        api_key = OPENAI_API_KEY,
        messages=messages,
        temperature=0.0,
        max_tokens=32,
        n=1)

    query_text = chat_completion.choices[0].message.content
    print(f"Query Text: {query_text}")

    dense = encode(query_text)

    matches = index.query(
        vector=dense,
        top_k=3,  # how many results to return
        include_metadata=True
    )

    contents = format_search_results(matches)

    if prompt_override is None:
        system_message = system_message_chat_conversation
    else:
        system_message = system_message_chat_conversation.format(injected_prompt=prompt_override)


    messages = get_messages_from_history(
        system_message + "\n\nSources:\n" + contents,
        'gpt-3.5-turbo',
        history,
        history[-1]["user"],
        max_tokens=4000)

    chat_completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        api_key = OPENAI_API_KEY,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        n=1)

    chat_content = chat_completion.choices[0].message.content

    history[-1]['bot'] = chat_content
    print(f"Chat Response: {chat_content}")

    return history, contents

history = []
query = 'When did the Tesla factory open in Nevada?'
prompt_override = None
history = search(query, history, prompt_override)

query = 'What about in Shanghai?'
history = search(query, history, prompt_override)

query = 'What about in Berlin?'
history = search(query, history, prompt_override)

query = 'What was the original production location for Tesla?'
history = search(query, history, prompt_override)

query = 'What is Tesla operating cash flow over the years?'
history = search(query, history, prompt_override)

query = 'How much was Tesla operating cash flow in 2019?'
history, contents = search(query, history, prompt_override)



