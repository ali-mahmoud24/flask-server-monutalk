import google.generativeai as genai

import textwrap
import numpy as np
import pandas as pd
import pickle

# Used to securely store your API key

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')

genai.configure(api_key=API_KEY)

df = pd.read_csv('./Chat/embeddingv3.csv')

with open('./Chat/embeddingdatav1.pickle', 'rb') as file:
    # Deserialize and retrieve the variable from the file
    loaded_data = pickle.load(file)

df['Embeddings'] = loaded_data

embed_model = 'models/embedding-001'
def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=embed_model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] # Return text from index with max value

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot for an interactive application that answers questions using text from the reference passage included below. \
  answer as if ur the character talking. for example if your asked about the sphinx, answer as if you are the sphinx talking \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a simple audience, so be sure to break down complicated big paragraphs and answer in small sentences relative to the question only and \
  strike a friendly and converstional tone. \
  additionally give question recomendations to continue the conversation. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

def gen_ans(x):
  query = x
  model = 'models/embedding-001'

  request = genai.embed_content(model=model,
                              content=query,
                              task_type="retrieval_query")
  passage = find_best_passage(query, df)
  prompt = make_prompt(query, passage)

  generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
  }

  model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=generation_config,

  )
  answer = model.generate_content(prompt)
  return answer.text


def output_of_to_genai(query):
    model = 'models/embedding-001'

    request = genai.embed_content(model=model,
                                  content=query,
                                  task_type="retrieval_query")
    passage = find_best_passage(query, df)
    prompt = make_prompt(query, passage)

    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash-latest",
      generation_config=generation_config,

    )
    answer = model.generate_content(prompt)
    return answer.text