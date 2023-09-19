#### imports
import streamlit as st
import pandas as pd
import numpy as np
import requests

from google.protobuf.struct_pb2 import Struct

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2


#### helper functions
def get_all_inputs(stub, dataset_id, user_app_id, page_size=1000):
  page_n = 1
  all_inputs = []

  while True:
    res_ldi = stub.ListDatasetInputs(
      service_pb2.ListDatasetInputsRequest(
        user_app_id = user_app_id,
        dataset_id = dataset_id,
        per_page = page_size,
        page = page_n
      )# ,
      # metadata = auth_meta
    )
    
    all_inputs.extend(res_ldi.dataset_inputs)
    
    if len(res_ldi.dataset_inputs) < page_size:
      break
    else:
      page_n += 1

  return all_inputs


def get_all_unlabeled_inputs(stub, dataset_id, user_app_id, page_size=1000, query={"labeled":True}):
  # only grabs the first 1000 filtered inputs, but that should be fine since this is a reductive process?

  query_struct = Struct()
  query_struct.update(query)

  res_pis = stub.PostInputsSearches(
    service_pb2.PostInputsSearchesRequest(
      user_app_id = user_app_id,
      searches = [
        resources_pb2.Search(
          query = resources_pb2.Query(
            filters = [
              # filter for already labeled data (via metadata)
              resources_pb2.Filter(
                negate = True,
                annotation = resources_pb2.Annotation(
                  data = resources_pb2.Data(
                    metadata = query_struct
                  )
                )
              ),
              # filter for dataset id
              resources_pb2.Filter(
                input = resources_pb2.Input(
                  dataset_ids = [dataset_id]
                )
              )
            ]
          )
        )
      ]
    )
  )

  return res_pis.hits


def get_text(auth, url):
  """Download the raw text from the url"""
  h = {"Authorization": f"Key {auth._pat}"}
  response = requests.get(url, headers=h)
  response.encoding = response.apparent_encoding

  return response.text if response else ""


#### streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


#### clarifai credentials/config
auth = ClarifaiAuthHelper.from_streamlit(st)
pat = auth._pat
stub = create_stub(auth)
user_app_id = auth.get_user_app_id_proto()


#### model credentials
model_user = 'openai'
model_app = 'chat-completion'
model_id = 'GPT-3_5-turbo'
model_version = '8ea3880d08a74dc0b39500b99dfaa376'


#### title description
st.title("generating Qs")
st.write('not q-anon')


#### select dataset (todo: confirm that it contains text inputs)

if 'load_dataset' not in st.session_state:
  st.session_state.load_dataset = False

def updated_dataset_id():
  st.session_state.dataset_id = dataset_id

if st.session_state.load_dataset == False:
  with st.spinner('Loading datasets...'):
    res_ld = stub.ListDatasets(
      service_pb2.ListDatasetsRequest(
        user_app_id = user_app_id
      )
    )

    st.session_state.list_datasets = res_ld
    st.session_state.load_dataset = True

if len(st.session_state.list_datasets.datasets) > 0:
  dataset_id = st.selectbox(
    'Select the dataset to label:',
    [x.id for x in st.session_state.list_datasets.datasets],
  )

  st.write(f'selected dataset_id: {dataset_id}')

else:
  st.write('No datasets found in app. This tool requires a dataset with text inputs to function.')

st.divider()




## get all dataset inputs
if 'clicked1' not in st.session_state:
  st.session_state.clicked1 = {1:False,2:False}

def clicked1(button):
  st.session_state.clicked1[button] = True

if st.button('get all unlabeled inputs', on_click=clicked1, args=[1], key='get-all-unlabeled-inputs'):
  st.write(dataset_id)
  all_inputs = get_all_unlabeled_inputs(stub, dataset_id, user_app_id)
  st.session_state.all_inputs = all_inputs

if st.session_state.clicked1[1]:
  st.write('get-all-unlabeled-inputs successful')
  st.write(len(st.session_state.all_inputs))

  if len(st.session_state.all_inputs) == 0:
    st.write('There are no unlabeled inputs in this dataset. Mission accomplished. :tada:')
  
  else:
    #### from dataset, select random input
    # if st.session_state.clicked1[1]:

    st.caption('#### from dataset, select random input')
    random_input = np.random.choice(st.session_state.all_inputs).input
    input_text = get_text(auth, random_input.data.text.url)
    st.caption('Input Text:')
    st.write(input_text)

    st.session_state.input_text = input_text

st.divider()


if 'clicked2' not in st.session_state:
  st.session_state.clicked2 = {1:False,2:False}

def clicked2(button):
  st.session_state.clicked2[button] = True

if st.button('Generate questions', on_click=clicked2, args=[1], key='generate-questions'):
  st.caption('#### call LLM#1 - generate seed questions')
  text_prompt = """
  Generate five questions relevant to the following passage:
  {}
  """

  res_pmo_llm1 = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
      user_app_id = resources_pb2.UserAppIDSet(
        user_id = model_user,
        app_id = model_app
      ),
      model_id = model_id,
      version_id = model_version,
      inputs = [
        resources_pb2.Input(
          data = resources_pb2.Data(
            text = resources_pb2.Text(
              raw = text_prompt.format(input_text)
            )
          )
        )
      ]
    )
  )

  # st.write(res_pmo_llm1.outputs[0].data.text.raw)

  llm1_questions = [
    x.lstrip('12345. ')
    for x
    in res_pmo_llm1.outputs[0].data.text.raw.split('\n')
  ]

  st.session_state.llm1_questions = llm1_questions

if 'llm1_validated' not in st.session_state:
  st.session_state.llm1_validated = []

if st.session_state.clicked2[1]:

  #### pass questions to the human in the loop #1
  st.caption('#### pass questions to human-in-the-loop, round1of2')
  val = [None]* len(st.session_state.llm1_questions)
  for i,llm1_question in enumerate(st.session_state.llm1_questions):
    val[i] = st.checkbox(llm1_question)

  st.write(val)
  st.session_state.llm1_validated = [
    x
    for i,x
    in enumerate(st.session_state.llm1_questions)
    if val[i]
  ]
  
st.divider()
st.write('st.session_state.llm1_validated')
st.write(st.session_state.llm1_validated)

#### call LLM #2 for each selected question
if 'clicked3' not in st.session_state:
  st.session_state.clicked3 = {1:False,2:False}

def clicked3(button):
  st.session_state.clicked3[button] = True

if st.button('Generate questions from questions', on_click=clicked3, args=[1], key='generate-questions-from-questions'):
  st.caption('#### call LLM#2 - generate questions from questions')
  text_prompt = """
  Provided below are a number of questions, each seperated by the pipe character, |. For each question, rewrite the question in five different formats:
  {}
  """

  res_pmo_llm2 = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
      user_app_id = resources_pb2.UserAppIDSet(
        user_id = model_user,
        app_id = model_app
      ),
      model_id = model_id,
      version_id = model_version,
      inputs = [
        resources_pb2.Input(
          data = resources_pb2.Data(
            text = resources_pb2.Text(
              raw = text_prompt.format('|'.join(st.session_state.llm1_validated))
            )
          )
        )
      ]
    )
  )

  st.write('res_pmo_llm2')
  st.write(res_pmo_llm2.outputs[0].data.text.raw)
  st.session_state.llm2_questions = res_pmo_llm2


#### pass questions to human in the loop #2
# if st.session_state.clicked3[1]:

#   #### pass questions to the human in the loop #1
#   st.caption('#### pass questions to human-in-the-loop, round1of2')
#   val = [None]* len(st.session_state.llm2_questions)
#   for i,llm1_question in enumerate(st.session_state.llm2_questions):
#     val[i] = st.checkbox(llm2_questions)


  #### post annotations