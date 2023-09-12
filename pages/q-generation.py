#### imports
import streamlit as st

import pandas as pd

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2


#### helper functions



#### streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


#### clarifai credentials/config
auth = ClarifaiAuthHelper.from_streamlit(st)
pat = auth._pat
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


#### title description
st.title("generating Qs")
st.write('not the racist kind')


#### select dataset, confirm that it contains text inputs
with st.sipper('Loading datasets...'):
  res_ld = stub.ListDatasets(
    service_pb2.ListDatasetsRequest(
      user_app_id = user_app_id
    )
  )

if res_ld:
  if len(res_ld.datasets) > 0:
    dataset = st.selectbox(
      'Select the dataset to label:'
      [x.id for x in res_ld.datasets]
    )
  else:
    st.write('No datasets found in app. This tool requires a dataset with text inputs to function.')
    

#### from dataset, select random input


#### call LLM #1 - generate seed questions



#### pass questions to the human in the loop #1



#### call LLM #2 for each selected question



#### pass questions to human in the loop #2



#### post annotations