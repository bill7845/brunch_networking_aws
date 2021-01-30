import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

import pandas as pd

## load csv
@st.cache(allow_output_mutation=True)
def load_data():
    credentials = service_account.Credentials.from_service_account_file(".credential/brunch-networking-07958d4e3d41.json")
    project_id = 'brunch-networking-303012'
    client = bigquery.Client(credentials = credentials, project=project_id)

    query_job = client.query(
    """
    SELECT * FROM `brunch-networking-303012.brunch_networking.brunch_all_text` LIMIT 100
    """
    )

    df = pd.query_job.to_dataframe()

    return df



## main ##
def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("",["Home", "App 실행"])
    
    st.table(load_data())

if __name__ == "__main__":
    main()
