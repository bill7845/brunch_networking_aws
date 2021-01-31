import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

import pandas as pd
import numpy as np
import pickle
from konlpy.tag import Mecab

## load csv
@st.cache(allow_output_mutation=True)
def load_data():
    credentials = service_account.Credentials.from_service_account_file(".credential/brunch-networking-07958d4e3d41.json")
    project_id = 'brunch-networking-303012'
    client = bigquery.Client(credentials = credentials, project=project_id)

    df = query_job.to_dataframe()

    return df

## laod tfidf vector
@st.cache(allow_output_mutation=True)
def load_tfidf_train_vect():
    tfidf_train_vect = pickle.load(open("pkl_objects/tfidf_train_vect.pkl", 'rb'))
    return tfidf_train_vect

## load classifier
@st.cache
def load_clf():
    clf = clf = pickle.load(open("model/classifier_lg.pkl", 'rb'))
    return clf

# classifier
# 입력받은 문서를 분류하여 분류결과,분류확률 반환
def classify(document, label_dict, tfidf_train_vect):

    # 형태소 분석기(Mecab)
    mecab = Mecab()
    # 입력받은 문서를 길이 2이상의 명사로만 추출하여 재구성
    document = [i[0] for i in mecab.pos(document) if ( ((i[1]=="NNG") or (i[1]=="NNP") and (len(i[0])>1)) )]
    document = " ".join(document)

    clf = load_clf() # 분류모델 load
    X = tfidf_train_vect.transform([document]) # train tfidf vector에 맞춰 입력문서 vectorize
    y = clf.predict(X)[0] # 입력문서의 예측 클래스

    proba = clf.predict_proba(X) # 입력문서의 예측 클래스(확률값) 반환
    proba_max = np.max(proba) # 입력문서의 예측 클래스(확률값) 중 가장 확률값 반환

    return label_dict[y],proba_max,y



## main ##
def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("",["Home", "App 실행"])
    
    tfidf_train_vect = load_tfidf_train_vect()
    label_dict = {0:'육아_이야기', 1:'요리_레시피', 2:'건강_운동', 3:'멘탈관리_심리탐구',4:'문화_예술', 5:'인문학_철학', 6:'쉽게_읽는_역사',
     7:'우리집_반려동물', 8:'사랑_이별',
    9:'감성_에세이', 10:'지구한바퀴_세계여행', 11:'시사_이슈', 12:'IT_트렌드',
    13:'취향저격_영화리뷰',14:'오늘은_이런책', 15:'뮤직_인사이드', 16:'직장인_현실조언',
    17:'스타트업_경험담'}

    ## 개요 페이지. (시작 페이지)
    if app_mode == "Home":
        st.title('< Brunch Networking >')
        st.write('')
        st.subheader('브런치 텍스트 자동분류 및 추천 시스템')
        st.write('')

    ## app 실행 페이지.
    elif app_mode == "App 실행":
        st.sidebar.success('앱 실행중입니다')
        st.title("환영합니다 작가님!")
        st.write("")
        st.write("")

        document = st.text_area("작성하신 글을 입력해주세요.") ## text 입력란
        submit_button = st.button("제출",key='document') # submit 버튼

    #######################################################################
    ## 1. 문서 입력 후 submit 버튼 클릭 시 분류 모델에 의해 분류라벨,확률값 출력
    #######################################################################
    if submit_button:
        label,proba_max,y = classify(document,label_dict,tfidf_train_vect) ## classify 함수에 의해 라벨,확률값
        st.write('작성하신 글은 %d퍼센트의 확률로 \'%s\' 카테고리로 분류됩니다.' %(round((proba_max)*100),label))

    #######################################################################
    ## 2. 분류 결과에 대한 맞춤,틀림 여부 입력받음
    ##      2.1 정답일 경우
    ##      2.2 오답일 경우
    #######################################################################
    category_list = ['<select>','지구한바퀴_세계여행', '시사_이슈',
        'IT_트렌드', '취향저격_영화리뷰', '뮤직_인사이드',
        '육아_이야기', '요리_레시피', '건강_운동', '멘탈관리_심리탐구',
        '문화_예술', '인문학_철학','쉽게_읽는_역사',
        '우리집_반려동물' , '오늘은_이런책', '직장인_현실조언','스타트업_경험담',
        '감성_에세이','사랑_이별']
    st.write("")
    status = st.radio("분류가 알맞게 되었는지 알려주세요!", ("<select>","맞춤", "틀림")) # <select> 기본값

if __name__ == "__main__":
    main()
