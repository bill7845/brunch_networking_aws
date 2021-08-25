import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import boto3

import pandas as pd
import numpy as np
import json 
import pymysql
import re
import pickle
from konlpy.tag import Mecab
import plotly as plt
from sklearn.metrics.pairwise import cosine_similarity

## load csv
@st.cache(allow_output_mutation=True)
def load_data(y):
    
    
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket="util-brunch-networking", Key="credential/gcp/brunch-networking-07958d4e3d41.json")
    body = obj['Body'].read()

    credentials = service_account.Credentials.from_service_account_info(json.loads(body))
    project_id = 'brunch-networking-303012'
    client = bigquery.Client(credentials = credentials, project=project_id)

    query = """
    SELECT title,publish_date,text,keyword,url FROM `brunch-networking-303012.brunch_networking.ori_brunch_data` WHERE class_num = {class_num} 
    """.format(class_num=y)

    query_job = client.query(query=query)
    df = query_job.to_dataframe()

    return df

def get_s3(key):

    s3 = boto3.client('s3')
    bucket = 'util-brunch-networking'

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj['Body'].read()
    data = pickle.loads(body)

    return data

def laod_data_keyword_sim(top_n_sim):
    credentials = service_account.Credentials.from_service_account_file(".credential/brunch-networking-07958d4e3d41.json")
    project_id = 'brunch-networking-303012'
    client = bigquery.Client(credentials = credentials, project=project_id)
    
    top_n_index = tuple(top_n_sim)
    query = """
    SELECT title,publish_date,text,keyword,url FROM `brunch-networking-303012.brunch_networking.ori_brunch_data` where pk in {top_n_index}
    """.format(top_n_index=top_n_index)

    query_job = client.query(query=query)
    df = query_job.to_dataframe()

    df['text'] = df['text'].apply(lambda x : x[:600])

    return df

## laod tfidf vector
@st.cache(allow_output_mutation=True)
def load_tfidf_train_vect():
    tfidf_train_vect = get_s3(key='train/tfidf_train_vect.pkl')
    return tfidf_train_vect

## load classifier
@st.cache
def load_clf():
    clf = get_s3(key="model_weight/classifier_lg.pkl")
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

## category에 해당하는 keyword 목록 반환
def get_categories(label):
    # category_dict = pickle.load(open("pkl_objects/keyword/keyword_dict.txt", 'rb'))
    with open('pkl_objects/keyword/keyword_dict.pkl', 'rb') as fr:
        category_dict = pickle.load(fr)
    return tuple(category_dict[label]) # streamlit의 multiselect box에서 사용위해 tuple로 반환

## 추천 시스템_1 작성 글 기반
def find_sim_document(df,input_document, y, top_n=3): # 전체 데이터프레임, 입력문서, 입력문서의 예측라벨, 추천글 수
    # 형태소 분석기(mecab)
    mecab = Mecab()

    # 예측 or 수정된 문서 라벨 별 vector,matrix 로드
    each_tfidf_vect = get_s3(key='each_vect/' + str(y) + "tfidf_vect.pkl")
    each_tfidf_matrix = get_s3(key='each_matrix/' + str(y) + "tfidf_matrix.pkl")

    # 입력받은 문서를 길이 2이상의 명사로만 추출하여 재구성
    input_document = [i[0] for i in mecab.pos(input_document) if ( ((i[1]=="NNG") or (i[1]=="NNP") and (len(i[0])>1)) )]
    input_document = " ".join(input_document)

    input_document_mat = each_tfidf_vect.transform([input_document]) # train tfidf vector에 맞춰 입력문서 vectorize
    # 입력문서의 tfidf 행렬과 기존 게시글(특정 label)간에 cosine_similarity
    document_sim = cosine_similarity(input_document_mat, each_tfidf_matrix)

    document_sim_sorted_ind = document_sim.argsort()[:,::-1] # 행별 유사도 값이 높은순으로 정렬

    top_n_sim = document_sim_sorted_ind[:1,:(top_n)] # 유사도가 높은순으로 top_n 만큼
    top_n_sim = top_n_sim.reshape(-1) # index

    df = df.iloc[top_n_sim]
    df.loc[:,'text'] = df['text'].apply(lambda x : x[:600]) # 지면상 300글자씩만

    return df

## 추천 시스템_2 Keyword 기반
def find_sim_keyword(count_vect, keyword_mat, input_keywords, top_n):

  input_keywords_mat = count_vect.transform(pd.Series(input_keywords)) # 입력 받은 키워드를 count_vectorizer
  keyword_sim = cosine_similarity(input_keywords_mat, keyword_mat) # 입력 키워드와 기존 키워드간 cosine_similarity

  keyword_sim_sorted_ind = keyword_sim.argsort()[:,::-1] # 유사도가 높은순으로 정렬

  top_n_sim = keyword_sim_sorted_ind[:1,:(top_n)]
  top_n_sim = top_n_sim.reshape(-1) # 키워드간 유사도가 가장 높은 top_n 게시글의 index 반환
  
  return top_n_sim

## keyword trend 차트
## 2020/01/01부터 입력된 키워드들의 주별 등장횟수를 구하여 반환함
def keyword_trend_chart(df, select_keyword):
    df.index = pd.to_datetime(df['publish_date'],format='%Y-%m-%d') # 게시글별 발행일을 index로
    df = df['keyword']['2020-01-01':].resample('M').sum()

    res_df = pd.DataFrame(columns=select_keyword,index=df.index)
    for keyword in select_keyword:
        keyword_week_count = []
        for week in range(len(df)):
            keyword_week_count.append(df.iloc[week].count(keyword))
        res_df[keyword] = keyword_week_count

    return res_df

## load keyword count_vector
@st.cache(allow_output_mutation=True)
def load_keyword_count_vect():
    keyword_count_vect = get_s3(key='keyword_vect/keyword_count_vect.pkl')
    return keyword_count_vect

## load keyword matrix
@st.cache(allow_output_mutation=True)
def load_keyword_mat():
    keyword_mat = get_s3(key='keyword_matrix/keyword_mat.pkl')
    return keyword_mat

## rds(mysql) query
## 입력문서, 정답여부, 예측라벨값, 수정라벨(오답일 경우), 입력 키워드 DB에 insert
def mysql_main(document, answer, pred_label, correction_label, keyword_select):

    data = (document, answer, pred_label, correction_label, keyword_select)
    
    conn = pymysql.connect(
        user='brunch_networking', 
        passwd='qwp5197705', 
        host='localhost', 
        db='brunch_networking', 
        charset='utf8'
    )

    c = conn.cursor(pymysql.cursors.DictCursor)
    ## check table exist

    check = "select 1 from information_schema.tables where table_name='log_basic' and table_schema='brunch_networking';"
    c.execute(check)
    check_res = c.fetchall()

    if len(check_res) == 1:
        query = """
            INSERT INTO log_basic(text_input, answer, pred_label, correction_label, keyword_select, date)
            VALUES (%s, %s, %s, %s, %s, now())
        """
        c.execute(query,data)
        conn.commit()
        conn.close()

    else :
        query = """
            CREATE TABLE log_basic(
                text_input MEDIUMTEXT,
                answer int,
                pred_label varchar(50),
                correction_label int,
                keyword_select varchar(50),
                date datetime
        );
        """
        c.execute(query)
        conn.commit()
        query = """
            INSERT INTO log_basic(text_input, answer, pred_label, correction_label, keyword_select, date)
            VALUES (%s, %s, %s, %s, %s, now());
        """
        c.execute(query,data)
        conn.commit()
        conn.close()

    
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
        st.subheader('텍스트 자동분류 및 추천 시스템')
        st.write('')

        st.write("-------------------------------------------------------")
        st.subheader("MANUAL")
        st.write("1. 아래의 예시글 혹은 원하는 텍스트를 복사하신 후 좌측 사이드바를 통해 APP 실행 페이지로 이동해주세요.")
        st.write("2. APP 실행 페이지의 텍스트 입력란에 복사하신 글을 붙여넣기 하신 후 제출 버튼을 클릭해주세요.")
        st.write("3. 입력한 텍스트가 분류 모델에 의해 브런치의 18가지 카테고리 중 하나로 분류되고, 입력 텍스트를 바탕으로 다른 작가의 글을 추천해줍니다.")
        st.write("4. 입력받은 키워드를 기반으로 다른 작가의 글을 추천해줍니다.")
        st.write("-------------------------------------------------------")
        st.subheader("<Sample Text>")
        st.write("이순신은 웃음이 적고 행동이 단아했으며 좌절과 포기를 모른 채 자신의 사명에만 충실하여 전장에서 싸우기를 멈추지 않았다. 탐관오리들이 자신의 위신을 높이고자 높으신 분들과 뇌물을 주고 받는 동안 부정행위를 하지 않고 무관의 본분에 충실하였다. 하지만 이러한 원리원칙적인 성향과 굉장히 청렴한 성격 탓에 당시 상사들과 갈등이 많아 임진왜란 전에는 인사 이동이 자주 있어 여러 지역을 옮겨 다녔다. 이순신은 사람이 갈망하는 권위나 권력같은 원초적인 욕망에 휘둘리지 않고 오직 자신의 신념으로 매사에 임하는 사람이었다. 그는 전장에서 싸우다 죽던 순간까지 누구의 인정과 보상도 바라지 않고 오직 나라와 백성을 구하고자 헌신하였으며, 몇몇 전투는 너무나 비현실적인 공적을 세워 어떻게 이뤄낸건지 아직도 학설이 분분할 정도이다. 가령 명량 해전의 초반부에서 물살이 바뀌기 전까지 약 2시간가량을 이순신은 대장선 1척으로 일본 측 함선 133척과 정면으로 붙어 하나하나 박살내고 있었다. 분명히 조선 측과 일본 측의 풍부한 사료로 교차검증이 가능한 기록임에도 너무 믿어지지 않아서 사람들이 오히려 왜곡된 유사역사학자의 주장을 믿고 마는 것이다. 일개 병졸 하나하나의 공을 세세히 적어 장계를 올려 포상을 받게 했으며 자신의 공적을 부하들에게 돌리는 경우도 허다했다. 대표적으로 명량 대첩 때 자신의 공로를 안위에게 준 것이 있는데 그 덕에 안위는 초고속 승진을 하게 된다. 허나 마냥 너그럽게 대했냐고 했다면 이것도 아닌데 훈련을 게을리하는 병사들을 매우 엄히 다스렸으며 심지어 조선 수군이 제일 무서워하는 것은 왜군이 아닌 이순신이라는 평가도 존재한다. 고로 당근과 채찍을 정확히 다루어 부하들을 부린다고 할 수 있겠다.")
        st.write("출처 - 나무위키")
        st.write("-------------------------------------------------------")

        

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

        if status == "맞춤" : # 정답일 경우
                st.write("분류가 알맞게 되었군요! 추천시스템을 이용해보세요 작성하신 글을 기반으로 다른 작가분의 글을 추천해드려요")
                # streamlit의 변수 공유기능 한계로, label,proba,max,y값을 다시 구함 * 향후 방법 찾을 시 수정
                label,proba_max,y = classify(document,label_dict,tfidf_train_vect)
                df = load_data(y)
                recommended_text = find_sim_document(df,document,y,top_n=3)

                st.write("")
                st.subheader("<작성글 기반 추천글 목록>")
                st.write("")

                st.write("Recommended Text 1.")
                st.write("<" + recommended_text['title'].iloc[1] + ">", "[원본](%s)" %(recommended_text['url'].iloc[0]))
                # st.write("[원문링크](%s)" %(recommended_text['url'].iloc[0]))
                st.write(recommended_text['text'].iloc[0] + str(" ... ..."))

                st.write(" ----------------------------------------------------------------------- ")

                st.write("Recommended Text 2.")
                st.write("<" + recommended_text['title'].iloc[1] + ">", "[원본](%s)" %(recommended_text['url'].iloc[1]))
                st.write(recommended_text['text'].iloc[1] + str(" ... ..."))

                st.write(" ----------------------------------------------------------------------- ")

                st.write("Recommended Text 3.")
                st.write("<" + recommended_text['title'].iloc[2] + ">", "[원본](%s)" %(recommended_text['url'].iloc[2]))
                st.write(recommended_text['text'].iloc[2] + str(" ... ..."))

                ## 추천 시스템 부분 시작
                st.write('---')
                st.write("## 추천 시스템")
                st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
                select_category = st.multiselect("keyword를 선택하세요.",get_categories(label))
                st.write(len(select_category), "가지 keyword를 선택했습니다.")

                keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

                if keyword_submit_button: ## keyword 선택 완료 시
                    # 기존 게시글들의 keyword vector와 keyword matrix 로드
                    keyword_count_vect = load_keyword_count_vect()
                    keyword_mat = load_keyword_mat()

                    st.write("")
                    st.write("")
                    st.write("키워드 트렌드")
                    line_chart_df = keyword_trend_chart(df,select_category)
                    st.line_chart(line_chart_df)
                    select_category_joined = (' ').join(select_category)
                
                    recommended_keyword_index = find_sim_keyword(keyword_count_vect, keyword_mat, select_category_joined, top_n=3)
                    recommended_keyword = laod_data_keyword_sim(recommended_keyword_index)
                    
                    st.write("")
                    st.subheader("<Keyword 기반 추천글 목록>")
                    st.write("")

                    st.write("Recommended by Keyword 1.")
                    st.write("<" + recommended_keyword['title'].iloc[0] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[0]))
                    st.write(recommended_keyword['text'].iloc[0] + str(" ... ..."))

                    st.write(" ----------------------------------------------------------------------- ")

                    st.write("Recommended by Keyword 2.")
                    st.write("<" + recommended_keyword['title'].iloc[1] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[1]))
                    st.write(recommended_keyword['text'].iloc[1] + str(" ... ..."))

                    st.write(" ----------------------------------------------------------------------- ")

                    st.write("Recommended by Keyword 3.")
                    st.write("<" + recommended_keyword['title'].iloc[2] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[2]))
                    st.write(recommended_keyword['text'].iloc[2] + str(" ... ..."))

                    answer = 1 # 맞춤/틀림 여부
                    mysql_main(document ,answer, label, None, select_category_joined) ## 결과 db 저장

        elif status == "틀림":
            st.write("분류가 잘못되었군요. 피드백을 주신다면 다음부턴 틀리지 않을거예요.")
            label,proba_max,y = classify(document,label_dict,tfidf_train_vect)
            category_correction = st.selectbox("category 수정하기", category_list) # 오답일 경우 정답을 새로 입력받음
            if category_correction != "<select>": # 오답 수정 부분이 입력 받았을 경우 (default가 아닐경우 => 값을 입력받은 경우)
                st.write("피드백을 주셔서 감사합니다. 이런 글은 어떠세요?")
                tmp_y = [key for key,val in label_dict.items() if val == category_correction][0]
                df = load_data(tmp_y)
                recommended_text = find_sim_document(df,document,tmp_y,top_n=3)

                st.write("")
                st.subheader("<작성글 기반 추천글 목록>")
                st.write("")

                st.write("Recommended Text 1.")
                st.write("<" + recommended_text['title'].iloc[0] + ">", "[원본](%s)" %(recommended_text['url'].iloc[0]))
                st.write(recommended_text['text'].iloc[0] + str(" ... ..."))

                st.write(" ----------------------------------------------------------------------- ")

                st.write("Recommended Text 2.")
                st.write("<" + recommended_text['title'].iloc[1] + ">", "[원본](%s)" %(recommended_text['url'].iloc[1]))
                st.write(recommended_text['text'].iloc[1] + str(" ... ..."))

                st.write(" ----------------------------------------------------------------------- ")

                st.write("Recommended Text 3.")
                st.write("<" + recommended_text['title'].iloc[2] + ">", "[원본](%s)" %(recommended_text['url'].iloc[2]))
                st.write(recommended_text['text'].iloc[2] + str(" ... ..."))

                st.write('---')
                st.write("## 추천 시스템")
                st.write("선택하신 키워드를 기반으로 다른 작가분의 글을 추천해드려요.")
                select_category = st.multiselect("keyword를 선택하세요.",get_categories(category_correction))
                st.write(len(select_category), "가지 keyword를 선택했습니다.")

                keyword_submit_button = st.button("keyword 선택 완료",key='select_category') # submit 버튼

                if keyword_submit_button: ## keyword 선택 완료 시
                    keyword_count_vect = load_keyword_count_vect()
                    keyword_mat = load_keyword_mat()

                    st.write("")
                    st.write("")
                    st.write("키워드 트렌드")
                    line_chart_df = keyword_trend_chart(df,select_category)
                    st.line_chart(line_chart_df)

                    select_category_joined = (' ').join(select_category)

                    recommended_keyword_index = find_sim_keyword(keyword_count_vect, keyword_mat, select_category_joined, top_n=3)
                    recommended_keyword = laod_data_keyword_sim(recommended_keyword_index)

                    st.write("")
                    st.subheader("<추천글 목록>")
                    st.write("")
                    
                    st.write("Recommended By Keyword 1.")
                    st.write("<" + recommended_keyword['title'].iloc[0] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[0]))
                    st.write(recommended_keyword['text'].iloc[0] + str(" ... ..."))

                    st.write(" ----------------------------------------------------------------------- ")

                    st.write("Recommended By Keyword 2.")
                    st.write("<" + recommended_keyword['title'].iloc[1] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[1]))
                    st.write(recommended_keyword['text'].iloc[1] + str(" ... ..."))

                    st.write(" ----------------------------------------------------------------------- ")

                    st.write("Recommended By Keyword 3.")
                    st.write("<" + recommended_keyword['title'].iloc[2] + ">", "[원본](%s)" %(recommended_keyword['url'].iloc[2]))
                    st.write(recommended_keyword['text'].iloc[2] + str(" ... ..."))

                    answer = 0 # 맞춤/틀림 여부
                    mysql_main(document ,answer, label, tmp_y, select_category_joined) ## 결과 db 저장


if __name__ == "__main__":
    main()
