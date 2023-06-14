import requests
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.express as px
import collections
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

url = 'https://api.lens.org/patent/search'

st.set_page_config(layout="wide", initial_sidebar_state='expanded')
st.title("Patenting trends for eVTOL")
st.subheader("This app uses data from Lens.org")

with st.sidebar:
    submit = st.button("Get patenting trends!")

if submit:
    # include fields
    include = '''["doc_key",
                  "jurisdiction",
                  "docdb_id",
                  "biblio.invention_title",
                  "biblio.cited_by",
                  "legal_status",
                  "biblio.priority_claims",
                  "biblio.parties",
                  "biblio.classifications_ipcr",
                  "biblio.classifications_cpc"
               ]'''

    # request body with scroll time of 1 minute
    request_body = '''{
      "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "match" : {
                                        "title": "eVTOL"
                                    }
                                },
                                {
                                    "match" : {
                                        "abstract": "eVTOL"
                                    }
                                },
                                {
                                    "match" : {
                                        "claim": "eVTOL"
                                    }
                                }
                            ]
                        }
                    },
                    
                    {
                        "range" : {
                            "date_published": {
                                "gte": "2020-01-01",
                                "lte": "2023-05-31" 
                            }
                        }
                    
                    }
                ]
            }
        },
      "size": 100,
      "include": %s,
      "scroll": "1m"
    }''' % include

    headers = {'Authorization': st.secrets["key_lens"], 'Content-Type': 'application/json'}
    df_total = pd.DataFrame(columns=["Document key","Title", "Jurisdiction","DOCDB ID", "Inventors", "Applicants", "Owners",
                                     "Inventor residences", "IPCR", "CPC", "Application filing date", "Application status",
                                     "Application grant date", "Number of citations"])

    # Recursive function to scroll through paginated results
    def scroll(scroll_id, df_total):
      # Change the request_body to prepare for next scroll api call
      # Make sure to append the include fields to make faster response
      if scroll_id is not None:
        global request_body
        request_body = '''{"scroll_id": "%s", "include": %s}''' % (scroll_id, include)

      df = pd.DataFrame(columns=df_total.columns)

      # make api request
      response = requests.post(url, data=request_body, headers=headers)

      # If rate-limited, wait for n seconds and proceed the same scroll id
      # Since scroll time is 1 minutes, it will give sufficient time to wait and proceed
      if response.status_code == requests.codes.too_many_requests:
        time.sleep(8)
        scroll(scroll_id)
      # If the response is not ok here, better to stop here and debug it
      elif response.status_code != requests.codes.ok:
        print(response.json())
      # If the response is ok, do something with the response, take the new scroll id and iterate
      else:
        json = response.json()
        if json.get('results') is not None and json['results'] > 0:
            scroll_id = json['scroll_id'] # Extract the new scroll id from response
            print(json)
            for i in range (0, json['results']):
                inventors = []
                inventor_residences = []
                applicants = []
                owners = []
                classifications = []
                classifications_cpc = []
                num_citations = 0
                try:
                    for n in range(0, len(json['data'][i]['biblio']['parties']['inventors'])):
                        inventors.append(json['data'][i]['biblio']['parties']['inventors'][n]['extracted_name']['value'])
                    for n in range(0, len(json['data'][i]['biblio']['parties']['inventors'])):
                        inventor_residences.append(json['data'][i]['biblio']['parties']['inventors'][n]['residence'])
                    for n in range (0, len(json['data'][i]['biblio']['parties']['applicants'])):
                        applicants.append(json['data'][i]['biblio']['parties']['applicants'][n]['extracted_name']['value'])
                    for n in range (0, len(json['data'][i]['biblio']['parties']['owners_all'])):
                        owners.append(json['data'][i]['biblio']['parties']['owners_all'][n]['extracted_name']['value'])
                    for n in range (0, len(json['data'][i]['biblio']['classifications_ipcr']['classifications'])):
                        classifications.append(json['data'][i]['biblio']['classifications_ipcr']['classifications'][n]['symbol'])
                    for n in range (0, len(json['data'][i]['biblio']['classifications_cpc']['classifications'])):
                        classifications_cpc.append(json['data'][i]['biblio']['classifications_cpc']['classifications'][n]['symbol'])

                except:
                    pass

                df.loc[str(i)] = [json['data'][i]['doc_key'],
                                  next(item for item in json['data'][i]['biblio']['invention_title'] if item["lang"] == "en")["text"],
                                  json['data'][i]['jurisdiction'],json['data'][i]["docdb_id"],inventors,applicants,
                                  owners, inventor_residences, classifications,classifications_cpc,
                                  json['data'][i]['legal_status']['calculation_log'][0][-10:],
                                  json['data'][i]['legal_status']['patent_status'],0,0
                                  #json['data'][i]['legal_status']['grant_date']
                                  #len(json['data'][i]['biblio']['cited_by']['patent_count'])
                                 ]

            df_total = pd.concat([df_total,df])
            scroll(scroll_id,df_total)
        else:
            owners_list = df_total["Owners"].sum()
            inventors_list = df_total["Inventors"].sum()
            ipc_list = df_total["CPC"].sum()

            df_total["Application filing date"] = pd.to_datetime(df_total["Application filing date"])
            df_total["Application filing year"] = df_total["Application filing date"].dt.year
            df_total.to_csv("Patent data.csv")

            jurIPCdf = pd.DataFrame()
            yearIPCdf = pd.DataFrame()
            def jurIPC_Yr(df):
                list = df["CPC"].sum()
                count = collections.Counter(list)
                ser = pd.Series(count).sort_values(ascending=False)[:10]
                return ser

            jursdns = df_total["Jurisdiction"].unique()
            for x in jursdns:
                df = df_total[df_total["Jurisdiction"] == x]
                ser = jurIPC_Yr(df)
                df_temp = pd.DataFrame(columns=["Jurisdiction", "IPCR", "Count"])
                df_temp["Jurisdiction"] = [x] * len(ser)
                df_temp["CPC"] = ser.index
                df_temp["Count"] = ser.values
                jurIPCdf = pd.concat([jurIPCdf,df_temp])
            #jurIPCdf.to_csv("JUR.csv")

            years = df_total["Application filing year"].unique()
            for x in years:
                df = df_total[df_total["Application filing year"] == x]
                ser = jurIPC_Yr(df)
                df_temp = pd.DataFrame(columns=["Year", "CPC", "Count"])
                df_temp["Year"] = [x] * len(ser)
                df_temp["CPC"] = ser.index
                df_temp["Count"] = ser.values
                yearIPCdf = pd.concat([yearIPCdf, df_temp])

            fig1, fig2 = st.columns(2)
            with fig1:
                try:
                    st.markdown("**:red[Patents by jurisdiction 2020-2023]**")
                    fig = px.bar(df_total.groupby("Jurisdiction").count()["Title"].sort_values(ascending=False))
                    # fig.update_traces(stackgroup=None, fill='tozeroy')
                    fig.update_layout(height=300)
                    fig.update_xaxes(tickangle=45)
                    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

                try:
                    st.markdown("**:red[Top 10 patent inventors]**")
                    count = collections.Counter(inventors_list)
                    inventors_count = pd.Series(count).sort_values(ascending=False)
                    fig = px.bar(inventors_count[0:10], orientation="h")
                    # fig.update_traces(stackgroup=None, fill='tozeroy')
                    fig.update_layout(height=300)
                    fig.update_xaxes(tickangle=45)
                    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

                try:
                    special_SW = ["eVTOL","aircraft","Aircraft","vertical takeoff", "vertical", "method", "vehicle", "electric",
                                  "System", "electric","electrical", "methods", "use", "take", "flight", "Systems", "High", "high"
                                  ]
                    st.markdown("**:red[Word cloud for active patents]**")
                    df = df_total[df_total["Application status"] == "ACTIVE"]
                    text = " ".join(title for title in df["Title"])
                    stopwords = set(STOPWORDS)
                    stopwords.update(special_SW)
                    wordcloud = WordCloud(max_font_size=50, min_font_size=8, max_words=100, stopwords=stopwords,
                                          background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    # plt.imshow(wordcloud, interpolation='bilinear')
                    ax.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(fig)
                except:
                    pass

            with fig2:
                try:
                    st.markdown("**:red[Top 10 patent owners]**")
                    count = collections.Counter(owners_list)
                    owner_count = pd.Series(count).sort_values(ascending=False)
                    fig = px.bar(owner_count[0:10], orientation="h")
                    # fig.update_traces(stackgroup=None, fill='tozeroy')
                    fig.update_layout(height=300)
                    fig.update_xaxes(tickangle=45)
                    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

                try:
                    st.markdown("**:red[Top 10 technologies (CPC)]**")
                    count = collections.Counter(ipc_list)
                    IPC_count = pd.Series(count).sort_values(ascending=False)
                    fig = px.bar(IPC_count[0:10])
                    # fig.update_traces(stackgroup=None, fill='tozeroy')
                    fig.update_layout(height=300)
                    fig.update_xaxes(tickangle=45)
                    # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

                try:
                    special_SW = ["eVTOL","aircraft","Aircraft","vertical takeoff", "vertical", "method", "vehicle", "electric",
                                  "System", "electric","electrical", "methods", "use", "take", "flight", "Systems", "High", "high"
                                  ]
                    st.markdown("**:red[Word cloud for patents awaiting approval]**")
                    df = df_total[df_total["Application status"] == "PENDING"]
                    text = " ".join(title for title in df["Title"])
                    stopwords = set(STOPWORDS)
                    stopwords.update(special_SW)
                    wordcloud = WordCloud(max_font_size=50, min_font_size=8, max_words=100, stopwords=stopwords,
                                          background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(5, 5))
                    # plt.imshow(wordcloud, interpolation='bilinear')
                    ax.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(fig)
                except:
                    pass

            st.markdown("**:red[Number of patents by year, by jurisdiction]**")
            df_yr_jur = df_total.groupby(["Application filing year", "Jurisdiction"]).count()
            df_yr_jur = df_yr_jur.reset_index(level=[0, 1]) #turning multi-index to new integer single-index
            fig = px.bar(df_yr_jur, x="Application filing year", y="Title", color="Jurisdiction")
            # fig.update_traces(stackgroup=None, fill='tozeroy')
            fig.update_layout(height=350)
            fig.update_xaxes(tickangle=45)
            # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**:red[Patenting trend by jurisdiction, by top 10 technologies (CPC)]**")
            fig = px.bar(jurIPCdf, x="Jurisdiction", y="Count", color="CPC")
            # fig.update_traces(stackgroup=None, fill='tozeroy')
            fig.update_layout(height=350)
            fig.update_xaxes(tickangle=45)
            # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**:red[Patenting trend by year, by top 10 technologies (CPC)]**")
            fig = px.bar(yearIPCdf, x="Year", y="Count", color="CPC")
            # fig.update_traces(stackgroup=None, fill='tozeroy')
            fig.update_layout(height=350)
            fig.update_xaxes(tickangle=45)
            # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
            st.plotly_chart(fig, use_container_width=True)

            def SNA(sna_series, sna_unit):
                nodes = pd.Series()
                for i in range(0, len(sna_series)):
                    nodes_list = sna_series[i]
                    for x in nodes_list:
                        list_temp = nodes_list[:]
                        list_temp.remove(x)
                        index = [x] * (len(nodes_list) - 1)
                        ser = pd.Series(data=list_temp, index=index)
                        nodes = pd.concat([nodes, ser])

                df_nodes = pd.DataFrame({"Node1": nodes.index, "Node2": nodes.values})
                df_nodes = df_nodes[df_nodes["Node1"] != df_nodes["Node2"]]
                sn = nx.from_pandas_edgelist(df_nodes, source="Node1", target="Node2")
                pos = nx.spring_layout(sn, k=0.15, iterations=20)
                degree_centrality = [sn.degree(n) for n in sn.nodes()]  # list of degrees
                node_list = list(sn.nodes())
                eigenvector_degree_centrality = nx.eigenvector_centrality(sn).values()
                # Eigenvector centrality measures a node's importance while giving consideration to the importance of its neighbors
                betweenness_centrality = nx.betweenness_centrality(sn, normalized=True).values()
                nodes_degrees = pd.DataFrame(list(zip(degree_centrality, eigenvector_degree_centrality, betweenness_centrality)),
                    columns=["Degree centrality", "Eigenvector centrality", "Betweenness centrality"],index=node_list)

                nx.draw(sn, pos, node_color='r', edge_color='b', node_size=[v * 2 for v in degree_centrality])
                nx.draw_networkx_labels(sn, font_size=4,pos=pos)
                plt.savefig(f"{sna_unit} graph.png", dpi=600)

                n_degrees = nodes_degrees.copy()
                n_degrees["Degree centrality"] *= 0.01
                nodes_degrees_dict = nodes_degrees["Degree centrality"].to_dict()
                nx.set_node_attributes(sn, nodes_degrees_dict, 'size')

                net = Network(height="1000px", width="100%", font_color="black")
                net.repulsion()
                net.from_nx(sn)
                net.show_buttons()  # (filter_=['physics'])

                # Save and read graph as HTML file (on Streamlit Sharing)
                try:
                    path = '/tmp'
                    net.save_graph(f'{path}/{sna_unit}.html')
                    HtmlFile = open(f'{path}/{sna_unit}.html', 'r', encoding='utf-8')

                # Save and read graph as HTML file (locally)
                except:
                    path = r"C:\Users\Ram.Kamath\Desktop\Modules_DIPAT\Patent_analysis\Patenting-trends"
                    net.save_graph(f'{path}/{sna_unit}.html')
                    HtmlFile = open(f'{path}/{sna_unit}.html', 'r', encoding='utf-8')

                st.markdown(f"**:red[(Social) network for {sna_unit}]**")
                # Load HTML file in HTML component for display on Streamlit page
                components.html(HtmlFile.read(), height=500)

                fig1, fig2 = st.columns(2)
                with fig1:
                    try:
                        st.markdown(f"**:red[Top 5 most connected {sna_unit}]**")
                        fig = px.bar(nodes_degrees.sort_values(["Degree centrality"], ascending=False)["Degree centrality"][:5])
                        # fig.update_traces(stackgroup=None, fill='tozeroy')
                        fig.update_layout(height=500)
                        fig.update_xaxes(tickangle=45)
                        # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                        st.plotly_chart(fig, use_container_width=True)
                        st.info(f"These are the {sna_unit} with the most connections to other {sna_unit}",icon="ℹ️")
                    except:
                        pass

                with fig2:
                    try:
                        st.markdown(f"**:red[Top 5 most 'influential' {sna_unit}]**")
                        fig = px.bar(nodes_degrees.sort_values(["Eigenvector centrality"], ascending=False)["Eigenvector centrality"][:5])
                        # fig.update_traces(stackgroup=None, fill='tozeroy')
                        fig.update_layout(height=500)
                        fig.update_xaxes(tickangle=45)
                        # fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0.0))
                        st.plotly_chart(fig, use_container_width=True)
                        st.info(f"These are the {sna_unit} with the most connections to other very influential {sna_unit}",icon="ℹ️")
                    except:
                        pass

            sna_inventor_series = df_total["Inventors"]
            SNA(sna_inventor_series, "Inventors")
            sna_tech_series = df_total["IPCR"]
            SNA(sna_tech_series, "Technologies (IPCR)")


    # start recursive scrolling
    scroll(None, df_total)




