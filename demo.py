import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
import seaborn as sns
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv(r"Final.csv", index_col=0, low_memory=False)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

index_list=[]
for index in df.index:
    if df['Pos'][index]=='GK' or df['MP'][index]<5:
        index_list.append(index)
    df['Player'][index] = df['Player'][index].partition("\\")[0]
player_values=[]

fixed_cols=['Gls','G-PK','G+A','G+A-PK','xG','xG+xA','npxG','npxG+xA','Sh','SoT','Sh/90','SoT/90','Dist','TotalCmp','TotalAtt',
            'TotalCmp%','TotDist','PrgDist','MediumCmp','MediumAtt','MediumCmp%','LongCmp','LongAtt','LongCmp%',
            'KP','1/3Pass','PPA','CrsPA','ProgPass','PassAtt','Live','Dead','FKPass','PressPass','Switch','Cross','Ground',
            'Low','High','OutcomeCmp','OutcomeBlocks','SCA','SCA90','SCAPassLive','Tkl','TklW','Tkl Def 3rd','Tkl Mid 3rd',
            'TklvsDrib','AttvsDrib','Tkl%vsDrib','PastvsDrib','Press','PressSucc','Press Def 3rd','Press Mid 3rd','Press Att 3rd',
            'Blocks','ShBlock','Pass','Int','Tkl+Int','Clr','Touches','Def Pen','Touch Def 3rd','Touch Mid 3rd','Touch Att 3rd',
            'Touch Att Pen','Touch Live','Drib Succ','Drib Att','Drib #Pl','Carries','TotDistCarries','PrgDistCarries',
            'ProgCarries','CarriesCPA','CarriesMis','CarriesDis','RecTarg','Rec%','RecProg','Mn/MP','Mn/Start','Compl',
            'Subs']

sample_values=['Player','Gls','xG','Ast','xA','SoT/90','ProgPass','KP','1/3Pass','PressPass','SCA90',
               'Tkl','TklW','PressSucc','Int','Drib Succ','Drib #Pl','ProgCarries']


df=df.drop(index_list)
df=df.reset_index(drop=True)
names=df['Player'].tolist()
pos=df['Pos'].tolist()
new_df=df[fixed_cols]


def normalization(test_df): #No need to execute seperately
    test_df = test_df.transform(lambda x: (x - x.mean()) / x.std())
    test_df = test_df.fillna(test_df.mean())
    x = test_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_norm = pd.DataFrame(x_scaled)
    pca = sklearnPCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(X_norm))
    return(transformed)

def no_of_clusters(): #Only for demo
    cluster_range = range(1, 11)
    cluster_errors = []
    for num_clusters in cluster_range:
        clusters = KMeans(num_clusters)
        clusters.fit(normalization(new_df))
        cluster_errors.append(clusters.inertia_)
    clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})
    #print(clusters_df)
    #print("\n")

    sns.set(style="white")
    plt.figure(figsize=(12, 6))
    plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")
    plt.tick_params(labelsize=15)

    plt.xlabel("Clusters", fontsize=20)
    plt.ylabel("Sum of squared errors", fontsize=20)
    plt.show()


def player_clustering():
    transformed=normalization(new_df)
    kmeans = KMeans(n_clusters=8)
    kmeans = kmeans.fit(transformed)
    labels = kmeans.predict(transformed)
    C = kmeans.cluster_centers_
    clusters = kmeans.labels_.tolist()

    cluster_list = []

    transformed['cluster'] = clusters
    transformed['name'] = names

    transformed.columns = ['x', 'y', 'cluster', 'name']
    pd.set_option('display.max_rows', None)
    transformed['pos'] = pos
    return(transformed)

def grouping_clusters(transformed): #To execute : grouping_clusters(player_clustering())
    transformed[['cluster','name','pos']].groupby('cluster').apply(print)


def identifying_cluster(req_name,temp_df):
    identified_cluster = ""
    transformed=temp_df
    player_grouping = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
    for i in transformed[['cluster', 'name', 'pos']].index:
        if transformed['cluster'][i] == 0:
            player_grouping['0'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 1:
            player_grouping['1'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 2:
            player_grouping['2'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 3:
            player_grouping['3'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 4:
            player_grouping['4'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 5:
            player_grouping['5'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 6:
            player_grouping['6'].append((transformed['name'][i], transformed['pos'][i]))
        if transformed['cluster'][i] == 7:
            player_grouping['7'].append((transformed['name'][i], transformed['pos'][i]))
    flag=0
    for i in player_grouping:
        #print(player_grouping.get(i))
        for j in player_grouping[i]:
            if req_name in j[0]:
                #print(player_grouping[i])
                #print("Yes ", i)
                identified_cluster = int(i)
                flag=1
    if(flag==0):
        print("No such player exists")
    return(identified_cluster)



def finding_similar_players(transformed,required_pos, identified_cluster,required_name): # To execute: finding_similar_players(player_clustering(),input(),input())
    if(required_pos not in ['FW','MF','DF']):
        print("No such position exists")
    else:
        new_transformed = transformed[transformed['cluster'] == int(identified_cluster)]
        new_transformed = new_transformed[new_transformed['pos'].str.contains(required_pos)]


        #print(new_transformed[['name','pos']])#Output for column 2
    if required_name in new_transformed.values:
        pass
    else:
        temp = transformed[transformed['name'].str.contains(required_name)]
        new_transformed=pd.concat([new_transformed,temp])
    return(new_transformed)

#finding_similar_players(player_clustering(),input("Position: "),int(input("Cluster number: ")))


#"""temp_df=player_clustering()
#print(temp_df)
#cluster=-1
#cluster_id=identifying_cluster(input("Enter Player Name"),temp_df)
#print(cluster_id)
#if(cluster_id!=-1):
#    finding_similar_players(temp_df,input("Enter Position Needed: "),identified_cluster=cluster_id)"""







def create_plot_1(transformed):
    sns.set(style="white")
    # pal =  sns.blend_palette(vapeplot.palette('vaporwave'))
    pal = sns.color_palette(palette="bright")
    ax = sns.lmplot(x="x", y="y", hue='cluster', data=transformed, legend=True,
                    fit_reg=False, height=8, scatter_kws={"s": 45}, palette=pal)
    texts = []

    for x, y, s in zip(transformed.x, transformed.y, transformed.name):
        texts.append(plt.text(x, y, s, fontsize=6.5))
    ax.set(ylim=(-2, 2))
    plt.tick_params(labelsize=15)
    plt.xlabel("PC 1", fontsize=20)
    plt.ylabel("PC 2", fontsize=20)
    return(ax)



def create_plot_2(new_transformed):
    sns.set(style="white")
    # pal =  sns.blend_palette(vapeplot.palette('vaporwave'))
    pal = sns.color_palette("bright",8)
    ax = sns.lmplot(x="x", y="y", hue='cluster', data=new_transformed, legend=True,
                    fit_reg=False, height=8, scatter_kws={"s": 250}, palette=pal)
    texts = []

    for x, y, s in zip(new_transformed.x, new_transformed.y, new_transformed.name):
        texts.append(plt.text(x, y, s, fontsize=6.5))
    ax.set(ylim=(-2, 2))
    plt.tick_params(labelsize=15)
    plt.xlabel("PC 1", fontsize=20)
    plt.ylabel("PC 2", fontsize=20)
    return(ax)

#create_plot_1(temp_df)

#"""temp_df=player_clustering()
#req_player=input("Enter a player name: ")
#cluster_id=identifying_cluster(req_player,temp_df)
#req_pos=input("Enter Position: ")
#sim_player=finding_similar_players(temp_df,req_pos,cluster_id)
#create_plot_2(sim_player)
#plt.show()"""

def create_plot_3(demo_df,req_player, sim_player):

    player_values = []
    for row in demo_df.values.tolist():
        if req_player.lower() in row[0].lower():
            req_pl = list(map(float, row[1:]))
    for row in demo_df.values.tolist():
        if sim_player.lower() in row[0].lower():
            sim_pl = list(map(float, row[1:]))
    x = np.arange(len(req_pl))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x + width / 2, sim_pl, width, label=sim_player)
    rects2 = ax.bar(x - width / 2, req_pl, width, label=req_player)
    ax.set_xticks(x, sample_values[1:],fontsize=3.5)
    ax.legend()

    ax.bar_label(rects1, padding=5, fontsize=3.5)
    ax.bar_label(rects2, padding=5, fontsize=3.5)
    return(fig)

#temp_df=player_clustering()
#print(temp_df)
demo_df=df[sample_values]

#create_plot_3(demo_df,"Jon Rowe", "Adam Idah")
#plt.show()



temp_df=player_clustering()

#col1= st.columns([1])

#with col1:
#    pass
team_select = st.multiselect(label="Choose a team of your choice", options=df.Team.unique())
st.write("List of existing players for reference: ")
st.dataframe(data=df[df['Team'].isin(team_select)][['Team', 'Player', 'Pos']], height=1000,)
final_required_player = st.selectbox(label="Choose Player from the list", options=df['Player'])
cluster_id=identifying_cluster(final_required_player,temp_df)
final_req_position = st.radio("Position needed", ["FW", "MF", "DF"])
sim_player = finding_similar_players(temp_df, final_req_position, cluster_id,final_required_player)





st.pyplot(create_plot_1(temp_df),clear_figure=False)
st.pyplot(create_plot_2(sim_player))







st.write("A list of similar players:")
st.write("The following players match the profile of the player you seek")
st.dataframe(data=sim_player[['name','pos']],height=950)

final_similar_player=st.selectbox(label="Choose a player you want to compare with", options=sim_player['name'])

submit_button = st.button('Submit', disabled=False)
if(submit_button==True):
    st.write("PLOT")
    st.pyplot(create_plot_3(demo_df, final_required_player, final_similar_player),height=150)
    #st.bar_chart(data=demo_df[['Gls','Ast']])

#transformed[transformed['name'].str.contains(required_name)]"""

