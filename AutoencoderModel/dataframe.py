import pandas as pd

#final_df = pd.DataFrame(columns=['query', 'distances', 'gallery'])#, 'prova'])
query_list = ['a','a','a','a','a']
distances_list = ['q','l','m','n','l']
gallery_list = ['1','2','4','4','8']

def display_df(query_list, distances_list, gallery_list):
  df = pd.DataFrame(list(zip(query_list, distances_list, gallery_list)),
               columns =['query', 'distances', 'gallery'], )
  df = df.set_index('query', append=True).swaplevel(0,1)
  print(df)
  #return df










'''import plotly.graph_objects as go

query_list = ['a','b','v']
distances_list = ['q','l','m','n','l']
gallery_list = ['1','2','4','4','8']

values = [query_list,distances_list, gallery_list]
'''



'''
fig = go.Figure(data=[go.Table(
  columnorder = [1,2,3],
  columnwidth = [80,400],
  header = dict(
    values = [['<b>QUERY'],
              ['<b>DISTANCES'],
              ['<b>GALLERY']],
    line_color='darkslategray',
    fill_color='white',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='darkslategray',
    fill=dict(color=['white', 'white', 'white']),
    align=['left', 'center'],
    font_size=12,
    height=30)
    )
])
fig.show()
'''