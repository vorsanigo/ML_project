import pandas as pd

def display_df(query_image, distances_list, gallery_list, k):
    query_list = [query_image for i in range(k)]
    distances_list = distances_list.flatten()
    df = pd.DataFrame(list(zip(query_list, distances_list, gallery_list)),
                      columns=['query', 'distances', 'gallery'])
    df.index += 1
    df = df.set_index('query', append=True).swaplevel(0,1)
    return df

def df_to_html(dataframe):
    f = open('results.html', 'w')
    a = dataframe.to_html()
    f.write(a)
    f.close()

    # TODO: CONVERT TO PDF
    # pdfkit.from_file('results.html', 'results.pdf')