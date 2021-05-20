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


def create_final_dict(res_dict):
    final_res = dict()
    final_res["groupname"] = "Innominati"
    final_res["images"] = res_dict
    return final_res


def create_results_dict(results_dict, query_img, gallery_list):
    results_dict[query_img] = []
    for gallery_img in gallery_list:
        results_dict[query_img].append(gallery_img)



