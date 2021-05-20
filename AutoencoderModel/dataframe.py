import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
from PIL import Image
from io import BytesIO
from IPython.core.display import HTML
from matplotlib.backends.backend_pdf import PdfPages
import pdfkit
from tensorflow.keras.preprocessing.image import array_to_img

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

# TODO: PROVARE A SITEMARE QUESTO CODICE
def prova(dataframe):
    print(dataframe.query)
    #dataframe.query = [get_thumbnail(f) for f in dataframe.query]
    dataframe.gallery = [get_thumbnail(f) for f in dataframe.gallery]

    HTML(dataframe.to_html(formatters={'gallery': image_formatter}, escape=False))


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
    return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpg;base64,{image_base64(im)}">'
