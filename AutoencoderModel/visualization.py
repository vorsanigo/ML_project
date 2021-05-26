import matplotlib.pyplot as plt

 
def plot_query_retrieval(img_query, imgs_retrieval, outFile):
    
    """Plots images in 2 rows: top row is query, bottom row is answer"""
    
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(2*n_retrieval, 4))
    fig.suptitle("Image Retrieval (k={})".format(n_retrieval), fontsize=25)
    ax = plt.subplot(2, n_retrieval, 0 + 1)

    plt.imshow(img_query)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)  
        ax.spines[axis].set_color('black')  
    ax.set_title("query",  fontsize=14)  

    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(2, n_retrieval, n_retrieval + i + 1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black') 
        ax.set_title("Rank #%d" % (i+1), fontsize=14) 

    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()

