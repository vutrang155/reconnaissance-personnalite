import numpy as np
import matplotlib.pyplot as plt
import csv 
import matplotlib

# Bar plot 
def bar_plot(x,y,xlabel='', ylabel='',title='', save=''):
    fig, ax = plt.subplots(figsize=(12,6))
    bar_plot = plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Rattacher les valeurs sur les bars
    for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2. # pos_x
                    , 1.01*height # pos_y
                    , y[idx] # valeur
                    , ha='center', va='bottom', rotation=0)
    if save != '' : 
        plt.savefig(save)
    plt.show()

def plot_fig2(y_seperated, n_lignes):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    fig.suptitle('Répartition des types', fontsize=16)
    ax1.pie(
        [np.sum(y_seperated[:,0]), n_lignes - np.sum(y_seperated[:, 0])], 
        labels=['Extraverting', 'Introverting'],
        autopct='%1.1f%%',
        startangle=90
        )   
    ax1.set_title('Extraverting - Introverting')
    ax2.pie(
        [np.sum(y_seperated[:,1]), n_lignes - np.sum(y_seperated[:, 1])], 
        labels=['Sensing', 'Intuiting'],
        autopct='%1.1f%%',
        startangle=90
        )   
    ax2.set_title('Sensing - Intuiting')
    ax3.pie(
        [np.sum(y_seperated[:,2]), n_lignes - np.sum(y_seperated[:, 2])], 
        labels=['Thinking', 'Feeling'],
        autopct='%1.1f%%',
        startangle=90
        )   
    ax3.set_title('Thinking - Feeling')
    ax4.pie(
        [np.sum(y_seperated[:,3]), n_lignes - np.sum(y_seperated[:, 3])], 
        labels=['Judging', 'Perceiving'],
        autopct='%1.1f%%',
        startangle=90
        )   
    ax4.set_title('Judging - Perceiving')
    plt.savefig("repartition.pdf")
    plt.show()

def plot_fig3(corr_types):
    fig, ax = plt.subplots()
    tick_labels = ['EI','SI','TF','JP']
    im, cbar = heatmap(corr_types, tick_labels, tick_labels, ax=ax,
                       cmap="YlGn", cbarlabel="Correlation entre les types")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    plt.savefig('correlation_type.pdf')
    plt.show()

# Remplacer les chaines speciales (http, emoji) par les chaines identiques pour standariser
def elimine_texte_special(X):
    # Rq :  C'est la même chose le fait qu'on élimine les sites web et rajouter un feature pour sa fréquence d'apparaition
    #       et le fait qu'on les remplace par une chaine (Les algos va detecter). On choisit de les remplace

    # Regex pour http : http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+
    import re
    http_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    np_sub_http = np.vectorize(http_pattern.sub) # Appliquer cette fonction pour utiliser sur un ensemble (np)
    X = np_sub_http('http_str', X)

    emoji_pattern = re.compile('\:([^\s]+)\:') # mots commencant par : et fini par :, entre les :.:, tous les caracteres sauf l'espace 
    np_sub_emoji = np.vectorize(emoji_pattern.sub)
    X = np_sub_emoji('emoji_str', X)

    return X


# Seperation des types en plusieurs sous-types (ie. ESTJ en E, S, T, J)jjj
#   On a : 
#       Extraverting    -   Introverting
#       Sensing         -   Intuiting
#       Thinking        -   Feeling     
#       Judging         -   Perceiving
#   Donc 4 colonnes pour les outputs
def separer_types(y, output_n_colonnes=4) :
    n_lignes = y.shape[0]
    y_seperated = np.zeros((n_lignes, output_n_colonnes)) 

    # Fonctions renvoient 1 si E, 0 si I, de memes pour les restes
    f_EI = np.vectorize(lambda s: 1 if s[0] == 'E' else 0) 
    f_SN = np.vectorize(lambda s: 1 if s[1] == 'S' else 0) 
    f_TF = np.vectorize(lambda s: 1 if s[2] == 'T' else 0) 
    f_JP = np.vectorize(lambda s: 1 if s[3] == 'J' else 0) 

    y_seperated[:,0] = f_EI(y)
    y_seperated[:,1] = f_SN(y)
    y_seperated[:,2] = f_TF(y)
    y_seperated[:,3] = f_JP(y)
    return y_seperated

# Lemmentization 
#   caring -> care
#   labels -> label

def lemmatization_(sentence):
    import nltk

    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer 

    """Map POS tag to first character lemmatize() accepts"""
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    # Init 
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]) 

def lemmatization(X, show_log=True):
    f = np.vectorize(lemmatization_)
    if show_log == True :
        for i in range(X.shape[0]):
            print("\t"+str(i+1)+"/"+str(X.shape[0]))
            X[i] = lemmatization_(X[i])            
    else : 
        X = f(X)

    return X 

######################## SOURCE : matplotlib.org #############################################3
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
######################## SOURCE : matplotlib.org #############################################3


def afficher_cv(results):
    '''
        Afficher l'optimisation de CV
    '''
    print(f'Les paramètres: {results.best_params_}')
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,2)} + ou -{round(std,2)} pour {params}')


def read_csv(filename, delim='\n'):
    '''
        Lire CSV
        Param:
            filename    nom du fichier
        Return:
            les données
    '''
    '''
    reader = csv.reader(open(filename))
    next(reader) # Sauter la première ligne (les noms de colonnes)   
    data = np.genfromtxt(
        ("\t".join(i) for i in reader) 
        # On peut noter ce param est par default de type fichier, mais il peut être un iterateur  
        # Ici, il s'agit d'un iterateur et pas un fichier !
        , delimiter="\t" 
        , dtype='unicode'# Par default, float, donc il faut caster 
    )
    '''
    import pandas as pd
    df = pd.read_csv(filename, na_values='')
    data = df.to_numpy(dtype="unicode")
    return data.ravel()