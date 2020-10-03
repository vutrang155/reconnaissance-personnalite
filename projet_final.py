# Importation
import numpy as np
import csv 
import matplotlib.pyplot as plt
import fonctions as mes_fonctions
from re import findall

# Configuration : 
config_preload = True 
config_train_filename = 'traindata'
config_test_filename = 'testdata'
config_tuning = False 

'''
# On peut passer cette etape si nltk est déjà installé
import nltk
print("Téléchargement des packages nltk...")
nltk.download('all')
''' 
############################################
# Lecture du fichier 
reader = csv.reader(open('./mbti-type/mbti_1.csv'))
next(reader) # Sauter la première ligne (les noms de colonnes)   
data = np.genfromtxt(
	("\t".join(i) for i in reader) 
	# On peut noter ce param est par default de type fichier, mais il peut être un iterateur  
	# Ici, il s'agit d'un iterateur et pas un fichier !
	, delimiter="\t" 
	, dtype='unicode'# Par default, float, donc il faut caster 
	)

n_lignes = data.shape[0] # Nombre de lignes 
n_colonnes =  data.shape[1] # Nombre de colonnes
Y, c_Y = np.unique(data[:,0], return_counts=True) # Ensemble de resultats et les nombre d'occurences de chaque element 
n_Y = Y.shape[0]

############################################
# Quelque stats :
print("Nombre de lignes : " + str(n_lignes))
print("Nombre de colonnes : " + str(n_colonnes))
print("Nombre de résultats : " + str(Y.shape[0]))
print("Résultats possibles:")
print("\ttype\tn_occurences\ttaux")
print("\t-----------------------------")
for i in range(n_Y) :
	taux = c_Y[i]/n_lignes*100
	print("\t" + Y[i] 
		+ "\t" + str(c_Y[i])
		+ "\t\t" + str(round(taux,1))+" %")

# Plotting
print("Fig 1 : La répartition des types")

mes_fonctions.bar_plot(Y, c_Y
	, xlabel="Type"
	, ylabel="Nombre d'occurences"
	, title="Répartition des types"
	, save='repartition_types.png')

# Nombre de mots 
# Attention ici il s'agit d'un split tout simple ! Il n'y a pas encore des filtres
n_words = np.array([len(s.split()) for s in data[:,1]]) # Liste de nombres de mots en totale
print('\n\n______________________________________')
print("Nombre total de mots (un lien http se compte 1) : ", np.sum(n_words))
print("Nombre moyen de mots : ", np.mean(n_words))
print("Synthèse :")
print("\ttype\tn_occurences\ttaux\tmoyen\ttotal")
print("\t---------------------------------------------")
for i, y in enumerate(Y) :
	taux = c_Y[i]/n_lignes*100
	m_type = n_words[np.where(data[:,0] == y)] # Nombres des mots ou la ligne corresponde a y
	print("\t" + Y[i] 
		+ "\t" + str(c_Y[i])
		+ "\t\t" + str(round(taux,1))+" %"
		+ "\t" + str(round(np.mean(m_type), 1)) # Moyen
		+ "\t" + str(np.sum(m_type))) # Somme

# Compteur Http
def compteur_http_(s) :
	return s.count('http')
compteur_http = np.vectorize(compteur_http_)

n_http = compteur_http(data[:, 1])
print("Nombre total de site web : " + str(np.sum(n_http)))
print("Nombre moyen de site web : " + str(round(np.mean(n_http),1)))
print("Nombre sites web moyen par type : ")
for y in Y:
	m_http = n_http[np.where(data[:, 0] == y)]
	print("\t" + y
		+ "\t" + str(round(np.mean(m_http), 1)))

# Processing Emojis 
print("Analyzing liste des emojis...")
liste_des_emojis = []
f_liste_des_emojis = open("liste_des_emojis_HTML.txt", "r")
text_liste_des_emojis = f_liste_des_emojis.read()
liste_emojis_et_non_emojis = text_liste_des_emojis.split(":")

#Récupérer la liste des emojis
for i in range(len(liste_emojis_et_non_emojis)):
	tempMot = liste_emojis_et_non_emojis[i]
	if " " not in tempMot and tempMot not in liste_des_emojis:
		liste_des_emojis.append(":" + tempMot + ":")

#print(liste_des_emojis)
print(str(len(liste_des_emojis)) + " emojis dans Personality Cafe")

################################# Separtion #################################
test_size = 0.2
random_seed = 42

# Si le config_preload == False, alors on effectue Lemmatization et on sauvegarde a nouveau les train set et test set
# Lemmatization prend bcp de temps donc on pre-charge au fichier
print("Chargement...")
if config_preload==False:
    X, y = data[:, 1], data[:,0]
    #X = X.reshape((n_lignes, 1))
    #y = y.reshape((n_lignes, 1))

    # Separation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)


    from pipeline import Preprocessing
    import pandas as pd
    processing_train = Preprocessing('./mbti-type/'+config_train_filename+'_X.csv',load_lemmatization=False)
    X_train = processing_train.transform(X_train)
    df = pd.DataFrame(y_train)
    df.to_csv('./mbti-type/'+config_train_filename+ '_y.csv', index=False, header=None)
    #np.save(config_train_filename+'_y.npy', y_train)
    #np.savetxt(config_train_filename+ '_y.csv', y_train,
    #        delimiter=',', fmt="%s") 

    processing_test = Preprocessing('./mbti-type/'+config_test_filename+'_X.csv',load_lemmatization=False)
    X_test= processing_test.transform(X_test)
    df = pd.DataFrame(y_test)
    df.to_csv('./mbti-type/'+config_test_filename+ '_y.csv', index=False, header=None)
    #np.save(config_test_filename+'_y.npy', y_test)
    #np.savetxt(config_test_filename+ '_y.csv', y_test,
    #        delimiter=',', fmt="%s") 

else :
    X_train = mes_fonctions.read_csv('./mbti-type/'+config_train_filename+'_X.csv')
    y_train = mes_fonctions.read_csv('./mbti-type/'+config_train_filename+'_y.csv')
    X_test = mes_fonctions.read_csv('./mbti-type/'+config_test_filename+'_X.csv')
    y_test = mes_fonctions.read_csv('./mbti-type/'+config_test_filename+'_y.csv')
    '''
    X_train = np.load(config_train_filename+'_X.npy')
    y_train = np.load(config_train_filename+'_y.npy')
    X_test = np.load(config_test_filename+'_X.npy')
    y_test = np.load(config_test_filename+'_y.npy')
    '''

# Separation de y :
# 'ESTJ' => [0, 1, 0, 1]...
# Probleme devient maintenant Multilabel Classification !
print("Seperation y... ")
y_train_separated = np.array(mes_fonctions.separer_types(y_train, output_n_colonnes=4))
y_test_separated = np.array(mes_fonctions.separer_types(y_test, output_n_colonnes=4))
print("Figure 2 : La répartion apres la séparation")
mes_fonctions.plot_fig2(y_train_separated, int(n_lignes*(1-test_size)))
corr_types = np.corrcoef([y_train_separated[:,0],y_train_separated[:, 1], y_train_separated[:, 2], y_train_separated[:, 3]])  
# Calculation de correlation entre les types
#   Ce metric est pour determiner la dependence entre les labels
#   Si les correlations sont proches de 0 => bon pour multi-labels
print("Figure 3 : Correlation entre les labels")
print(corr_types)
mes_fonctions.plot_fig3(corr_types)

#################################################################################
# Pipeline
# On construit un pipeline simple pour traiter les donnees test.
# Par ailleurs, le preprocessing est fait avant la seperation pour optimiser la temps d'execution du programme, on n'a pas à rajouter l'etape en Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

# SVM :
# https://scikit-learn.org/stable/modules/sgd.html
# https://en.wikipedia.org/wiki/Stochastic_gradient_descent
# Rq : C_svc proportionnelle a 1/alpha_sgd
sgd_clf = Pipeline([
    ('vect', CountVectorizer(lowercase=False)), #lowercase = False car on l'a fait deja ET Countvectorizer cherche np.array.lower(), ce qui n'existe pas 
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(SGDClassifier(
        loss='hinge', # SVM utilise hinge loss pour loss function 
        penalty='l1', 
        alpha= 0.0001,
        random_state=42
        )))
    ])
if config_tuning == False:
    sgd_clf.fit(X_train, y_train_separated)
else:
    parameters = {
            'clf__estimator__penalty':['l1', 'l2'],
            'clf__estimator__alpha' : [10**(-4), 10**(-3), 10**(-2), 10**(-1)],
            'clf__estimator__max_iter': [100,200,500,1000] 
            }

    clf = GridSearchCV(sgd_clf, parameters, scoring='f1_micro',cv=10, n_jobs=-1)
    clf.fit(X_train, y_train_separated)
    mes_fonctions.afficher_cv(clf)


########## Test ################
if config_tuning==True:
    y_test_pred = np.array(clf.predict(X_test))
else:
    y_test_pred = np.array(sgd_clf.predict(X_test))

cm = multilabel_confusion_matrix(y_test_pred, y_test_separated) # Donne 4 matrix de 2x2
print("SGD Test F1-Score : ", np.mean(f1_score(y_test_separated, y_test_pred, average='micro')))
column_name = ["EI", "SN", "TF", "JP"]
for column_number in range(4): # On a EI, SN, TF, JP
    print("\t"+column_name[column_number]+" : ") 
    print("\t\tF1-Score : ", f1_score(y_test_pred[:,column_number], y_test_separated[:, column_number]))
    print("\t\tConfusion Matrix : ")
    print("\t\t\t" + str(cm[column_number]).replace('\n','\n\t\t\t'))
print(classification_report(y_test_pred, y_test_separated))
