from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
import fonctions as mes_fonctions

class Preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, filename, load_lemmatization=True):
        self.load_lemmatization = load_lemmatization
        self.filename = filename
        self.n_maj = 0
        self.n_ponc = 0
    
    def compter_maj(self, X):
        '''
        Compter le nombre des caracteres en majuscule dans npdarray de string
        Parametres :
            X - ndarray 
        Return :
            Nombre d'occurences des caracteres en majuscule
        '''
        def n_maj_(s):
            return sum(1 for ch in s if ch.isupper())
        f_n_maj = np.vectorize(n_maj_)
        return f_n_maj(X)

    def compter_ponc(self, X):
        '''
        Compter le nombre des signes de ponctuation 
        Parametres :
            X - ndarray 
        Return :
            Nombre d'occurences des signes de ponctuation
        '''
        import re 
        def n_ponc_(s):
            return len(re.findall(r'[^\w\s]', s))
        f_n_ponc = np.vectorize(n_ponc_)

        return f_n_ponc(X)

    def get_n_maj(self):
        return self.n_maj
    
    def get_n_ponc(self):
        return self.n_ponc

    def transform(self, X, y=None):
        # Compter Maj
        self.n_maj = self.compter_maj(X) 

        # Eliminer Maj
        X = np.char.lower(X)

        # Remplacement ||| par ...
        X = np.core.defchararray.replace(X, '|||', '.')

        # Remplacer les chaines speciales
        # emojis et https vont être pris en compte
        X = mes_fonctions.elimine_texte_special(X)

        # Compter Ponc
        self.n_ponc = self.compter_ponc(X)
        
        # Eliminter Ponc
        #   Regex : tous ce qui sont de poncuation 
        X = np.array(list(map(lambda c: re.sub(r'[^\w\s]',' ', c) ,X))) 

        # Lematization
        import pandas as pd
        if self.load_lemmatization == False :
            mes_fonctions.lemmatization(X, show_log=True)

            #np.save(self.filename + '.npy', X)
            #np.savetxt(self.filename + '.csv', X,
            #        delimiter=',', fmt="%s") 

            df = pd.DataFrame(X)
            df.to_csv(self.filename, index=False, header=None)
        else :
            ''' 
            Pour but de optimiser la performance
            '''
            df = pd.read_csv(self.filename, header=None)
            X = df.to_numpy()
            
            #X = np.load(self.filename+'.npy')
            
        return X
