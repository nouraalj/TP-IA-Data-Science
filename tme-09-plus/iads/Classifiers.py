# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd
import graphviz as gv
import math

# ---------------------------

class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """ 
        raise NotImplementedError("Please Implement this method")
        
        
            
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # cette qualité est mesurée par le taux de bonne classification du classifieur sur le dataset.
        # C'est donc une valeur de [0,1] qui s'obtient divisant le nombre d'exemples du dataset 
        # qui sont bien classés par le classifieur par le nombre total d'exemples du dataset-
        well_classed =0.0
        for i in range(len(desc_set)):
            if (self.predict(desc_set[i]) == label_set[i]) :
                well_classed += 1;
        return well_classed/len(desc_set)



class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        self.desc_set = None
        self.label_set = None
  
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        tab_dist = []
        desc = self.desc_set 
        
        
        for i in range(len(desc)):
            somme = 0
            for j in range(self.input_dimension):
                 somme += (desc[i][j] - x[j])**2
            dist = math.sqrt(somme)
            tab_dist.append(dist)
            
        tab_ind = np.argsort(tab_dist)
        
        cpt = 0
        for i in range(self.k):
            if self.label_set[tab_ind[i]] == 1:
                cpt += 1
                
        return 2*((cpt/self.k)-0.5)
                
        
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        y = self.score(x)
        if(y<0):
            return -1
        else :
            return 1
        
        

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        
        
        
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        v = np.random.uniform(-1, 1, input_dimension)
        self.w = v / np.linalg.norm(v)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        y = self.score(x)
        if(y<0):
            return -1
        else :
            return 1
            
            
            
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
       
        if (init):
            self.w = np.zeros(input_dimension)
        else:
            self.w = (np.random.uniform(0,1,input_dimension)*2-1)*0.001
        self.allw =[self.w.copy()]
        #print("Init perceptron: w=",self.w)
        
    def get_allw(self):
    	return self.allw
          
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        l = [i for i in range (len(desc_set))]
        desc_rand = np.random.shuffle(l)
        for i in l:
            if self.predict(desc_set[i]) != label_set[i]:
                self.w = self.w + desc_set[i]*label_set[i]*self.learning_rate
                self.allw.append(np.copy(self.w))
        
        
            
            
     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        norm_list = []
        w_original = self.w.copy()
        for i in range(nb_max):
            self.train_step(desc_set, label_set)
            w_prim = self.w.copy()
            norm = np.linalg.norm(w_original - w_prim)
            norm_list.append(norm)
            if norm<seuil:
                break
        return norm_list
                
            
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        y = self.score(x)
        if np.all(y<0):
            return -1
        else :
            return 1
            
class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        #print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        # Ne pas oublier d'ajouter les poids à allw avant de terminer la méthode
        l = [i for i in range (len(desc_set))]
        desc_rand = np.random.shuffle(l)
        for i in l:
            if np.all((self.score(desc_set[i]) * label_set[i])<1):
                self.w += desc_set[i]*self.learning_rate*(label_set[i]-self.score(desc_set[i]))
                self.allw.append(np.copy(self.w))    

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, index_list, nb_fois = np.unique(Y,return_index = True,return_counts=True)
    a = np.unique(nb_fois)
    if len(a)==1:
        res = Y[np.sort(index_list)[0]]
        return res
    else:
        i = np.argmax(nb_fois)
        return valeurs[i]
    
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        rem: la fonction utilise le log dont la base correspond à la taille de P
    """
    H = 0
    if len(P)<=1:
            return 0.0
    for p in P:
        
        if p > 0:
            H -= p * math.log(p, len(P))
    return H
    
def entropie(labels):
    """proba_list = []
    valeurs, nb_fois = np.unique(labels,return_counts=True)
    for j in nb_fois:
        proba_list.append(j/len(labels))
    return shannon(proba_list)"""
    _,nb_fois = np.unique(labels,return_counts=True)
    return shannon(nb_fois/len(labels))
    


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    (nb_lignes, nb_colonnes) = X.shape
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        
        
        ############
        
        """for i in range(nb_colonnes):
            ent = 0
            valeurs, nb_fois = np.unique(X[:,i],return_counts=True)
            prob = nb_fois/sum(nb_fois)
            for v,p in zip(valeurs,prob):
                ent += p * entropie(Y[X[:,i] == v])
            if ent<min_entropie:
                min_entropie = ent
                i_best = i
        Xbest_valeurs = np.unique(X[:,i_best])"""
        for attribut in LNoms:
            index = LNoms.index(attribut) # indice de l'attribut dans LNoms
            attribut_valeurs = np.unique([x[index] for x in X]) #liste des valeurs (sans doublon) prises par l'attribut
            # Liste des entropies de chaque valeur pour l'attribut courant
            entropies = []
            # Liste des probabilités de chaque valeur pour l'attribut courant
            probas_val = []
            
            for v in attribut_valeurs:
                # on construit l'ensemble des exemples de X qui possède la valeur v ainsi que l'ensemble de leurs labels
                X_v = [i for i in range(len(X)) if X[i][index] == v]
                Y_v = np.array([Y[i] for i in X_v])
                e_v = entropie(Y_v)
                entropies.append(e_v)
                probas_val.append(len(X_v)/len(X))
            
            entropie_cond = 0
            
            for i in range(len(attribut_valeurs)):
                entropie_cond += probas_val[i]*entropies[i]
                
            Is = entropie_ens - entropie_cond
            
            if entropie_cond < min_entropie:
                min_entropie = entropie_cond
                i_best = index
                Xbest_valeurs = attribut_valeurs
        """for i in range(nb_colonnes):
            HsYXj =0 
            valeurs, nb_fois = np.unique(X[:,i],return_counts=True) #donne les différentes valeurs de la colonne Xj
            proba_valeurs = nb_fois/np.sum(nb_fois)
            for j in range(len(valeurs)):
                yjl, nb_yjl = np.unique(Y[np.where(X[:,i]==valeurs[j])[0]], return_counts=True)
                proba_Yval = nb_yjl/np.sum(nb_yjl)
                HsYXj += proba_valeurs[j] * shannon(proba_Yval)
            gain_info= entropie_ens - HsYXj
            if gain_info < min_entropie:
                i_best=i
                min_entropie = gain_info
                Xbest_valeurs = valeurs
            
        for i in range(nb_colonnes):
            gain = 0
            U, C = np.unique(X[:, i], return_counts=True)
            P = C / np.sum(C)
            for j in range(len(U)):
                uuY, cuY = np.unique(Y[np.where(X[:,i] == U[j])[0]], return_counts=True)
                puY = cuY / np.sum(cuY)
                gain += P[j] * shannon(puY)
            gain = entropie_ens- gain
            if gain > min_entropie:
                min_entropie = gain
                i_best = i
                Xbest_valeurs = U"""
               
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
    
    
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine =construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
