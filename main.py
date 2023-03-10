from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar
from lib.utils import mass_to_labels
from lib.eknn_imperfect import EKNN
from sklearn import preprocessing
import matplotlib.pyplot as plt
from lib import ibelief
import numpy as np
import math

DATASET = "IRIS" # IMP, IRIS, LINE, SIN, CIRCLE, LOG, TRIPLE, DOG
CERTAINTY = "UNC" # EV_UNC, UNC, PL, EP

# Size of the uncertainty grid
# can be reduced to go much faster
SIZE_X1 = 60
SIZE_X2 = 50

# Show all certainties when multiple
PRINT_ALL = True 

# Number of neighbors for K-NN
N_NEIGHBORS = 8

def __main__():
    # Load dataset
    X, y, y_cred = load_data()
    
    # Scale data for K-NN
    X = preprocessing.scale(X)

    # Generate the uncertainty grid
    X_test = load_Xtest(X)

    y = mass_to_labels(y_cred)
    
    plot_dataset(X, y_cred)
    print("Loading uncertainties...")

    certainties, cert1, cert2 = compute_certainties(X, y, y_cred, X_test)

    if(PRINT_ALL and CERTAINTY != "UNC"):
        plot_uncertainty(X_test, cert1)
        plot_uncertainty(X_test, cert2)
        plot_uncertainty(X_test, certainties)
    elif(CERTAINTY != "PL" and CERTAINTY != "EP"):
        plot_uncertainty(X_test, certainties)
    else:
        plot_uncertainty(X_test, cert1)


# Load grid used to plot uncertainties
def load_Xtest(X):
    X_test = np.zeros((SIZE_X1 * SIZE_X2, 2))

    minx1 = np.min(X[:,0])
    minx1 = minx1 + max(minx1, -minx1) * 0.2
    maxx1 = np.max(X[:,0])
    maxx1 = maxx1 - max(maxx1, -maxx1) * 0.2
    minx2 = np.min(X[:,1])
    minx2 = minx2 + max(minx2, -minx2) * 0.2
    maxx2 = np.max(X[:,1])
    maxx2 = maxx2 - max(maxx2, -maxx2) * 0.2

    for i in range(SIZE_X1):
        for j in range(SIZE_X2):
            X_test[i + j * SIZE_X1][0] = minx1 + i * (maxx1 - minx1) / SIZE_X1
            X_test[i + j * SIZE_X1][1] = minx2 + j * (maxx2 - minx2) / SIZE_X2

    return X_test

# Compute uncertainties for the given dataset
def compute_certainties(X, y, y_cred, X_test):
    classes = np.array(list(set(y)))
    nb_classes = classes.shape[0]

    certainties = np.zeros(X_test.shape[0])
    cert1 = np.zeros(X_test.shape[0])
    cert2 = np.zeros(X_test.shape[0])

    # Uncertainty sampling
    if CERTAINTY == "UNC" or CERTAINTY == "EP":
        # Train K-NN
        classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
        classifier.fit(X, y)

        # Compute certainties
        for i in range(X_test.shape[0]):
            dist, indices = classifier.kneighbors(np.array([X_test[i]]), N_NEIGHBORS)
            proba = classifier.predict_proba(np.array([X_test[i]]))
            
            if CERTAINTY == "UNC":
                certainties[i] = uncertainy(proba)
            elif CERTAINTY == "EP":
                certainties[i], cert1[i], cert2[i] = epistemic(dist[0], indices[0], X, y, proba)
        
    else:
        
        classifier = EKNN(nb_classes, n_neighbors=N_NEIGHBORS)
        classifier.fit(X, y_cred, alpha=1, beta=2)

        for i in range(X_test.shape[0]):
            # Evidential uncertainty
            if CERTAINTY == "EV_UNC":
                _, bba = classifier.predict(np.array([X_test[i]]), return_bba=True)
                certainties[i], cert1[i], cert2[i] = evidential_uncertainty(bba)
            # Evidential epistemic uncertainty
            if CERTAINTY == "PL":
                _, bba = classifier.predict(np.array([X_test[i]]), return_bba=True)
                certainties[i], cert1[i], cert2[i] = plausibility(bba, nb_classes)

    return certainties, cert1, cert2

# Load datasets
def load_data():
    if DATASET == "IRIS":
        X = np.load("datasets/Iris/x.npy").astype(float)
        y = np.load("datasets/Iris/y.npy").astype(int)
        y_cred = np.load("datasets/Iris/y_cred.npy").astype(float)
    elif DATASET == "LINE":
        X = np.load("datasets/Line/x.npy").astype(float)
        y = np.load("datasets/Line/y.npy").astype(int)
        y_cred = np.load("datasets/Line/y_cred.npy").astype(float)
    elif DATASET == "IMP":
        X = np.load("datasets/Imp/x.npy").astype(float)
        y = np.load("datasets/Imp/y.npy").astype(int)
        y_cred = np.load("datasets/Imp/y_cred.npy").astype(float)
    elif DATASET == "CIRCLE":
        X = np.load("datasets/Circle/x.npy").astype(float)
        y = np.load("datasets/Circle/y.npy").astype(int)
        y_cred = np.load("datasets/Circle/y_cred.npy").astype(float)
    elif DATASET == "LOG":
        X = np.load("datasets/Log/x.npy").astype(float)
        y = np.load("datasets/Log/y.npy").astype(int)
        y_cred = np.load("datasets/Log/y_cred.npy").astype(float)
    elif DATASET == "SIN":
        X = np.load("datasets/Sin/x.npy").astype(float)
        y = np.load("datasets/Sin/y.npy").astype(int)
        y_cred = np.load("datasets/Sin/y_cred.npy").astype(float)
    elif DATASET == "TRIPLE":
        X = np.load("datasets/Triple/x.npy").astype(float)
        y = np.load("datasets/Triple/y.npy").astype(int)
        y_cred = np.load("datasets/Triple/y_cred.npy").astype(float)
    elif DATASET == "DOG":
        X = np.load("datasets/dog-2/X_reduced.npy").astype(float)
        X = X[:,:2]
        y = np.load("datasets/dog-2/y_real.npy").astype(int)
        y_cred = np.load("datasets/dog-2/y.npy").astype(float)

    return X, y, y_cred

# Plot dataset
def plot_dataset(X, y):
    for i in range(X.shape[0]):
        X1 = X[i,0]
        X2 = X[i,1]

        alpha = 1

        if DATASET == "DOG":
            if y[i][2] > y[i][1]:
                colors = np.array([0.8, 0, 0])
            else:
                colors = np.array([0, 0.8, 0])
            alpha = 1 - (0.7 * y[i][3])**0.4

        else:
            if(y[i][1] == 1):
                colors = np.array([0, 0.8, 0])
            elif(y[i][2] == 1):
                colors = np.array([0.8, 0, 0])
            elif(y.shape[1] > 4 and y[i][4] == 1):
                colors = np.array([0, 0, 0.8])
            elif(y.shape[1] <= 4):
                if(y[i][1] < y[i][2]):
                    colors = np.array([1, 0.75, 0.75])
                else:
                    colors = np.array([0.75, 1, 0.75])
            else:
                if(y[i][1] < y[i][2] and y[i][4] < y[i][2]):
                    colors = np.array([1, 0.75, 0.75])
                elif(y[i][2] < y[i][1] and y[i][4] < y[i][1]):
                    colors = np.array([0.75, 1, 0.75])
                else:
                    colors = np.array([0.75, 0.75, 1])
        
        plt.scatter(X1, X2, color=colors, alpha=alpha)

    plt.xticks([])
    plt.yticks([])
    plt.show()

# Plot uncertainties
def plot_uncertainty(X, certainties):
    certainties = certainties**1.4
    if(np.max(certainties) != 0):
        certainties = certainties / np.max(certainties)

    for i in range(X.shape[0]):
        X1 = X[i,0]
        X2 = X[i,1]

        colors =  np.array([1, 1 - certainties[i],  1 - certainties[i]])

        alpha = 1
        if(certainties[i] < 0.1):
            alpha = 0

        plt.scatter(X1, X2, color=colors, alpha=alpha, marker=",", s = 80)

    plt.xticks([])
    plt.yticks([])
    plt.show()

# Compute uncertainty
def uncertainy(proba):
    return 1 - np.max(proba)

# Compute evidential uncertainty
def evidential_uncertainty(bbas):
    card = np.zeros(bbas.shape[1])
    for i in range(1, bbas.shape[1]):
        card[i] = math.log2(bin(i).count("1"))

    pign_prob = np.zeros((bbas.shape[0], bbas.shape[1]))
    for k in range(bbas.shape[0]): 
            betp_atoms = ibelief.decisionDST(bbas[k].T, 4, return_prob=True)[0]
            for i in range(1, bbas.shape[1]):
                for j in range(betp_atoms.shape[0]):
                        if ((2**j) & i) == (2**j):
                            pign_prob[k][i] += betp_atoms[j]

                if pign_prob[k][i] != 0:
                    pign_prob[k][i] = math.log2(pign_prob[k][i])

    return np.sum((0.5 * bbas[0] * card) + (-0.5 * bbas[0] * pign_prob[0])), np.sum(bbas[0] * card), -np.sum(bbas[0] * pign_prob[0])

# Compute evidential plausibility
def epistemic(dist, indices, X, y, proba):
    nb_classes = 2
    res = np.zeros(nb_classes)

    for i in range(indices.shape[0]):
        res[y[indices[i]]] += (1/dist[i])

    p = res[0]
    n = res[1]

    opt = minimize_scalar(f_objective_1, bounds=(0, 1), method='bounded', args=(p, n))
    pl1 = opt.x

    opt = minimize_scalar(f_objective_2, bounds=(0, 1), method='bounded', args=(p, n))
    pl2 = 1 - opt.x

    ue = min(pl1, pl2) - 0.5
    ua = 1 - max(pl1, pl2)

    return ue + ua, ue, ua

def f_objective_1(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 2 * theta - 1

    res = min(left, right)

    return -res

def f_objective_2(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 1 - (2 * theta)
        
    res = min(left, right)

    return -res

# Compute evidential plausibility
def plausibility(bba, nb_classes):
    pl = np.zeros(nb_classes)
    bel = np.zeros(nb_classes)

    plaus = ibelief.mtopl(bba.T)
    for i in range(nb_classes):
        pl[i] = plaus[2**i]


    cred = ibelief.mtobel(bba.T)
    for i in range(nb_classes):
        bel[i] = cred[2**i]

    # epistemic
    epistemic = 0
    for i in range(nb_classes):
        epistemic += min(pl[i], 1 - bel[i])


    # aleatoric
    aleatoric = 0
    for i in range(nb_classes):
        if min(bel[i], 1 - pl[i]) > 0:
            aleatoric += min(bel[i], 1 - pl[i])

    return epistemic + aleatoric, epistemic, aleatoric

# Run
__main__()
