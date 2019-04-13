import glob
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class ImageToVector:
    def __init__(self, path_ka, path_zd):
        self.path_karcinom = path_ka
        self.path_zdravo = path_zd
        self.lista_karcinom = []
        self.lista_zdravo = []
        self.vector_karcinom = []
        self.vector_zdravo = []
        self.x_karcinom = []
        self.x_zdravo = []
        self.value = []

    def reading_karcinom(self):
        for image in glob.glob(self.path_karcinom):
            pic = cv2.imread(image)
            pic_bw = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)      # maknuti konverziju u crbo-bijelo za original bazu
            self.lista_karcinom.append(pic_bw)
        print("Slika karcinom: ", len(self.lista_karcinom))

    def reading_zdravo(self):
        for image in glob.glob(self.path_zdravo):
            pic = cv2.imread(image)
            pic_bw = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)      # maknuti konverziju u crbo-bijelo za original bazu
            self.lista_zdravo.append(pic_bw)
        print("Slika zdravo: ", len(self.lista_zdravo))

    def transform_to_vector(self, lista):
        if lista == 'karcinom':
            for image in self.lista_karcinom:
                self.vector_karcinom = []
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        self.value = image[i, j]
                        self.vector_karcinom.append(self.value)

                self.x_karcinom.append(self.vector_karcinom)
            # print(len(self.x_karcinom))

        elif lista == 'zdravo':
            for image in self.lista_zdravo:
                self.vector_zdravo = []
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        self.value = image[i, j]
                        self.vector_zdravo.append(self.value)

                self.x_zdravo.append(self.vector_zdravo)
            # print(len(self.x_zdravo))


# Putanje sa kojih se ucitavaju slike ----------------------------------------------------------------------------------
path_karciom = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Train_set\Prewitt\karcinom\*.jpg'
path_zdravo = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Train_set\Prewitt\zdravo\*.jpg'

# ------- Stvaranje objekta image_to_vector ----------------------------------------------------------------------------
image_to_vector = ImageToVector(path_karciom, path_zdravo)

# Ucitavanje slika zdravog tkiva te transformiranje u vektore unutar liste (x_zdravo)
image_to_vector.reading_zdravo()
image_to_vector.transform_to_vector('zdravo')

# Ucitavanje slika karcinoma te transformiranje u vektore unutar liste (x_karcinom)
image_to_vector.reading_karcinom()
image_to_vector.transform_to_vector('karcinom')
# ----------------------------------------------------------------------------------------------------------------------

# defininranje ulaza i izlaza za neuronsku mrezu -----------------------------------------------------------------------
ulaz_karcinom = image_to_vector.x_karcinom
izlaz_karcinom = [1] * len(ulaz_karcinom)

ulaz_zdravo = image_to_vector.x_zdravo
izlaz_zdravo = [0] * len(ulaz_zdravo)

# uzimanje oko 80% slika za treniranje
split_karcinom = round(len(ulaz_karcinom) * 0.8)
split_zdravo = round(len(ulaz_zdravo) * 0.8)

# ulazni i izlazni podaci za treniranje i testiranje (cjelina 100% - treniranje 80% - testiranje 20%)
train_x = ulaz_karcinom[:split_karcinom] + ulaz_zdravo[:split_zdravo]
train_y = izlaz_karcinom[:split_karcinom] + izlaz_zdravo[:split_zdravo]

test_x = ulaz_karcinom[split_karcinom:] + ulaz_zdravo[split_zdravo:]
test_y = izlaz_karcinom[split_karcinom:] + izlaz_zdravo[split_zdravo:]
# ----------------------------------------------------------------------------------------------------------------------

# Parametri neuronske mreze --------------------------------------------------------------------------------------------
hidden_layers = (50, 50, 50)
activation_function = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
alpha = 0.0001
batch_size = 'auto'
learn_rate = ['constant', 'invscaling', 'adaptive']
learn_rate_init = 0.001
power_t = 0.5
max_iter = 200
shuffle = True
random_state = None
tol = float(1/10000)
verbose = True                                                                                         # za ispis stanja
warm_start = False
momentum = 0.9
nesterovs_momentum = True
early_stopping = False
validation_fraction = 0.1
beta_1 = 0.9
beta_2 = 0.999
epsilon = float(1/1000000000)
n_iter_no_change = 10
# ----------------------------------------------------------------------------------------------------------------------

NN = MLPClassifier(max_iter=1000)
parameter_space = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)]
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(NN, parameter_space, cv=3,scoring='accuracy')
print("begin train")
clf.fit(train_x, train_y)
print("done train")

# best parameter set
print('Best parameters found: \n', clf.best_params_)

# Provjera
y_true, y_pred = test_y , clf.predict(test_x)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

#
# # Result
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Stvaranje neuronske mreze Multi-layer perceptron ---------------------------------------------------------------------
# NN = MLPClassifier(hidden_layers, activation_function[0], solver[1], alpha, batch_size, learn_rate[0], learn_rate_init,
#                     power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
#                     early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change)
# ----------------------------------------------------------------------------------------------------------------------
print("begin train")
NN.fit(train_x, train_y)
print("done train")

print("begin test")
y_predict = NN.predict(test_x)
print("done test")

score = accuracy_score(test_y, y_predict)
print("accuracy: ")
print(score)
# ----------------------------------------------------------------------------------------------------------------------

