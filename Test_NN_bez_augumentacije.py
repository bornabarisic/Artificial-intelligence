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

# Učitavanje podataka za TRAIN-iranje mreže
TRAIN_path_karcinom = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\TRAIN\AREA\PREWITT\karcinom\*.jpg'
TRAIN_path_zdravo = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\TRAIN\AREA\PREWITT\zdravo\*.jpg'

# Učitavanje podataka za TEST-iranje mreže
TEST_path_karcinom = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\TEST\AREA\PREWITT\karcinom\*.jpg'
TEST_path_zdravo = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\TEST\AREA\PREWITT\zdravo\*.jpg'

# Stvaranje objekta za rad
TRAIN_set = ImageToVector(TRAIN_path_karcinom, TRAIN_path_zdravo)
TEST_set = ImageToVector(TEST_path_karcinom, TEST_path_zdravo)

# Rad sa objektima. učitavnje slika i pretvaranje u vektore
TRAIN_set.reading_karcinom()
TRAIN_set.transform_to_vector('karcinom')
TRAIN_set.reading_zdravo()
TRAIN_set.transform_to_vector('zdravo')

TEST_set.reading_karcinom()
TEST_set.transform_to_vector('karcinom')
TEST_set.reading_zdravo()
TEST_set.transform_to_vector('zdravo')

# Definiranje ulaza
# ulazi za TRAIN
TRAIN_input_karcinom = TRAIN_set.x_karcinom
TRAIN_input_zdravo = TRAIN_set.x_zdravo

# ulazi za TEST
TEST_input_karcinom = TEST_set.x_karcinom
TEST_input_zdravo = TEST_set.x_zdravo

# Definiranje izlaza
# izlazi za TRAIN
TRAIN_output_karcinom = [1] * len(TRAIN_input_karcinom)
TRAIN_output_zdravo = [0] *len(TRAIN_input_zdravo)

# izlazi za TEST
TEST_output_karcinom = [1] * len(TEST_input_karcinom)
TEST_output_zdravo = [0] * len(TEST_input_zdravo)

# Definiranje ulaza i izlaza u NN
train_X = TRAIN_input_karcinom + TRAIN_input_zdravo
train_y = TRAIN_output_karcinom + TRAIN_output_zdravo

test_x = TEST_input_karcinom + TEST_input_zdravo
test_y = TEST_output_karcinom + TEST_output_zdravo

#"""
# Namještanje parametara
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

NN = MLPClassifier(max_iter=1000)
parameter_space = {
    #'hidden_layer_sizes':[(20,20,20),(50,100,50),(50,40,30,20,10)],
    #'activation':['tanh','relu','logistic'],
    #'solver':['sgd','adam','lbfgs'],
    'alpha':[0.0001,0.00001],
    #'learning_rate':['constant','adaptive','invscaling'],
}

clf = GridSearchCV(NN, parameter_space, cv=3)
print("begin train")
clf.fit(train_X, train_y)
print("done train")

# best parameter set
print('Best parameters found: \n', clf.best_params_)

# Result
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Provjera
y_true, y_pred = test_y , clf.predict(test_x)

print('Results on the test set:')
print(classification_report(y_true, y_pred))
#""" # ZA PRONAĆ PARAMETRE

"""
hidden_layers = (100, 75, 50, 25)
activation_function = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
alpha = 0.0001
batch_size = 'auto'
learn_rate = ['constant', 'invscaling', 'adaptive']
learn_rate_init = 0.001
power_t = 0.5
max_iter = 1000
shuffle = True
random_state = None
tol = float(1/10000) #def 1/10000
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


NN = MLPClassifier(hidden_layers, activation_function[2], solver[1], alpha, batch_size, learn_rate[0], learn_rate_init,
                    power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
                    early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change)


print("begin train")
NN.fit(train_X, train_y)
print("done train")

print("begin test")
y_predict = NN.predict(test_x)
print("done test")

score = accuracy_score(test_y, y_predict)
print("accuracy: ")
print(score)

""" # ZA POKRENUT TRENIRANJE I UČENJE MREŽE
