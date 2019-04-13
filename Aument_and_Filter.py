import glob
import Augmentor
import cv2
import numpy as np

"""
# Putanje za ucitavanja originalnog seta
load_original_karcinom_path = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\original_set\karcinom'
load_original_zdravo_path = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\original_set\zdravo'

# ----------------------------------------------------------------------------------------------------------------------

# Putanje za spremanje augumentiranog originalnog seta
# resize AREA
save_augment_karcinom_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Augumentirani_set\karcinom'
save_augment_zdravo_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Augumentirani_set\zdravo'

# resize CUBIC
save_augment_karcinom_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Augumentirani_set\karcinom'
save_augment_zdravo_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Augumentirani_set\zdravo'

# resize MAX
save_augment_karcinom_path_3 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Augumentirani_set\karcinom'
save_augment_zdravo_path_3 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Augumentirani_set\zdravo'

# ----------------------------------------------------------------------------------------------------------------------

# Putanje za učitavanje augumentiranih slika za filtriranje
#resize AREA
load_augment_karcinom_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Augumentirani_set\karcinom\*.jpg'
load_augment_zdravo_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Augumentirani_set\zdravo\*.jpg'

# resize CUBIC
load_augment_karcinom_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Augumentirani_set\karcinom\*.jpg'
load_augment_zdravo_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Augumentirani_set\zdravo\*.jpg'

# resize MAX
load_augment_karcinom_path_3 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Augumentirani_set\karcinom\*.jpg'
load_augment_zdravo_path_3 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Augumentirani_set\zdravo\*.jpg'

# ----------------------------------------------------------------------------------------------------------------------

# Putanje za spremanje filtriranih slika slika
#resize AREA
save_karcinom_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Train_set\karcinom'
save_zdravo_path_1 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_area\Train_set\zdravo'

# resize CUBIC
save_karcinom_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Train_set\karcinom'
save_zdravo_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_cubic\Train_set\zdravo'

# resize MAX
save_karcinom_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Train_set\karcinom'
save_zdravo_path_2 = 'C:\\Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\Resize_max\Train_set\zdravo'

# ----------------------------------------------------------------------------------------------------------------------
""" # definirane putanje

# Broj stvaranja uzoraka prilikom Augmentacije
karcinom_uzoraka = 6000;
zdravo_uzoraka = 450;

# Dimenzije slika, skaliranje
dimenzija = [40, 40]

class Augmentacija():
    def __init__(self, load_path, save_path, sample):
        self.load = load_path
        self.save = save_path
        self.sample = sample

    def augmenting(self):
        p = Augmentor.Pipeline(self.load, self.save)

        for i in range(1, 11):
            value = i * 5
            p.random_distortion(probability=0.9, grid_width=value, grid_height=value, magnitude=5)

        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.skew_tilt(probability=0.5)
        p.skew(probability=0.5)
        p.skew_top_bottom(probability=0.5)
        p.skew_corner(probability=0.5)

        for j in range(1, 6):
            value = j * 2
            p.rotate(probability=0.5, max_left_rotation=value, max_right_rotation=value)
            # p.shear(probability=0.1, max_shear_left=value, max_shear_right=value)
            # p.crop_random(probability=0.5, percentage_area=0.9)

        p.sample(self.sample)

class Obrada():
    def __init__(self, image, image_dimension, save_path, num):
        self.image = image
        self.image_dimension = image_dimension
        self.save_path = save_path
        self.num = num

        self.gauss_image = np.matrix
        self.canny_image = np.matrix
        self.laplace_image = np.matrix
        self.prewitt_image = np.matrix
        self.roberts_image = np.matrix
        self.sobel_image = np.matrix

        self.save_canny = ''
        self.save_laplace = ''
        self.save_prewitt = ''
        self.save_roberts = ''
        self.save_sobel = ''

        self.high_thresh = 0
        self.low_thresh = 0
        self.thresh_img = 0

    def filter(self):
        G = 1 / 159
        gauss_array = G * np.array([[2, 4, 5, 4, 2],
                                    [4, 9, 12, 9, 4],
                                    [5, 12, 15, 12, 5],
                                    [4, 9, 12, 9, 4],
                                    [2, 4, 5, 4, 2]])

        px = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])

        py = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

        rx = np.array([[1, 0],
                       [0, -1]])

        ry = np.array([[0, -1],
                       [1, 0]])

        sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        sy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

        self.gauss_image = cv2.filter2D(self.image, -1, gauss_array)

        # self.high_thresh, self.thresh_img = cv2.threshold(self.gauss_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # self.low_thresh = 0.5 * self.high_thresh
        # self.canny_image = cv2.Canny(self.gauss_image, self.low_thresh, self.high_thresh)

        # self.laplace_image = cv2.Laplacian(self.gauss_image, cv2.CV_8U, -1)
        self.prewitt_image = cv2.filter2D(self.gauss_image, cv2.CV_32F, px) + cv2.filter2D(self.gauss_image, cv2.CV_32F, py)
        self.roberts_image = cv2.filter2D(self.gauss_image,cv2.CV_32F, rx) + cv2.filter2D(self.gauss_image,cv2.CV_32F, ry)
        self.sobel_image = cv2.filter2D(self.gauss_image, cv2.CV_8U, sx) + cv2.filter2D(self.gauss_image, cv2.CV_8U, sy)

    def resize(self):
        # self.canny_image = cv2.resize(self.canny_image, (self.image_dimension[0], self.image_dimension[1]),
        #                               interpolation=cv2.INTER)  # probati INTER_CUBIC, i jos mlatit
        #
        # self.laplace_image = cv2.resize(self.laplace_image, (self.image_dimension[0], self.image_dimension[1]),
        #                                 interpolation=cv2.INTER_AREA)

        self.prewitt_image = cv2.resize(self.prewitt_image, (self.image_dimension[0], self.image_dimension[1]),
                                        interpolation=cv2.INTER_AREA) # INTER_AREA , INTER_CUBIC ,INTER_LANCZOS4

        self.roberts_image = cv2.resize(self.roberts_image, (self.image_dimension[0], self.image_dimension[1]),
                                        interpolation=cv2.INTER_AREA)

        self.sobel_image = cv2.resize(self.sobel_image, (self.image_dimension[0], self.image_dimension[1]),
                                      interpolation=cv2.INTER_AREA)

    def save(self):
        # self.save_canny = self.save_path + '\Canny_' + str(self.num) + '.jpg'
        # cv2.imwrite(self.save_canny, self.canny_image)
        #
        # self.save_laplace = self.save_path + '\Laplace_' + str(self.num) + '.jpg'
        # cv2.imwrite(self.save_laplace, self.laplace_image)

        self.save_prewitt = self.save_path + '\Prewitt' + str(self.num) + '.jpg'
        cv2.imwrite(self.save_prewitt, self.prewitt_image)

        self.save_roberts = self.save_path + '\Roberts_' + str(self.num) + '.jpg'
        cv2.imwrite(self.save_roberts, self.roberts_image)

        self.save_sobel = self.save_path + '\Sobel_' + str(self.num) + '.jpg'
        cv2.imwrite(self.save_sobel, self.sobel_image)

def ucitavanje(load_path, save_path):
    num = 0
    for objekt in glob.glob(load_path):
        slika = cv2.imread(objekt)
        cb_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
        num = num + 1

        filter_karcinom = Obrada(cb_slika, dimenzija, save_path, num)
        filter_karcinom.filter()
        filter_karcinom.resize()
        filter_karcinom.save()

# print('Augumentiranje slika tkiva koje predstavaja karcinom')
# print('Augumentiranje seta')
# augment_karcinom = Augmentacija('C:/Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\TRAIN_original\karcinom','C:/Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\AUGMENTIRANO\karcinom',karcinom_uzoraka)
# augment_karcinom.augmenting()
# print('završeno')
#
# print('Augumentacija zdravog tkiva')
# print('Augumentiranje seta')
# augment_zdravo = Augmentacija('C:/Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\TRAIN_original\zdravo','C:/Users\db\Desktop\BORNA\Faks\Diplomski\setpodataka\set_S_augmentacijom\AUGMENTIRANO\zdravo',zdravo_uzoraka)
# augment_zdravo.augmenting()
# print('završeno')

print('Obrađivanje slika tkiva koje predstavaja karcinom:')
print('Obrađivanje seta')
ucitavanje('\*.jpg','')
print('završeno')

print('\nObrađivanje slika zdravog tkiva')
print('Obrađivanje seta')
ucitavanje('\*.jpg','')
print('završeno')
