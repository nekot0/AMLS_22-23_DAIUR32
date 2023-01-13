import os
import numpy as np
import cv2
import dlib
import time
from tqdm import tqdm

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Datasets'
images_dir = {
    'celeba-train': os.path.join(basedir, 'celeba'),
    'celeba-test': os.path.join(basedir, 'celeba_test'),
    'cartoon-train': os.path.join(basedir, 'cartoon_set'), 
    'cartoon-test': os.path.join(basedir, 'cartoon_set_test')
}
labels1_column = {
    'celeba-train': 2,
    'celeba-test': 2,
    'cartoon-train': 2,
    'cartoon-test': 2
}
labels2_column = {
    'celeba-train': 3,
    'celeba-test': 3,
    'cartoon-train': 1,
    'cartoon-test': 1
}
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('A1/shape_predictor_68_face_landmarks.dat')



###facial landmark detection using SVM on HoG features (using dlib libraries)
###Code based on lab3_data.py

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels(mode='celeba-train'):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :input:
        mode: 'celeba-train', 'celeba-test', 'cartoon-train', 'cartoon-test'
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        task1_labels:   an array containing labels for each image in which a face was detected
                            if 'celeba-train' or 'celeba-test' -> the gender label (male=0 and female=1)
                            if 'cartoon-train' or 'cartoon-test' -> face shape labels
        task2_labels:   an array containing labels for each image in which a face was detected
                            if 'celeba-train' or 'celeba-test' -> smiling or not (smiling=0 and not smiling=1)
                            if 'cartoon-train' or 'cartoon-test' -> eye colour labels
    """
    jpg_dir = os.path.join(images_dir[mode], 'img')
    image_paths = [os.path.join(jpg_dir, l) for l in os.listdir(jpg_dir)]
    labels_file = open(os.path.join(images_dir[mode], labels_filename), 'r')
    lines = labels_file.readlines()
    task1_labels = {line.split('\t')[0] : int(line.strip().split('\t')[labels1_column[mode]]) for line in lines[1:]}
    task2_labels = {line.split('\t')[0] : int(line.strip().split('\t')[labels2_column[mode]]) for line in lines[1:]}
    
    start = time.time()
    if os.path.isdir(images_dir[mode]):
        all_features = []
        all_labels1 = []
        all_labels2 = []
        undetected_files = []
        
        for img_path in tqdm(image_paths):
            file_name= img_path.split('.')[1].split('\\')[-1]

            # load image
            img = cv2.imread(img_path)
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels1.append(task1_labels[file_name])
                all_labels2.append(task2_labels[file_name])
            else:
                undetected_files.append(file_name)
        
    end = time.time()
    total_time = end - start

    landmark_features = np.array(all_features)

    if mode == 'celeba-train' or mode == 'celeba-test':
        task1_labels = (np.array(all_labels1) + 1)/2
        task2_labels = (np.array(all_labels2) + 1)/2
    elif mode == 'cartoon-train' or mode == 'cartoon-test':
        task1_labels = all_labels1
        task2_labels = all_labels2
    
    return landmark_features, task1_labels, task2_labels, undetected_files, total_time


def get_data(mode='celeba-train'):
    """
    This function sets train/test images in (X, Y) where X represents facial landmarks and Y is the labels. 
    :input:
        mode: 'celeba-train', 'celeba-test', 'cartoon-train', 'cartoon-test'
    :return:
        X:  an array containing 68 landmark points for each image in which a face was detected
        Y1: an array containing labels for each image in which a face was detected
                if 'celeba-train' or 'celeba-test' -> the gender label (Y1[:,0]=1 if male and Y1[:,1]=1 if female)
                if 'cartoon-train' or 'cartoon-test' -> face shape labels (Y1[:,i]=1 if labeled as class i)
        Y2:   an array containing labels for each image in which a face was detected
                            if 'celeba-train' or 'celeba-test' -> smiling or not (Y2[:,0]=1 if smiling and Y2[:,1]=1 if not smiling)
                            if 'cartoon-train' or 'cartoon-test' -> eye colour labels (Y2[:,i]=1 if labeled as class i)
    """

    X, y1, y2, undetected_files, total_time = extract_features_labels(mode)
    
    if mode == 'celeba-train' or mode == 'celeba-test':
        Y1 = np.array([y1, -(y1-1)]).T
        Y2 = np.array([y2, -(y2-1)]).T
    elif mode == 'cartoon-train' or mode == 'cartoon-test':
        Y1 = np.zeros((len(y1), 5))
        Y2 = np.zeros((len(y2), 5))
        for i in range(len(y1)):
            Y1[i,y1[i]] = 1
        for i in range(len(y2)):
            Y2[i,y2[i]] = 1
    
    return X, Y1, Y2, undetected_files, total_time