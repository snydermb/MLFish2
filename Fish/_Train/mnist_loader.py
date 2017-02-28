#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
from PIL import Image, ImageOps
import glob
import random


VALIDATION_SIZE = 100
TEST_SIZE = 100
fishNames = ['Bass', 'Catfish', 'Eel', 'Founder', 'Salmon', 'Shark', 'Trout', 'Tuna']


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    #Create Usable Training Data
    image_list = []
    labels = []
    for fish in fishNames:
        path = fish + '/*.jpg'
        for filename in glob.glob(path):
            im=Image.open(filename)
            img = np.asarray(to_grayscale(im))
            image_list.append(img)
            labels.append(fish)
	
    #Create Validation Data
    val_imgs = []
    val_labels = []
    for i in range(VALIDATION_SIZE):
        inx = random.randint(0,len(image_list)-1)
        img = np.asarray(image_list.pop(inx))
        lab = labels.pop(inx)
        val_imgs.append(img)
        val_labels.append(fishNames.index(lab))
    
    #Create Test Data
    tst_imgs = []
    tst_labels = []
    for i in range(TEST_SIZE):
        inx = random.randint(0,len(image_list)-1)
        img = np.asarray(image_list.pop(inx))
        lab = labels.pop(inx)
        tst_imgs.append(img)
        tst_labels.append(fishNames.index(lab))

    
    training_data = (image_list, labels)
    validation_data = (val_imgs, val_labels)	
    test_data = (tst_imgs, tst_labels)
	
	#debugging
    """print "TRAINING EXAMPLES ------------------------------------------------------"
    for x in range(len(image_list)):
        pic = to_grayscale(image_list[x])
        picArr = np.asarray(pic)
        print picArr, ' IS A ', labels[x]
		
    print "Validation EXAMPLES --++++++++++++++++++++++++++++++++++++++++++++++++--"
    for x in range(len(val_imgs)):
        pic = to_grayscale(val_imgs[x])
        print pic, ' IS A ', val_labels[x]
	
    print "Testing EXAMPLES --++++++++++++++++++++++++++++++++++++++++++++++++--"
    for x in range(len(tst_imgs)):
        pic = to_grayscale(tst_imgs[x])
        print pic, ' IS A ', tst_labels[x]"""	

    '''print "Imgs? -  - - - - - - - -  -", str(len(image_list))
    print 'training_data[0] = ', training_data[0]
    print 'training_data[1] = ', training_data[1]
    print 'Validation_data[0] = ', validation_data[0]
    print 'Validation_data[1] = ', validation_data[1]
    print 'testing_data[0] = ', test_data[0]
    print 'testing_data[1] = ', test_data[1]'''
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e"""
    #Returns 8D unit vector with 1 in position of approprate label
    #['Bass', 'Catfish', 'Eel', 'Founder', 'Salmon', 'Shark', 'Trout', 'Tuna']
    inx = fishNames.index(j)
    e = np.zeros((8,1))
    e[inx] = 1.0
    return e
	

def to_grayscale(img):
    resized = img.resize((28,28),Image.BILINEAR)
    greypic = ImageOps.grayscale(resized)
    return greypic