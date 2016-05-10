__author__ = 'Daniel'

from PIL import Image, ImageDraw
import random
import os
import cPickle
#from compressing_an_image import compress_me
from dwave_sapi import local_connection

class Neuron:
    '''A neuron is a node in the network.'''
    def __init__(self, node_num, status, activation_input):
        self.node_num = node_num
        self.status = status
        self.activation_input = activation_input

    def get_node_num(self):
        '''Returns the number of the node in a layer.'''
        return self.node_num

    def get_status(self):
        '''Returns the activation status of a node - either 0 or 1.'''
        return self.status

    def set_status(self, status):
        '''Sets a node active(1) or inactive (0)'''
        if status == 1 or status == 0:
            self.status = int(status)
        else:
            print 'Invalid status setting: Should be 0 or 1 only'

class Synapse:
    '''A synapse is a connection between two nodes.'''
    def __init__(self, conn_num, status):
        self.conn_num = conn_num
        self.status = status

    def get_conn_num(self):
        '''Returns the number of the synapse in a layer.'''
        return self.conn_num

    def get_status(self):
        '''Returns the status of a connection - either 0 or 1.'''
        return self.status

    def set_status(self, status):
        '''Sets the status of a connection - either 0 or 1.'''
        if status == 1 or status == 0:
            self.status = int(status)
        else:
            print 'Invalid status setting: Should be 0 or 1 only'

def load_net_parameters():

##############################################################################################
######################         CREATE THE NET              ###################################

# create the neural net - all nodes in all layers stored in a single list.
#Note that numbering of the nodes is done uniquely. Here is an example of how a net is numbered:

#      16 17 18 19
#   9  10 11 12 13 14
#  1  2  3  4  5  6  7  8
##############################################################################################

    number_nodes_layer_1 = 8
    number_nodes_layer_2 = 6
    number_nodes_layer_3 = 4

    threshold_activation = 0

    dim_x_viz = 23 #For the visualization, sets the size of the little images
    dim_y_viz = 33
    dim_x_viz_big = 80 # The larger versions of the same dictionary atoms
    dim_y_viz_big = 112

    nodes = []
    conns = dict()

    total_number_nodes = number_nodes_layer_1+number_nodes_layer_2+number_nodes_layer_3
    total_number_conns = number_nodes_layer_1*number_nodes_layer_2+number_nodes_layer_2*number_nodes_layer_3

    layer1_index = range(number_nodes_layer_1)
    layer2_index = range(number_nodes_layer_1, number_nodes_layer_1+number_nodes_layer_2)
    layer3_index = range(number_nodes_layer_1+number_nodes_layer_2, number_nodes_layer_1 \
                         + number_nodes_layer_2+number_nodes_layer_3)

    for i in range(total_number_nodes):
        nodes.append(Neuron(i, 0, 0))

    nodes_layer1 = []
    nodes_layer2 = []
    nodes_layer3 = []

    for each in layer1_index:
        nodes_layer1.append(nodes[each])

    for each in layer2_index:
        nodes_layer2.append(nodes[each])

    for each in layer3_index:
        nodes_layer3.append(nodes[each])

    #create the synaptic connection list, populates a dictionary to hold the synaptic objects

    for i in layer1_index: #make connections between layer 1 and layer 2 nodes
        for j in layer2_index:
            conns[(i,j)]=Synapse((i,j), 1)

    for i in layer2_index: #make connections between layer 2 and layer 3 nodes
        for j in layer3_index:
            conns[(i,j)]=Synapse((i,j), 1)


    return nodes_layer1, nodes_layer2, nodes_layer3, number_nodes_layer_1, number_nodes_layer_2, \
            number_nodes_layer_3, threshold_activation, dim_x_viz, dim_y_viz, dim_x_viz_big, \
            dim_y_viz_big, nodes, conns, total_number_nodes, total_number_conns, \
            layer1_index, layer2_index, layer3_index

#def randomize_net(conns):
#    randomized_connections = []
#    for i, j in conns.iteritems():
#        if random.choice([0,1]):
#            j.set_status(1)
#            randomized_connections.append(1)
#        else:
#            j.set_status(0) #Set to 0 for randomly connected net, 1 for a fully connected net
#            randomized_connections.append(0)

def randomize_net(nodes_layer1, conns): # TUTORIAL2 - Add nodes_layer1

    for i in range(len(nodes_layer1)): # TUTORIAL2 - Add loop to randomize inputs
        if random.choice([0,1]):
            nodes_layer1[i].set_status(1)
        else:
            nodes_layer1[i].set_status(0)

    randomized_connections = []
    for i, j in conns.iteritems():
        if random.choice([0,1]):
            j.set_status(1)
            randomized_connections.append(1)
        else:
            j.set_status(0) #Set to 0 for randomly connected net, 1 for a fully connected net
            randomized_connections.append(0)
    return randomized_connections

def update_status(node, input, threshold_activation):
    if input > threshold_activation:
        node.set_status(1)
    else:
        node.set_status(0)

def update_layer2(nodes_layer1, nodes_layer2, conns, threshold_activation):
    for node_layer2 in nodes_layer2:
        counter = 0
        for node_layer1 in nodes_layer1:
            if (node_layer1.get_node_num(),node_layer2.get_node_num()) in conns \
            and conns[(node_layer1.get_node_num(),node_layer2.get_node_num())].get_status()==1:
                if node_layer1.get_status() == 1:
                    counter +=1
        update_status(node_layer2, counter, threshold_activation)

def update_layer3(nodes_layer2, nodes_layer3, conns, threshold_activation):
    for node_layer3 in nodes_layer3:
        counter = 0
        for node_layer2 in nodes_layer2:
            if (node_layer2.get_node_num(),node_layer3.get_node_num()) in conns \
            and conns[(node_layer2.get_node_num(),node_layer3.get_node_num())].get_status()==1:
                if node_layer2.get_status() == 1:
                    counter +=1
        update_status(node_layer3, counter, threshold_activation)


def visualizeNet(nodes_layer1, nodes_layer2, nodes_layer3, conns):
    bitmap_dimension = (600,400)
    im = Image.new('RGB', bitmap_dimension, 'white')
    draw = ImageDraw.Draw(im)
    circlesize = 6

    node_coords = [] # the co-ordinate list will have the same indexing as the node ID

    for j in range(len(nodes_layer1)):
        if nodes_layer1[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 100
        y = j*bitmap_dimension[1]/(len(nodes_layer1))+0.5*(bitmap_dimension[1]/(len(nodes_layer1)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    for j in range(len(nodes_layer2)):
        if nodes_layer2[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 300
        y = j*bitmap_dimension[1]/(len(nodes_layer2))+0.5*(bitmap_dimension[1]/(len(nodes_layer2)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    for j in range(len(nodes_layer3)):
        if nodes_layer3[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 500
        y = j*bitmap_dimension[1]/(len(nodes_layer3))+0.5*(bitmap_dimension[1]/(len(nodes_layer3)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    for node_layer2 in nodes_layer2:
        for node_layer1 in nodes_layer1:
            if (node_layer1.get_node_num(),node_layer2.get_node_num()) in conns \
                and conns[(node_layer1.get_node_num(),node_layer2.get_node_num())].get_status()==1:
#                print 'connection exists'
                start_coord = node_coords[node_layer1.get_node_num()]
                end_coord = node_coords[node_layer2.get_node_num()]
                draw.line((start_coord,end_coord), fill='black', width=1)

    for node_layer3 in nodes_layer3:
        for node_layer2 in nodes_layer2:
            if (node_layer2.get_node_num(),node_layer3.get_node_num()) in conns \
            and conns[(node_layer2.get_node_num(),node_layer3.get_node_num())].get_status()==1:
#                print 'connection exists'
                start_coord = node_coords[node_layer2.get_node_num()]
                end_coord = node_coords[node_layer3.get_node_num()]
                draw.line((start_coord,end_coord), fill='black', width=1)

    for j in range(len(nodes_layer1)):
        if nodes_layer1[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 100
        y = j*bitmap_dimension[1]/(len(nodes_layer1))+0.5*(bitmap_dimension[1]/(len(nodes_layer1)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    for j in range(len(nodes_layer2)):
        if nodes_layer2[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 300
        y = j*bitmap_dimension[1]/(len(nodes_layer2))+0.5*(bitmap_dimension[1]/(len(nodes_layer2)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    for j in range(len(nodes_layer3)):
        if nodes_layer3[j].get_status() == 1:
            color = 'red'
        else:
            color = 'black'
        x = 500
        y = j*bitmap_dimension[1]/(len(nodes_layer3))+0.5*(bitmap_dimension[1]/(len(nodes_layer3)))
        draw.ellipse((x-circlesize,y-circlesize, x+circlesize, y+circlesize), color)
        node_coords.append((x,y))

    im.save('deepnet-random.bmp')
    #im.show() # - You can add this command if you want a separate window to pop up \
        # with the visualized net output

def training_data_setup():

    dim_x_viz = 23 #For the visualization, sets the size of the little images
    dim_y_viz = 33
    dim_x_viz_big = 80 # The larger versions of the same dictionary atoms
    dim_y_viz_big = 112

    # Load the dictionary ------------------------------------------------------------------------
    cwd = os.getcwd()
    filepath = "../images-for-QUFL"

    filename_to_read_final_D_values = cwd + "\DX.txt"
    mypicklefile = open(filename_to_read_final_D_values, 'r')
    D = cPickle.load(mypicklefile)
    mypicklefile.close()

    filename_to_read_mean_pixel_values = cwd + "\mean_pixels.txt"
    mypicklefile = open(filename_to_read_mean_pixel_values, 'r')
    mean_value_of_raw_pixels = cPickle.load(mypicklefile)
    mypicklefile.close()

    #Load the training and test data from the txt files ------------------------------------------
    numbers_train = ['000', '010', '020', '030', '040', '050', '060', '070', '080', '090', '100', '110', \
                     '120', '130', '140', '150', '160', '170', '180', '190', '200']
    numbers_test = ['005', '015', '025', '035', '045', '055', '065', '075', '085', '095', '105', '115', \
                    '125', '135', '145', '155', '165', '175', '185', '195', '205']

    folders = ['GR', 'MukMuk', 'suz', 'Apple']
    actual_labels_list = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    vector_of_twos = [2,2,2,2]
    numpixels = dim_x_viz_big*dim_y_viz_big; N_raw = numpixels * 3
    K = 8

    return dim_x_viz, dim_y_viz, dim_x_viz_big, dim_y_viz_big, filepath, cwd, D, folders, \
           numbers_train, numbers_test, actual_labels_list, vector_of_twos, numpixels, \
           mean_value_of_raw_pixels, N_raw, K

def turn_D_into_a_picture(D, K, number_of_pixels_in_input_images_x, number_of_pixels_in_input_images_y):

    mypicklefile = open('mean_pixels.txt', 'r')
    mean_value_of_raw_pixels = cPickle.load(mypicklefile)
    mypicklefile.close()
    dictionary_atoms=Image.new('RGB',(number_of_pixels_in_input_images_x*K,number_of_pixels_in_input_images_y))

    for y_pixels in range(number_of_pixels_in_input_images_y):
        for x_pixels in range(number_of_pixels_in_input_images_x):
            for image_number in range(K):
                pixelnumber=x_pixels+y_pixels*number_of_pixels_in_input_images_x
                pixel_value=(D[pixelnumber*3,image_number]+mean_value_of_raw_pixels[pixelnumber*3,0], \
                             D[pixelnumber*3+1,image_number]+mean_value_of_raw_pixels[pixelnumber*3+1,0], \
                             D[pixelnumber*3+2,image_number]+mean_value_of_raw_pixels[pixelnumber*3+2,0])
                dictionary_atoms.putpixel((x_pixels+number_of_pixels_in_input_images_x*\
                                                    image_number,y_pixels), pixel_value)
    dictionary_atoms.save('K8_dict.bmp')

def resize_the_K8_dict(dim_x_viz, dim_y_viz):
    im = Image.open('K8_dict.bmp')
    im2 = Image.new('RGB', (dim_x_viz*8, dim_y_viz))
    im2.save('resized_K8_dict.bmp')
    imResize = im.resize((dim_x_viz*8,dim_y_viz))
    imResize.save('resized_K8_dict.bmp')

def put_atoms_from_K8_dict_on_net(w_vector, dim_x_viz, dim_y_viz):
    im_dict = Image.open('resized_K8_dict.bmp')
    y_offset = 17
    raw_data_dict = list(im_dict.getdata())
    im = Image.open('deepnet.bmp')
    im_num = len(w_vector)
    for i in range(im_num):
        if w_vector[i]:
            for x in range(0,dim_x_viz):
                for y in range(0,dim_y_viz):
                    im.putpixel((x,((y_offset*i)+y+(i*dim_y_viz))), \
                                raw_data_dict[(i*dim_x_viz)+x+y*(im_num*dim_x_viz)])
    im.save('deepnet.bmp')

def put_correct_label_on_net(folder):
    im = Image.open('deepnet.bmp')
    draw = ImageDraw.Draw(im)
    circlesize = 6
    y_offsets = [50,150,250,350]
    draw.ellipse((550-circlesize,y_offsets[folder]-circlesize, 550+circlesize, \
                  y_offsets[folder]+circlesize), 'green')
    im.save('deepnet.bmp')

def compress_training_and_test_data(filepath, folders, numbers_train, numbers_test, D, \
                                    mean_value_of_raw_pixels, numpixels, N_raw, \
                                    K, blackbox_parameter_compress):
    training_data_outer = []
    test_data_outer = []
    use_blackbox = 0
    solver_flag = 0

    for j in range(len(folders)):
        training_data_inner = []
        for i in range(len(numbers_train)):
            base_file_path = filepath+'/'+folders[j]+'/image'+numbers_train[i]+'.bmp'
            print 'compressing training datapoint', base_file_path
            answer = compress_me(base_file_path, D, mean_value_of_raw_pixels, numpixels, N_raw, \
                                 K, use_blackbox, solver_flag, blackbox_parameter_compress)
            training_data_inner.append(answer)
        training_data_outer.append(training_data_inner)

    for j in range(len(folders)):
        test_data_inner = []
        for i in range(len(numbers_test)):
            base_file_path = filepath+'/'+folders[j]+'/image'+numbers_test[i]+'.bmp'
            print 'compressing test datapoint', base_file_path
            answer = compress_me(base_file_path, D, mean_value_of_raw_pixels, numpixels, N_raw, \
                                 K, use_blackbox, solver_flag, blackbox_parameter_compress)
            test_data_inner.append(answer)
        test_data_outer.append(test_data_inner)

    mypicklefile = open('training_data_w_vectors.txt', 'w')
    cPickle.dump(training_data_outer, mypicklefile)
    mypicklefile.close()
    mypicklefile = open('test_data_w_vectors.txt', 'w')
    cPickle.dump(test_data_outer, mypicklefile)
    mypicklefile.close()

    raw_input('Training and test data compressed. Continue?')

def load_training_and_test_data():

    mypicklefile = open('training_data_w_vectors.txt', 'r')
    training_data_outer = cPickle.load(mypicklefile)
    mypicklefile.close()

    mypicklefile = open('test_data_w_vectors.txt', 'r')
    test_data_outer = cPickle.load(mypicklefile)
    mypicklefile.close()

    return training_data_outer, test_data_outer

def initialize_net(nodes_layer1, conns, w_vector, connections_list):
    for i in range(len(nodes_layer1)):
        if i < len(w_vector):
            nodes_layer1[i].set_status(w_vector[i])
        else:
            nodes_layer1[i].set_status(0)
    counter = 0
    for i, j in conns.iteritems():
        j.set_status(connections_list[counter])
        counter +=1

def send_an_image_through_net(nodes_layer1, nodes_layer2, nodes_layer3, \
                              conns, threshold_activation, dim_x_viz, dim_y_viz):

    input = raw_input('Assign labels to images? Hit enter')
    while input == '':
        mypicklefile = open('test_data_w_vectors.txt', 'r')
        test_data_outer = cPickle.load(mypicklefile)
        mypicklefile.close()

        mypicklefile = open('neural-net-conns-random.txt', 'r')
        best_net = cPickle.load(mypicklefile)
        mypicklefile.close()

        print 'Enter folder of the image to process, in range 0 to', len(folders)-1, ':'
        folder = int(raw_input('folder?'))
        print 'Enter number of the image to process, in range 0 to', len(test_data_outer[0])-1, ':'
        number = int(raw_input('number?'))
        test_data_w = test_data_outer[int(folder)][int(number)]

        bit_string_column_vector = test_data_w
        w_vector = []
        for datum in bit_string_column_vector:
        # Converts from a column vector to a row vector
                raw_value = datum[0]
                w_vector.append(raw_value)

        initialize_net(nodes_layer1, conns, w_vector, best_net)
        update_layer2(nodes_layer1, nodes_layer2, conns, threshold_activation)
        update_layer3(nodes_layer2, nodes_layer3, conns, threshold_activation)
        visualizeNet(nodes_layer1, nodes_layer2, nodes_layer3, conns)
        put_atoms_from_K8_dict_on_net(w_vector, dim_x_viz, dim_y_viz)
        put_correct_label_on_net(folder)
        input = raw_input('Again? Press enter. Type "q" and enter to quit')
#nodes_layer1, nodes_layer2, nodes_layer3, number_nodes_layer_1, number_nodes_layer_2, \
#number_nodes_layer_3, threshold_activation, dim_x_viz, dim_y_viz, dim_x_viz_big, \
#dim_y_viz_big, nodes, conns, total_number_nodes, total_number_conns, \
#layer1_index, layer2_index, layer3_index = load_net_parameters()
##
##randomize_net(conns)
##
##visualizeNet(nodes_layer1, nodes_layer2, nodes_layer3, conns)
#
#randomize_net(nodes_layer1, conns)
#update_layer2(nodes_layer1, nodes_layer2, conns, threshold_activation)
#update_layer3(nodes_layer2, nodes_layer3, conns, threshold_activation)
#visualizeNet(nodes_layer1, nodes_layer2, nodes_layer3, conns)

nodes_layer1, nodes_layer2, nodes_layer3, number_nodes_layer_1, number_nodes_layer_2, \
        number_nodes_layer_3, threshold_activation, nodes, conns, total_number_nodes, \
        total_number_conns, layer1_index, layer2_index, layer3_index = load_net_parameters()

dim_x_viz, dim_y_viz, dim_x_viz_big, dim_y_viz_big, filepath, cwd, D, folders, \
           numbers_train, numbers_test, actual_labels_list, vector_of_twos, numpixels, \
           mean_value_of_raw_pixels, N_raw, K = training_data_setup()

#Next line only needs to run once:
compress_training_and_test_data(filepath, folders, numbers_train, numbers_test, D, \
                                mean_value_of_raw_pixels, numpixels, N_raw, K, blackbox_parameter_compress)

training_data_outer, test_data_outer = load_training_and_test_data()

#Next line only needs to run once:
turn_D_into_a_picture(D, K, dim_x_viz_big, dim_y_viz_big)

#Next line only needs to run once:
resize_the_K8_dict(dim_x_viz,dim_y_viz)

randomized_connections = randomize_net(nodes_layer1, conns)

mypicklefile = open('neural-net-conns-random.txt', 'w')
cPickle.dump(randomized_connections, mypicklefile)
mypicklefile.close()

send_an_image_through_net(nodes_layer1, nodes_layer2, nodes_layer3, conns, \
                          threshold_activation, dim_x_viz, dim_y_viz)
