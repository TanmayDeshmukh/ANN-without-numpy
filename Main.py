import math
import random
import time
import cv2
import math
from tkinter import *

root = Tk()
root.title("Neural Network")

window_width=600
window_height=750
canvas = Canvas(root, width=window_width, height=window_height, background="black")
canvas.grid()

def der_sig(x):
    return sig(x) * (1 - sig(x))


def sig(x):
    val = 1 / (1 + 1 / pow(math.e, x))
    return val


'''
def der_sig(x):
    return 1.0/(1+pow(math.e, -x))


def sig(x):
    return math.log((1+pow(math.e, x)))

def der_sig(x):
    return -1.0*2.0*x/(x**2+1)**2
def sig(x):
    return 1.0/(x**2+1)


def der_sig(x):
    if x == 0:
        return 0
    else:
        return math.cos(x)/x-math.sin(x)/(x**2)

def sig(x):
    if x == 0:
        return 1
    else:
        return math.sin(x)/x
'''
mse = 0
epochs = 0
learning_rate = 0.5
momentum = 0.1

NN_window_x1 = 100
NN_window_x2 = 500

NN_window_y1 = 50
NN_window_y2 = window_height-50

num_layers = 3

inputs = [[0], [0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9],[1]]
expected =   [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0]]
newinput = [[0.25]]

#inputs = [[0], [0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7]]
#expected =   [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
#newinput = [[0.25]]

#inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
#expected = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]]
#newinput = [[0, 0, 0]]


#inputs = [[0,0], [0,1],[1,0],[1,1]]
#expected = [[0],[1],[1],[0]]
#newinput = [[0,1]]

#inputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],]
#expected = [[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0]]
#newinput = [[0,0,0]]

#inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
#expected = [[0,0], [1,0], [1,0], [0, 1]]
#newinput = [[1, 1]]


inputs = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], \
          [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
expected = [  [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], \
          [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0],[1, 1, 1, 1],[0, 0, 0, 0]]

newinput = [[1, 0, 0, 0]]


l1 = len(inputs[0])

l2 = 10

l3 = len(expected[0])

widths = [l1, l2, l3]

def create_circle(x, y, r, **kwargs):
    canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)


class Neuron:
    num_neurons = 0

    def __init__(self, layer, neuron_number):
        self.name = "inited"
        self.layer = layer
        self.neuron_num = neuron_number
        self.dendrites = []  # list of connections
        self.sum = 0
        self.sumout = 0
        self.delta = 0
        Neuron.num_neurons += 1

        self.graphic_prop = Graphic_prop(0, 0)

    def init_graphic(self):
        self.graphic_prop.x1 = NN_window_x1 + (NN_window_x2 - NN_window_x1) * (self.layer-1) / (num_layers-1)
        self.graphic_prop.y1 = NN_window_y1 + (NN_window_y2 - NN_window_y1) * self.neuron_num / (widths[self.layer-1])
        create_circle(self.graphic_prop.x1, self.graphic_prop.y1, 15, fill="#BBB", outline="")
        canvas.create_text(self.graphic_prop.x1, self.graphic_prop.y1+15, text="{}".format(self.delta), anchor=N,
                           fill='white')

        #canvas.update()

    def activation_fn(self):
        # self.sumout = 1 / (1 + pow(math.e, -self.sum))
        self.sumout = sig(self.sum)

    def summer(self):
        self.name = "summing"
        self.sum = 0
        for conn in self.dendrites:
            self.sum += conn.output  # output after applying weights
        Neuron.activation_fn(self)
        # print("layer : {0} sumout : {1}".format(self.layer, self.sumout))

    def num_dendrites(self):
        return len(self.dendrites)


class Connection:
    num_connections = 0
    max_weight = 1
    min_weight = 0

    max_weight_wip = 1  #wt with ip
    min_weight_wip = -1

    wip = False

    def __init__(self, prev_neuron, next_neuron, layer):
        self.weight = random.uniform(Connection.min_weight, Connection.max_weight)
        #print(self.weight)
        self.prev_neuron = prev_neuron
        self.sum = 0
        self.output = 0
        self.gradient = 0
        self.prev_gradient = 0
        self.del_weight = 0
        self.sum_del_weight = 0
        self.prev_del_weight = 0
        next_neuron.dendrites.append(self)
        self.next_neuron_graphic_prop = next_neuron.graphic_prop
        self.layer = layer
        Connection.num_connections += 1

        # graphic prop
        self.graphic_prop = Graphic_prop(0, 0)

    def init_graphic(self):
        r = 20
        w = 2
        if self.wip:

            if self.output < 0:
                g = 20
                r = int((self.output - Connection.min_weight_wip) * (160 - 100) / (Connection.max_weight_wip - Connection.min_weight_wip) + 100)
                tk_rgb = "#%02x%02x%02x" % (int(0.8*r), int(g), int(0.2 * g))
            else:
                r = 50
                g = int((self.output - Connection.min_weight_wip) * (175 - 100) / (Connection.max_weight_wip - Connection.min_weight_wip) + 100)
                tk_rgb = "#%02x%02x%02x" % (int(0.3 * r), int(g), int(0.4 * g))
            if self.output == 0:
                r=130
                tk_rgb = "#%02x%02x%02x" % (50,50,50)
        else:
            if self.weight < 0:
                g = 20
                r = int((self.output - Connection.min_weight) * (160 - 100) / (Connection.max_weight - Connection.min_weight) + 100)
                tk_rgb = "#%02x%02x%02x" % (int(r*0.9), g, int(0.2 * g))
            else:
                r = 50
                g = int((self.weight - Connection.min_weight) * (170 - 100) / (Connection.max_weight - Connection.min_weight) + 100)
                tk_rgb = "#%02x%02x%02x" % (int(0.2 * r), g, int(0.2 * g))
            if self.weight == 0:
                r=130
                tk_rgb = "#%02x%02x%02x" % (50,40,40)
        #(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
        #print(g)


        if g>135 or r>115:
            w=3
        if g>145 or r>125:
            w=4
        if g>155 or r>135:
            w=5
        if g>165 or r>145:
            w=6

        canvas.create_line(self.prev_neuron.graphic_prop.x1, self.prev_neuron.graphic_prop.y1,\
                           self.next_neuron_graphic_prop.x1, self.next_neuron_graphic_prop.y1, width=w, fill=tk_rgb)
        #canvas.update()

    def apply_weight(self):
        self.output = self.prev_neuron.sumout * self.weight
        if Connection.max_weight < self.weight:
            Connection.max_weight = self.weight
        else:
            if Connection.min_weight > self.weight:
                Connection.min_weight = self.weight
        if Connection.max_weight_wip < self.output:
            Connection.max_weight_wip = self.output
        else:
            if Connection.min_weight_wip > self.output:
                Connection.min_weight_wip = self.output
            # print("{0}*{1}={2}:".format(self.prev_neuron.sumout,self.weight,self.output))


class Graphic_prop:
    def __init__(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.x2 = 0
        self.y2 = 0
        self.color = Graphic_prop.rgb_to_hex(255, 255, 255)

    @staticmethod
    def rgb_to_hex(r, g, b):
        return '#%02x%02x%02x' % (r, g, b)


conn_layer1 = []
conn_layer2 = []
conn_layer3 = []

conns = []

i = 0
j = 0

Layer1 = [Neuron(1, i) for i in range(0, l1)]
Layer2 = [Neuron(2, i) for i in range(0, l2)]
Layer3 = [Neuron(3, i) for i in range(0, l3)]


for i in range(l1):
    for j in range(l2):
        conn_layer1.append(Connection(Layer1[i], Layer2[j], 1))
for i in range(l2):
    for j in range(l3):
        conn_layer2.append(Connection(Layer2[i], Layer3[j], 2))

#for i in range(l3):
#    conn_layer3.append(Connection(Layer3[i], Layer1[i]))
#    conn_layer3[i].weight = 1


def redraw_del():
    canvas.delete(ALL)

    for i in range(0, l1):
        Layer1[i].init_graphic()

    for i in range(0, l2):
        Layer2[i].init_graphic()

    for i in range(0, l3):
        Layer3[i].init_graphic()

    for i in range(len(conn_layer1)):

        conn_layer1[i].apply_weight()
    for i in range(len(conn_layer1)):
        conn_layer1[i].init_graphic()
        time.sleep(0.1)
        canvas.update()

    for i in range(len(conn_layer2)):
        conn_layer2[i].apply_weight()
    for i in range(len(conn_layer2)):
        conn_layer2[i].init_graphic()

        time.sleep(0.1)
        canvas.update()


def redraw(ip=-1):
    canvas.delete(ALL)

    for i in range(len(conn_layer1)):
        conn_layer1[i].init_graphic()

    for i in range(len(conn_layer2)):
        conn_layer2[i].init_graphic()

    for i in range(0, l1):
        Layer1[i].init_graphic()

    for i in range(0, l2):
        Layer2[i].init_graphic()

    for i in range(0, l3):
        Layer3[i].init_graphic()

    canvas.create_text(50, window_height-50, text="MSE: {}".format(mse), anchor=SW, fill='white')
    canvas.create_text(50, window_height-25, text="Epochs: {}".format(epochs), anchor=SW, fill='white')

    if ip >= 0:
        for i in range(0, l1):
            x1 = NN_window_x1 - 30
            y1 = NN_window_y1 + (NN_window_y2 - NN_window_y1) * i / (widths[0])
            canvas.create_text(x1, y1, text="{}".format(inputs[ip][i]), anchor=SW, fill='white')
    for i in range(0, l3):
        x1 = NN_window_x2 + 30
        y1 = NN_window_y1 + (NN_window_y2 - NN_window_y1) * i / (widths[2])
        canvas.create_text(x1, y1, text="{0:.2f}".format(Layer3[i].sumout,2), anchor=SW, fill='white')

    canvas.update()


def feed_forward():
    # print(" ")
    # print(" ")
    for i in range(len(conn_layer1)):
        conn_layer1[i].apply_weight()
        # print("conn1: {0}".format(conn_layer1[i].output))

    for j in range(l2):
        Layer2[j].summer()
    # print(" ")
    for i in range(len(conn_layer2)):
        conn_layer2[i].apply_weight()
        # print("conn2: {0}".format(conn_layer2[i].output))

    for j in range(l3):
        Layer3[j].summer()

    for i in range(len(conn_layer3)):
        conn_layer3[i].apply_weight()


#feed_forward()


def fn():
    print(Layer2[0].num_dendrites())
    print(Layer3[0].num_dendrites())
    print(len(conn_layer1))
    print(len(conn_layer2))


def backprop(expected):
    g_errors = []
    # delta of layer 3:
    for i in range(len(Layer3)):
        g_errors.append(Layer3[i].sumout - expected[i])
        Layer3[i].delta = -g_errors[i] * der_sig(Layer3[i].sum)  # derivative of sigmoid
        #Layer3[i].delta = -(Layer3[i].sumout - expected[i])*learning_rate
    # delta of layer 2:
    for hidden_layer in range(1, num_layers - 1):
        for i in range(widths[num_layers - hidden_layer - 1]):  # 1
            sum_portion = 0
            for j in range(widths[num_layers - hidden_layer]):  # 2
                weight = conn_layer2[i * widths[num_layers - hidden_layer] + j].weight  # 2
                sum_portion += weight * Layer3[j].delta
            Layer2[i].delta = der_sig(Layer2[i].sum) * sum_portion

    # grad of conn layer2  :
    # for connlayer in range(num_layers - 2,0,-1):
    for i in range(widths[1]):  # 1
        for j in range(widths[2]):  # 2
            conn_layer2[i * widths[2] + j].gradient = Layer3[j].delta * Layer2[i].sumout  # 2

    # change weights in layer 2
    for i in range(widths[1]):  # 1
        for j in range(widths[2]):  # 2
            conn_layer2[i * widths[2] + j].del_weight = learning_rate * conn_layer2[i * widths[2] + j].gradient + \
                                                        momentum * learning_rate * conn_layer2[
                                                            i * widths[2] + j].prev_del_weight  # 222
            # conn_layer2[i * widths[2] + j].weight += conn_layer2[i*widths[2]+j].del_weight
            conn_layer2[i * widths[2] + j].sum_del_weight += conn_layer2[i * widths[2] + j].del_weight  # 22
            conn_layer2[i * widths[2] + j].prev_del_weight = conn_layer2[i * widths[2] + j].del_weight  # 22

    # grad of conn layer1
    for i in range(widths[0]):
        for j in range(widths[1]):
            conn_layer1[i * widths[1] + j].gradient = Layer2[j].delta * Layer1[i].sumout

    # change weights in conn layer1
    for i in range(widths[0]):
        for j in range(widths[1]):
            conn_layer1[i * widths[1] + j].del_weight = learning_rate * conn_layer1[i * widths[1] + j].gradient + \
                                                        momentum * learning_rate * conn_layer1[
                                                            i * widths[1] + j].prev_del_weight
            # conn_layer1[i * widths[1] + j].weight += conn_layer1[i*widths[1]+j].del_weight
            conn_layer1[i * widths[1] + j].sum_del_weight += conn_layer1[i * widths[1] + j].del_weight
            conn_layer1[i * widths[1] + j].prev_del_weight = conn_layer1[i * widths[1] + j].del_weight


count = 0


def clear_sum_deltas():
    for i in range(widths[1]):
        for j in range(widths[2]):
            conn_layer2[i * widths[2] + j].sum_del_weight = 0
    for i in range(widths[0]):
        for j in range(widths[1]):
            conn_layer1[i * widths[1] + j].sum_del_weight = 0


def add_deltas():
    for i in range(widths[1]):
        for j in range(widths[2]):
            conn_layer2[i * widths[2] + j].weight += conn_layer2[i * widths[2] + j].sum_del_weight
    for i in range(widths[0]):
        for j in range(widths[1]):
            conn_layer1[i * widths[1] + j].weight += conn_layer1[i * widths[1] + j].sum_del_weight



sq_sum = 0

def apply_input(ip_no):
    for i in range(len(inputs[0])):
        Layer1[i].sumout = inputs[ip_no][i]


# while Layer3[0].sumout > 0.45:

def stop(event):
    global close
    close = True
root.bind("<Button-1>", stop)


def train():
    global mse
    global epochs
    global close
    close = False
    sq_sum = 1
    count = 0
    print("\nTraining...")
    # for i in range(5000):
    while not close: #sq_sum > 0.001:
        clear_sum_deltas()


        for j in range(len(inputs)):
            apply_input(j)
            Connection.max_weight = 1
            Connection.min_weight = 0
            feed_forward()
            for neuron_num in range(widths[2]):
                sq_sum += (Layer3[neuron_num].sumout - expected[j][neuron_num]) ** 2
            backprop(expected[j])
        sq_sum /= len(inputs)
        #print("MSE: {}".format(sq_sum))
        mse = sq_sum

        count += 1
        epochs=count
        add_deltas()
        if count%100 == 0:
            redraw(j)

    print("Epochs : {}".format(count))
    print("MSE: {}".format(sq_sum))


def printall(layer):
    print(" : ", end="")
    for i in range(len(layer)):
        print("{0:.2f}".format(layer[i].sumout), end="")
        print(", ", end="")
    print()


def print_weights():
    print("Layer 1 \t\t\t Layer 2")
    count = 0

    while count < len(conn_layer1) or count < len(conn_layer2):
        if count < len(conn_layer1):
            print(conn_layer1[count].weight, end=" \t ")
        if count < len(conn_layer2):
            print(conn_layer2[count].weight, end="")
        print()
        count += 1


#print_weights()

#redraw_del()
waiting = True
print("Number of neurons ")
print(Neuron.num_neurons)
print("Number of connections ")
print(Connection.num_connections)
print("before: ")
for j in range(len(inputs)):
    apply_input(j)
    for i in range(len(inputs[0])):
        print("{} ".format(inputs[j][i]), end="")
    feed_forward()
    printall(Layer3)
    for neuron_num in range(widths[2]):
        sq_sum += (Layer3[neuron_num].sumout - expected[j][neuron_num]) ** 2
    backprop(expected[j])
sq_sum /= len(inputs)
print("MSE: {}".format(sq_sum))


train()
print("after training:")

for j in range(len(inputs)):
    apply_input(j)
    for i in range(len(inputs[0])):
        print("{} ".format(inputs[j][i]), end="")
    feed_forward()
    printall(Layer3)

print("new ip: ")
for j in range(len(newinput)):
    for i in range(len(newinput[0])):
        Layer1[i].sumout = newinput[j][i]
    for i in range(len(newinput[0])):
        print("{} ".format(newinput[j][i]), end="")
    feed_forward()
    printall(Layer3)
# printWeights()

Connection.wip = True
close = False

print("Min {} , max {}".format(Connection.min_weight, Connection.max_weight))
print("Min {} , max {}".format(conn_layer1[0].min_weight, conn_layer1[0].max_weight))

print("Min {} , max {}".format(conn_layer1[1].min_weight, conn_layer1[1].max_weight))
print("Min {} , max wip {}".format(Connection.min_weight_wip, Connection.max_weight_wip))

def doSomething():
    # check if saving
    # if not:
    global close
    close = True
    root.destroy()
root.protocol('WM_DELETE_WINDOW', doSomething)  # root is your root window

ip_app=0
def key(event):
    global ip_app
    ip_app += 1
    if ip_app >= len(inputs):
        ip_app = 0
    Connection.max_weight = 1
    Connection.min_weight = 0
    apply_input(ip_app)
    feed_forward()
    redraw(ip_app)

root.bind("<Button-1>", key)

root.mainloop()







