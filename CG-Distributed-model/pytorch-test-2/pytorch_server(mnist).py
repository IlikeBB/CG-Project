import socket, copy, pickle, threading, time, numpy

import pygad.torchga
import pygad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def close_socket(*args):
    soc.close()
    print("Socket Closed")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.linear1 = nn.Linear(28*28, 100) 
        self.linear2 = nn.Linear(100, 50) 
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=6144000, recv_timeout=30):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        all_data_received_flag = False
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                try:
                    pickle.loads(received_data)
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop. The flag all_data_received_flag is set to True to signal all data is received.
                    all_data_received_flag = True
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.

                elif all_data_received_flag:
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0

                else:
                    # In case data is received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, model, best_model_weights_matrix):
        model_weights_vector = pygad.torchga.model_weights_as_vector(model=model)
        model_weights_matrix = pygad.torchga.model_weights_as_dict(model=model,
                                                                     weights_vector=model_weights_vector)
        # new_weights = numpy.array(model_weights_matrix + best_model_weights_matrix)/2

        # Notice!!!!!!!!!!!!!!!!!!!!! Please Check model index name . Example: old_model['0 weight'] -> new_modl['0 weight']
        temp_model = copy.deepcopy(model)
        temp_model.load_state_dict(best_model_weights_matrix)
        print("------Client Weight Matrix-----------------------------")
        # print(temp_model.state_dict())

        new_weights = model_weights_matrix
        temp_model = copy.deepcopy(model)
        temp_model.load_state_dict(model_weights_matrix)
        print("------Server Weight Matrix----------------------------")
        # print(temp_model.state_dict())
        for idx in (new_weights):
            new_weights[idx] = new_weights[idx] + best_model_weights_matrix[idx]
            new_weights[idx] = new_weights[idx] / 2
        model.load_state_dict(new_weights)
        print("------model_averaging pass--------------------------")
        # print(model.state_dict())
    def reply(self, received_data):
        global torch_ga, data_inputs, data_outputs, model
        # Notice!!!!!!!!!!!!!!! Please check pytorch dict list had key index.
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                msg_model = received_data["data"]
                print("Client's Message Subject is {subject}.".format(subject=subject))

                print("Replying to the Client.")
                if subject == "echo":
                    if msg_model is None:
                        data_dict = {"population_weights": torch_ga.population_weights,
                                     "model_json": model.state_dict(),
                                     "num_solutions": torch_ga.num_solutions}
                        data = {"subject": "model", "data": data_dict}
                    else:
                        with torch.no_grad():
                                    valid_loss = 0.0
                                    model.eval()
                                    _p = []
                                    _y = []
                                    for step, (b_x, b_y) in enumerate(trainloader):
                                        b_x_, b_y_= b_x.float(), b_y
                                        r_out = model(b_x_)
                                        _, preds = torch.max(r_out, 1) 
                                        _p.extend(preds.view(-1).detach().cpu().numpy())    
                                        _y.extend(b_y_.cpu().view(-1).detach().cpu().numpy())
                                        torch.cuda.empty_cache()
                                        
                        accuracy = accuracy_score(numpy.array(_p),  numpy.array(_y))
                        print('Echo Accuracy: ', accuracy)
                        # In case a client sent a model to the server despite that the model accuracy is 1.0. In this case, no need to make changes in the model.
                        if accuracy == 1.0:
                            data = {"subject": "done", "data": None}
                        else:
                            data_dict = {"population_weights": torch_ga.population_weights,
                                         "model_json": model.state_dict(),
                                         "num_solutions": torch_ga.num_solutions}
                            data = {"subject": "model", "data": data_dict}
                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    print("------client   -----------------------------")
                    try:
                        best_model_weights_vector = received_data["data"]["best_model_weights_vector"]
                        best_model_weights_matrix = pygad.torchga.model_weights_as_dict(model=model,
                                                                                          weights_vector=best_model_weights_vector)
                        if model is None:
                            print("Model is None")
                        else:
                            new_model = copy.deepcopy(model)
                            new_model.load_state_dict(best_model_weights_matrix)
                            
                            predictions = model(data_inputs.float())
                            predictions2 = numpy.argmax(predictions.detach().numpy(), axis=1)
                            data_outputs2= numpy.argmax(data_outputs.detach().numpy(), axis=1)
                            accuracy = accuracy_score(predictions2, data_outputs2)
                            
                            # In case a client sent a model to the server despite that the model accuracy is 1.0. In this case, no need to make changes in the model.
                            if accuracy == 1.0:
                                data = {"subject": "done", "data": None}
                                response = pickle.dumps(data)
                                return

                            self.model_averaging(model, best_model_weights_matrix)
                        predictions = model(data_inputs.float())
                        predictions2 = numpy.argmax(predictions.detach().numpy(), axis=1)
                        data_outputs2= numpy.argmax(data_outputs.detach().numpy(), axis=1)
                        accuracy = accuracy_score(predictions2, data_outputs2)
                        print("Accuracy = {accuracy}\n".format(accuracy=accuracy))
                        if accuracy != 1.0:
                            data_dict = {"population_weights": torch_ga.population_weights,
                                         "model_json": model.state_dict(),
                                         "num_solutions": torch_ga.num_solutions}
                            data = {"subject": "model", "data": data_dict}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("reply(): Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
                print("Error Parsing Received Dictionary")
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))

    def run(self):
        print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break
            # print(received_data)
            self.reply(received_data)

class ListenThread(threading.Thread):
    

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            try:
                connection, client_info = soc.accept()
                print("\nNew Connection from {client_info}".format(client_info=client_info))
                socket_thread = SocketThread(connection=connection,
                                             client_info=client_info, 
                                             buffer_size=6144000,
                                             recv_timeout=30)
                socket_thread.start()
            except BaseException as e:
                soc.close()
                print("Error in the run() of the ListenThread class: {msg}.\n".format(msg=e))
                print("Socket is No Longer Accepting Connections")
                break

model = None
# Preparing the NumPy array of the inputs.
data_inputs = numpy.load('./dataset/train_images.npy')
data_inputs =  torch.tensor(data_inputs)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.load('./dataset/train_labels.npy')
data_outputs =  torch.tensor(data_outputs)

_zip = TensorDataset(data_inputs, data_outputs)
trainloader = torch.utils.data.DataLoader(_zip, batch_size=8, shuffle=True)
num_classes = 10
# Build the keras model using the functional API.
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet()
print(model)
num_solutions = 10
# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                                num_solutions=10)

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created")

ipv4_address = "localhost"
port_number = "5556"
soc.bind((ipv4_address, int(port_number)))
print("Socket Bound to IPv4 & Port Number")

soc.listen(1)
print("Socket is Listening for Connections")
print("-----------------processing--------------------")
listenThread = ListenThread()
listenThread.start()