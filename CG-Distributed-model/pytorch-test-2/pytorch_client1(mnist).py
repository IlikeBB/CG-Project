import socket, copy, pickle, threading, time, numpy

import pygad.torchga
import pygad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
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

def fitness_func(solution, sol_idx):
    global torch_ga, data_inputs, data_outputs
    
    model = torch_ga.model
    
    predictions = pygad.torchga.predict(model, 
                                        solution, 
                                        data_inputs.float())                  
    loss_function = torch.nn.CrossEntropyLoss()
    data_outputs2 = torch.argmax(data_outputs, dim=1)
    # NOTICE change Loss Tensor to Loss Numpy type
    solution_fitness = (1.0 / (loss_function(predictions, data_outputs2) + 0.00000001)).detach().numpy()
    return solution_fitness

def prepare_GA(server_data):
    global torch_ga
    population_weights = server_data["population_weights"]
    model_json = server_data["model_json"]
    num_solutions = server_data["num_solutions"]
    print("------Server Weight Matrix----------------------------")
    # print(model_json)

    model = ConvNet()
    model.load_state_dict(model_json)
    torch_ga = pygad.torchga.TorchGA(model=model,
                                     num_solutions=num_solutions)

    torch_ga.population_weights = population_weights

    ga_instance = pygad.GA(num_generations=150, 
                           num_parents_mating=4, 
                           initial_population=torch_ga.population_weights.copy(),
                           fitness_func=fitness_func)
    return ga_instance

# Preparing the NumPy array of the inputs.
data_inputs = numpy.load('./dataset/test_images.npy')[0:10]
data_inputs =  torch.tensor(data_inputs)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.load('./dataset/test_labels.npy')[0:10]
data_outputs =  torch.tensor(data_outputs)
_zip = TensorDataset(data_inputs, data_outputs)
trainloader = torch.utils.data.DataLoader(_zip, batch_size=8, shuffle=True)

class RecvThread(threading.Thread):

    def __init__(self, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                soc.settimeout(self.recv_timeout)
                received_data += soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    # print("All data ({data_len} bytes) is received from the server.".format(data_len=len(received_data)), end="\n")
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop and a break statement should be excuted.
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=self.recv_timeout))
                print("{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=self.recv_timeout))
                return None, 0
            except BaseException as e:
                return None, 0
                print("Error While Receiving Data from the Server: {msg}.".format(msg=e))

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            return None, 0

        return received_data, 1

    def run(self):
        global server_data
        subject = "echo"
        server_data = None
        best_sol_idx = -1
        best_model_weights_vector = None

        while True:
            data_dict = {"best_model_weights_vector": best_model_weights_vector}
            data = {"subject": subject, "data": data_dict}
            # data = {"subject": subject, "data": keras_ga, "best_solution_idx": best_sol_idx}
            data_byte = pickle.dumps(data)
            print("Sending a Message of Type {subject} to the Server".format(subject=subject))
            # print(data)
            # print(data_byte)
            try:
                soc.sendall(data_byte)
            except BaseException as e:
                print("Error Connecting to the Server. The server might has been closed: {msg}".format(msg=e))
                break

            print("Receiving Reply from the Server")
            received_data, status = self.recv()
            if status == 0:
                print("Nothing Received from the Server")
                break
            else:
                print("New Message from the Server with subject {sub}".format(sub=received_data["subject"]))

            subject = received_data["subject"]
            if subject == "model":
                server_data = received_data["data"]
            elif subject == "done":
                print("Model is Trained")
                break
            else:
                print("Unrecognized Message Type: {subject}".format(subject=subject))
                break
            
            ga_instance = prepare_GA(server_data)
            
            ga_instance.run()
            subject = "model"
            best_sol_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)[2]
            best_model_weights_vector = ga_instance.population[best_sol_idx, :]
            predictions = torch_ga.model(data_inputs.float())
            predictions2 = numpy.argmax(predictions.detach().numpy(), axis=1)
            data_outputs2= numpy.argmax(data_outputs.detach().numpy(), axis=1)
            # print("predictions", predictions2)
            # print("data_outputs",data_outputs2)
            accuracy = accuracy_score(predictions2, data_outputs2)
            print("------client   -----------------------------")
            best_model_weights_matrix2 = pygad.torchga.model_weights_as_dict(model = ConvNet(), 
                                                                                                    weights_vector = best_model_weights_vector)
            print("------Client Weight Matrix-----------------------------")
            new_model = copy.deepcopy(ConvNet())
            new_model.load_state_dict(best_model_weights_matrix2)
            # print(new_model.state_dict())
            print("Accuracy = {accuracy}\n".format(accuracy=accuracy))
soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created")

try:
    ipv4_address = "localhost"
    port_number = "5556"
    soc.connect((ipv4_address, int(port_number)))
    print("Successful Connection to the Server")
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))

recvThread = RecvThread(buffer_size=6144000, recv_timeout=30)
recvThread.start()