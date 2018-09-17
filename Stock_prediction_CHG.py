
# Import numpy for arrays and matplotlib for drawing the numbers
import scipy.special, numpy, datetime
import matplotlib.pyplot as plt
class NeuralNetwork:

    # Init network (run eachtime we make new instance of class)
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Set the number of nodes in each input
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight Matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logisitic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propogation of errors
    def train(self, inputs_list, targets_list):

        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin = 2).T
        targets_array = numpy.array(targets_list, ndmin = 2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# Load the training samples
data_file = open("chg_2011.csv", 'r')
data_list = data_file.readlines()
data_file.close()
data_list = numpy.flip(data_list, 0)
all_prices = []

# loop around to eliminate unwanted data, output an array (all_prices) of useable data 
for record in data_list[:-1]:
    # Split the record by the commas
    all_values = record.split(',')
    # Set input values
    if any(x == '-' for x in all_values):
        continue
    open_price = float(all_values[1])
    high_price = float(all_values[2])
    low_price = float(all_values[3])
    close_price = float(all_values[4])
    #temp_array = [open_price, low_price, high_price, close_price]
    temp_array = [open_price, close_price]
    #Obtain difference between previous and current price
    all_prices = numpy.append(all_prices, temp_array)

# Initialise parameters for time variables
number_sample_days = 3 #Define a number of days to sample over for training and testing
prices_per_day = len(temp_array) #number of prices that are being analysed

# Create Neural network Instance
input_nodes = number_sample_days*prices_per_day - prices_per_day # number of input nodes relies on timing interval
hidden_nodes = 8
output_nodes = 1
learning_rate = 0.9
testNeuralNet = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

inputs_temp = []
inputs = []
# divide all_prices in to separate arrays for each day
all_prices = [all_prices[i:i+prices_per_day] for i  in range(0, len(all_prices), prices_per_day)]
all_prices = all_prices[1:] #slice first value from array as it is not a difference value
total_days = len(all_prices) #Total number of days from provided data
day_start_test = round(total_days * 0.8) #Day at which we stop training the algorithm and start testing it (0.9 = 90% training data = 9 years training out of 10 years of data)
# Initialise x and y data to plot
x = []
y = []

# Run test for a time for different confidence factors. Do this with same weights each time
for a in range(1, 50):
    # Train the data over number of sample days, incrementing the interval to train on by 1 each time
    for i in range(0, day_start_test):
        for train in all_prices[i:i+number_sample_days]:
            inputs = numpy.append(inputs, train)
        # Normalise data between 0 and 1 (min/max normalisation)
        inputs_temp = numpy.array( (inputs - numpy.amin(inputs))/(numpy.amax(inputs) - numpy.amin(inputs)) * 0.99 + 0.01 )
        inputs = inputs_temp.astype(numpy.float)
        # subtract closing price of previous day to new day and evaluate evolution over given interval
        diff_24hour = inputs[-1] - inputs[-prices_per_day-1] # current_closing$ - previous_closing$
        # Set targets for single output
        if diff_24hour > 0: 
            #targets = diff_24hour/2 + 0.5 # share price has gone up from previous days closing price
            targets = 0.99
        #elif -0.2 <= diff_24hour <= 0.2:
        #    targets = 0.5
        else:
            #targets = (1-diff_24hour)/2  # share price has gone down from previous days closing price
            targets = 0.01
        inputs = inputs[0:-prices_per_day] # slice last 4 prices of inputs as this is the value we want to predict
        # Train the network with inputs and targets
        if len(inputs) != input_nodes:
            break
        testNeuralNet.train(inputs, targets)
        inputs = [] # Re-initialise inputs after training
    pass
    inputs = []
    j = a *0.01
    # Scorecard list for how well the network performs, initially empty
    scorecard = []

############# UNCOMMENT TO SEE HOW NET PERFORMS OVER A TEST DATA SET #############
#    # Loop through all of the records in the test data set
#    for i in range(day_start_test, total_days):
#        for test in all_prices[i:i+number_sample_days]:
#            inputs = numpy.append(inputs, test)
#        # subtract closing price of previous day to new day and evaluate evolution over given interval
#        diff_24hour = inputs[-1] - inputs[-prices_per_day-1] # current_closing$ - previous_closing$
#        if diff_24hour > 0:
#            correct_label = 0.99 # The share price has gone UP
#        #elif -0.2 <= diff_24hour <= 0.2:
#        #    correct_label = 0.5
#        else:
#            correct_label = 0.01 # The share price has gone DOWN
#        inputs = inputs[0:-prices_per_day] # slice last prices off inputs as this is the days value(s) we want to predict
#        # Normalise data between 0 and 1 (min/max normalisation)
#        inputs_temp = numpy.array( (inputs - numpy.amin(inputs))/(numpy.amax(inputs) - numpy.amin(inputs)) * 0.99 + 0.01 )
#        inputs = inputs_temp.astype(numpy.float)
#        if len(inputs) != input_nodes:
#            break
#        # Query the network
#        outputs = testNeuralNet.query(inputs)
#        inputs = [] # Re-initialise inputs after testing
#        # Append either a 1 or a 0 to the scorecard list
#        interval = [0.5-j, 0.5+j] #test interval of confidence factor
#        if ((outputs>interval[1]) and (correct_label==0.99)) or ((outputs<=interval[0]) and (correct_label==0.01)): # ie if the NN output and corect label are the same 
#            scorecard.append(1)
#        elif ((outputs>interval[1]) and (correct_label==0.01)) or ((outputs<=interval[0]) and (correct_label==0.99)): # ie if the NN output and corect label are different
#            scorecard.append(0)
#        else:
#            pass
#        ## Test whether the performance is better nearer the end of the training data (it would make sense if it was)
#        #scorecard_array = numpy.asarray(scorecard)
#        #performance = (scorecard_array.sum() / scorecard_array.size)*100
#        #print(performance)
#        pass
#    pass
#    # Calculate the performance score, the fraction of correct answers
#    scorecard_array = numpy.asarray(scorecard)
#    performance = (scorecard_array.sum() / scorecard_array.size)*100
#    print('For', len(scorecard_array), 'attempts over', total_days - day_start_test, 'days, Performance=', performance, '%, ', 'Confidence=', a*2, '%')
#    x = numpy.append(x, a*2)
#    y = numpy.append(y, performance)
#plt.xlabel('Confidence (%)')
#plt.ylabel('Performance (%)')
##reverse_x = x[::-1] # reverse x axis to get percentage confidence factor corressponding to correct values
#plt.plot(x,y)
#plt.show()


############# COMMENT TO SEE HOW NET PERFORMS OVER A TEST DATA SET #############
# TESTING NEXT DAY DATA
    for i in range(day_start_test, total_days):
        for test in all_prices[i:i+number_sample_days]:
            inputs = numpy.append(inputs, test)
        # Normalise data between 0 and 1 (min/max normalisation)
        inputs_temp = numpy.array( (inputs - numpy.amin(inputs))/(numpy.amax(inputs) - numpy.amin(inputs)) * 0.99 + 0.01 )
        inputs = inputs_temp.astype(numpy.float)
        inputs = inputs[prices_per_day:]
        if len(inputs) != input_nodes:
            break
        # Query the network
        outputs = testNeuralNet.query(inputs)
        inputs = []

print(outputs)
