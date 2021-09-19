import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/*
 * This class defines a neural network consisting of any number input nodes in the input layer, any number of
 * hidden nodes in any number of hidden layers, and any number of outputs nodes in the output layer. The user has
 * the option of either manually inputting the weights or randomly initializing these weights. The goal of the
 * neural network is to minimize the error function for any number of hidden layers, using backpropagation.
 *
 * Additionally, the Perceptron class provides the following methods:
 * void initializeWeights()
 * void initializeActivations()
 * void setWeights()
 * double randomizer(double min, double max)
 * double activationFunction(double val)
 * double activationFunctionDerivative(double val)
 * void calculateOutputValue(int indexBeingTested)
 * void backProp(int indexBeingTested)
 * double calculateError(int index)
 * void printHyperParams()
 * void train()
 * void main(String[] args)
 *
 * @author Srivishnu Pyda
 * @version 5/5/2020
 */
public class Perceptron
{
   private double[][][] all_weights;
   private double[][] all_activations;

   private static double[][] all_thetas;
   private static double[][] all_psis;

   private int[] networkStructure;
   private int numberOfLayers;

   private static double[][] userInputs;
   private static double[][] userOutputs;
   private static int numHiddenLayers;
   private static int userInputsArrayLength;
   private static int userInputColumnLength;
   private static int userOutputsColumnLength;

   private static int numTrainingCases;
   private static int numIterations;
   private static double lambda;
   private static double errorThreshold;
   private static double minWeight;
   private static double maxWeight;

   /*
    * Constructor for Perceptron. The constructor takes in an array representing the
    * structure of the network.
    *
    * @param networkStructure array containing the number of nodes in the input layer,
    * number of nodes in each hidden layer, and the number of nodes in the output later.
    */
   public Perceptron(int[] networkStructure)
   {
      this.networkStructure = networkStructure;
      this.numberOfLayers = networkStructure.length;

      all_thetas = new double[numberOfLayers][];

      for (int ind = 0; ind < numberOfLayers; ind++)
      {
         all_thetas[ind] = new double[networkStructure[ind]];
      }

      all_psis = new double[numberOfLayers][];

      for (int n = 0; n < numberOfLayers; n++)
      {
         all_psis[n] = new double[networkStructure[n]];
      }

      initializeWeights();
      initializeActivations();

      for (int ind = 0; ind < userInputColumnLength; ind++)
      {
         for (int k = 0; k < userInputsArrayLength; k++)      // Assigns network's inputs as current target input values
         {
            all_activations[0][k] = userInputs[ind][k];
         }
      }

      setWeights();
   } // public Perceptron(int[] networkStructure)

   /*
    * This method, initializeWeights, initializes the dimensions of the 3-dimensional,
    * jagged, weight matrix.
    */
   public void initializeWeights()
   {
      all_weights = new double[numberOfLayers - 1][][];

      for (int n = 0; n < numberOfLayers - 1; n++)
      {
         all_weights[n] = new double[networkStructure[n]][];
         for (int start = 0; start < networkStructure[n]; start++)
         {
            all_weights[n][start] = new double[networkStructure[n + 1]];
         }
      }
   } // public void initializeWeights()

   /*
    * This method, initializeActivations, intializes the dimensions of the 2-dimensional,
    * jagged, activation matrix.
    */
   public void initializeActivations()
   {
      all_activations = new double[numberOfLayers][];

      for (int n = 0; n < numberOfLayers; n++)
      {
         all_activations[n] = new double[networkStructure[n]];
      }
   } // public void initializeActivations()

   /*
    * The method, setWeights, gives the user two options: either using random weights (1)
    * or manually inputting the weights (2). If the user inputs 1, then the network populates
    * the 3-dimensional weight matrix, all_weights, with random weights. Else, if the user
    * inputs 2, the network populates all_weights with the weights that the user manually inputs.
    * This method uses a Scanner to read user input.
    */
   public void setWeights()
   {
      Scanner sc = new Scanner(System.in);

      // Prompting user to either input 1 to randomly initialize the weights
      // or 2 to manually input the weights
      System.out.println("");
      System.out.println("Do you want to randomize weights (1) or input the weights (2).");
      int answer = sc.nextInt();

      if (answer == 1)
      {
         for (int n = 0; n < numberOfLayers - 1; n++)
         {
            for (int start = 0; start < networkStructure[n]; start++)
            {
               for (int end = 0; end < networkStructure[n + 1]; end++)
               {
                  all_weights[n][start][end] = randomizer(minWeight, maxWeight);
               } // for (int end = 0; end < networkStructure[n+1]; end++)
            } // for (int start = 0; start < networkStructure[n]; start++)
         } // for (int n = 0; n < numberOfLayers - 1; n++)
      } // if (answer == 1)
      else
      {
         for (int n = 0; n < numberOfLayers - 1; n++)
         {
            for (int start = 0; start < networkStructure[n]; start++)
            {
               for (int end = 0; end < networkStructure[n + 1]; end++)
               {
                  System.out.println("Input the weights:");
                  all_weights[n][start][end] = sc.nextDouble();
               } // for (int end = 0; end < networkStructure[n+1]; end++)
            } // for (int start = 0; start < networkStructure[n]; start++)
         } // for (int n = 0; n < numberOfLayers - 1; n++)
      } // else
   } // public void setWeights()

   /*
    * This method, randomizer, that returns a random number of the data-type double
    * within a user-defined range.
    *
    * @param min lower bound of the range.
    * @param max upper bound of the range.
    * @return randomly generated double within the given range.
    */
   public double randomizer(double min, double max)
   {
      double range = (max - min) + 1.0;
      return (Math.random() * range) + min;
   } // public double randomizer(double min, double max)

   /*
    * This method performs the activation function on a
    * given value, and returns the resulting value.
    *
    * @param val This parameter represents the input value to the activation function.
    * @return This method returns the output value of the activation function.
    */
   public double activationFunction(double val)
   {
      return 1.0 / (1.0 + Math.exp(-val));
   } // public double activationFunction(double val)

   /*
    * This method performs the derivative of the activation function on a
    * given value, and returns the resulting value.
    *
    * @param val This parameter represents the input value to the derivative of the activation function
    * @return This method returns the output value of the derivative of the activation function
    */
   public double activationFunctionDerivative(double val)
   {
      double temp = activationFunction(val);
      return (temp * (1.0 - temp));
   } // public double activationFunctionDerivative(double val)

   /*
    * This method forward propagates through the network. Each
    * activation is calculated by finding the weighted sum
    * of the nodes and weights in the previous layer and having this weighted
    * sum passed through an activation or threshold function.
    * The corresponding theta value of each node is the same as the weighted
    * sum calculated.
    *
    * @param indexBeingTested index of which input training set is being tested
    */
   public void calculateOutputValue(int indexBeingTested)
   {
      for (int i = 0; i < userInputsArrayLength; i++)
      {
         all_activations[0][i] = userInputs[indexBeingTested][i];
      }

      for (int n = 0; n < numberOfLayers - 1; n++)
      {
         for (int end = 0; end < networkStructure[n+1]; end++)
         {
            all_thetas[n+1][end] = 0.0;

            for (int start = 0; start < networkStructure[n]; start++)
            {
               all_thetas[n+1][end] += all_activations[n][start] * all_weights[n][start][end];
            }

            all_activations[n+1][end] = activationFunction(all_thetas[n+1][end]);
         } // for (int end = 0; end < networkStructure[n+1]; end++)
      } // for (int n = 0; n < numberOfLayers-1; n++)
   } // public void calculateOutputValue(int indexBeingTested)

   /*
    * Performs the backpropagation algorithm, which aims to optimize the
    * learning process and minimize the error function by adjusting the
    * network's weights.
    *
    * @param indexBeingTested index of which input training set is being tested
    */
   public void backProp(int indexBeingTested)
   {
      double deltaChange;
      double smallOmega;
      double bigOmega = 0.0;
      double thetaDeriv;

      for (int i = 0; i < networkStructure[numberOfLayers - 1]; i++)
      {
         smallOmega = (userOutputs[i][indexBeingTested] - all_activations[numberOfLayers-1][i]);
         all_psis[numberOfLayers-1][i] =  smallOmega * activationFunctionDerivative(all_thetas[numberOfLayers-1][i]);
      }

      for (int n = numberOfLayers - 2; n >= 0; n--)
      {
         for (int start = 0; start < networkStructure[n]; start++)
         {
            bigOmega = 0.0;

            for (int end = 0; end < networkStructure[n+1]; end++)
            {
               bigOmega += all_psis[n+1][end] * all_weights[n][start][end];
               deltaChange = all_activations[n][start] * all_psis[n+1][end];
               all_weights[n][start][end] += lambda * deltaChange;
            }

            thetaDeriv = activationFunctionDerivative(all_thetas[n][start]);
            all_psis[n][start] = bigOmega * thetaDeriv;
         } // for (int start = 0; start < networkStructure[n]; start++)
      } // for (int n = numLayers - 2; n >= 0; n--)
   } // public void backProp(int indexBeingTested)

   /*
    * This method, calculateError, calculates the error for the
    * current training set.
    *
    * @param index index of which training set the error
    * is being calculated for.
    * @return half of the square root of the error between the
    * true value and the output produced by the Perceptron for that training set.
    */
   public double calculateError(int index)
   {
      double error = 0.0;
      double temp = 0.0;

      for (int i = 0; i < userOutputsColumnLength; i++)
      {
         temp = userOutputs[i][index] - all_activations[numberOfLayers-1][i];
         error += 0.5 * temp * temp;
      }

      return error;
   } // public double calculateError(double output)

   /*
    * This methods prints the hyper parameters of the
    * networks, including the weight range, the configuration
    * of the network, and the number of nodes in each of the layers.
    */
   public void printHyperParams()
   {
      System.out.println("The weight range is between: " + minWeight + " and " + maxWeight);
      System.out.print("Network Configuration: " + networkStructure[0]);

      for (int n = 1; n < numberOfLayers; n++)
      {
         System.out.print("x" + networkStructure[n]);
      }

      System.out.println("");
      System.out.println("Number of nodes in Input Layer: " + networkStructure[0]);
      System.out.println("Number of Hidden Layers: " + numHiddenLayers);

      for (int n = 1; n < numberOfLayers - 1; n++)
      {
         System.out.println("Number of nodes in Hidden Layer " + n + ": " + networkStructure[n]);
      }

      System.out.println("Number of nodes in Output Layer: " + networkStructure[numberOfLayers-1]);
   } // public void printHyperParams()

   /*
    * Training the network to minimize the error function. The calculateOutputValue method is
    * called to calculate the output values with the given input training set. The backProp
    * method readjusts the weights to minimize the error between the calculated and actual
    * output values. The network runs for the number of iterations specified by the user. The training
    * terminates if the number of iterations (of the while loop) is greater than the number
    * of iterations specified by the user. Additionally, if the error threshold exceeds the total error or
    * the lower learning rate threshold exceeds the given learning rate, the training terminates.
    */
   public void train()
   {
      int trackingIterations = 0;
      double totalError = 0.0;
      int caseNumber = 1;
      boolean flag = true;

      while (flag)
      {
         trackingIterations++;
         totalError = 0.0;

         for (int k = 0; k < numTrainingCases; k++)
         {
            calculateOutputValue(k);
            backProp(k);
            calculateOutputValue(k);
            totalError += calculateError(k);
            totalError /= numTrainingCases;
         } // for (int k = 0; k < numTrainingCases; k++)



         if (totalError < errorThreshold)
         {
            System.out.println("");
            System.out.println("Training completed with error below the error threshold of " + errorThreshold);
            System.out.println("Total error is " + totalError);
            System.out.println("Number of iterations:" + trackingIterations);
            System.out.println("Max number of iterations: " + numIterations);
            System.out.println("Learning rate: " + lambda);
            System.out.println("");

            for (int i = 0; i < userInputColumnLength; i++)
            {
               calculateOutputValue(i);

               for (int j = 0; j < userInputsArrayLength; j++)
               {
                  all_activations[0][j] = userInputs[i][j];
               }

               //System.out.println("Inputs for case " + caseNumber + " are: " + userInputs[i][0] + " and " + userInputs[i][1]);
               System.out.println("Calculated Outputs for case " + caseNumber + " is: " + all_activations[numberOfLayers-1][0]
                       + ", " + all_activations[numberOfLayers-1][1] + ", " + all_activations[numberOfLayers-1][2]
                       + ", " + all_activations[numberOfLayers-1][3] + ", " + all_activations[numberOfLayers-1][4]);
               System.out.println("True Outputs for case " + userOutputs[0][i] + ", " + userOutputs[1][i] + ", " +
                       ", " + userOutputs[2][i] + ", " + userOutputs[3][i] + ", " + userOutputs[4][i]);
               System.out.println("");
               caseNumber++;
            } // for (int i = 0; i < userInputColumnLength; i++)

            flag = false;
         } // if (totalError < errorThreshold)
         System.out.println("Iteration: " + trackingIterations + " Total Error: " + totalError);

         if (trackingIterations >= numIterations)
         {
            caseNumber = 1;
            System.out.println("");
            System.out.println("The training timed out with " + trackingIterations + " iterations");
            System.out.println("Total error was " + totalError);
            System.out.println("Learning rate: " + lambda);
            System.out.println("");

            for (int i = 0; i < userInputColumnLength; i++)
            {
               calculateOutputValue(i);

               for (int j = 0; j < userInputsArrayLength; j++)
               {
                  all_activations[0][j] = userInputs[i][j];
               }

               System.out.println("Calc");
               //System.out.println("Inputs for case " + caseNumber + " are: " + userInputs[i][0] + " and " + userInputs[i][1]);
               System.out.println("Calculated Outputs for case " + i + " is: " + all_activations[numberOfLayers-1][0]
                       + ", " + all_activations[numberOfLayers-1][1] + ", " + all_activations[numberOfLayers-1][2]
                       + ", " + all_activations[numberOfLayers-1][3] + ", " + all_activations[numberOfLayers-1][4]);
               System.out.println("True Outputs for case " + userOutputs[0][i] + ", " + userOutputs[1][i] + ", " +
                       + userOutputs[2][i] + ", " + userOutputs[3][i] + ", " + userOutputs[4][i]);
               System.out.println("");
               caseNumber++;
            } // for (int i = 0; i < userInputColumnLength; i++)

            flag = false;
         } // if (trackingIterations >= numIterations)
      } // while (flag)

      printHyperParams();
   } // public void train()

   /*
    * The main method asks the user for the desired number of nodes in the input layer
    * and number of nodes in the hidden layer. Additionally, this method prompts the user
    * to enter the values for the input activations. Then, a perceptron object is created
    * in order to calculate the outputValue using the given input activation values.
    *
    * @param args an array of strings which stores arguments passed by command line while
    * starting a program. All the command line arguments are stored in that array.
    */
   public static void main(String[] args) throws FileNotFoundException
   {

      DibDump dib = new DibDump();
      dib.bmpToPelFile("FingerFiles/one.bmp", "DibDumpInputs/DibDump0.txt");
      dib.bmpToPelFile("FingerFiles/one_diagonal.bmp", "DibDumpInputs/DibDump1.txt");
      dib.bmpToPelFile("FingerFiles/two.bmp", "DibDumpInputs/DibDump2.txt");
      dib.bmpToPelFile("FingerFiles/two_diagonal.bmp", "DibDumpInputs/DibDump3.txt");
      dib.bmpToPelFile("FingerFiles/three.bmp", "DibDumpInputs/DibDump4.txt");
      dib.bmpToPelFile("FingerFiles/four.bmp", "DibDumpInputs/DibDump5.txt");
      dib.bmpToPelFile("FingerFiles/five.bmp", "DibDumpInputs/DibDump6.txt");
      dib.bmpToPelFile("FingerFiles/three_diagonal.bmp", "DibDumpInputs/DibDump7.txt");
      dib.bmpToPelFile("FingerFiles/four_diagonal.bmp", "DibDumpInputs/DibDump8.txt");
      dib.bmpToPelFile("FingerFiles/five_diagonal.bmp", "DibDumpInputs/DibDump9.txt");
      Scanner sc = new Scanner(System.in);

      System.out.println("");
      System.out.println("Enter max number of iterations: ");
      numIterations = sc.nextInt();

      System.out.println("Enter lambda: ");
      lambda = sc.nextDouble();

      System.out.println("Enter min weight: ");
      minWeight = sc.nextDouble();

      System.out.println("Enter max weight: ");
      maxWeight = sc.nextDouble();

      System.out.println("Enter the error threshold: ");
      errorThreshold = sc.nextDouble();

      File filename = new File("config");
      sc = new Scanner(filename);

      int numInputNodes = sc.nextInt();
      numHiddenLayers = sc.nextInt();

      // 2 is added to account for the input and output layers
      int totalNumLayers = numHiddenLayers + 2;

      // This array is used to represent the structure of the network
      int[] userNetworkStructure = new int[totalNumLayers];

      userNetworkStructure[0] = numInputNodes;

      for (int n = 1; n < totalNumLayers - 1; n++)
      {
         userNetworkStructure[n] = sc.nextInt();
      }

      userNetworkStructure[totalNumLayers-1] = sc.nextInt();

      numTrainingCases = sc.nextInt();
      System.out.println("Number training cases: " + numTrainingCases);

      userInputs = new double[numTrainingCases][userNetworkStructure[0]];

      userInputsArrayLength = userInputs[0].length;
      userInputColumnLength = userInputs.length;

      for (int i = 0; i < numTrainingCases; i++)
      {
            Scanner newsc = new Scanner(new File(sc.next()));
            for (int n = 0; n < numInputNodes; n++)
            {
               double input = newsc.nextDouble();
               userInputs[i][n] = input;
               System.out.print(userInputs[i][n] + " ");
            }
            System.out.println("");
      }

      userOutputs = new double[userNetworkStructure[totalNumLayers-1]][numTrainingCases];
      userOutputsColumnLength = userOutputs.length;

      for (int i = 0; i < numTrainingCases; i++)
      {
         for (int n = 0; n < userNetworkStructure[totalNumLayers-1]; n++)
         {
            double outputval = sc.nextDouble();
            userOutputs[n][i] = outputval;
            System.out.print(userOutputs[n][i] + " ");
         }
         System.out.println("");
      }

      // Creating a new Perceptron object in order to train the neural network to minimize the error function.
      Perceptron p = new Perceptron(userNetworkStructure);
      p.train();

   } // public static void main(String[] args)
} // public class Perceptron