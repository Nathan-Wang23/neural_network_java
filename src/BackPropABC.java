import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class BackPropABC
{

      /**
       * The number of user-input nodes.
       */
      private int numInputNodes;

      /**
       * An array of doubles that represent the inputs received from the user. Used for my printAllValues function,
       * so therefore not integral to the running of the neural network.
       */
      private double[] inputs;

      /**
       * The number of hidden layers given by the constructor parameter, hiddenLayer's, length.
       */
      private int numHiddenLayers;

      /**
       * An array of integers that represent the number of nodes in each hidden layer. Used for my printAllValues
       * function, so therefore not integral to the running of the neural network.
       */
      private int[] hiddenLayer;

      /**
       * The number of output nodes; Represents how many outputs (values that are received when user inputs are passed
       * through the connectivity pattern, threshold function, and activation functions) are anticipated.
       */
      private int numOutputNodes;

      /**
       * 2D array of doubles that is the activation values of each node. These are the values resulting from either inputs
       * given or calculated with the dot product of weights and activation nodes (put in a threshold function).
       */
      private double[][] activation;

      /**
       * Array of doubles that is the weight values connecting nodes. The first set of brackets represent the layer,
       * the second represents the input node index, and the third represents the hidden node index.
       */
      private double[][][] weight;

      /**
       * Array of integers that represents the number of vertical nodes within each n layer. This is to help keep track
       * of layer iterations (mainly for j and k) since these two for loops must rely on number of nodes in each
       * vertical column and each layer can have different number of nodes.
       */
      private int[] verticalLayers;

      /**
       * The total number of layers in the network. Given by numHiddenActivation layers + 2. (The 2 is for input and
       * output layers). This is just an easier way to visualize the numHiddenLayers + 2 in the for loops.
       */
      private int totalLayers;

      /**
       * Expected output for the training set test. This is provided by the user while testing.
       */
      private double expectedOutput;

      /**
       * Threshold value to determine whether the error is within acceptable range.
       */
      private double lambda;

      /**
       * Maximum number of iterations before stopping the steepest descent.
       */
      private int maxIterations;

      /**
       * Minimum error to stop the steepest descent.
       */
      private double errorThreshold;

      /**
       * Number of iterations the training has undergone.
       */
      private int iterations;

      /**
       * The most number of nodes in a vertical layer. Needed to know the smallest possible size of the arrays to make.
       * Can be used as a variable within a method, but it's included here for future design options.
       */
      private int maxActivations;

      /**
       * Minimum lambda for adaptive learning.
       */
      private double minLambda;

      /**
       * Minimum weight for randomized weights.
       */
      private double minWeight;

      /**
       * Maximum weight for randomized weights.
       */
      private double maxWeight;

      /**
       * Used in the tester so I can decide if I want to test the getOutput() or gradient descent.
       */
      private boolean gradient;

      /**
       * An array that holds the expected outputs of each training set. For multiple outputs, I should make it a 2D array.
       */
      private double[][] expected;

      /**
       * 2D array that stores the inputs for each set of training values.
       */
      private double[][] inputSets;

      /**
       * The number of training sets there are.
       */
      private int numSets;

      /**
       * Holds the starting lambda to print for debug. Test if adaptive learning is working.
       */
      private double startingLambda;


      /**
       * Used in First version of findWeightChange() train neural to determine max Error and if weights should revert
       * to original weights (if the changed error exceeds the new error).
       */
      private double[] trainingErrors;

      /**
       * How much each weight should change by during steepest descent training.
       */
      private double[][][] delta;

      /**
       * Mulitplier to change the learning factor by during adaptive learning.
       */
      private double multiplier;

      private boolean testingAll;

      private double[][] thetas;
      private double[] omegas;
      private double[] psis;
      private double[] psis2;
      private double[] layer2;

      /**
       * Constructor for the neural network. Assigns number of Inputs.txt, number of nodes in each hidden layer, number of
       * layers in the network, size of the arrays, and sets the default weights.
       *
       * @param input       the number of input nodes.
       * @param hiddenLayer an array with each integer value representing the number of nodes in each hidden layer.
       * @param output      the number of output nodes.
       */
      public BackPropABC(int input, int[] hiddenLayer, int output)
      {
         gradient = true;
         testingAll = false;

         numInputNodes = input;
         this.hiddenLayer = hiddenLayer;
         numHiddenLayers = hiddenLayer.length;
         numOutputNodes = output;

         totalLayers = numHiddenLayers + 2;

         int maxActivations = 0;                      // This stores the largest number of nodes in a vertical layer.
         for (int i = 0; i < numHiddenLayers; i++)
         {
            maxActivations = Math.max(maxActivations, hiddenLayer[i]);
         }
         maxActivations = Math.max(maxActivations, numInputNodes);
         maxActivations = Math.max(maxActivations, numOutputNodes);
         this.maxActivations = maxActivations;        //Not needed but included for possible future design options.

         activation = new double[totalLayers][maxActivations];
         thetas = new double[totalLayers][maxActivations];

         weight = new double[numHiddenLayers + 1][maxActivations][maxActivations];
         double[] inputs = setInputs();

         for (int j = 0; j < numInputNodes; j++) //Sets inputs into activation.
         {
            activation[0][j] = inputs[j];
         }
         delta = new double[totalLayers - 1][maxActivations][maxActivations];
         trainingErrors = new double[numSets];

         verticalLayers = new int[totalLayers];       //Number of nodes in each layer
         verticalLayers[0] = numInputNodes;
         verticalLayers[numHiddenLayers + 1] = output;

         for (int j = 1; j < numHiddenLayers + 1; j++)
         {
            verticalLayers[j] = hiddenLayer[j - 1];
         }

         omegas = new double[numOutputNodes];
         psis = new double[numOutputNodes];
         psis2 = new double[maxActivations];
         layer2 = new double[maxActivations];

         setWeightsDefault();
      } // Constructor public NeuralNet(int input, int[] hiddenLayer, int output)

      /**
       * Sets the gradient parameters to the default values. Does not include adaptive learning.
       */
      public void setGradientDefault()
      {
         lambda = 5.0;
         startingLambda = 5.0;
         multiplier = 1.0;
         maxIterations = 10000;
         errorThreshold = 0.01;
         minWeight = -2.0;
         maxWeight = 2.0;
         minLambda = 0.0;
      }

      /**
       * Sets the inputs to the user's inputted values. Can input with typing inputs or a file name.
       * Also sets expected outputs for gradient testing.
       *
       * @return an array of the double values of the inputs received from the user.
       */
      public double[] setInputs()
      {
         Scanner inputs = new Scanner(System.in);
         double[] input = new double[numInputNodes];

         if (gradient)
         {
            String inputsFile = "inputs.txt";
            try
            {
               Scanner in = new Scanner(new File(inputsFile));
               System.out.println("How many training sets are there?");
               numSets = inputs.nextInt();
               inputSets = new double[numSets][numInputNodes];
               for (int t = 0; t < numSets; t++)
               {
                  for (int k = 0; k < numInputNodes; k++)
                  {
                     inputSets[t][k] = in.nextDouble();
                  }
               }
               in.close();
            } // Reads the input file.
            catch (FileNotFoundException e)
            {
               throw new RuntimeException(e);
            }

            System.out.println("Which logic function do you want to test for? \nIf you want to put in your own " +
               "expected outputs file name, type it.");
            String answer = inputs.next().toLowerCase();

            if (answer.equals("and")) //AND, OR, AND XOR ARE JUST HARD-CODED IN TO MAKE MY TESTING EASIER.
            {
               try
               {
                  Scanner in = new Scanner(new File("and_expectedoutputs.txt"));
                  expected = new double[numSets][numOutputNodes];
                  for (int t = 0; t < numSets; t++) //change to numSets
                  {
                     double ex = in.nextDouble();
                     for (int i = 0; i < numOutputNodes; i++)
                     {
                        expected[t][i] = ex;
                     }
                  }
                  in.close();

               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            } //Sets up network to test AND. if (answer.equals("and"))

            else if (answer.equals("or"))
            {
               try
               {
                  Scanner in = new Scanner(new File("or_expectedoutputs.txt"));
                  expected = new double[numSets][numOutputNodes];
                  for (int t = 0; t < numSets; t++) //change to numSets
                  {
                     double ex = in.nextDouble();
                     for (int i = 0; i < numOutputNodes; i++)
                     {
                        expected[t][i] = ex;
                     }
                  }
                  in.close();
               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            } //Sets up network to test OR. else if (answer.equals("or"))

            else if (answer.equals("xor"))
            {
               try
               {
                  Scanner in = new Scanner(new File("xor_expectedoutputs.txt"));
                  expected = new double[numSets][numOutputNodes];
                  for (int t = 0; t < numSets; t++) //change to numSets
                  {
                     double ex = in.nextDouble();
                     for (int i = 0; i < numOutputNodes; i++)
                     {
                        expected[t][i] = ex;
                     }
                  }
                  in.close();
               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            } // Sets up network to test XOR. else if (answer.equals("xor"))

            else if (answer.equals("all"))
            {
               testingAll = true;
               expected = new double[numSets][numOutputNodes];
               try
               {
                  Scanner in = new Scanner(new File("all_expectedoutputs.txt"));
                  for (int t = 0; t < numSets; t++)
                  {
                     for (int i = 0; i < numOutputNodes; i++)
                     {
                        expected[t][i] = in.nextDouble();
                     }
                  }
                  in.close();
               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            } // Sets up the network to test OR, AND, and XOR. else if (answer.equals("all"))
            else //User inputs own outputs.
            {
               try
               {
                  Scanner in = new Scanner(new File(answer));
                  expected = new double[numSets][numOutputNodes];
                  for (int t = 0; t < numSets; t++)
                  {
                     for (int i = 0; i < numOutputNodes; i++)
                     {
                        expected[t][i] = in.nextDouble();
                     }
                  }
                  in.close();
               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            } //Scans in a user's special file.
            System.out.println("DEBUG: ");
            for (int t = 0; t < numSets; t++)
            {
               for (int k = 0; k < numInputNodes; k++)
               {
                  System.out.print(inputSets[t][k] + " ,");
               }
            }
         } // For testing Steepest descent.  if (gradient)

         else //Not testing the steepest descent.
         {
            System.out.println("If you want to type in the inputs, type \"Type.\" If you want to select a file, type " +
               "\"File\"");
            String user = inputs.next().toLowerCase();

            if (user.equals("type"))
            {
               System.out.println("Type in the inputs you would like to use.");
               for (int k = 0; k < numInputNodes; k++)
               {
                  input[k] = inputs.nextDouble();
               }
               this.inputs = input;
            }
            else if (user.equals("file"))
            {
               System.out.println("Type in the file name you would like to use.");
               try
               {
                  String filename = inputs.next();
                  if (filename.equals("."))
                  {
                     filename = "UserInputs.txt";
                  }

                  Scanner in = new Scanner(new File(filename));
                  List<Double> list = new ArrayList<Double>();

                  while (in.hasNextDouble())
                     list.add(in.nextDouble());
                  in.close();

                  for (int i = 0; i < list.size(); i++)
                  {
                     System.out.println(list.get(i));
                     input[i] = list.get(i);
                  }
                  this.inputs = input;
               }
               catch (FileNotFoundException e)
               {
                  throw new RuntimeException(e);
               }
            }
            else //neither file nor a valid input was typed into the console.
            {
               throw new RuntimeException("Type file or type");
            }
         } //For testing getOutput() first step.  else //Not testing the steepest descent.
         return input;
      } // Input and Expected Value Setter.    public double[] setInputs()

      /**
       * Sets the weights to the default values. Default is a 2-2-1 network. (with 6 weights)
       */
      public void setWeightsDefault()
      {
         weight[0][0][0] = 0.3;
         weight[0][0][1] = 0.5;
         weight[0][1][0] = 0.4;
         weight[0][1][1] = 0.2;
         weight[1][0][0] = 0.8;
         weight[1][1][0] = 0.2;
      }

      /**
       * Sets the weights to the user-provided weights.
       */
      public void setUserWeights()
      {
         Scanner sc = new Scanner(System.in);
         for (int n = 0; n < numHiddenLayers + 1; n++)
         {
            for (int k = 0; k < verticalLayers[n]; k++)
            {
               for (int j = 0; j < verticalLayers[n + 1]; j++)
               {
                  System.out.println("Please type weight value for w" + n + k + j + ".");
                  double userNum = sc.nextDouble();
                  weight[n][k][j] = userNum;
                  System.out.println("DEBUG: Weight w" + n + k + j + " set to " + userNum + ".\n");
               }
            }
         }
      } // Sets Weights to User's choice. public void setUserWeights()

      /**
       * Randomizes a number between min and max.
       * @param min the minimum number
       * @param max the maximum number
       * @return the random number between min and max.
       */
      public double randomize(double min, double max)
      {
         return (Math.random() * (max - min)) + min;
      }

      /**
       * Sets the weights to a random double between the min and max weight.
       */
      public void setWeightsRandom()
      {
         for (int n = 0; n < numHiddenLayers + 1; n++)
         {
            for (int k = 0; k < verticalLayers[n]; k++)
            {
               for (int j = 0; j < verticalLayers[n + 1]; j++)
               {
                  weight[n][k][j] = randomize(minWeight, maxWeight);
                  System.out.println("DEBUG: Weight w" + n + k + j + " set to " + weight[n][k][j] + ".\n");
               }
            }
         }
      } // Sets weights to random. Iterates through each layer. public void setWeightsRandom()

      /**
       * The wrapper function to limit/control the activation values.
       *
       * @param x the value to pass through the function.
       * @return the value passed through the threshold function.
       */
      private double thresholdFunction(double x)
      {
         return 1.0 / (1.0 + Math.exp(-x));
      }

      /**
       * Returns the derivative of the threshold function for a value x.
       *
       * @param x the value to find f'(x) of.
       * @return the derivative of the threshold function at x.
       */
      private double derivativeThreshold(double x)
      {
         return thresholdFunction(x) * (1.0 - thresholdFunction(x));
      }

      /**
       * Calculates weight change and applies it to the network. Uses the math from the document Minimization of the
       * Single output error function.
       * @param trainingTest the test case being run.
       */
      public void deltaWeights(int trainingTest)
      {
         double[] omegas = new double[numOutputNodes];
         double[] psis = new double[numOutputNodes];
         for (int i = 0; i < verticalLayers[2]; i++)
         {
            omegas[i] = expected[trainingTest][i] - activation[totalLayers - 1][i];
            double thetaI = thetas[2][i];

            psis[i] = omegas[i] * derivativeThreshold(thetaI);     //ψ0 = ω0 * derivative of theta0

            for (int j = 0; j < verticalLayers[1]; j++)             //iterates through each hidden layer node
            {
               double hJ = activation[1][j];                        //Thresholded thetaJ
               double derivErrorJI = hJ * psis[i];                     //derivJ0 = -hJ * ψ0
               delta[1][j][i] = lambda * derivErrorJI;              //delta = -λ * derivJ0
            }
         } //Loop for output layer delta weights. for (int i = 0; i < verticalLayers[2]; i++)

         for (int j = 0; j < verticalLayers[1]; j++)                //For First Layer Weights.
         {
            double layer2 = 0.0;
            double fI = 0.0;
            for (int i = 0; i < numOutputNodes; i++)
            {
               fI = activation[numHiddenLayers + 1][i];
               double thetaI = thetas[2][i];

               double psiI = weight[1][j][i] * derivativeThreshold(thetaI);//Ψj0 = Ωj0 * derivative of thetaJ
               layer2 += omegas[i] * psiI;
            }

            double thetaJ = thetas[1][j];

            for (int k = 0; k < verticalLayers[0]; k++)
            {
               double derivErrorKJ = activation[0][k] * derivativeThreshold(thetaJ);   //derivKJ = -Ak * Ψj0
               delta[0][k][j] = lambda * derivErrorKJ * layer2;           // delta = −λ * derivKJ
            }
         } // Loop for Hidden Layer delta weights. for (int j = 0; j < verticalLayers[1]; j++)


      } // Calculates the Delta Weights. public void deltaWeights()

      /**
       * Backpropagation of the delta weights. First evaluates network in FillActivations, then runs another forward
       * loop to calcuate omegas and psis. Finally it runs back prop to train the network and change weights on the fly.
       */
      public void backProp()
      {
         for (int k = 0; k < numInputNodes; k++)
         {

            for (int j = 0; j < verticalLayers[1]; j++)
            {
               psis2[k] = 0.0;
               layer2[j] = 0.0;
               for (int i = 0; i < numOutputNodes; i++)
               {
                  layer2[i] += psis[i] * weight[1][j][i];
                  weight[1][j][i] += lambda * activation[1][j] * psis[i];
                  delta[1][j][i] = lambda * activation[1][j] * psis[i];
               }
               psis2[j] += layer2[j] * derivativeThreshold(thetas[1][j]);
               weight[0][k][j] += lambda * psis2[j] * activation[0][k] * derivativeThreshold(thetas[1][j]);
               delta[0][k][j] = lambda * psis2[j] * activation[0][k] * derivativeThreshold(thetas[1][j]);
            }
         }
      } // Calculates the Delta Weights. public void backProp(int testCase)


      /**
       * Fills the activation 2d array with the calculated activations given weights and inputs.
       * THIS VERSION ONLY DOES A 2 LAYER
       */
      public void fillActivations(int trainingTest)
      {
         for (int j = 0; j < verticalLayers[1]; j++)
         {
            double dotFirst = 0.0;
            for (int k = 0; k < verticalLayers[0]; k++)
            {
               dotFirst += activation[0][k] * weight[0][k][j];
            }
            thetas[1][j] = dotFirst;
            activation[1][j] = thresholdFunction(dotFirst);
         }

         for (int i = 0; i < verticalLayers[2]; i++)
         {
            double dotOutput = 0.0;
            for (int j = 0; j < verticalLayers[1]; j++)
            {
               dotOutput += activation[1][j] * weight[1][j][i];
            }
            thetas[2][i] = dotOutput;
            activation[2][i] = thresholdFunction(dotOutput);
            omegas[i] = expected[trainingTest][i] - activation[totalLayers - 1][i];
            psis[i] = omegas[i] * derivativeThreshold(thetas[2][i]);
         }

      } // Fills the Activations. public void fillActivations()

      /**
       * Calculates the error of the given training test. 1/2 * (T-F)^2.
       * @param trainingTest the test case being tested.
       * @return the calculated error of the training Test.
       */
      public double calculateError(int trainingTest)
      {
         double err = 0.0;
         for (int i = 0; i < verticalLayers[2]; i++)
         {
            double omegaI = 0.0;
            omegaI = (expected[trainingTest][i] - activation[numHiddenLayers + 1][i]);
            err += omegaI * omegaI;
         }

         return 0.5 * err;
      }

      /**
       * Trains the A-B-C neural network by running train Weights until lambda reaches min lambda, the error threshold is
       * reached, or the max iterations is reached. Adaptive Learning has been taken out. Trains by using each test case
       * and calculating delta weights for each.
       */
      public void trainNeural()
      {
         int iterations = 0;
         lambda = startingLambda;                     //Used in my DEBUG Print.
         double err = 0.0;
         setWeightsRandom();
         boolean finish = false;

         while (!finish)
         {
            err = 0.0;
            iterations++;

            for (int test = 0; test < numSets; test++) //runs through each training test.
            {
               for (int j = 0; j < numInputNodes; j++) //Sets inputs into activation.
               {
                  activation[0][j] = inputSets[test][j];
               }
               fillActivations(test);                      //Fills with original activations.

               backProp();

               for (int j = 0; j < numInputNodes; j++) //Sets inputs into activation.
               {
                  activation[0][j] = inputSets[test][j];
               }
               fillActivations(test);                       // Fills the activations with changed weights.

               err += calculateError(test);             //Calculates total Error.
            } // Loop that iterates through each test case.  for (int test = 0; test < numSets; test++)
            System.out.println("Total Error: " + err);

            if (iterations >= maxIterations)
            {
               System.out.println("DEBUG: Max Iterations Reached");
               finish = true;
            }

            if (err < errorThreshold)
            {
               System.out.println("DEBUG: Error Threshold Reached" + err);
               finish = true;
            }
         /*
         if (lambda <= minLambda)         //For adaptive lambda, NOT YET IMPLEMENTED
         {
            System.out.println("DEBUG: Lambda Reached minimum");
            finish = true;
         }
          */
         } //Loop that determines when the training should stop.  while (!finish)
         this.iterations = iterations;
      } // public void trainNeural()



      /**
       * Prints the output of the network. The output is given by the activation[number of layers - 1] [i],
       * where i represents each output node. To get the output(s), the method iterates through each layer n, not
       * counting the last layer. Within each iteration, it iterates through the input nodes and also iterates
       * through each hidden node. To find activation[n+1][k] it takes the dot product: ex. a[0][0] * w[0][0][0] +
       * a[0][1] * w[0][0][1]. The output is given printed by taking the values at activations[numHiddenLayers + 1][i].
       */
      public void getOutput()
      {
         for (int r = 0; r < totalLayers; r++) // initializes activation[a][b].
         {
            for (int c = 0; c < maxActivations; c++)
            {
               activation[r][c] = 0.0;
            }
         }
         for (int j = 0; j < numInputNodes; j++) //Sets inputs into activation.
         {
            activation[0][j] = inputs[j];
         }
      /*
      for (int n = 0; n < numHiddenLayers + 1; n++) // Iterates through each layer - 1. GENERALIZED VERSION.
      {
         for (int k = 0; k < verticalLayers[n]; k++) // Iterates through each input node index.
         {
            for (int j = 0; j < verticalLayers[n + 1]; j++) // Iterates through each hidden node index.
            {
               activation[n + 1][j] += (activation[n][k] * weight[n][k][j]);
            }
         }

         for (int h = 0; h < verticalLayers[n]; h++) //Threshold function of each activation.
         {
            activation[n + 1][h] = thresholdFunction(activation[n + 1][h]); //Start at a[1] cuz we don't want f(inputs).
         }
      }
       */
         for (int m = 0; m < numInputNodes; m++) // Iterates through each input node index. A-B-C VERSION.
         {
            for (int k = 0; k < numHiddenLayers; k++)
            {
               for (int j = 0; j < hiddenLayer[k]; j++) // Iterates through each hidden node index.
               {
                  activation[1][j] += (activation[0][m] * weight[0][m][j]);
                  System.out.println("DEBUG: a" + (1) + j + " = " + "a" + 0 + m + ":" + activation[0][m] + " * " + "w" +
                     0 + m + j + ":" + weight[0][m][j] + " a" + (1) + j + " ==> " + activation[1][m]);
               }
            }
         }

         for (int h = 0; h < numInputNodes; h++)
         {
            activation[1][h] = thresholdFunction(activation[1][h]);
         }

         for (int m = 0; m < hiddenLayer[numHiddenLayers - 1]; m++) // Iterates through each input node index.
         {
            for (int k = 0; k < numHiddenLayers; k++)
            {
               for (int j = 0; j < hiddenLayer[k]; j++) // Iterates through each hidden node index.
               {
                  activation[2][j] += (activation[1][m] * weight[1][m][j]);
                  System.out.println("DEBUG: a" + (2) + j + " = " + "a" + 1 + m + ":" + activation[1][m] + " * " + "w" +
                     1 + m + j + ":" + weight[1][m][j] + " a" + (2) + j + " ==> " + activation[2][m]);
               }
            }
         }

         for (int h = 0; h < numInputNodes; h++)
         {
            activation[2][h] = thresholdFunction(activation[2][h]);
         }

         System.out.println("\nActual Output(s): ");
         for (int i = 0; i < 1; i++) // 1 output only. Change "1" to numOutputs for general.
         {
            System.out.println("a" + (numHiddenLayers + 1) + i + ": " + activation[numHiddenLayers + 1][i]);
         }

      } // public void getOutput()

      /**
       * Prints all activations of the neural network. Used for my testing and visualization. Not integral for network.
       */
      public void printActivations()
      {
         System.out.println("\n\nDEBUG: Activations: ");
         for (int n = 0; n < totalLayers; n++)
         {
            for (int k = 0; k < verticalLayers[n]; k++)
            {
               System.out.print("a" + n + k + ": " + activation[n][k] + ", ");
            }
            System.out.println("");
         }
      }

      /**
       * Prints the weights of the neural network. Used for my testing and visualization.
       */
      public void printWeights()
      {
         System.out.println("\nWeights: ");
         for (int n = 0; n < numHiddenLayers + 1; n++)
         {
            for (int k = 0; k < verticalLayers[n]; k++)
            {
               for (int j = 0; j < verticalLayers[n+1]; j++)
               {
                  System.out.print("w" + n + k + j + ": " + weight[n][k][j] + ", ");
               }
               System.out.println("");
            }
         }
      } // public void printWeights()

      /**
       * Prints all values for the current test run of the network. This method is to help my testing and visualization
       * of the network and how it runs.
       */
      public void printAllValues()
      {
      /*
      System.out.println("\n------------------------ DEBUG ------------------------");
      System.out.println("DEBUG: Number of inputs: " + numInputNodes);
      System.out.print("DEBUG: Your Inputs.txt: ");
      for (int i = 0; i < numInputNodes; i++)
      {
         System.out.print(inputs[i] + ", ");
      }
      System.out.print("\nDEBUG: Hidden layer array: ");
      for (int j = 0; j < numHiddenLayers; j++)
      {
         System.out.print(hiddenLayer[j] + ", ");
      }
      System.out.print("so Number of Hidden Layers: " + numHiddenLayers);
      System.out.println("\nDEBUG: Number of outputs: " + numOutputNodes);
      System.out.print("DEBUG: Vertical layer array: ");
      for (int i = 0; i < totalLayers; i++)
      {
         System.out.print(verticalLayers[i] + ", ");
      }
      printActivations();
      System.out.println("\n------------------------ Your Values ------------------------");
      printWeights();
      System.out.println("\nExpected Output(s): ");
      System.out.println(expectedOutput);

       */
         System.out.println("\nActual Output(s): ");

         for (int i = 0; i < 1; i++) // For 1 output. Use "numOutputNodes" instead of "1" for generalized.
         {
            System.out.println("a" + (numHiddenLayers + 1) + i + ": " + activation[numHiddenLayers + 1][i]);
         }

         System.out.println("\nIterations: " + iterations);
         System.out.println("\nLambda: " + lambda);

      } // public void printAllValues()

      /**
       * Method to test the gradient descent of an A-B-C network. Asks the user for specifics and runs the training
       * network method.
       */
      public void testGradientDescent()
      {
         Scanner user = new Scanner(System.in);

         System.out.println("\nDo you want to set your own Gradient Descent test values? Yes or no.");
         String ans = user.next().toLowerCase();

         if (ans.equals("yes"))
         {
            System.out.println("\nWhat is the starting lambda?");
            lambda = user.nextDouble();
            startingLambda = lambda;

            System.out.println("\nWhat is minimum lambda?");
            minLambda = user.nextDouble();

            System.out.println("\nWhat is the learning factor multiplier?");
            multiplier = user.nextDouble();

            System.out.println("\nHow many iterations?");
            maxIterations = user.nextInt();

            System.out.println("\nWhat is the error threshold?");
            errorThreshold = user.nextDouble();

            System.out.println("\nWhat is maximum weight value?");
            maxWeight = user.nextDouble();

            System.out.println("\nWhat is minimum weight value?");
            minWeight = user.nextDouble();
         } //For when the user wants to input their own gradient parameters. if (ans.equals("yes"))
         else
         {
            setGradientDefault();
            System.out.println("Starting Lambda: " + lambda);
            System.out.println("Min Lambda: " + minLambda);
            System.out.println("Error threshold: " + errorThreshold);
            System.out.println("Max Weight: " + maxWeight);
            System.out.println("Min Weight: " + minWeight);
         }

         System.out.println("\nOutput: ");

         trainNeural(); //Trains the network

         System.out.println("\nTesting a " + numInputNodes + "-" + verticalLayers[1] + "-" + numOutputNodes + " network" +
            ".");
         System.out.println("Starting lambda: " + startingLambda);
         System.out.println("\nMin Lambda: " + minLambda);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Max Weight: " + maxWeight);
         System.out.println("Min Weight: " + minWeight);
         System.out.println("Max iterations: " + maxIterations);

         printAllValues();
         printWeights();

         double error = 0.0;
         for (int test = 0; test < numSets; test++)
         {
            System.out.println("\nTest " + test + ": ");

            for (int j = 0; j < numInputNodes; j++)
            {
               activation[0][j] = inputSets[test][j];
            }
            fillActivations(test);

            for (int i = 0; i < numOutputNodes; i++)
            {
               System.out.println("Test outputs for output case " + i + ": " + expected[test][i]);
               System.out.println("Actual Outputs for output case " + i + ": " + activation[totalLayers - 1][i]);
            }
            System.out.println("\nError is " + calculateError(test));
            error += calculateError(test);
         } //Prints out the training data.
         System.out.println("Total Error: " + error);


      } //public void testGradientDescent()

      /**
       * Method to test the neural network. Includes the user interface and restart function. Restarting the test does
       * not erase anything but you can reset the weights (same inputs).
       */
      public void testNetwork()
      {
         gradient = false;
         setGradientDefault();
         Scanner user = new Scanner(System.in);
         System.out.println("\nWould you like to set weights randomly? Type \"Yes\" or \"No.\"");
         String u = user.next();
         if (u.toLowerCase().equals("yes"))
         {
            setWeightsRandom();
         }
         else if (u.toLowerCase().equals("no"))
         {
            System.out.println("\nWould you like to set weights? Type \"Yes\" or \"No.\"");
            if (user.next().toLowerCase().equals("yes"))
            {
               setUserWeights();
            }
         }

         System.out.println("\nWhat is the expected output?");
         expectedOutput = user.nextDouble();

         System.out.println("\nOutput: ");
         getOutput();
         printAllValues();

         System.out.println("\nTo retry the test with different weights and expected output, type \"Retry.\"");
         System.out.println("To finish this test iteration, type anything else.");

         if (user.next().toLowerCase().equals("retry"))
         {
            testNetwork();
         }
      } //public void testNetwork()

      /**
       * Main class that tests the outputs and error given sets of inputs. Creates the network and calls the tester.
       * Can restart the entire tester without recompiling by typing "Restart."
       * @param args the main method parameter.
       */
      public static void main(String[] args)
      {
         Scanner user = new Scanner(System.in);
         System.out.println("How many input nodes?");
         int inputLayers = user.nextInt();

         System.out.println("How many hidden activation layers?");
         int[] hiddenLayer = new int[user.nextInt()];

         System.out.println("How many hidden activation nodes in each layer? You're entering the " +
            "number of nodes for " + hiddenLayer.length + " hidden layers.");

         for (int i = 0; i < hiddenLayer.length; i ++)
         {
            hiddenLayer[i] = user.nextInt(); // Since just 1 hidden layer I can just set hiddenLayer to {2}
         }

         System.out.println("How many output nodes?");
         int outputLayers = user.nextInt();

         BackPropABC neuro = new BackPropABC(inputLayers, hiddenLayer, outputLayers); //Sets inputs inside constructor.

         neuro.testGradientDescent();
         System.out.println("Thank you for testing");
      } // public static void main
   } //public class NeuralNet

