using System;
using System.IO;

namespace ExampleNaiveBayes
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("\nNaive Bayes Classification demo\n");
                Console.WriteLine("Demo will classify graduation success based on grade in two courses (programming and discrete mathematics) and gym attendance\n");


                string fileName = ".\\Data\\data.txt";
                int numberVar = 3;          // Number of predictor variables
                int numberClassLabels = 2;  // Number of class labels 
                int N = 40;                 // Number of data points

                Console.WriteLine("\nBegin simple naive Bayes pediction of graduation success based on: \n");

                string[] attributes = new string[] { "grade in COMP 1510, ", "grade in COMP 2121 and", "gym attendance", "graduation sucess" };
                for (int i = 0; i < numberVar; ++i)
                    Console.Write(attributes[i] + " ");
                Console.WriteLine("\n");

                string[][] attributeValues = new string[attributes.Length][];
                attributeValues[0] = new string[] { "fail", "low", "medium", "high" };
                attributeValues[1] = new string[] { "fail", "low", "medium", "high" };
                attributeValues[2] = new string[] { "no", "monthly", "weekly", "daily" };
                attributeValues[3] = new string[] { "0", "1", " ", " " };
                for (int i = 0; i < numberVar + 1; ++i)
                {
                    Console.Write("attribute values" + " " + attributes[i] + ": ");
                    for (int j = 0; j < numberVar + 1; ++j)
                    {
                        Console.Write(attributeValues[i][j] + " ");
                    }
                    Console.WriteLine("\n");
                }

                Console.WriteLine("\n");


                string[][] data = LoadData(fileName, N, numberVar + 1, ',');
                Console.WriteLine("Training data:");
                for (int i = 0; i < N; ++i)
                {
                    Console.Write("[" + i + "] ");
                    for (int j = 0; j < numberVar + 1; ++j)
                    {
                        Console.Write(data[i][j] + " ");
                    }
                    Console.WriteLine("");
                }

                Console.WriteLine("------------------------------------ \n");

                int[][] jointCts = MatrixInt(numberVar, numberClassLabels);
                int[] yCts = new int[numberClassLabels];
                string[] X = new string[] { "high", "low", "no" };
                Console.WriteLine("Item to classify: ");
                for (int i = 0; i < numberVar; ++i)
                    Console.Write(X[i] + " ");
                Console.WriteLine("\n");

                Console.WriteLine("------------------------------------ \n");

                // Calculation of the class counts for both situations: graduation success (“1”)
                // and non-graduation success (“0”)
                for (int i = 0; i < N; ++i)
                {
                    int y = int.Parse(data[i][numberVar]);
                    ++yCts[y];
                    for (int j = 0; j < numberVar; ++j)
                    {
                        if (data[i][j] == X[j])
                            ++jointCts[j][y];
                    }
                }

                Console.WriteLine("Joint counts BEFORE Laplacian smoothing: ");
                for (int i = 0; i < numberVar; ++i)
                {
                    for (int j = 0; j < numberClassLabels; ++j)
                    {
                        Console.Write(jointCts[i][j] + " ");
                    }
                    Console.WriteLine("");
                }
                // Laplacian smoothing (adding one to all counts to prevent multiplication by zero)
                for (int i = 0; i < numberVar; ++i)
                    for (int j = 0; j < numberClassLabels; ++j)
                        ++jointCts[i][j];

                Console.WriteLine("Joint counts AFTER Laplacian smoothing: ");
                for (int i = 0; i < numberVar; ++i)
                {
                    for (int j = 0; j < numberClassLabels; ++j)
                    {
                        Console.Write(jointCts[i][j] + " ");
                    }
                    Console.WriteLine("");
                }
                Console.WriteLine("\nClass counts: ");
                for (int k = 0; k < numberClassLabels; ++k)
                    Console.Write(yCts[k] + " ");
                Console.WriteLine("\n");

                // Compute evidence terms; note adding the number of predictive variables (here is 3) to denominator 
                // The addition is needed because of the Laplacian smoothing 
                double[] evidenceTerms = new double[numberClassLabels];
                for (int k = 0; k < numberClassLabels; ++k)
                {
                    double v = 1.0;
                    for (int j = 0; j < numberVar; ++j)
                    {
                        v *= (double)(jointCts[j][k]) / (yCts[k] + numberVar);
                    }
                    v *= (double)(yCts[k]) / N;
                    evidenceTerms[k] = v;
                }
                Console.WriteLine("Evidence terms:");
                for (int k = 0; k < numberClassLabels; ++k)
                    Console.Write(evidenceTerms[k].ToString("F4") + " ");
                Console.WriteLine("\n");
                double evidence = 0.0;
                for (int k = 0; k < numberClassLabels; ++k)
                    evidence += evidenceTerms[k];
                double[] probs = new double[numberClassLabels];
                for (int k = 0; k < numberClassLabels; ++k)
                    probs[k] = evidenceTerms[k] / evidence;
                Console.WriteLine("Probabilities: ");
                for (int k = 0; k < numberClassLabels; ++k)
                    Console.Write(probs[k].ToString("F4") + " ");
                Console.WriteLine("\n");
                int pc = ArgMax(probs);

                Console.WriteLine("Predicted class: ");
                Console.WriteLine(pc);
                if (pc == 0)
                    Console.WriteLine("low graduation success");
                else
                    Console.WriteLine("high graduation success");
                Console.WriteLine("\nEnd Naive Bayes ");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        }

        static string[][] MatrixString(int rows, int cols)
        {
            string[][] result = new string[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new string[cols];
            return result;
        }
        static int[][] MatrixInt(int rows, int cols)
        {
            int[][] result = new int[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new int[cols];
            return result;
        }
        static string[][] LoadData(string fn, int rows, int cols, char delimit)
        {
            string[][] result = MatrixString(rows, cols);
            FileStream ifs = new FileStream(fn, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            string[] temp = null;
            string line = null;
            int i = 0;
            while ((line = sr.ReadLine()) != null)
            {
                temp = line.Split(delimit);
                for (int j = 0; j < cols; ++j)
                    result[i][j] = temp[j];
                ++i;
            }
            sr.Close(); ifs.Close();
            return result;
        }
        static int ArgMax(double[] vector)
        {
            int result = 0;
            double maxVector = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxVector)
                {
                    maxVector = vector[i];
                    result = i;
                }
            }
            return result;
        }
    }
}
