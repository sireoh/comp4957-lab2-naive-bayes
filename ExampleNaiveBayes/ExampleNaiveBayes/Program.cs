using System;
using System.IO;
using System.Linq;

namespace ExampleNaiveBayes
{
    /**
    * Naive Bayes Classifier for predicting food storage freshness.
    * 
    * This program implements a Naive Bayes classifier to predict whether stored food items
    * are fresh or spoiled based on various factors such as storage location, storage duration,
    * item type, and packaging quality.
    * 
    * The classifier is trained on a dataset and can classify new instances based on the learned model.
    * It includes Laplacian smoothing to handle zero-frequency problems and provides detailed output
    * of the classification process.
    */
    public class NaiveBayesClassifier
    {
        private readonly int numberVar;
        private readonly int numberClassLabels;
        private readonly string[] attributes;
        private readonly string[][] attributeValues;
        private string[][] trainingData;
        private int N; // Number of data points

        /**
        * Constructor to initialize the Naive Bayes Classifier.
        * 
        * @param numberVar Number of predictor variables.
        * @param numberClassLabels Number of class labels.
        * @param attributes Array of attribute names.
        * @param attributeValues 2D array of possible values for each attribute.
        */
        public NaiveBayesClassifier(int numberVar, int numberClassLabels, string[] attributes, string[][] attributeValues)
        {
            this.numberVar = numberVar;
            this.numberClassLabels = numberClassLabels;
            this.attributes = attributes;
            this.attributeValues = attributeValues;
        }

        /**
        * Loads training data from a file.
        * @param fileName Path to the training data file.
        */
        public void LoadTrainingData(string fileName)
        {
            // Count lines first to determine N
            N = File.ReadAllLines(fileName).Length;
            trainingData = LoadData(fileName, N, numberVar + 1, ',');
        }

        public ClassificationResult Classify(string[] input)
        {
            if (trainingData == null)
                throw new InvalidOperationException("Training data must be loaded first.");

            if (input.Length != numberVar)
                throw new ArgumentException($"Input must have {numberVar} variables.");

            int[][] jointCts = MatrixInt(numberVar, numberClassLabels);
            int[] yCts = new int[numberClassLabels];

            // Calculate class counts and joint counts
            for (int i = 0; i < N; ++i)
            {
                int y = int.Parse(trainingData[i][numberVar]);
                ++yCts[y];
                for (int j = 0; j < numberVar; ++j)
                {
                    if (trainingData[i][j] == input[j])
                        ++jointCts[j][y];
                }
            }

            // Apply Laplacian smoothing
            for (int i = 0; i < numberVar; ++i)
                for (int j = 0; j < numberClassLabels; ++j)
                    ++jointCts[i][j];

            // Compute evidence terms
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

            // Calculate probabilities
            double evidence = evidenceTerms.Sum();
            double[] probs = new double[numberClassLabels];
            for (int k = 0; k < numberClassLabels; ++k)
                probs[k] = evidenceTerms[k] / evidence;

            int predictedClass = ArgMax(probs);

            return new ClassificationResult
            {
                Input = input,
                JointCounts = jointCts,
                ClassCounts = yCts,
                EvidenceTerms = evidenceTerms,
                Probabilities = probs,
                PredictedClass = predictedClass,
                Confidence = probs[predictedClass]
            };
        }

        public void PrintModelInfo()
        {
            Console.WriteLine("=== Food Storage Freshness Prediction Model ===");
            Console.WriteLine($"Number of predictor variables: {numberVar}");
            Console.WriteLine($"Number of class labels: {numberClassLabels}");
            Console.WriteLine($"Training data points: {N}");
            Console.WriteLine();

            Console.WriteLine("Predictor variables and their possible values:");
            for (int i = 0; i < numberVar; ++i)
            {
                Console.Write($"{attributes[i]}: ");
                for (int j = 0; j < attributeValues[i].Length && !string.IsNullOrEmpty(attributeValues[i][j]); ++j)
                {
                    Console.Write(attributeValues[i][j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.WriteLine("Class labels:");
            Console.WriteLine("0 = Spoiled (unsafe to consume, discard item)");
            Console.WriteLine("1 = Fresh (safe to consume, good quality)");
            Console.WriteLine();
        }

        private static string[][] LoadData(string fn, int rows, int cols, char delimit)
        {
            string[][] result = MatrixString(rows, cols);
            string[] lines = File.ReadAllLines(fn);

            for (int i = 0; i < Math.Min(rows, lines.Length); i++)
            {
                string[] temp = lines[i].Split(delimit);
                for (int j = 0; j < Math.Min(cols, temp.Length); ++j)
                    result[i][j] = temp[j].Trim();
            }
            return result;
        }

        private static string[][] MatrixString(int rows, int cols)
        {
            string[][] result = new string[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new string[cols];
            return result;
        }

        private static int[][] MatrixInt(int rows, int cols)
        {
            int[][] result = new int[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new int[cols];
            return result;
        }

        private static int ArgMax(double[] vector)
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

    public class ClassificationResult
    {
        public string[] Input { get; set; }
        public int[][] JointCounts { get; set; }
        public int[] ClassCounts { get; set; }
        public double[] EvidenceTerms { get; set; }
        public double[] Probabilities { get; set; }
        public int PredictedClass { get; set; }
        public double Confidence { get; set; }

        public void PrintDetailedResults()
        {
            Console.WriteLine("=== Classification Results ===");
            Console.Write("Input to classify: ");
            foreach (var item in Input)
                Console.Write(item + " ");
            Console.WriteLine();
            Console.WriteLine();

            Console.WriteLine("Joint counts (after Laplacian smoothing):");
            for (int i = 0; i < JointCounts.Length; ++i)
            {
                Console.Write($"Variable {i}: ");
                for (int j = 0; j < JointCounts[i].Length; ++j)
                {
                    Console.Write(JointCounts[i][j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.Write("Class counts: ");
            foreach (var count in ClassCounts)
                Console.Write(count + " ");
            Console.WriteLine();
            Console.WriteLine();

            Console.Write("Evidence terms: ");
            foreach (var term in EvidenceTerms)
                Console.Write(term.ToString("F6") + " ");
            Console.WriteLine();
            Console.WriteLine();

            Console.Write("Probabilities: ");
            foreach (var prob in Probabilities)
                Console.Write(prob.ToString("F4") + " ");
            Console.WriteLine();
            Console.WriteLine();

            Console.WriteLine($"Predicted class: {PredictedClass}");
            Console.WriteLine($"Prediction: {(PredictedClass == 1 ? "Fresh (Safe to consume)" : "Spoiled (Do not consume)")}");
            Console.WriteLine($"Confidence: {Confidence:F4} ({Confidence * 100:F1}%)");
            Console.WriteLine();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("===============================================");
                Console.WriteLine("Food Storage Freshness Predictor");
                Console.WriteLine("Using Naive Bayes Classification");
                Console.WriteLine("===============================================");
                Console.WriteLine();

                // Model configuration
                int numberVar = 4;          // Number of predictor variables
                int numberClassLabels = 2;  // Number of class labels 

                string[] attributes = new string[]
                {
                    "Storage Location",
                    "Storage Duration",
                    "Item Type",
                    "Packaging Quality",
                    "Freshness Status"
                };

                string[][] attributeValues = new string[attributes.Length][];
                attributeValues[0] = new string[] { "fridge", "freezer", "pantry", "left_outside" };
                attributeValues[1] = new string[] { "short", "medium", "long", "very_long" };
                attributeValues[2] = new string[] { "meat", "vegetable", "dairy", "grain", "canned" };
                attributeValues[3] = new string[] { "sealed", "loose", "vacuum_packed", "damaged" };
                attributeValues[4] = new string[] { "0", "1", "", "" };

                // Create classifier
                NaiveBayesClassifier classifier = new NaiveBayesClassifier(numberVar, numberClassLabels, attributes, attributeValues);

                // Load training data
                string fileName = ".\\Data\\food_freshness_data.txt";
                classifier.LoadTrainingData(fileName);

                // Print model information
                classifier.PrintModelInfo();

                Console.WriteLine("=== TESTING DIFFERENT SCENARIOS ===");
                Console.WriteLine();

                // Test Case 1: High freshness item (freezer, short storage, quality packaging)
                Console.WriteLine("--- Test Case 1: High freshness item ---");
                string[] test1 = new string[] { "freezer", "short", "meat", "vacuum_packed" };
                var result1 = classifier.Classify(test1);
                result1.PrintDetailedResults();

                // Test Case 2: Borderline freshness item 
                Console.WriteLine("--- Test Case 2: Borderline freshness item ---");
                string[] test2 = new string[] { "fridge", "medium", "dairy", "loose" };
                var result2 = classifier.Classify(test2);
                result2.PrintDetailedResults();

                // Test Case 3: Clearly spoiled item
                Console.WriteLine("--- Test Case 3: Clearly spoiled item ---");
                string[] test3 = new string[] { "left_outside", "very_long", "vegetable", "damaged" };
                var result3 = classifier.Classify(test3);
                result3.PrintDetailedResults();

                // Interactive mode
                Console.WriteLine("=== INTERACTIVE MODE ===");
                Console.WriteLine("Enter food item details to predict freshness:");
                Console.WriteLine("Press Enter with empty input to exit.");
                Console.WriteLine();

                while (true)
                {
                    try
                    {
                        Console.Write("Storage Location (fridge/freezer/pantry/left_outside): ");
                        string location = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(location)) break;

                        Console.Write("Storage Duration (short/medium/long/very_long): ");
                        string duration = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(duration)) break;

                        Console.Write("Item Type (meat/vegetable/dairy/grain/canned): ");
                        string itemType = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(itemType)) break;

                        Console.Write("Packaging Quality (sealed/loose/vacuum_packed/damaged): ");
                        string packaging = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(packaging)) break;

                        string[] userInput = new string[] { location, duration, itemType, packaging };
                        var userResult = classifier.Classify(userInput);

                        Console.WriteLine();
                        Console.WriteLine("--- Your Prediction ---");
                        userResult.PrintDetailedResults();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                        Console.WriteLine("Please try again with valid inputs.");
                        Console.WriteLine();
                    }
                }

                Console.WriteLine("Thank you for using the Food Freshness Predictor!");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }
    }
}