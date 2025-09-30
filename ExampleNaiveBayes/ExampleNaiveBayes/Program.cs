using System;
using System.IO;
using System.Linq;

namespace ExampleNaiveBayes
{
    /**
    * Naive Bayes Classifier for predicting software developer career success.
    * 
    * This program implements a Naive Bayes classifier to predict whether a software developer
    * will have high or low career success based on various attributes such as programming skills,
    * math performance, team collaboration, and problem-solving abilities.
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

        /**
        * Loads data from a file into a 2D array.
        * @param fn File name.
        * @param n Number of rows.
        * @param m Number of columns.
        * @param delimit Delimiter character.
        */
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
            Console.WriteLine("=== Software Developer Career Success Prediction Model ===");
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
            Console.WriteLine("0 = Low Career Success (difficulty finding jobs, lower performance)");
            Console.WriteLine("1 = High Career Success (good job prospects, strong performance)");
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
            Console.WriteLine($"Prediction: {(PredictedClass == 1 ? "High Career Success" : "Low Career Success")}");
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
                Console.WriteLine("Software Developer Career Success Predictor");
                Console.WriteLine("Using Naive Bayes Classification");
                Console.WriteLine("===============================================");
                Console.WriteLine();

                // Model configuration
                int numberVar = 4;          // Number of predictor variables
                int numberClassLabels = 2;  // Number of class labels 

                string[] attributes = new string[]
                {
                    "Programming Skills",
                    "Math Performance",
                    "Team Collaboration",
                    "Problem Solving",
                    "Career Success"
                };

                string[][] attributeValues = new string[attributes.Length][];
                attributeValues[0] = new string[] { "beginner", "intermediate", "advanced", "expert" };
                attributeValues[1] = new string[] { "poor", "fair", "good", "excellent" };
                attributeValues[2] = new string[] { "poor", "average", "good", "excellent" };
                attributeValues[3] = new string[] { "weak", "moderate", "strong", "exceptional" };
                attributeValues[4] = new string[] { "0", "1", "", "" };

                // Create classifier
                NaiveBayesClassifier classifier = new NaiveBayesClassifier(numberVar, numberClassLabels, attributes, attributeValues);

                // Load training data
                string fileName = ".\\Data\\career_success_data.txt";
                classifier.LoadTrainingData(fileName);

                // Print model information
                classifier.PrintModelInfo();

                // Test scenarios
                // Run unit tests first
                NaiveBayesClassifierTests.RunAllTests();

                Console.WriteLine("=== TESTING DIFFERENT SCENARIOS ===");
                Console.WriteLine();

                // Test Case 1: High-performing student
                Console.WriteLine("--- Test Case 1: High-performing student ---");
                string[] test1 = new string[] { "expert", "excellent", "excellent", "exceptional" };
                var result1 = classifier.Classify(test1);
                result1.PrintDetailedResults();

                // Test Case 2: Average student
                Console.WriteLine("--- Test Case 2: Average student ---");
                string[] test2 = new string[] { "intermediate", "good", "average", "moderate" };
                var result2 = classifier.Classify(test2);
                result2.PrintDetailedResults();

                // Test Case 3: Struggling student
                Console.WriteLine("--- Test Case 3: Struggling student ---");
                string[] test3 = new string[] { "beginner", "poor", "poor", "weak" };
                var result3 = classifier.Classify(test3);
                result3.PrintDetailedResults();

                // Interactive mode
                Console.WriteLine("=== INTERACTIVE MODE ===");
                Console.WriteLine("Enter student characteristics to predict career success:");
                Console.WriteLine("Press Enter with empty input to exit.");
                Console.WriteLine();

                while (true)
                {
                    try
                    {
                        Console.Write("Programming Skills (beginner/intermediate/advanced/expert): ");
                        string prog = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(prog)) break;

                        Console.Write("Math Performance (poor/fair/good/excellent): ");
                        string math = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(math)) break;

                        Console.Write("Team Collaboration (poor/average/good/excellent): ");
                        string team = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(team)) break;

                        Console.Write("Problem Solving (weak/moderate/strong/exceptional): ");
                        string problem = Console.ReadLine().Trim().ToLower();
                        if (string.IsNullOrEmpty(problem)) break;

                        string[] userInput = new string[] { prog, math, team, problem };
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

                Console.WriteLine("Thank you for using the Career Success Predictor!");
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