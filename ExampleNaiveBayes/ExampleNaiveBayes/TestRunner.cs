using System;
using System.IO;

namespace ExampleNaiveBayes
{
    public static class NaiveBayesClassifierTests
    {
        public static void RunAllTests()
        {
            Console.WriteLine("=== RUNNING UNIT TESTS ===");
            Console.WriteLine();

            int passedTests = 0;
            int totalTests = 0;

            // Test 1: Constructor Test
            totalTests++;
            if (Test_Constructor_ValidParameters_CreatesInstance())
            {
                Console.WriteLine("✓ Test 1: Constructor with valid parameters - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 1: Constructor with valid parameters - FAILED");
            }

            // Test 2: Load Training Data Test
            totalTests++;
            if (Test_LoadTrainingData_ValidFile_LoadsDataSuccessfully())
            {
                Console.WriteLine("✓ Test 2: Load training data - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 2: Load training data - FAILED");
            }

            // Test 3: No Training Data Exception Test
            totalTests++;
            if (Test_Classify_NoTrainingDataLoaded_ThrowsException())
            {
                Console.WriteLine("✓ Test 3: Exception when no training data - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 3: Exception when no training data - FAILED");
            }

            // Test 4: Invalid Input Length Test
            totalTests++;
            if (Test_Classify_InvalidInputLength_ThrowsException())
            {
                Console.WriteLine("✓ Test 4: Exception for invalid input length - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 4: Exception for invalid input length - FAILED");
            }

            // Test 5: High Performance Prediction Test
            totalTests++;
            if (Test_Classify_HighPerformanceProfile_PredictsHighSuccess())
            {
                Console.WriteLine("✓ Test 5: High performance profile prediction - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 5: High performance profile prediction - FAILED");
            }

            // Test 6: Low Performance Prediction Test
            totalTests++;
            if (Test_Classify_LowPerformanceProfile_PredictsLowSuccess())
            {
                Console.WriteLine("✓ Test 6: Low performance profile prediction - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 6: Low performance profile prediction - FAILED");
            }

            // Test 7: Valid Probabilities Test
            totalTests++;
            if (Test_Classify_ReturnsValidProbabilities())
            {
                Console.WriteLine("✓ Test 7: Valid probabilities returned - PASSED");
                passedTests++;
            }
            else
            {
                Console.WriteLine("✗ Test 7: Valid probabilities returned - FAILED");
            }

            Console.WriteLine();
            Console.WriteLine($"=== TEST RESULTS: {passedTests}/{totalTests} TESTS PASSED ===");

            if (passedTests == totalTests)
            {
                Console.WriteLine("🎉 ALL TESTS PASSED! 🎉");
            }
            else
            {
                Console.WriteLine($"⚠️  {totalTests - passedTests} TESTS FAILED");
            }
            Console.WriteLine();
        }

        private static bool Test_Constructor_ValidParameters_CreatesInstance()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var testClassifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);
                return testClassifier != null;
            }
            catch
            {
                return false;
            }
        }

        private static bool Test_LoadTrainingData_ValidFile_LoadsDataSuccessfully()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string testData = "expert,excellent,excellent,exceptional,1\nbeginner,poor,poor,weak,0\nintermediate,good,average,moderate,1";
                string tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, testData);

                classifier.LoadTrainingData(tempFile);

                // Test classification to ensure data was loaded
                string[] testInput = new string[] { "expert", "excellent", "excellent", "exceptional" };
                var result = classifier.Classify(testInput);

                File.Delete(tempFile);

                return result != null && result.Input.Length == 4 && result.Input[0] == "expert";
            }
            catch
            {
                return false;
            }
        }

        private static bool Test_Classify_NoTrainingDataLoaded_ThrowsException()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string[] testInput = new string[] { "expert", "excellent", "excellent", "exceptional" };
                classifier.Classify(testInput); // Should throw exception

                return false; // If we reach here, exception was not thrown
            }
            catch (InvalidOperationException)
            {
                return true; // Exception was correctly thrown
            }
            catch
            {
                return false; // Wrong exception type
            }
        }

        private static bool Test_Classify_InvalidInputLength_ThrowsException()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string testData = "expert,excellent,excellent,exceptional,1\nbeginner,poor,poor,weak,0";
                string tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, testData);
                classifier.LoadTrainingData(tempFile);

                // Test with wrong number of input variables (should be 4, giving 3)
                string[] testInput = new string[] { "expert", "excellent", "exceptional" };
                classifier.Classify(testInput); // Should throw exception

                File.Delete(tempFile);
                return false; // If we reach here, exception was not thrown
            }
            catch (ArgumentException)
            {
                return true; // Exception was correctly thrown
            }
            catch
            {
                return false; // Wrong exception type or other error
            }
        }

        private static bool Test_Classify_HighPerformanceProfile_PredictsHighSuccess()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string testData = CreateTestDataset();
                string tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, testData);
                classifier.LoadTrainingData(tempFile);

                // Test high-performance profile
                string[] testInput = new string[] { "expert", "excellent", "excellent", "exceptional" };
                var result = classifier.Classify(testInput);

                File.Delete(tempFile);

                return result != null &&
                       result.PredictedClass == 1 &&
                       result.Confidence > 0.5;
            }
            catch
            {
                return false;
            }
        }

        private static bool Test_Classify_LowPerformanceProfile_PredictsLowSuccess()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string testData = CreateTestDataset();
                string tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, testData);
                classifier.LoadTrainingData(tempFile);

                // Test low-performance profile
                string[] testInput = new string[] { "beginner", "poor", "poor", "weak" };
                var result = classifier.Classify(testInput);

                File.Delete(tempFile);

                return result != null &&
                       result.PredictedClass == 0 &&
                       result.Confidence > 0.5;
            }
            catch
            {
                return false;
            }
        }

        private static bool Test_Classify_ReturnsValidProbabilities()
        {
            try
            {
                var attributes = GetTestAttributes();
                var attributeValues = GetTestAttributeValues();
                var classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

                string testData = CreateTestDataset();
                string tempFile = Path.GetTempFileName();
                File.WriteAllText(tempFile, testData);
                classifier.LoadTrainingData(tempFile);

                string[] testInput = new string[] { "intermediate", "good", "average", "moderate" };
                var result = classifier.Classify(testInput);

                File.Delete(tempFile);

                if (result == null || result.Probabilities == null || result.Probabilities.Length != 2)
                    return false;

                // Probabilities should sum to approximately 1.0
                double sum = result.Probabilities[0] + result.Probabilities[1];
                bool sumIsValid = Math.Abs(sum - 1.0) < 0.001;

                // Each probability should be between 0 and 1
                bool prob0Valid = result.Probabilities[0] >= 0 && result.Probabilities[0] <= 1;
                bool prob1Valid = result.Probabilities[1] >= 0 && result.Probabilities[1] <= 1;

                return sumIsValid && prob0Valid && prob1Valid;
            }
            catch
            {
                return false;
            }
        }

        private static string[] GetTestAttributes()
        {
            return new string[]
            {
                "Programming Skills",
                "Math Performance",
                "Team Collaboration",
                "Problem Solving",
                "Career Success"
            };
        }

        private static string[][] GetTestAttributeValues()
        {
            string[][] attributeValues = new string[5][];
            attributeValues[0] = new string[] { "beginner", "intermediate", "advanced", "expert" };
            attributeValues[1] = new string[] { "poor", "fair", "good", "excellent" };
            attributeValues[2] = new string[] { "poor", "average", "good", "excellent" };
            attributeValues[3] = new string[] { "weak", "moderate", "strong", "exceptional" };
            attributeValues[4] = new string[] { "0", "1", "", "" };
            return attributeValues;
        }

        private static string CreateTestDataset()
        {
            return @"expert,excellent,excellent,exceptional,1
advanced,good,good,strong,1
intermediate,fair,average,moderate,0
beginner,poor,poor,weak,0
expert,good,excellent,strong,1
advanced,excellent,good,exceptional,1
intermediate,good,average,strong,1
beginner,fair,poor,weak,0
expert,excellent,good,exceptional,1
advanced,good,excellent,strong,1
intermediate,fair,good,moderate,1
beginner,poor,average,weak,0
expert,good,good,strong,1
advanced,excellent,excellent,exceptional,1
intermediate,good,average,moderate,0
beginner,fair,poor,weak,0
expert,excellent,excellent,strong,1
advanced,good,good,exceptional,1
intermediate,fair,average,moderate,0
beginner,poor,poor,weak,0";
        }
    }
}