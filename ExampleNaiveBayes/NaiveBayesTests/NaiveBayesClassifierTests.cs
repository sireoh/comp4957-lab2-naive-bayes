using ExampleNaiveBayes;
using System.IO;

namespace NaiveBayesTests
{
    [TestClass]
    public class NaiveBayesClassifierTests
    {
        private NaiveBayesClassifier? classifier;
        private string[] attributes = null!;
        private string[][] attributeValues = null!;
        private string testDataContent = null!;
        private string tempDataFile = null!;

        [TestInitialize]
        public void Setup()
        {
            // Initialize test data
            attributes = new string[]
            {
                "Programming Skills",
                "Math Performance",
                "Team Collaboration",
                "Problem Solving",
                "Career Success"
            };

            attributeValues = new string[attributes.Length][];
            attributeValues[0] = new string[] { "beginner", "intermediate", "advanced", "expert" };
            attributeValues[1] = new string[] { "poor", "fair", "good", "excellent" };
            attributeValues[2] = new string[] { "poor", "average", "good", "excellent" };
            attributeValues[3] = new string[] { "weak", "moderate", "strong", "exceptional" };
            attributeValues[4] = new string[] { "0", "1", "", "" };

            classifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

            // Create test dataset
            testDataContent = CreateTestDataset();
            tempDataFile = Path.GetTempFileName();
            File.WriteAllText(tempDataFile, testDataContent);
        }

        [TestCleanup]
        public void Cleanup()
        {
            // Clean up temporary files
            if (File.Exists(tempDataFile))
            {
                File.Delete(tempDataFile);
            }
        }

        [TestMethod]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var testClassifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);

            // Assert
            Assert.IsNotNull(testClassifier);
        }

        [TestMethod]
        public void LoadTrainingData_WithValidFile_LoadsDataSuccessfully()
        {
            // Arrange
            var simpleData = "expert,excellent,excellent,exceptional,1\nbeginner,poor,poor,weak,0\nintermediate,good,average,moderate,1";
            var tempFile = Path.GetTempFileName();
            File.WriteAllText(tempFile, simpleData);

            try
            {
                // Act
                classifier!.LoadTrainingData(tempFile);

                // Test classification to ensure data was loaded
                string[] testInput = new string[] { "expert", "excellent", "excellent", "exceptional" };
                var result = classifier.Classify(testInput);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(4, result.Input.Length);
                Assert.AreEqual("expert", result.Input[0]);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void Classify_WithoutTrainingData_ThrowsException()
        {
            // Arrange
            var freshClassifier = new NaiveBayesClassifier(4, 2, attributes, attributeValues);
            string[] testInput = new string[] { "expert", "excellent", "excellent", "exceptional" };

            // Act & Assert (ExpectedException handles the assertion)
            freshClassifier.Classify(testInput);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Classify_WithInvalidInputLength_ThrowsException()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);

            // Act & Assert (ExpectedException handles the assertion)
            // Test with wrong number of input variables (should be 4, giving 3)
            string[] invalidInput = new string[] { "expert", "excellent", "exceptional" };
            classifier.Classify(invalidInput);
        }

        [TestMethod]
        public void Classify_HighPerformanceProfile_PredictsHighSuccess()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] highPerformanceInput = new string[] { "expert", "excellent", "excellent", "exceptional" };

            // Act
            var result = classifier.Classify(highPerformanceInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(1, result.PredictedClass, "Expert profile should predict high career success");
            Assert.IsTrue(result.Confidence > 0.5, "Should have high confidence for clear case");
            Assert.AreEqual("expert", result.Input[0]);
            Assert.AreEqual("excellent", result.Input[1]);
            Assert.AreEqual("excellent", result.Input[2]);
            Assert.AreEqual("exceptional", result.Input[3]);
        }

        [TestMethod]
        public void Classify_LowPerformanceProfile_PredictsLowSuccess()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] lowPerformanceInput = new string[] { "beginner", "poor", "poor", "weak" };

            // Act
            var result = classifier.Classify(lowPerformanceInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.PredictedClass, "Beginner profile should predict low career success");
            Assert.IsTrue(result.Confidence > 0.5, "Should have high confidence for clear case");
            Assert.AreEqual("beginner", result.Input[0]);
            Assert.AreEqual("poor", result.Input[1]);
            Assert.AreEqual("poor", result.Input[2]);
            Assert.AreEqual("weak", result.Input[3]);
        }

        [TestMethod]
        public void Classify_ReturnsValidProbabilities()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "intermediate", "good", "average", "moderate" };

            // Act
            var result = classifier.Classify(testInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Probabilities);
            Assert.AreEqual(2, result.Probabilities.Length);

            // Probabilities should sum to approximately 1.0
            double sum = result.Probabilities[0] + result.Probabilities[1];
            Assert.IsTrue(Math.Abs(sum - 1.0) < 0.001, $"Probabilities should sum to 1.0, but sum was {sum}");

            // Each probability should be between 0 and 1
            Assert.IsTrue(result.Probabilities[0] >= 0 && result.Probabilities[0] <= 1,
                $"Probability 0 should be between 0 and 1, but was {result.Probabilities[0]}");
            Assert.IsTrue(result.Probabilities[1] >= 0 && result.Probabilities[1] <= 1,
                $"Probability 1 should be between 0 and 1, but was {result.Probabilities[1]}");
        }

        [TestMethod]
        public void ClassificationResult_HasValidConfidence()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "advanced", "good", "good", "strong" };

            // Act
            var result = classifier.Classify(testInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.Confidence >= 0 && result.Confidence <= 1,
                $"Confidence should be between 0 and 1, but was {result.Confidence}");
            Assert.AreEqual(result.Probabilities[result.PredictedClass], result.Confidence,
                "Confidence should equal the probability of the predicted class");
        }

        [TestMethod]
        public void ClassificationResult_PredictedClassIsValid()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "advanced", "good", "good", "strong" };

            // Act
            var result = classifier.Classify(testInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.PredictedClass == 0 || result.PredictedClass == 1,
                $"Predicted class should be 0 or 1, but was {result.PredictedClass}");
        }

        [TestMethod]
        public void ClassificationResult_JointCountsHaveCorrectDimensions()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "intermediate", "fair", "average", "moderate" };

            // Act
            var result = classifier.Classify(testInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.JointCounts);
            Assert.AreEqual(4, result.JointCounts.Length, "Should have joint counts for 4 variables");

            foreach (var jointCount in result.JointCounts)
            {
                Assert.AreEqual(2, jointCount.Length, "Each variable should have counts for 2 classes");
                Assert.IsTrue(jointCount[0] > 0, "Joint count for class 0 should be positive (due to Laplacian smoothing)");
                Assert.IsTrue(jointCount[1] > 0, "Joint count for class 1 should be positive (due to Laplacian smoothing)");
            }
        }

        [TestMethod]
        public void PrintModelInfo_DoesNotThrowException()
        {
            // Arrange & Act
            try
            {
                // Redirect console output to prevent cluttering test output
                var originalOut = Console.Out;
                using var stringWriter = new StringWriter();
                Console.SetOut(stringWriter);

                classifier!.PrintModelInfo();

                var output = stringWriter.ToString();

                Console.SetOut(originalOut);

                // Assert
                Assert.IsTrue(output.Contains("Software Developer Career Success Prediction Model"));
                Assert.IsTrue(output.Contains("Number of predictor variables"));
                Assert.IsTrue(output.Contains("Programming Skills"));
            }
            catch (Exception ex)
            {
                Assert.Fail($"PrintModelInfo should not throw exception, but threw: {ex.Message}");
            }
        }

        [TestMethod]
        public void PrintDetailedResults_DoesNotThrowException()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "advanced", "good", "good", "strong" };
            var result = classifier.Classify(testInput);

            // Act
            try
            {
                // Redirect console output to prevent cluttering test output
                var originalOut = Console.Out;
                using var stringWriter = new StringWriter();
                Console.SetOut(stringWriter);

                result.PrintDetailedResults();

                var output = stringWriter.ToString();

                Console.SetOut(originalOut);

                // Assert
                Assert.IsTrue(output.Contains("Classification Results"));
                Assert.IsTrue(output.Contains("Input to classify"));
                Assert.IsTrue(output.Contains("Predicted class"));
            }
            catch (Exception ex)
            {
                Assert.Fail($"PrintDetailedResults should not throw exception, but threw: {ex.Message}");
            }
        }

        private string CreateTestDataset()
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