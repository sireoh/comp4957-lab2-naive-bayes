using ExampleNaiveBayes;
using System.IO;

namespace NaiveBayesTests
{
    [TestClass]
    public class FoodFreshnessClassifierTests
    {
        private NaiveBayesClassifier? classifier;
        private string[] attributes = null!;
        private string[][] attributeValues = null!;
        private string testDataContent = null!;
        private string tempDataFile = null!;

        [TestInitialize]
        public void Setup()
        {
            // Initialize test data for food freshness prediction
            attributes = new string[]
            {
                "Storage Location",
                "Storage Duration",
                "Item Type",
                "Packaging Quality",
                "Freshness Status"
            };

            attributeValues = new string[attributes.Length][];
            attributeValues[0] = new string[] { "fridge", "freezer", "pantry", "left_outside" };
            attributeValues[1] = new string[] { "short", "medium", "long", "very_long" };
            attributeValues[2] = new string[] { "meat", "vegetable", "dairy", "grain", "canned" };
            attributeValues[3] = new string[] { "sealed", "loose", "vacuum_packed", "damaged" };
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
            var simpleData = "freezer,short,meat,vacuum_packed,1\nleft_outside,very_long,vegetable,damaged,0\nfridge,medium,dairy,loose,0";
            var tempFile = Path.GetTempFileName();
            File.WriteAllText(tempFile, simpleData);

            try
            {
                // Act
                classifier!.LoadTrainingData(tempFile);

                // Test classification to ensure data was loaded
                string[] testInput = new string[] { "freezer", "short", "meat", "vacuum_packed" };
                var result = classifier.Classify(testInput);

                // Assert
                Assert.IsNotNull(result);
                Assert.AreEqual(4, result.Input.Length);
                Assert.AreEqual("freezer", result.Input[0]);
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
            string[] testInput = new string[] { "freezer", "short", "meat", "vacuum_packed" };

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
            string[] invalidInput = new string[] { "freezer", "short", "meat" };
            classifier.Classify(invalidInput);
        }

        [TestMethod]
        public void Classify_HighFreshnessProfile_PredictsFresh()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] highFreshnessInput = new string[] { "freezer", "short", "meat", "vacuum_packed" };

            // Act
            var result = classifier.Classify(highFreshnessInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(1, result.PredictedClass, "Well-preserved item should predict fresh");
            Assert.IsTrue(result.Confidence > 0.5, "Should have high confidence for clear case");
            Assert.AreEqual("freezer", result.Input[0]);
            Assert.AreEqual("short", result.Input[1]);
            Assert.AreEqual("meat", result.Input[2]);
            Assert.AreEqual("vacuum_packed", result.Input[3]);
        }

        [TestMethod]
        public void Classify_LowFreshnessProfile_PredictsSpoiled()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] lowFreshnessInput = new string[] { "left_outside", "very_long", "vegetable", "damaged" };

            // Act
            var result = classifier.Classify(lowFreshnessInput);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.PredictedClass, "Poorly stored item should predict spoiled");
            Assert.IsTrue(result.Confidence > 0.5, "Should have high confidence for clear case");
            Assert.AreEqual("left_outside", result.Input[0]);
            Assert.AreEqual("very_long", result.Input[1]);
            Assert.AreEqual("vegetable", result.Input[2]);
            Assert.AreEqual("damaged", result.Input[3]);
        }

        [TestMethod]
        public void Classify_ReturnsValidProbabilities()
        {
            // Arrange
            classifier!.LoadTrainingData(tempDataFile);
            string[] testInput = new string[] { "fridge", "medium", "dairy", "loose" };

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
            string[] testInput = new string[] { "pantry", "medium", "grain", "sealed" };

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
            string[] testInput = new string[] { "fridge", "long", "vegetable", "sealed" };

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
            string[] testInput = new string[] { "pantry", "short", "canned", "sealed" };

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
                Assert.IsTrue(output.Contains("Food Storage Freshness Prediction Model"));
                Assert.IsTrue(output.Contains("Number of predictor variables"));
                Assert.IsTrue(output.Contains("Storage Location"));
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
            string[] testInput = new string[] { "fridge", "medium", "meat", "sealed" };
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
            return @"freezer,short,meat,vacuum_packed,1
freezer,short,vegetable,sealed,1
fridge,short,dairy,sealed,1
pantry,short,grain,sealed,1
pantry,short,canned,sealed,1
left_outside,short,meat,damaged,0
left_outside,short,vegetable,damaged,0
left_outside,medium,dairy,loose,0
left_outside,long,vegetable,damaged,0
left_outside,very_long,meat,damaged,0
fridge,medium,dairy,loose,0
fridge,long,meat,sealed,0
fridge,very_long,vegetable,sealed,0
pantry,long,grain,loose,0
pantry,very_long,grain,sealed,0
freezer,medium,meat,vacuum_packed,1
freezer,long,vegetable,sealed,1
fridge,short,meat,vacuum_packed,1
fridge,medium,vegetable,sealed,1
pantry,medium,canned,sealed,1";
        }
    }
}