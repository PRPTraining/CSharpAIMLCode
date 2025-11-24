using MachinelearningClass.ModelNLP;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public class Week3 // NLP
    {
        public static void Lab11_OneHotEncoding()
        {
            var ml = new MLContext();

            var data = new[]
            {
            new FruitData { Fruit = "Mango" },
            new FruitData { Fruit = "Apple" },
            new FruitData { Fruit = "Berry" }
        };

            var dataView = ml.Data.LoadFromEnumerable(data);

            var pipeline = ml.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "FruitEncoded",
                inputColumnName: "Fruit");

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);

            var encoded = ml.Data.CreateEnumerable<FruitFeatures>(
                transformedData, reuseRowObject: false);

            Console.WriteLine("One-Hot Encoded Vectors:");
            foreach (var row in encoded)
            {
                Console.WriteLine($"[{string.Join(",", row.FruitEncoded)}]");
            }
        }
        public static void Lab12and13_BowTFIDF()
        {
            var ml = new MLContext();

            // Sample data
            var samples = new[]
            {
            new InputText { Text = "This camera camera is good" },
            new InputText { Text = "This camera is bad" }
        };

            var data = ml.Data.LoadFromEnumerable(samples);


            var bowPipeline =
                ml.Transforms.Text.TokenizeIntoWords("Tokens", "Text")
                .Append(ml.Transforms.Conversion.MapValueToKey("KeyTokens", "Tokens"))
                .Append(ml.Transforms.Text.ProduceNgrams(
                    outputColumnName: "Features",
                    inputColumnName: "KeyTokens",
                    ngramLength: 1,        
                    useAllLengths: false,  // Do NOT create bigrams
                    weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.TfIdf
                ));

            var bowModel = bowPipeline.Fit(data);
            var bowTransformed = bowModel.Transform(data);
            var bowResults = ml.Data.CreateEnumerable<TextFeatures>(bowTransformed, reuseRowObject: false);
            VBuffer<ReadOnlyMemory<char>> slotNames = default;

            bowTransformed.Schema["Features"]
                .Annotations.GetValue("SlotNames", ref slotNames);

            var vocab = slotNames.DenseValues()
                .Select(v => v.ToString())
                .ToArray();
            Console.WriteLine("Vocabulary: " + string.Join(", ", vocab));
            foreach (var row in bowResults)
            {

                Console.WriteLine(string.Join(", ", row.Features));
            }

            int docIndex = 1;
            foreach (var row in bowResults)
            {
                Console.WriteLine($"--- Document {docIndex++} ---");
                for (int i = 0; i < vocab.Length; i++)
                {
                    if (row.Features[i] != 0)
                    {
                        Console.WriteLine($"{vocab[i]} : {row.Features[i]}");
                    }
                }
            }

        }
        public static void Lab14_Embedding()
        {
            var ml = new MLContext();

            var samples = new[]
            {
            new InputText { Text = "king" },
            new InputText { Text = "queen" },
            new InputText { Text = "camera" }
        };

            var data = ml.Data.LoadFromEnumerable(samples);

            var tokenizationPipeline = ml.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "Text");

            var embeddingPipeline = ml.Transforms.Text.ApplyWordEmbedding(
                outputColumnName: "Features",
                inputColumnName: "Tokens", 
                modelKind: Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.GloVe50D
            );

            var pipeline = tokenizationPipeline.Append(embeddingPipeline);
            var model = pipeline.Fit(data);
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<TextFeatures>(transformed, false).ToList();


            for (int i = 0; i < results.Count; i++)
            {
                Console.WriteLine($"\nWord: {samples[i].Text}");
                Console.WriteLine("Vector (first 10 values):");
                Console.WriteLine(string.Join(", ", results[i].Features.Take(10)) + " ...");
            }
            var resultsList = results.ToList();

            var kingVector = resultsList[0].Features;
            var queenVector = resultsList[1].Features;
            var cameraVector = resultsList[2].Features;
            double distanceKingQueen = CalculateCosineSimilarity(kingVector, queenVector);
            double distanceKingCamera = CalculateCosineSimilarity(kingVector, cameraVector);

            Console.WriteLine($"\nDistance (King vs. Queen): {distanceKingQueen:F4}");
            Console.WriteLine($"Distance (King vs. Camera): {distanceKingCamera:F4}");
        }
        public static double CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            double dotProduct = 0.0;
            double magnitude1 = 0.0;
            double magnitude2 = 0.0;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            if (magnitude1 == 0.0 || magnitude2 == 0.0)
            {
                // Handle cases where one or both vectors are zero vectors
                return 0.0;
            }

            return dotProduct / (magnitude1 * magnitude2);
        }

        public static double CalculateEuclideanDistance(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same length.");
            }

            double sumOfSquares = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                double difference = vector1[i] - vector2[i];
                sumOfSquares += difference * difference;
            }

            return Math.Sqrt(sumOfSquares);
        }
    }
}
