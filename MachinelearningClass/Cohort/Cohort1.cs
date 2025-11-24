using MachinelearningClass.Cohort;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public class NiftyData
    {
        [LoadColumn(1)]
        public float Nifty { get; set; }   // Historical price value
    }

    public class NiftyForecast
    {
        [ColumnName("ForecastedNifty")]
        public float[] ForecastedNifty { get; set; }
    }

    public static class  CohortLabs
    {
        
        public static void PredictNiftySSA()
        {
           


                var mlContext = new MLContext();

                // Load data (ensure CSV has header: Price)
                var dataView = mlContext.Data.LoadFromTextFile<NiftyData>(
                    path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\Nifty50.csv",
                    // path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\Nifty50.csv",
                    hasHeader: true,
                    separatorChar: ',');
           
                // Train SSA forecaster
                int windowSize = 12;   // Look at last 12 months pattern
                int seriesLength = 120; // Total historical span window
                int trainSize = 240;   // Total rows used for training
                int horizon = 3;       // Predict next 1 month

                var pipeline = mlContext.Forecasting.ForecastBySsa(
                    outputColumnName: nameof(NiftyForecast.ForecastedNifty),
                    inputColumnName: nameof(NiftyData.Nifty),
                    windowSize: windowSize,
                    seriesLength: seriesLength,
                    trainSize: trainSize,
                    horizon: horizon,
                    confidenceLevel: 0.95f
                );

                var model = pipeline.Fit(dataView);

                // Create prediction engine
                var forecastEngine = model.CreateTimeSeriesEngine<NiftyData, NiftyForecast>(mlContext);
                
                var result = forecastEngine.Predict();

                for (int i = 0; i < result.ForecastedNifty.Length; i++)
                {
                    Console.WriteLine($"Month +{i + 1}: {result.ForecastedNifty[i]:N2}");
                }

            Console.ReadLine();
        }
        public static void PredictNiftyUsingLags()
        {
            var mlContext = new MLContext();

            // Load the lagged CSV
            var dataView = mlContext.Data.LoadFromTextFile<NiftyLagData>(
                path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\Nifty50_with_lags.csv",
                hasHeader: true,
                separatorChar: ',');


            var validRows = mlContext.Data.FilterRowsByMissingValues(dataView, "NiftyLag1");


            var pipeline = mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(NiftyLagData.NiftyLag1),
                    nameof(NiftyLagData.NiftyLag2),
                    nameof(NiftyLagData.NiftyLag3)
                )
                .Append(mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(NiftyLagData.Nifty),
                    featureColumnName: "Features"));


            var model = pipeline.Fit(validRows);


            var engine = mlContext.Model.CreatePredictionEngine<NiftyLagData, NiftyPrediction>(model);

            var lastRow = mlContext.Data.CreateEnumerable<NiftyLagData>(validRows, reuseRowObject: false).Last();

            var prediction = engine.Predict(lastRow);


            Console.WriteLine($"Predicted Next Month Nifty: {prediction.PredictedValue:N2}");
            Console.WriteLine("====================================");

            Console.ReadLine();
        }
    }
   

}


