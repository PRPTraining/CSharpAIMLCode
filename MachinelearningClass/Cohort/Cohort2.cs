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
   

    public static class  CohortLabs2
    {
        
        public static void CreditCardFaultDetection()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromTextFile<CreditCardInputs>(
                path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\creditcard.csv",
                hasHeader: true,
                separatorChar: ',');

            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = ml.Transforms.Conversion.ConvertType(
                                inputColumnName: "Label",
                                outputColumnName: "Label",
                                outputKind: DataKind.Boolean)
                .Append(ml.Transforms.Concatenate("Features",
                    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "Amount"))
                .Append(ml.BinaryClassification.Trainers.FastTree());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            var metrics = ml.BinaryClassification.Evaluate(predictions);

            var predictor = ml.Model.CreatePredictionEngine<CreditCardInputs, FraudPrediction>(model);
            //4462,1.633048245,3.0577729,-0.21230852,-1.112095975,
            //-1.868420609,1.315005882,1.303208654,0.483606008,18.98,0
            
            var sample = new CreditCardInputs
            {
                Time = 4462,
                V1 = 1.633048245f,
                V2 = 3.0577729f,
                V3 = -0.21230852f,
                V4 = -1.112095975f,
                V5 = -1.868420609f,
                V6 = 1.315005882f,
                V7 = 1.303208654f,
                V8 = 0.483606008f,
                Amount = 18.98f
            };

            var result = predictor.Predict(sample);
            string fraudLabel = result.IsFraud ? "Fraud" : "Not Fraud";

            Console.WriteLine($"Result: {fraudLabel}");
            Console.WriteLine($"Probability: {result.Probability}");
            Console.ReadLine();
        }
      
    }
   

}


