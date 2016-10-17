package com.berlinsmartdata.example.mnist;


import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;

import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class MLPKaggleMnistExample extends ModelUtils {

    private static Logger logger = LoggerFactory.getLogger(MLPKaggleMnistExample.class);

    public static void main(String[] args) throws Exception {
        // images specs
        final int numRows = 28;
        final int numCols = 28;
        // kaggle MNIST dataset is streched into one single vector, thus:
        final int numInputs = numRows * numRows;
        final int depth = 1; // not used in traditional NN model, as we have no convolutions
        final int fullDatasetNumRows = 42000;
        final int numClasses = 10; // num of labels
        final int batchSize = 128; // for stochastic gradient descent => num examples to use on a given step;
        final int seed = 123; // pseudo-random number generator, which is used to randomly generate weights initially
        final int numEpochs = 15; // an epoch is a complete pass through the dataset
        final int numIterations = 1;
        final int labelIndex = 0;

        final int layer0NumHiddenNodes = 1000;
        final int layer1NumHiddenNodes = 1000;

        final int numLinesToSkip = 1; // header
        final String delimiter = ",";
        //hyperparam
        final Double l2RegParam = 1e-4;
        final Double learningRate= 0.006;

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("/kaggleMnist/train.csv").getFile()));
        List<DataSet> dataSetList = getTrainTestDatasets(recordReader, fullDatasetNumRows, labelIndex, numClasses);
        DataSet trainingData = dataSetList.get(0);
        DataSet testData = dataSetList.get(1);

        logger.info("Starting to build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(numIterations)
                .weightInit(WeightInit.XAVIER)
                .learningRate(learningRate)
                .regularization(true).l2(l2RegParam)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(layer0NumHiddenNodes)
                        .activation("relu")
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(layer0NumHiddenNodes)
                        .nOut(layer1NumHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(layer1NumHiddenNodes)
                        .nOut(numClasses)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        logger.info("Starting to train the model for a total of " + numEpochs +" Epochs...");
        trainModel(trainingData, batchSize, numEpochs, model);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        logger.info(eval.stats());
        logger.info("****************Example finished********************");

    }
}
