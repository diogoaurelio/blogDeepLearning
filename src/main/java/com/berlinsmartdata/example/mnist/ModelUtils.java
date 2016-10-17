package com.berlinsmartdata.example.mnist;


import org.apache.avro.generic.GenericData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.List;

public abstract class ModelUtils {

    protected static void trainModel(DataSet training, int batchSize, int numEpochs, MultiLayerNetwork model) {
        for(Integer i=0; i < numEpochs; i++) {
            DataSet trainData = training.sample(batchSize);
            model.fit(trainData);
        }
    }

    protected static List<DataSet> getTrainTestDatasets(RecordReader recordReader, int fullDatasetNumRows, int labelIndex, int numClasses) {
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, fullDatasetNumRows, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70);
        List<DataSet> result = new ArrayList<DataSet>(2);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        result.add(trainingData);
        result.add(testData);
        return result;
    }

}
