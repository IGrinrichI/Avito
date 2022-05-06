import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.examples.download.DownloaderUtility;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.stachek66.nlp.mystem.holding.MyStemApplicationException;

import java.io.*;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class NewNetwork {
    private static Logger log = LoggerFactory.getLogger(NewNetwork.class);

    final static int labelIndex = 100;
    final static int numClasses = 49;
    final static int batchSize = 1000;
    final static int epoch = 20;
    final static double learningRate = 0.025;
    final static boolean load = true;

    private MultiLayerNetwork my_model;

    NewNetwork(String model) throws IOException {
        this.my_model = MultiLayerNetwork.load(new File(MyStemJavaExample.resources + model), true);
    }

    public String getKategory(INDArray input){
        if(input == null){
            return "Не определено";
        }
        Integer argmax = my_model.output(input.reshape(1,100)).argMax(1).getInt(0);
        Double max = my_model.output(input.reshape(1,100)).getDouble(0,argmax);
        if(max < 0.1){
            return "Не определено (" + MyStemJavaExample.textAnalyzer.getKategory(argmax) + ")";
        }
        return MyStemJavaExample.textAnalyzer.getKategory(argmax);
    }

    /*
    1 layer:
    2000
    learningRate = 0.025
    40 epochs
    79%

    1 layer:
    1000
    learningRate = 0.025
    20 epochs
    78%

    2 layer:
    1000
    250
    learningRate = 0.025
    20 epochs
    78%

    2 layer:
    2000
    500
    learningRate = 0.025
    30 epochs
    78%

    3 layer:
    1000
    500
    250
    learningRate = 0.1
    20 epochs
    70%


     */

    public static void newmain() throws  Exception {

        final int numInputs = labelIndex;
        int outputNum = numClasses;
        long seed = 6;

        final int first = 1000;
        final int second = 250;
        final int third = 250;
        int score_each = 100;

        MultiLayerNetwork model;

        if(load){
            model = MultiLayerNetwork.load(new File(MyStemJavaExample.resources + "iter_1.nn"), true);

        }
        else{
            //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
            int numLinesToSkip = 0;
            char delimiter = ',';
            char quote = '"';
            System.out.println("make iterator");
//        File trainDataFile = new File("G:\\TextMF\\MyNN\\train_vectorize.csv");
//        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter,quote);
//        recordReader.initialize(new FileSplit(trainDataFile));


            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
            //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

            System.out.println("train/test");
//        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);//(recordReader,batchSize,labelIndex,numClasses);
//        System.out.println(iterator.next().getLabels().toIntMatrix()[0][0] + "\n" +
//                iterator.next().getLabels().toIntMatrix()[1][1] + "\n" +
//                iterator.next().getLabels().toIntMatrix()[2][2] + "\n" +
//                iterator.next().getLabels().toIntMatrix()[3][3] + "\n" +
//                iterator.next().getLabels().toIntMatrix()[4][4]);

//        DataSet allData = iterator.next();
            //System.out.println(allData.getLabels().toIntMatrix()[0][0]);
//        allData.shuffle();
//        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);  //Use 80% of data for training
//
//        DataSet trainingData = testAndTrain.getTrain();
//        DataSet testData = testAndTrain.getTest();
        /*
        System.out.println("normalize data?");
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


         */





            System.out.println("build model");
            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .updater(new Adam(learningRate,0.9,0.999,1e-8))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(first).activation(Activation.SOFTMAX)
                            .build())
//                .layer(new DenseLayer.Builder().nIn(first).nOut(second).activation(Activation.SOFTMAX)
//                        .build())
                    .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nIn(first).nOut(outputNum).build())
                    .build();

            System.out.println("run model");
            //run the model
            model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(score_each));


//        recordReader.close();

            DataSet trainData = new DataSet();
            trainData.load(new File("./src/main/resources/vectorized_labeled.csv"));//complectDataSet("train_data.csv");
            trainData.shuffle(seed);
            DataSetIterator dsi = new ListDataSetIterator<DataSet>(trainData.asList(), batchSize);
            model.fit(dsi,epoch);
            model.save(new File(MyStemJavaExample.resources + "iter_1.nn"));

        }







        System.out.println("Evaluation");
        //evaluate the model on the test set
        Evaluation eval = new Evaluation(numClasses);
        DataSet testData = new DataSet();
        testData.load(new File("./src/main/resources/vectorized_unlabeled.csv"));//complectDataSet("test_data.csv");
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats(false,true));
//        TextAnalyzer.save("stats.csv",eval.getConfusionMatrix().toCSV().replace(',',';'));
//        System.out.println(model.output(testData.get(100).getFeatures()));
//        int which = 13010;
//        for(int i = which; i < which + 100; i++){
//            System.out.println(model.output(testData.get(i).getFeatures()).argMax(1));
//            System.out.println(model.output(testData.get(i).getFeatures()).getDouble(0,model.output(testData.get(i).getFeatures()).argMax(1).getInt(0)));
//            System.out.println(testData.get(i).getLabels().argMax(1));
//        }

        //log.info(eval.stats());
    }

    static DataSet complectDataSet(String file) throws IOException, MyStemApplicationException {
        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/" + file), ',' , '"' );
//        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/vectorize_" + file));
        List<DataSet> data = new LinkedList<>();

        String[] nextline;
        int idx = 0;
        int percent = 0;
        int n = 49000;
        while ((nextline = reader.readNext()) != null){
            data.add(MyStemJavaExample.textAnalyzer.toVec(MyStemJavaExample.getClearWords(nextline[0] + " " + nextline[1], false),labelIndex,numClasses, nextline[2]));
            idx++;
            if(idx >= (percent+1) * n/100){
                percent = (idx * 100) / n;
                System.out.println("Data preprocessing complete " + percent + "%");
            }
        }

        DataSet result = DataSet.merge(data);
        result.save(new File("./src/main/resources/vectorize_" + file));

        reader.close();
//        writer.close();
        System.out.println("trainset have size " + data.size());

        return result;
    }
}
