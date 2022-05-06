import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import ru.stachek66.nlp.mystem.holding.MyStemApplicationException;

import java.util.List;

/**
 * This is basic example for documents classification done with DL4j ParagraphVectors.
 * The overall idea is to use ParagraphVectors in the same way we use LDA:
 * topic space modelling.
 * <p>
 * In this example we assume we have few labeled categories that we can use
 * for training, and few unlabeled documents. And our goal is to determine,
 * which category these unlabeled documents fall into
 * <p>
 * <p>
 * Please note: This example could be improved by using learning cascade
 * for higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsClassifierExample {

    private static ParagraphVectors paragraphVectors;
    private static LabelAwareIterator iterator;
    private static TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.class);

    public static String dataLocalPath;


    public static void newmain() throws Exception {


        dataLocalPath = "./src/main/resources/";
        ParagraphVectorsClassifierExample app = new ParagraphVectorsClassifierExample();
//        app.makeParagraphVectors("labeled");
        app.loadVectors("iter_6.pv");
        app.checkUnlabeledData("unlabeled");
//        app.complectDataSet("labeled");
//        app.complectDataSet("unlabeled");
        /*
                Your output should be like this:
                Document 'health' falls into the following categories:
                    health: 0.29721372296220205
                    science: 0.011684473733853906
                    finance: -0.14755302887323793
                Document 'finance' falls into the following categories:
                    health: -0.17290237675941766
                    science: -0.09579267574606627
                    finance: 0.4460859189453788
                    so,now we know categories for yet unseen documents
         */
    }
    /*
    iter_1.pv 1 iter raw

    iter_2.pv 10 iter raw

    iter_3.pv 20 iter raw

    iter_4.pv 20 iter clear (without eng)
paragraphVectors = new ParagraphVectors.Builder()
                .minWordFrequency(2)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .useAdaGrad(true)
                .seed(4872127)
                .build();

    iter_5.pv 10 iter clear (without eng)
    batchsize = 100;

    iter_6.pv 10 iter clear (without eng)
    without AdaGrad(?)
    batchsize = 1000;
    learningRate = 0.005;
    minLearningRate = 0.0002;
    with epochs = 10
    Total 13074/19600
    Fails 46
    with epochs = 20
    Total 13202/19600
    Fails 46

    iter_7.pv 10 iter clear (without eng)
    learningRate = 0.001;
    minLearningRate = 0.00005;
    Total 11276/19600
    Fails 46

    iter_8.pv 20 iter clear (without eng)
    Total 12365/19600
    Fails 46

     */
    void makeParagraphVectors(String file) throws Exception {
        File resource = new File(dataLocalPath + file);
        int seed = 4872127;
        int batchSize = 1000;
        int epochs = 20;
        double learningRate = 0.001;
        double minLearningRate = 0.00005;
        int minWordFrequency = 2;

        // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(resource)
                .build();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        System.out.println("Build model");
        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
                .minWordFrequency(minWordFrequency)
                .learningRate(learningRate)
                .minLearningRate(minLearningRate)
                .batchSize(batchSize)
                .epochs(epochs)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .seed(seed)
                .build();

        System.out.println("Training");
        // Start model training
        paragraphVectors.fit();
        File pv = new File("./src/main/resources/iter_8.pv");
        pv.createNewFile();
        WordVectorSerializer.writeParagraphVectors(paragraphVectors,pv);
    }

    void loadVectors(String file) throws Exception {
        paragraphVectors = WordVectorSerializer.readParagraphVectors("./src/main/resources/" + file);
        File resource = new File(dataLocalPath + "labeled");

        // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(resource)
                .build();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        paragraphVectors.setTokenizerFactory(tokenizerFactory);
    }

    void checkUnlabeledData(String file) throws IOException {
      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */
        File unClassifiedResource = new File(dataLocalPath, file);
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(unClassifiedResource)
                .build();

     /*
      Now we'll iterate over unlabeled data, and check which label it could be assigned to
      Please note: for many domains it's normal to have 1 document fall into few labels at once,
      with different "weight" for each.
     */
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
                tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        System.out.println("Evaluation");

        int iter = 0;
        String item = "";
        int total = 0;
        int fails = 0;
        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
//            System.out.println(document.toString());
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            if(documentAsCentroid == null){
                fails++;
                continue;
            }
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            if(iter == 0){
                item = document.getLabels().get(0);
            }
         /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
            Collections.sort(scores, new Comparator<Pair<String, Double>>() {
                @Override
                public int compare(final Pair<String, Double> o1, final Pair<String, Double> o2) {
                    if (o1.getValue() > o2.getValue()) {
                        return -1;
                    } else if (o1.getValue().equals(o2.getValue())) {
                        return 0; // You can change this to make it then look at the
                        //words alphabetical order
                    } else {
                        return 1;
                    }
                }
            });

            if (!document.getLabels().get(0).equals(item)){
                System.out.println("Kategory " + item + " " + iter + "/400");
                total += iter;
                iter = 0;
                item = document.getLabels().get(0);
            }
            if(scores.get(0).getFirst().equals(item)){
                iter++;
            }

//         System.out.println("Document '" + document.getLabels() + "' falls into the following categories: ");
            //log.info("Document '" + document.getLabels() + "' falls into the following categories: ");
//            for (Pair<String, Double> score : scores) {
//                System.out.println("        " + score.getFirst() + ": " + score.getSecond());
                //log.info("        " + score.getFirst() + ": " + score.getSecond());
//                break;
//            }
        }
        System.out.println("Kategory " + item + " " + iter + "/400");
        total += iter;
        System.out.println("Total " + total + "/" + (400*49));
        System.out.println("Fails " + fails);

    }

    public INDArray getVector(String text){
        boolean has = false;
        for(String word : text.split(" ")){
            if (paragraphVectors.hasWord(word)){
                has = true;
                break;
            }
        }
        if(has){
            return paragraphVectors.inferVector(text);
        }
        else{
            return null;
        }
    }

    private DataSet complectDataSet(String file) throws IOException, MyStemApplicationException {
//        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/vectorize_" + file));
        List<DataSet> data = new LinkedList<>();


        File unClassifiedResource = new File(dataLocalPath, file);
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(unClassifiedResource)
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
                tokenizerFactory);


        int idx = 0;
        int percent = 0;
        int n = 49000;
        Integer numClasses = 49;
        while (unClassifiedIterator.hasNextDocument()){
            LabelledDocument document = unClassifiedIterator.nextDocument();

            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);

            if(documentAsCentroid == null){
                continue;
            }
            data.add(MyStemJavaExample.textAnalyzer.toVec(documentAsCentroid.reshape(1,100),numClasses, document.getLabels().get(0)));
            idx++;
            if(idx >= (percent+1) * n/100){
                percent = (idx * 100) / n;
                System.out.println("Data preprocessing complete " + percent + "%");
            }
        }

        DataSet result = DataSet.merge(data);
        new File("./src/main/resources/vectorized_" + file + ".csv").createNewFile();
        result.save(new File("./src/main/resources/vectorized_" + file + ".csv"));

//        writer.close();
        System.out.println("trainset have size " + data.size());

        return result;
    }

}

