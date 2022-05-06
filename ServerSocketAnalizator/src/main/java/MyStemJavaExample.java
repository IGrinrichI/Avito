import au.com.bytecode.opencsv.CSVWriter;
import org.json.JSONArray;
import org.json.JSONObject;
import ru.stachek66.nlp.mystem.holding.Factory;
import ru.stachek66.nlp.mystem.holding.MyStem;
import ru.stachek66.nlp.mystem.holding.MyStemApplicationException;
import ru.stachek66.nlp.mystem.holding.Request;
import ru.stachek66.nlp.mystem.model.Info;
import scala.Option;
import scala.collection.JavaConversions;

import au.com.bytecode.opencsv.CSVReader;

import java.io.*;
import java.util.*;

public class MyStemJavaExample {

    public static String resources = "./src/main/resources/";

    private final static MyStem mystemAnalyzer =
            new Factory("-igd --eng-gr --format json --weight")
                    .newMyStem("3.0", Option.<File>empty()).get();

    //игнорируемые части речи
    private static List<String> blacklist = new ArrayList<>(Arrays.asList("ADVPRO","ANUM","APRO","CONJ","INTJ","NUM","PART","PR","SPRO",
    "parenth","geo","persn","patrn","famn"));

    static TextAnalyzer textAnalyzer;


    public static void main(final String[] args) throws Exception {

//        mycsv();

//        make_dictionary(50000);
//        show_dictionary("example.txt",100);
//        show_dictionary("exampleheader.txt",100);
//        show_dictionary("exampleclasses.txt", 100);


//        convertationProcess();

//        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\MyNN\\train_vectorize.csv"), ',' , '"' );
//        String[] nextline;
//
//        for (int i = 0; i < 100; i++){
//            nextline = reader.readNext();
//            String t = "";
//            for (int j = 10; j > 0; j--){
//                t += nextline[65395 - j] + " ";
//            }
//            System.out.println(t);
//        }
//        reader.close();

//        Map<String,String> map = getmap();
//
//        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\MyNN\\train_vectorize.csv"), ',' , '"' );
//        CSVWriter writer = new CSVWriter(new FileWriter("G:\\TextMF\\MyNN\\train_vectorize1.csv"));
//        String[] nextline;
//        int iter = 0;
//        int percent = 0;
//        int n = 50000;
//        while ((nextline = reader.readNext()) != null){
//            iter++;
//            if(iter >= (percent+1) * n/100){
//                percent = (iter * 100) / n;
//                System.out.println("dictionary complete " + percent + "%");
//            }
//            nextline[65394] = map.get(nextline[65394]);
//            writer.writeNext(nextline);
//
//        }
//        reader.close();
//        writer.close();

        //System.out.println(new JSONObject(TextAnalyzer.read("exampleclasses.txt")).keySet().size());

//        Network.process();

        String text = "Статья́ — это жанр журналистики, в котором автор ставит задачу проанализировать общественные ситуации, процессы, явления, прежде всего с точки зрения закономерностей, лежащих в их основе.\n" +
                "\n" +
                "Такому жанру как статья присуща ширина практических обобщений, глубокий анализ фактов и явлений, четкая социальная направленность[источник не указан 3864 дня]. В статье автор рассматривает отдельные ситуации как часть более широкого явления. Автор аргументированно пишет о своей точке зрения.\n" +
                "\n" +
                "В статье выражается развернутая обстоятельная аргументированная концепция автора или редакции по поводу актуальной социологической проблемы. Также в статье журналист обязательно должен интерпретировать факты (это могут быть цифры, дополнительная информация, которая будет правильно расставлять акценты и ярко раскрывать суть вопроса).\n" +
                "\n" +
                "Отличительным аспектом статьи является её готовность. Если подготавливаемый материал так и не был опубликован (не вышел в тираж, не получил распространения), то такой труд относить к статье некорректно. Скорее всего данную работу можно назвать черновиком или заготовкой. Поэтому целью любой статьи является распространение содержащейся в ней информации.";


        //getClassesFromData();
        //makeCsvFromScratch(1400);
        //splitData(1000);
/*
//Очистка TF-IDF

        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/tf_idf_dic.csv"));
        writer.writeNext(new String[]{"counter","0"});
        writer.close();


//Постановка нового TF-IDF
        setNewIDF("train_data.csv", 49000);

 */

//        Word2Vector.newmain();



//        splitDataToDirectory("train_data.csv","labeled", true);
//        splitDataToDirectory("test_data.csv", "unlabeled_raw", true);

//        rewriteAllText("labeled");
//        rewriteAllText("unlabeled");



        //showWords(getClearWords(text));
        //System.out.println(Arrays.toString("a,b(c)d=e|f".split("[^A-Za-z0-9]")));
        //HERE NETWORK
        textAnalyzer = new TextAnalyzer(true);
//        ParagraphVectorsClassifierExample.newmain();
//        NewNetwork.newmain();

        String text1 = "";

        NewNetwork network = new NewNetwork("iter_1.nn");
        ParagraphVectorsClassifierExample doc2vec = new ParagraphVectorsClassifierExample();
        doc2vec.loadVectors("iter_6.pv");

        BufferedReader obj = new BufferedReader(new InputStreamReader(System.in));

        System.out.println("Started");
        while (true){
            text1 = obj.readLine();
            long t = System.currentTimeMillis();
            String cleartext = infoToText(getClearWords(text1,true),true);
            String kat = network.getKategory(doc2vec.getVector(cleartext));

            System.out.println(kat);
            System.out.println(System.currentTimeMillis() - t);
        }

    }

    static void splitDataToDirectory(String file, String dir, boolean toclasses) throws IOException {
        CSVReader reader = new CSVReader(new FileReader(resources + file), ',' , '"' );
        CSVReader class_reader = new CSVReader(new FileReader(resources + "total_classes.csv"), ',' , '"' );

        File load_to = new File(resources + dir);

        Map<String,String> classes = new HashMap<>();

        String[] nl;

        while((nl = class_reader.readNext()) != null && toclasses){
            classes.put(nl[0],nl[1]);
            if(!new File(resources + dir + "/" + nl[1]).exists()){
                new File(resources + dir + "/" + nl[1]).mkdir();
            }
        }
        class_reader.close();

        String name = "DOC_";
        String ex = ".txt";
        int counter = 0;

        while ((nl = reader.readNext()) != null){
            File newfile = new File(resources + dir + "/" + (toclasses ? classes.get(nl[2]) + "/" : "") + name + counter + ex);
            if(newfile.createNewFile()){
                TextAnalyzer.save(dir + "/" + (toclasses ? classes.get(nl[2]) + "/" : "") + name + counter + ex, nl[0] + " " + nl[1]);
            }
            counter++;
        }


        reader.close();
    }

    static void rewriteAllText(String file) throws MyStemApplicationException {
        File dir = new File(resources + file);
        String text;

        for(File kategory : dir.listFiles()){
            if(kategory.isDirectory()){
                for(File doc : kategory.listFiles()){
                    if(doc.isFile()){
                        text = TextAnalyzer.read(doc.getAbsolutePath(),true);

                        Iterable<Info> infos = getClearWords(text,true);
                        text = "";
                        for(Info info : infos){
                            if(info.lex().get().equals("None")){
                                text += info.initial().toLowerCase() + " ";
                            }
                            else {
                                text += info.lex().get() + " ";
                            }
                        }

                        TextAnalyzer.save(doc.getAbsolutePath(),text,true);
                    }
                }
            }
        }
    }

    static void splitData(Integer count_of_train) throws IOException {
        Map<String,Integer> train = getClassesWithCounter(count_of_train);
        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/all_data.csv"), ',' , '"' );
        CSVWriter train_writer = new CSVWriter(new FileWriter("./src/main/resources/train_data.csv"));
        CSVWriter test_writer = new CSVWriter(new FileWriter("./src/main/resources/test_data.csv"));
        String[] nextline;


        while ((nextline = reader.readNext()) != null){
            try {
                if(train.get(nextline[2]) > 0){
                    train_writer.writeNext(nextline);
                    train.put(nextline[2], train.get(nextline[2]) - 1);
                }
                else{
                    test_writer.writeNext(nextline);
                }
            }
            catch (Exception e){
                System.out.println(nextline);
            }
        }

        reader.close();
        test_writer.close();
        train_writer.close();

    }

    static Map<String,Integer> getClassesWithCounter(Integer counter) throws IOException {
        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/total_classes.csv"), ',' , '"' );

        Map<String,Integer> map = new HashMap<>();

        String[] nextline;

        while ((nextline = reader.readNext()) != null){
            map.put(nextline[0],counter);
        }

        reader.close();
        return map;
    }

    static void makeCsvFromScratch(Integer first_n) throws IOException {

        Map<String,Integer> map = getClassesWithCounter(first_n);
        String[] nextline;

        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\Avito\\train.csv\\train.csv"), ',' , '"' );
        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/all_data.csv"));


        int countclass = map.keySet().size();

        while ((nextline = reader.readNext()) != null){
            try {
                if(map.containsKey(nextline[3]) && map.get(nextline[3]) > 0){
                    writer.writeNext(new String[]{nextline[0],nextline[1],nextline[3]});
                    map.put(nextline[3],map.get(nextline[3]) - 1);
                    if(map.get(nextline[3]) == 0){
                        countclass--;
                        System.out.println("Index " + nextline[3] + " full, remain " + countclass + " classes");
                        if(countclass == 0){
                            break;
                        }
                    }
                }
            }
            catch (Exception e){

            }
        }
        reader.close();

    }

    static void getClassesFromData() throws IOException {
        Map<String,Integer> map = new HashMap<>();
        Map<String,String> map1 = new HashMap<>();

        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\Avito\\train.csv\\train.csv"), ',' , '"' );

        String[] nextline;
        int errors = 0;
        int items = 0;
        while ((nextline = reader.readNext()) != null){
            try {
                if(map.containsKey(nextline[3])){
                    map.put(nextline[3], map.get(nextline[3]) + 1);
                }
                else if (Integer.parseInt(nextline[3]) > 0){
                    map.put(nextline[3], 1);
                    map1.put(nextline[3], nextline[2]);
                }
                items++;
                if(items % 10000 == 0){
                    System.out.println("Already find " + items);
                    System.out.println("Errors " + errors);
                }
            }
            catch (Exception e){
                errors++;
            }
        }
        reader.close();

        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/total_classes.csv"));

        for (String key : map.keySet()){
            if(map.get(key) > 500){
                writer.writeNext(new String[]{key,map1.get(key),map.get(key).toString()});
            }
        }
        writer.close();

        System.out.println("Errors total " + errors);
        System.out.println("Items total " + items);
    }

    static String getPart(String parts){
        return parts.split("=")[0];
    }

    static List<String> getSeparatePart(String parts){
        return Arrays.asList(parts.split("=")[0].split(","));
    }

    static void showWords(Iterable<Info> infos){

        for (final Info info : infos) {
//            System.out.println(info.initial() + " -> " + (info.lex().toString().equals("None") ? "none" : info.lex().get() + info.rawResponse()/* + " | " + new JSONObject(info.rawResponse()).getJSONArray("analysis").getJSONObject(0).get("wt")*/));
//            System.out.println(info.initial() + " -> " + (info.lex().toString().equals("None") ? "none" : info.lex().get()/* + info.rawResponse()*/ + " | " + new JSONObject(info.rawResponse()).getJSONArray("analysis").getJSONObject(0).get("gr")));
            String parts = (String)new JSONObject(info.rawResponse()).getJSONArray("analysis").getJSONObject(0).get("gr");
            System.out.println(info.initial() + " -> " + (info.lex().toString().equals("None") ? "none" : info.lex().get()/* + info.rawResponse()*/ + " | " + getPart(parts)));
        }

    }

    static Map<String,String> getmap(){
        Map<String,String> map = new HashMap<>();

        JSONObject json = new JSONObject(TextAnalyzer.read("exampleclasses.txt"));
        int index = 0;
        for (Object key : json.keySet()){
            String id = (String) key;
            map.put(id, Integer.toString(index));
            index++;
        }

        return map;
    }

    static void show_dictionary(String dictionary,int n){
        JSONObject jobject = new JSONObject(TextAnalyzer.read(dictionary));
        Map<String,Integer> unsortedmap = new HashMap<>();

        Iterator iterator = jobject.keys();
        while (iterator.hasNext()){
            String next = (String) iterator.next();
            unsortedmap.put(next, (Integer) jobject.get(next));
        }

        Map<String,Integer> sortedmap = TextAnalyzer.sortByValuei(unsortedmap);

        System.out.println(sortedmap.keySet().size());
        int for_t = 0;
        for (String key:sortedmap.keySet()){
            for_t++;
            System.out.println(key + " : " + sortedmap.get(key));
            if(for_t >= n){
                break;
            }
        }
    }

    static Iterable<Info> getWords(String text) throws MyStemApplicationException {
        return JavaConversions.asJavaIterable(
                        mystemAnalyzer
                                .analyze(Request.apply(text))
                                .info()
                                .toIterable());
    }

    static Iterable<Info> getClearWords(String text, boolean withExtra) throws MyStemApplicationException {


        Iterable<Info> result = JavaConversions.asJavaIterable(
                mystemAnalyzer
                        .analyze(Request.apply(text))
                        .info()
                        .toIterable());

        List<Info> list = new ArrayList<>();

        for (Info info : result){

            try {
                if (Collections.disjoint(blacklist,getSeparatePart((String)new JSONObject(info.rawResponse()).getJSONArray("analysis").getJSONObject(0).get("gr")))){
                    list.add(info);
                }
                else if (info.lex().get().equals("None") && withExtra){
                    list.add(info);
                }
            }
            catch (Exception e){

            }

        }
        return list;
    }

    static String infoToText(Iterable<Info> infos, boolean withExtra){
        String text = "";

        for (Info info : infos){
            if(!info.lex().get().equals("None")){
                text += info.lex().get() + " ";
            }
            else if (withExtra){
                text += info.initial().toLowerCase() + " ";
            }
        }

        return text;
    }

    static void make_dictionary(int n) throws IOException, MyStemApplicationException {
//        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\Avito\\train.csv\\train.csv"), ',' , '"' , 1);
        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/train.csv"), ',' , '"' );

        int items = 0;

        String [] nextline;
        JSONObject json = new JSONObject();
        Map<String, Integer> map = new HashMap<>();
        JSONObject json1 = new JSONObject();
        Map<String, Integer> map1 = new HashMap<>();
        JSONObject json2 = new JSONObject();
        Map<Integer, Integer> map2 = new HashMap<>();

        int percent = 0;
        int errors = 0;

//        List<String[]> csv = new LinkedList<>();

        while((nextline = reader.readNext()) != null){


            if(items > n){
                break;
            }

            if(items >= (percent+1) * n/100){
                percent = (items * 100) / n;
                System.out.println("dictionary complete " + percent + "%");
            }

            Iterable<Info> infos = null;
            Iterable<Info> infos1 = null;
                    
            try {
                infos = getWords(nextline[1]);
                infos1 = getWords(nextline[0]);
                Integer class_type = Integer.parseInt(nextline[3]);
                map2.put(class_type, (map2.get(class_type) == null ? 1 : map2.get(class_type) + 1));
//                csv.add(nextline);
            }
            catch (Exception e){
                errors++;
                //System.out.println(Arrays.toString(nextline));
                continue;
                //System.exit(1);
            }
            items++;

            for (Info info:infos){
                String key;
                try{
                    key = info.lex().get();
                    Integer value = map.get(key);
                    map.put(key, (value == null ? 1 : value + 1));
                }
                catch (Exception e){
                    //key = info.initial();

                }

            }

            for (Info info:infos1){
                String key;
                try{
                    key = info.lex().get();
                    Integer value = map1.get(key);
                    map1.put(key, (value == null ? 1 : value + 1));
                }
                catch (Exception e){
                    //key = info.initial();

                }

            }

//            System.out.println(nextline[1]);
//            System.out.println(Arrays.toString(nextline));



        }
        //System.out.println(items);

        for(String key:map.keySet()){
            json.put(key, map.get(key));
        }

        for(String key:map1.keySet()){
            json1.put(key, map1.get(key));
        }

        for(Integer key:map2.keySet()){
            json2.put(key.toString(), map2.get(key));
        }

        TextAnalyzer.save("example.txt", json.toString());
        TextAnalyzer.save("exampleheader.txt", json1.toString());
        TextAnalyzer.save("exampleclasses.txt", json2.toString());
//        csvwriter("train.csv",csv);
        reader.close();
        System.out.println("Errors: " + errors);
    }

    static void mycsv() throws IOException {
        CSVReader reader = new CSVReader(new FileReader("G:\\TextMF\\Avito\\train.csv\\train.csv"), ',' , '"' , 1);
        String[] nextline;
        List<String[]> mylist = new LinkedList<>();
        int items = 0;
        while((nextline = reader.readNext()) != null){
            try{
                int class_type = Integer.parseInt(nextline[3]);
                items++;
                mylist.add(nextline);
                if(items >= 50000){
                    break;
                }
            }
            catch (Exception e){

            }
        }
        csvwriter("train.csv", mylist);
    }

    static void csvwriter(String file, List<String[]> data) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/" + file));
        //Create record
        //Write the record to file
        for (String[] raw:data){
            writer.writeNext(raw);
        }

        //close the writer
        writer.close();
    }

    static void mainprocess() throws MyStemApplicationException, IOException {
        String initText = "Думаю об этом - меня даже звали, если оголодаю. И собственно, даже звать не надо было, я знаю, что если совсем будет плохо, я могу зайти просто так.\n" +
                "Но я тут с ужасом думаю об обещанной алкогольном вечере сегодняшнем, если честно. Приехала, тут тишина (я даже плеер в доме оставила), пока приводила хотя бы часть участка в порядок, чтобы можно было жить... Так хорошо, честно. То есть, я мокрая замёрзшая, уставшая (физически), но мне хорошо~ в комнате, правда, все ещё +12, но к ночи, надеюсь, хотя бы +17 будет.";

        final Iterable<Info> result =
                JavaConversions.asJavaIterable(
                        mystemAnalyzer
                                .analyze(Request.apply(initText))
                                .info()
                                .toIterable());

        String text = "";

        for (final Info info : result) {
            System.out.println(info.initial() + " -> " + (info.lex().toString().equals("None") ? "none" : info.lex().get() + info.rawResponse()/* + " | " + new JSONObject(info.rawResponse()).getJSONArray("analysis").getJSONObject(0).get("wt")*/));
            try{
                text += info.lex().get() + " ";
            }
            catch (Exception e){
                text += info.initial() + " ";
            }
        }
        System.out.println(text);
        TextAnalyzer.clearHistory();
        TextAnalyzer textAnalyzer = new TextAnalyzer(true);
        //System.out.println(TextAnalyzer.read("statistic.txt"));

        List<String> blacklist = new ArrayList<String>();
        blacklist.add("");
        textAnalyzer.setBlacklist(blacklist);
        Map<String,Float> keywords = TextAnalyzer.sortByValue(textAnalyzer.getKeywords(text));
        for (String key:keywords.keySet()){
            System.out.println(key + " " + keywords.get(key));
        }

        TextAnalyzer.save("historytext.txt", new JSONArray(TextAnalyzer.read("historytext.txt")).put(initText).toString());
    }
//Создание нового IDF словаря
    static void setNewIDF(String file, Integer size) throws IOException, MyStemApplicationException {
        Iterable<Info> result;
        TextAnalyzer.clearHistory();
        TextAnalyzer textAnalyzer = new TextAnalyzer(true);
        List<String> blacklist = new ArrayList<String>();
        blacklist.add("");
        textAnalyzer.setBlacklist(blacklist);

        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/" + file), ',' , '"' );

        String[] nextline;

        int n = size;
        int progress = 0;
        int percent = 0;
        while ((nextline = reader.readNext()) != null){
            result = getClearWords(nextline[0] + " " + nextline[1], false);
                    /*JavaConversions.asJavaIterable(
                            mystemAnalyzer
                                    .analyze(Request.apply(nextline[0]+nextline[1]))
                                    .info()
                                    .toIterable());

                     */
            //textAnalyzer.getVector(result);



            textAnalyzer.setVector(result);
            progress++;
            if(progress >= (percent+1) * n/100){
                percent = (progress * 100) / n;
                System.out.println("TF-IDF dictionary complete " + percent + "%");
            }
        }
        reader.close();
        textAnalyzer.saveStatistic();

    }

    //Использует веса TF-IDF для просчета весов для текстов
    static void convertationProcess() throws IOException, MyStemApplicationException {
        CSVWriter writer = new CSVWriter(new FileWriter("G:\\TextMF\\MyNN\\train_vectorize.csv"));
        CSVReader reader = new CSVReader(new FileReader("./src/main/resources/train.csv"), ',' , '"' );

        TextAnalyzer textAnalyzer = new TextAnalyzer(true);

        String[] nextline;
        Iterable<Info> infos;
        String[] towrite;
        int iter = 0;
        int percent = 0;
        int n = 50000;
        while ((nextline = reader.readNext()) != null){
            iter++;
            if(iter >= (percent+1) * n/100){
                percent = (iter * 100) / n;
                System.out.println("dictionary complete " + percent + "%");
            }

            infos = getClearWords(nextline[0] + " " + nextline[1], false);
            towrite = textAnalyzer.getVector(infos);
            towrite[towrite.length - 1] = nextline[3];
            writer.writeNext(towrite);
        }
        writer.close();
        reader.close();
    }
}
