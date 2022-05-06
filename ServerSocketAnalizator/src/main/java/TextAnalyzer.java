import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.jfree.data.general.Dataset;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import ru.stachek66.nlp.mystem.model.Info;

import java.io.*;
import java.util.*;


public class TextAnalyzer {

    private List<String> blacklist = new ArrayList<String>();
    private Map<String,Integer> statistic;
    private Integer docCounter;
    private Map<String,Integer> tf_idf_array;
    private Map<String,Integer> class_token;
    private Map<String,Integer> raw_class_token;

    TextAnalyzer(boolean isCSV) throws IOException {
        if(isCSV){
            CSVReader reader = new CSVReader(new FileReader("./src/main/resources/tf_idf_dic.csv"), ',' , '"' );
            CSVReader class_reader = new CSVReader(new FileReader("./src/main/resources/total_classes.csv"), ',' , '"' );

            docCounter = Integer.parseInt(reader.readNext()[1]);

            Map<String, Integer> map = new HashMap<>();

            Map<String, Integer> map1 = new HashMap<>();

            class_token = new HashMap<>();
            raw_class_token = new HashMap<>();
            int ctcounter = 0;

            String[] nextline;
            int iterator = 0;
            while ((nextline = reader.readNext()) != null){
                map.put(nextline[0], Integer.parseInt(nextline[1]));
                iterator++;
                map1.put(nextline[0], iterator);
            }
            while ((nextline = class_reader.readNext()) != null){
                class_token.put(nextline[0],ctcounter);
                raw_class_token.put(nextline[1],ctcounter);
                ctcounter++;
            }

            statistic = map;
            tf_idf_array = map1;
            reader.close();
            class_reader.close();
        }
        else {
            JSONObject json = new JSONObject(read("statistic.txt"));
            docCounter = (Integer) json.get("counter");

            Map<String,Integer> map = new HashMap<>();
            JSONObject items = json.getJSONObject("items");

            for (Object key:items.keySet()){
                map.put((String) key,(Integer) items.get((String) key));
            }
            statistic = map;
            //System.out.println(docCounter + "\n" + map.keySet());
        }

    }

    public String getKategory(Integer arg){
        for(String key : raw_class_token.keySet()){
            if(raw_class_token.get(key) == arg){
                return key;
            }
        }
        return null;
    }

    public void setBlacklist(List<String> list){
        this.blacklist = list;
    }

    public Map<String,Float> getKeywords(String text) throws IOException {
        List<String> words = clearText(text);

        Map<String,Integer> map = new HashMap<String,Integer>();

        Integer max = 1;

        for (String word:words){
            if (map.containsKey(word)){
                map.put(word, map.get(word) + 1);
                if (max < map.get(word)){
                    max = map.get(word);
                }
            }
            else{
                map.put(word, 1);
            }
        }


        Map<String,Float> localWeights = new HashMap<String, Float>();

        for (String word:map.keySet()){
            Float TF = map.get(word).floatValue() / max;
            Float IDF = (float) (docCounter + 1) / (statistic.get(word) == null ? 1 : statistic.get(word));
            localWeights.put(word, TF * IDF);
            if(statistic.containsKey(word)){
                statistic.put(word, statistic.get(word) + 1);
            }
            else{
                statistic.put(word, 1);
            }
        }

        //saveMap("history.txt",localWeights);

        JSONObject json = new JSONObject();

        json.put("counter", docCounter + 1);
        JSONObject items = new JSONObject();

        for (String key:statistic.keySet()){
            items.put((String) key,(Integer) statistic.get((String) key));
        }
        json.put("items", items);

        save("statistic.txt", json.toString());

        //Map<String,Float> result = new HashMap<String, Float>();
        //result = sortByValue(localWeights);


        return localWeights;//sortByValue(localWeights);
    }

    public String[] getVector(Iterable<Info> info){
        String[] result = new String[statistic.keySet().size() + 1];

        Arrays.fill(result, "0");
        Map<String, Integer> map = new HashMap<>();
        int max = 1;
        for(Info inf : info){
            try {
                String word = inf.lex().get();
                Integer value = map.get(word);
                map.put(word, (value == null ? 1 : value + 1));
                if (max < map.get(word)){
                    max = map.get(word);
                }
            }
            catch (Exception e){

            }
        }

        for(String key : map.keySet()){
            if(statistic.containsKey(key)){
                Double TF = Double.valueOf(map.get(key)) / max;
                Double IDF = Double.valueOf(docCounter) / statistic.get(key);
                result[tf_idf_array.get(key)] = Double.toString(TF * IDF);
            }
        }

        return result;
    }

    public void setVector(Iterable<Info> info){
        Integer value;
        Set<String> set = new HashSet<>();
        for (final  Info inf : info){
            try{
                set.add(inf.lex().get());
            }
            catch (Exception e){
                //inf.initial();
            }
        }
        for (String key : set) {
            value = statistic.get(key);
            if(value == null){
                statistic.put(key, 1);
            }
            else{
                statistic.put(key, value + 1);
            }
        }
        docCounter++;
    }

    public void saveStatistic() throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter("./src/main/resources/tf_idf_dic.csv"));

        writer.writeNext(new String[] {"counter", docCounter.toString()});

        double count = 0;
        double finalcount = 0;
        for (String key:statistic.keySet()){
            if(statistic.get(key) > 1 && key.length() > 1){
                writer.writeNext(new String[] {key, statistic.get(key).toString()});
                finalcount++;
            }
            count += statistic.get(key);
        }
        writer.close();
        System.out.println(finalcount + " words of " + docCounter + " texts have been saved");
        System.out.println("mediana " + (count /statistic.keySet().size()));
    }

    public static Map<String, Float> sortByValue(Map<String, Float> unsortMap) {

        // 1. Convert Map to List of Map
        List<Map.Entry<String, Float>> list =
                new LinkedList<Map.Entry<String, Float>>(unsortMap.entrySet());

        // 2. Sort list with Collections.sort(), provide a custom Comparator
        //    Try switch the o1 o2 position for a different order
        Collections.sort(list, new Comparator<Map.Entry<String, Float>>() {
            public int compare(Map.Entry<String, Float> o1,
                               Map.Entry<String, Float> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // 3. Loop the sorted list and put it into a new insertion order Map LinkedHashMap
        Map<String, Float> sortedMap = new LinkedHashMap<String, Float>();
        for (Map.Entry<String, Float> entry : list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        /*
        //classic iterator example
        for (Iterator<Map.Entry<String, Integer>> it = list.iterator(); it.hasNext(); ) {
            Map.Entry<String, Integer> entry = it.next();
            sortedMap.put(entry.getKey(), entry.getValue());
        }*/


        return sortedMap;
    }

    public static Map<String, Double> sortByValued(Map<String, Double> unsortMap) {

        // 1. Convert Map to List of Map
        List<Map.Entry<String, Double>> list =
                new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());

        // 2. Sort list with Collections.sort(), provide a custom Comparator
        //    Try switch the o1 o2 position for a different order
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> o1,
                               Map.Entry<String, Double> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // 3. Loop the sorted list and put it into a new insertion order Map LinkedHashMap
        Map<String, Double> sortedMap = new LinkedHashMap<String, Double>();
        for (Map.Entry<String, Double> entry : list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        /*
        //classic iterator example
        for (Iterator<Map.Entry<String, Integer>> it = list.iterator(); it.hasNext(); ) {
            Map.Entry<String, Integer> entry = it.next();
            sortedMap.put(entry.getKey(), entry.getValue());
        }*/


        return sortedMap;
    }

    public static Map<String, Integer> sortByValuei(Map<String, Integer> unsortMap) {

        // 1. Convert Map to List of Map
        List<Map.Entry<String, Integer>> list =
                new LinkedList<Map.Entry<String, Integer>>(unsortMap.entrySet());

        // 2. Sort list with Collections.sort(), provide a custom Comparator
        //    Try switch the o1 o2 position for a different order
        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
            public int compare(Map.Entry<String, Integer> o1,
                               Map.Entry<String, Integer> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // 3. Loop the sorted list and put it into a new insertion order Map LinkedHashMap
        Map<String, Integer> sortedMap = new LinkedHashMap<String, Integer>();
        for (Map.Entry<String, Integer> entry : list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        /*
        //classic iterator example
        for (Iterator<Map.Entry<String, Integer>> it = list.iterator(); it.hasNext(); ) {
            Map.Entry<String, Integer> entry = it.next();
            sortedMap.put(entry.getKey(), entry.getValue());
        }*/


        return sortedMap;
    }

    List<String> clearText(String text){
        List<String> words = new ArrayList<>();


        String modifText = text.toLowerCase();
        String[] simbols = ("! @ # $ % ^ & * ( ) _ - + = \\ | / . , ? > < ' ; : ] } [ { ~ ` № \"").split(" ");


        for (int i = 0; i < simbols.length; i++){
            modifText = modifText.replace(simbols[i], "");
        }

        String[] splitedText = modifText.split(" ");

        for (String word:splitedText){
            if (!blacklist.contains(word) && !word.equals("")){
                words.add(word);
            }
        }

        return words;
    }

    public DataSet toVec(Iterable<Info> infos, Integer size, Integer classes, String kategory){
        INDArray input = Nd4j.zeros(new int[] {1,size});
        INDArray output = Nd4j.zeros(new int[] {1,classes});


        Map<String, Double> map = new HashMap<>();
        Double max = 1d;
        for(Info info : infos){
            try {
                String word = info.lex().get();
                if (statistic.containsKey(word)){
                    Double value = map.get(word);
                    map.put(word, (value == null ? 1 : value + 1));
                    if (max < map.get(word)){
                        max = map.get(word);
                    }
                }
            }
            catch (Exception e){

            }
        }


        for(String key : map.keySet()){
            if(statistic.containsKey(key)){
                Double TF = map.get(key) / max;
                Double IDF = Double.valueOf(docCounter) / statistic.get(key);
                map.put(key,TF * IDF);
            }
        }

        map = sortByValued(map);
        int size_s = 0;
        for(String key : map.keySet()){
            input.put(0, size_s, tf_idf_array.get(key));
            size_s++;
            if(size_s == size){
                break;
            }
        }
        output.put(0,class_token.get(kategory),1d);

        return new DataSet(input, output);
    }

    public DataSet toVec(INDArray input, Integer classes, String raw_kategory){
        INDArray output = Nd4j.zeros(new int[] {1,classes});

        output.put(0,raw_class_token.get(raw_kategory),1d);

        return new DataSet(input, output);
    }

    public static void save(String file, String text) throws IOException {
        new File(MyStemJavaExample.resources + file).createNewFile();
        try (FileOutputStream fos = new FileOutputStream("./src/main/resources/" + file)) {
            fos.write((text).getBytes());
        } catch (Exception ignored) {

        }
    }
    public static void save(String file, String text, boolean abspath){
        if(abspath){
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write((text).getBytes());
            } catch (Exception ignored) {

            }
        }
        else{
            try (FileOutputStream fos = new FileOutputStream("./src/main/resources/" + file)) {
                fos.write((text).getBytes());
            } catch (Exception ignored) {

            }
        }

    }

    public static void saveMap(String file, Map<String,Float> map) throws JSONException, IOException {
        JSONObject text = new JSONObject();

        for (String key:map.keySet()){
            text.put(key,map.get(key));
        }
        JSONArray json;

        String fileText = read(file);

        if(fileText.equals("")){
            json = new JSONArray();
        }
        else{
            json = new JSONArray(fileText);
        }


        json.put(text);

        save(file, json.toString());
    }

    public static String read(String file){
        String text = "";

        if (new File("./src/main/resources/" + file).exists()){
            try (FileInputStream fin = new FileInputStream("./src/main/resources/" + file)) {
                byte[] bytes = new byte[fin.available()];
                fin.read(bytes);
                text = new String(bytes);
            } catch (IOException ignored) {

            }
        }

        return text;
    }

    public static String read(String file, boolean abspath){
        String text = "";

        if(abspath){
            if (new File(file).exists()){
                try (FileInputStream fin = new FileInputStream(file)) {
                    byte[] bytes = new byte[fin.available()];
                    fin.read(bytes);
                    text = new String(bytes);
                } catch (IOException ignored) {

                }
            }
        }
        else{
            if (new File("./src/main/resources/" + file).exists()){
                try (FileInputStream fin = new FileInputStream("./src/main/resources/" + file)) {
                    byte[] bytes = new byte[fin.available()];
                    fin.read(bytes);
                    text = new String(bytes);
                } catch (IOException ignored) {

                }
            }
        }


        return text;
    }

    public static void clearHistory() throws IOException {
        save("statistic.txt", "{\"counter\": 0, \"items\": {}}");
        save("history.txt", "[]");
        save("historytext.txt", "[]");
    }
}
