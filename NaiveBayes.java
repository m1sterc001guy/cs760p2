import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.Map;

public class NaiveBayes {

  private HashMap<String, Integer> attributeToIndex;
  private HashMap<Integer, String> indexToAttribute;
  private HashMap<String, Integer> priors;
  private HashMap<String, HashMap<String, HashMap<String, Integer>>> conditionals;
  

  public NaiveBayes(String trainSetName, String testSetName) {
    attributeToIndex = new HashMap<String, Integer>();
    indexToAttribute = new HashMap<Integer, String>();
    priors = new HashMap<String, Integer>();
    conditionals = new HashMap<String, HashMap<String, HashMap<String, Integer>>>();
    readInTrainingSet(trainSetName);

    System.out.println("Count: " + conditionals.get("malign_lymph").get("block_of_affere").get("no"));
  }

  private void readInTrainingSet(String trainSetName) {

    HashMap<String, ArrayList<String>> attributeToValues = 
        new HashMap<String, ArrayList<String>>();

    try {
      File file = new File(trainSetName);
      BufferedReader br = new BufferedReader(new FileReader(file));
      String line = null;
      boolean processData = false;
      int attributeIndex = 0;
      while ((line = br.readLine()) != null) {
        if (!processData) {
          int firstParenIndex = line.indexOf("{");
          if (firstParenIndex != -1) {
            String firstHalf = line.substring(0, firstParenIndex);
            String[] firstHalfTokens = firstHalf.split("\\s+");
            if (firstHalfTokens[0].toLowerCase().equals("@attribute")) {
              String attribute = firstHalfTokens[1].substring(1, firstHalfTokens[1].length() - 1);
              attributeToIndex.put(attribute, attributeIndex);
              indexToAttribute.put(attributeIndex, attribute);
              attributeIndex++;

              String secondHalf = line.substring(firstParenIndex + 1, line.length() - 1);
              String[] secondHalfTokens = secondHalf.split(",");
              for (int i = 0; i < secondHalfTokens.length; i++) {
                if (attributeToValues.containsKey(attribute)) {
                  ArrayList<String> values = attributeToValues.get(attribute);
                  values.add(secondHalfTokens[i].trim());
                } else {
                  ArrayList<String> values = new ArrayList<String>();
                  values.add(secondHalfTokens[i].trim());
                  attributeToValues.put(attribute, values);
                }
              }
            }
          } else {
            if (line.toLowerCase().equals("@data")) {
              processData = true;

              ArrayList<String> labels = attributeToValues.get("class"); 
              for (String label : labels) {
                if (!conditionals.containsKey(label)) {
                  conditionals.put(label, new HashMap<String, HashMap<String, Integer>>());
                }

                for (Map.Entry<String, ArrayList<String>> attribute : attributeToValues.entrySet()) {
                  String attributeKey = attribute.getKey();
                  if (!attributeKey.equals("class")) {
                    HashMap<String, HashMap<String, Integer>> counts = conditionals.get(label);
                    counts.put(attributeKey, new HashMap<String, Integer>());

                    ArrayList<String> values = attribute.getValue();
                    HashMap<String, Integer> initCounts = conditionals.get(label).get(attributeKey);
                    for (String value : values) {
                      initCounts.put(value, 0);
                    }
                  }
                }
              }
            }
          }
        } else {
          //System.out.println("Processing data");
          // process the data
          String[] tokens = line.split(",");
          String label = tokens[tokens.length - 1];
          if (priors.containsKey(label)) {
            Integer labelCount = priors.get(label);
            labelCount++;
            //System.out.println("labelCount: " + labelCount);
            priors.put(label, labelCount);
          } else {
            //System.out.println("labelCount: " + 1);
            priors.put(label, 1);
          }

          for (int i = 0; i < tokens.length - 1; i++) {
            String attributeName = indexToAttribute.get(i);
            String attributeValue = tokens[i];
            HashMap<String, Integer> counts = conditionals.get(label).get(attributeName);
            Integer count = counts.get(attributeValue);
            count++;
            counts.put(attributeValue, count);
          }
        }
      }
      br.close();
    } catch (FileNotFoundException e) {
      Bayes.exitWithError("FileNotFoundException occurred when reading training set", e);
    } catch (IOException e) {
      Bayes.exitWithError("IOException occurred when reading training set", e);
    }
  }

}
