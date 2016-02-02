

public class Bayes {

  public static void main(String[] args) {
    if (args.length != 3) {
      exitWithError("Error. Wrong number of arguments. Quitting...");
    }
    String trainSetFileName = args[0];
    String testSetFileName = args[1];
    String algo = args[2];

    if (algo.toLowerCase().equals("n")) {
      NaiveBayes naiveBayes = new NaiveBayes(trainSetFileName, testSetFileName);
    } else if (algo.toLowerCase().equals("t")) {
      System.out.println("Implement TAN here");
    } else {
      exitWithError("Error. Invalid algorithm specified. Quitting...");
    }
  }

  public static void exitWithError(String errorMessage) {
    System.err.println(errorMessage);
    System.exit(-1);
  }

  public static void exitWithError(String errorMessage, Exception e) {
    System.err.println(errorMessage);
    e.printStackTrace();
    System.exit(-1);
  }

}
