public class Main {

	public static void main(String [] args) {
		String absPath = "/SETME";
		NaiveBayesClassifier classifier = new NaiveBayesClassifier(absPath+"train.txt");
		classifier.learn();
		classifier.test(absPath+"test.txt");
	}
}
