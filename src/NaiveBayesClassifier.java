import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class NaiveBayesClassifier {

	class Triplet {
		
		int i,j,k;
		
		public Triplet(int i, int j, int k) {
			this.i = i;
			this.j = j;
			this.k = k;
		}
		
		@Override
	    public boolean equals(Object object) {
	        if (object == null)   return false;
	        if (this == object)   return true;
	        if (!(object instanceof Triplet))   return false;
	        final Triplet other = (Triplet) object;
	        if (this.i == other.i && this.j == other.j && this.k == other.k) {
	        	return true;
	        }
	        return false;
		}
		
		 @Override
		    public int hashCode() {
		        int hash = 17;
		        hash = hash * 13 + this.i;
		        hash = hash * 13 + this.j;
		        hash = hash * 13 + this.k;
		        return hash;
		    }
	}
	
	ArrayList<Document> docs;
	TreeSet<Integer> allFeatures;
	HashMap<Triplet, Double> thetas;
	HashMap<Integer, Double> pis;
	int [] classKs;
	HashMap<Integer, Integer> classCounts;
	double smoothingParameter = 1;
	
	public NaiveBayesClassifier(String pathToTrainingData) {
		String sCurrentLine;
		BufferedReader br;
		HashSet<Integer> classSet;
		
		try {
			allFeatures = new TreeSet<Integer>();
			docs = new ArrayList<Document>();
			thetas = new HashMap<Triplet,Double>();
			pis = new HashMap<Integer, Double>();
			classSet = new HashSet<Integer>();
			classCounts = new HashMap<Integer, Integer>();
			br = new BufferedReader(new FileReader(pathToTrainingData));
			while ((sCurrentLine = br.readLine()) != null) {
				ArrayList<Integer> classes = new ArrayList<Integer>();
				HashMap<Integer, Integer> features = new HashMap<Integer, Integer>();
				
				StringTokenizer st = new StringTokenizer(sCurrentLine);
				boolean endOfClassLable = false;
			    while (st.hasMoreTokens()) {
			    	 String token = st.nextToken();
			    	 if (token.contains(":")) {
			    		 endOfClassLable = true;
			    	 }
			    	 if (endOfClassLable) {
			    		 String [] featureStr = token.split(":");
			    		 Integer wordIndex = Integer.parseInt(featureStr[0]);
			    		 allFeatures.add(wordIndex);
			    		 features.put(wordIndex, Integer.parseInt(featureStr[1]));
			    		 
			    	 } else {
			    		 classes.add(Integer.parseInt(token.replace(",","")));
			    	 }
			    }
			    
			    Document doc = new Document(classes, features);
			    docs.add(doc);
			    classSet.addAll(doc.getClasses());
			}
			
			for (Integer classNumber : classSet) {
				int count = 0;
				for (int z = 0; z < docs.size(); z++) {
					if (docs.get(z).containsClass(classNumber)) {
						count++;
					}
				}
				classCounts.put(classNumber, count);
			}
		} catch(Exception e) {
			docs = null;
			classSet = null;
			e.printStackTrace();
		}
	}
	
	//store theta value
	private void storeTheta(int i, int xij, int k, double thetaVal) {
		Triplet triplet = new Triplet(i,xij,k);
		thetas.put(triplet, thetaVal);
	}
	
	

	public double getPi(int classNumber) {
		if (pis.containsKey(classNumber)) {
			return pis.get(classNumber);
		}
		return 0;
	}
	
	
	public double getTheta(int i, int xij, int k) {
		Triplet triplet = new Triplet(i,xij,k);
		if (thetas.containsKey(triplet)) {
			return thetas.get(triplet);
		}
		
		double pseduoResult = smoothingParameter/(classCounts.get(k));
		return pseduoResult;
	}
	
	
	
	public void learn() {
		// calcuate pis
		
		for (Integer classNumber : classCounts.keySet()) {
			int numerator = 0;
			int denominator = docs.size();
			double smoothingParameter = 1;
			
			for (int z = 0; z <docs.size(); z++) {
				if (docs.get(z).containsClass(classNumber)) {
					numerator++;
				}
			}
			
			double pi = (1.0 * numerator + smoothingParameter) / (denominator + smoothingParameter * docs.size());
			pis.put(classNumber, pi);
		}
		
		// calculate thetas
		
		System.out.println("size of features "+ allFeatures.size());
		System.out.println("docSize "+docs.size());
		System.out.println("classCounts "+ classCounts.size());
		for (Integer wordIndex : allFeatures) {
			for (int j = 0; j < docs.size(); j++) {
				for (Integer classNumber : classCounts.keySet()) {
					int numerator = 0;
					int denominator = 0;
					
					// calculate the number of intersection Xij and Yk documents
					int Xij = docs.get(j).getFeatureWeight(wordIndex);
					for (int z = 0; z < docs.size(); z++) {
						if (docs.get(z).getFeatureWeight(wordIndex) == Xij && docs.get(z).containsClass(classNumber)) {
							numerator++;
						}
					}
	
					// calculate the number of class Yk documents
					for (int z = 0; z < docs.size(); z++) {
						if (docs.get(z).containsClass(classNumber)) {
							denominator += 1;
						}
					}
					
					// calculate the number of distinct values Xi can take on
					HashSet<Integer> XiValues = new HashSet<Integer>();
					for (int z = 0; z < docs.size(); z++) {
						XiValues.add(docs.get(z).getFeatureWeight(wordIndex));
					}
					
					int J = XiValues.size();
					
					double thetaVal = (numerator + smoothingParameter) / (denominator + smoothingParameter * J);
					storeTheta(wordIndex, docs.get(j).getFeatureWeight(wordIndex), classNumber, thetaVal);
				}
			}
		}
	}
	
	
	private double calculatePosteriorProb(int knumber, HashMap<Integer, Integer> X) {
		
		// calculate product of probabilities P(Xi| Y = y[classIndex]) 0 <= i <= maximumWordIndex
		double PXi_Yk_Product = 1.0;
		for (Integer wordIndex : allFeatures) {
			int xij;
			if (X.containsKey(wordIndex)) {
				xij = X.get(wordIndex);
			} else {
				xij = 0;
			}
			PXi_Yk_Product *= getTheta(wordIndex, xij, knumber);
		}
		
		double denominator = 0;
		for (Integer classNumber : classCounts.keySet()) {
			double PXi_Yj_Product = 1.0;
			for (Integer wordIndex : allFeatures) {
				int xij;
				if (X.containsKey(wordIndex)) {
					xij = X.get(wordIndex);
				} else {
					xij = 0;
				}
				PXi_Yj_Product *= getTheta(wordIndex, xij, classNumber);
			}
			denominator += getPi(classNumber) * PXi_Yj_Product;
		}
		
		return getPi(knumber) * PXi_Yk_Product/ denominator;
	}
	
	
	public void test(String pathToTestData) {
		String sCurrentLine;
		BufferedReader br;
		ArrayList<ArrayList<Integer>> classificationResult = new ArrayList<ArrayList<Integer>>();
		
		try {
			br = new BufferedReader(new FileReader(pathToTestData));
			while ((sCurrentLine = br.readLine()) != null) {
				HashMap<Integer,Double> scores = new HashMap<Integer,Double>();
				HashMap<Integer, Integer> features = new HashMap<Integer, Integer>();
				
				StringTokenizer st = new StringTokenizer(sCurrentLine);
			    while (st.hasMoreTokens()) {
			    	 String token = st.nextToken();
			    	 String [] featureStr = token.split(":");
			    	 features.put(Integer.parseInt(featureStr[0]), Integer.parseInt(featureStr[1]));
			    }
			    
			    double maxScore = 0;
			    for (Integer classNumber : classCounts.keySet()) {
			    	double score = calculatePosteriorProb(classNumber, features);
			    	if (score > 1) {
			    		throw new RuntimeException("> 1");
			    	}
			    	scores.put(classNumber, score);
			    	if (score > maxScore) {
			    		maxScore = score;
			    	}
			    }
			    ArrayList<Integer> result = new ArrayList<Integer>();
			    for (Integer classNumber : classCounts.keySet()) {
			    	if (Math.abs(scores.get(classNumber) - maxScore) < 0.05) {
			    		result.add(classNumber);
			    	}
			    }	    
			    classificationResult.add(result);		
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("classification result:");
		for (int i = 0; i < classificationResult.size(); i++) {
			ArrayList<Integer> list = classificationResult.get(i);
			for (int j = 0; j < list.size(); j++) {
				System.out.print(list.get(j));
				if (list.size() > 1 && j < list.size() - 1)
				System.out.print(" ");
			}
			if (classificationResult.size() > 0 && i < classificationResult.size() -1)
				System.out.println("");
		}
	}
	
}
