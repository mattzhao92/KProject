import java.util.HashMap;
import java.util.*;


public class Document {

	ArrayList<Integer> classes;
	HashMap<Integer, Integer> features;
	
	
	public Document(ArrayList<Integer> classes, HashMap<Integer, Integer> features) {
		this.classes = classes;
		this.features = features;
	}
	
	public int getFeatureWeight(int featureIndex) {
		if (features.containsKey(featureIndex)) {
			return features.get(featureIndex);
		} 
		return 0;
	}

	public ArrayList<Integer> getClasses() {
		return classes;
	}
	
	public boolean containsClass(int classIndex) {
		return classes.contains(classIndex);
	}
}
