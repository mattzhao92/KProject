package tree;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;


import mtree.DistanceMetric;
import mtree.MTreeMap;
import mtree.Result;



public class sdf {

	static int num_nearest_neighbors = 20;
	static int num_row_X_test = 100;
	static int num_column_X_test = 500;
	static int num_threads = 3;


	static MTreeMap<Double[], String []> mTree = new MTreeMap<Double[], String []>(new PointMetric());
	static HashMap<String, ArrayList<String>> child_to_parents = new HashMap<String, ArrayList<String>>(); 
	static HashMap<String, ArrayList<String>> parent_to_children = new HashMap<String, ArrayList<String>>(); 

	static Double[][] X_test;


	/** This DistanceMetric computes the distance between to (x,y) points. */
	static class PointMetric implements DistanceMetric<Double[]> {

		@Override
		public double distanceBtw(Double[] item1, Double[] item2) {
			return	( ( 1- cosine_similarity(item1,item2) ));

		}
	}

	private static double cosine_similarity(Double[] vec1, Double[] vec2) { 
		double dp = dot_product(vec1,vec2); 
		double magnitudeA = find_magnitude(vec1); 
		double magnitudeB = find_magnitude(vec2); 
		return (dp)/(magnitudeA*magnitudeB); 
	} 

	private static double find_magnitude(Double[] vec) { 
		double sum_mag=0; 
		for(int i=0;i<vec.length;i++) 
		{ 
			sum_mag = sum_mag + vec[i]*vec[i]; 
		} 
		return Math.sqrt(sum_mag); 
	} 

	private static double dot_product(Double[] vec1, Double[] vec2) { 
		double sum=0; 
		for(int i=0;i<vec1.length;i++) 
		{ 
			sum = sum + vec1[i]*vec2[i]; 
		} 
		return sum; 
	} 


	/*
	 * loading training data and training labels
	 */
	private static void load_trainingData() throws Exception{
		String line1;
		String line2;

		BufferedReader data_br = new BufferedReader(new FileReader("X_train_reduced.txt"));
		BufferedReader label_br = new BufferedReader(new FileReader("X_train_labels.txt"));

		System.out.println("loading training data and training labels");
		while ((line1 = data_br.readLine()) != null) {
			line2 = label_br.readLine();
			assert line2 != null;
			String [] strArr = line1.split("\\s+");
			assert strArr.length == num_column_X_test;
			Double [] doubleArr = new Double[strArr.length];
			for (int i = 0; i < strArr.length; i++) {
				doubleArr[i] = Double.parseDouble(strArr[i]);
			}
			String [] classLabels = line2.split("\\s+");
			mTree.put(doubleArr, classLabels);
		}
		data_br.close();
		label_br.close();
	}

	/*
	 * loading testing data
	 */
	private static void load_testingData() throws Exception{
		String line1;

		BufferedReader data_br = new BufferedReader(new FileReader("X_test_reduced.txt"));

		X_test = new Double[num_row_X_test][num_column_X_test];
		int i = 0;
		System.out.println("loading testing data");
		while ((line1 = data_br.readLine()) != null) {
			String [] strArr = line1.split("\\s+");
			for (int j = 0; j < strArr.length; j++) {
				X_test[i][j] = Double.parseDouble(strArr[j]);
			}
			i++;
		}

		assert i == num_row_X_test;
		data_br.close();
	}

	/*
	 * loading tree hierarchy
	 */
	private static void load_treeHierarchy() throws Exception{
		String line1;

		BufferedReader data_br = new BufferedReader(new FileReader("two_level_hierarchy.txt"));

		System.out.println("loading tree hierarchy");
		while ((line1 = data_br.readLine()) != null) {
			String [] strArr = line1.split("\\s+");
			String parent = strArr[0];
			String child = strArr[1];

			if (!child_to_parents.containsKey(child)) {
				child_to_parents.put(child, new ArrayList<String>());
			}
			child_to_parents.get(child).add(parent);

			if (!parent_to_children.containsKey(parent)) {
				parent_to_children.put(parent, new ArrayList<String>());
			}
			parent_to_children.get(parent).add(child);

		}
		data_br.close();
	}

	private static List<Map.Entry<String, Double>> sortScoreMap(HashMap<String, Double> unsortMap) {

		List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());

		// sort list based on comparator
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			@Override
			public int compare(Entry<String, Double> o1,
					Entry<String, Double> o2) {
				return ((Map.Entry<String, Double>) (o1)).getValue()
						.compareTo(((Map.Entry<String, Double>) (o2)).getValue());
			}
		});

		return list;
	}


	static void classify(int workerId, int start_index, int end_index) throws Exception {

		System.out.println(String.format("Worker[%d] start classifying in range[%d, %d] ", workerId, start_index, end_index));
		int one_percent_load = end_index - start_index + 1;

		File file = new File("output"+workerId);
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);

		Collection<Result<Double[], String[]>> results;

		for (int i = start_index; i <= end_index; i++) {

			if ((i-start_index + 1) % one_percent_load == 0) {
				System.out.println(String.format("Worker[%d]  %d", workerId, (i-start_index+1)/ one_percent_load -1) +"% done");
			}

			results = mTree.getNClosest(X_test[i], num_nearest_neighbors);

			// get all labels in results and give them a score of Double.MAX_VALUE
			HashMap<String, ArrayList<Double>> label_to_dists = new HashMap<String, ArrayList<Double>>();


			HashMap<String, Double> label_to_score = new HashMap<String, Double>();
			for (Result<Double[], String[]> result : results) {
				for (String label : result.value()) {
					if (!label_to_dists.containsKey(label)) {
						label_to_dists.put(label, new ArrayList<Double>());
					}
					label_to_dists.get(label).add(result.distance());
					label_to_score.put(label, Double.MAX_VALUE);
				}
			}

			// calculating scores

			for (String label : label_to_score.keySet()) {
				double minimum_distance_knn = 2.0;
				double average_distance_knn_in_tree = 0;



				int num_nodes = 0;

				Set<String> visited = new HashSet<String>();
				for (String myParent : child_to_parents.get(label)) {
					for (String myPeer : parent_to_children.get(myParent)) {
						if (label_to_dists.containsKey(myPeer) && visited.contains(myPeer) == false) {
							num_nodes += 1;
							visited.add(myPeer);
							double total_distance = 0.0;

							ArrayList<Double> dists = label_to_dists.get(myPeer);
							for (Double dist : dists) {
								total_distance += dist;
							}

							average_distance_knn_in_tree += total_distance/dists.size();
						}
					}
				}

				if (label_to_dists.containsKey(label)) {
					for (Double dist : label_to_dists.get(label)) {
						if (dist < minimum_distance_knn) {
							minimum_distance_knn = dist;
						}
					}
					average_distance_knn_in_tree = average_distance_knn_in_tree / num_nodes;
				} else {
					average_distance_knn_in_tree = 2.0;
				}

				double theta0 = 0.7;
				double theta1 = 0.3;
				double score = minimum_distance_knn * theta0 + average_distance_knn_in_tree * theta1;
				label_to_score.put(label, score);
			}



			// get the labels with the smallest scores
			List<Map.Entry<String, Double>> sortedList = sortScoreMap(label_to_score);
			double alpha = 1.02;
			double max_num_labels_per_line = 5;

			ArrayList<String> classified_labels = new ArrayList<String>();
			Map.Entry<String, Double> firstEle = sortedList.get(0);
			for (int j = 0; j < sortedList.size(); j++) {
				Map.Entry<String, Double> currEle = sortedList.get(j);

				if (currEle.getValue() > firstEle.getValue() * alpha || classified_labels.size() > max_num_labels_per_line) {
					break;
				}
				classified_labels.add(currEle.getKey());
			}

			for (int z = 0; z <  classified_labels.size(); z++) {
				if (classified_labels.size() == 1) {
					bw.write(classified_labels.get(z)+"\n");
				} else {
					if (z != classified_labels.size() - 1) {
						bw.write(classified_labels.get(z)+" ");
					} else {
						bw.write(classified_labels.get(z)+"\n");
					}
				}
			}

		}

		System.out.println(String.format("Worker[%d]", workerId) +" 100% done");

		bw.close();
	}


	static class ClassifierWorker extends Thread {

		int workerId = 0;
		Range range = null;

		public ClassifierWorker(int workerIdIn, Range rangeIn) {
			workerId = workerIdIn;
			range = rangeIn;
		}

		public void run() {
			try {
				classify(workerId, range.start, range.end);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	static class Range {
		private int start;
		private int end;

		public Range(int startIn, int endIn) {
			this.start = startIn;
			this.end = endIn;
		}
	}

	private static ArrayList<Range> getChunks(int end, int n_chunks) {
		int chunk_size = (end+1) / n_chunks;

		ArrayList<Range> retVal = new ArrayList<Range>();

		int i = 0;
		while (i < end) {
			retVal.add(new Range(i, Math.min(i + chunk_size -1, end)));
			i += chunk_size;
		}

		Range last = retVal.get(retVal.size() - 1);
		last.end = end;

		return retVal;
	}



	public static void merge() throws Exception {
		System.out.println("merging output files");
		File file = new File("output");


		// if file doesnt exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);

		bw.write("Id,Predicated \n");
		int count = 1;
		for (int i = 0; i < num_threads; i++) {
			BufferedReader br = new BufferedReader(new FileReader("output"+i));
			String line1;
			while ((line1 = br.readLine()) != null) {
				bw.write(""+(count++)+","+line1+"\n");
			}
			br.close();
		}
		bw.close();

		System.out.println("Done");		
	}

	public static void main(String[] args) throws Exception {

		if (args.length == 1){
			num_threads = Integer.parseInt(args[0]);
		}

		load_trainingData();
		load_testingData();
		load_treeHierarchy();


		ArrayList<Range> ranges = getChunks(num_row_X_test-1, num_threads);
		ArrayList<ClassifierWorker> workers = new ArrayList<ClassifierWorker>();

		for (int i = 0; i < num_threads; i++) {
			workers.add(new ClassifierWorker(i, ranges.get(i)));
		}

		for (int i = 0; i < num_threads; i++) {
			workers.get(i).start();
		}

		for (int i = 0; i < num_threads; i++) {
			workers.get(i).join();
		}

		merge();

	}
}
